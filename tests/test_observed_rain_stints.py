"""Reported stint lengths describe exposure, not fitted wear or stop causes."""

import importlib.util
import io
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError

import pytest


@pytest.fixture
def diagnostic():
    path = Path(__file__).resolve().parents[1] / "examples" / "check_weather_calibration.py"
    spec = importlib.util.spec_from_file_location("stint_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def summarize(diagnostic):
    return diagnostic.rain_stint_summary


def row(**updates):
    return dict(session_key=10, driver_number=7, stint_number=2, compound="INTERMEDIATE",
                lap_start=12, lap_end=30, tyre_age_at_start=3) | updates


def test_used_set_age_inclusive_laps_and_unknown_termination(summarize):
    result = summarize([row()], 10)
    assert result == {"source_records": 1, "excluded_records": {}, "stints": [{
        "driver_number": 7, "stint_number": 2, "compound": "INTERMEDIATE",
        "lap_start": 12, "lap_end": 30, "reported_laps": 19,
        "tyre_age_at_start": 3, "tyre_age_at_end": 22, "end_reason": "unknown",
    }]}


@pytest.mark.parametrize("age", [None, -1, True, "3"])
def test_unknown_initial_age_is_not_fabricated_as_fresh(summarize, age):
    stint = summarize([row(tyre_age_at_start=age)], 10)["stints"][0]
    assert stint["reported_laps"] == 19
    assert stint["tyre_age_at_start"] is None
    assert stint["tyre_age_at_end"] is None


def test_dry_empty_rain_sample_differs_from_missing_evidence(summarize):
    assert summarize([row(compound="HARD")], 10)["stints"] == []
    with pytest.raises(RuntimeError, match="No stint evidence"):
        summarize([], 10)


def test_bad_ranges_and_duplicate_rows_are_reported_not_counted_twice(summarize):
    rows = [row(), row(), row(stint_number=3, lap_end=None),
            row(stint_number=4, lap_end=11), row(stint_number=5, compound=None)]
    result = summarize(rows, 10)
    assert len(result["stints"]) == 1
    assert result["excluded_records"] == {
        "duplicate_record": 1, "incomplete_lap_range": 2, "unknown_compound": 1,
    }


def test_conflicting_or_wrong_session_evidence_fails(summarize):
    with pytest.raises(RuntimeError, match="Conflicting"):
        summarize([row(), row(lap_end=31)], 10)
    with pytest.raises(RuntimeError, match="unexpected session"):
        summarize([row(session_key=11)], 10)


@pytest.mark.parametrize("retry", ["7", "Tue, 08 Sep 2026 00:00:07 GMT"])
def test_observed_rate_limit_honors_retry_after(diagnostic, monkeypatch, retry):
    sleeps, calls = [], []
    monkeypatch.setattr(diagnostic.time, "sleep", sleeps.append)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 8, tzinfo=timezone.utc)

    monkeypatch.setattr(diagnostic, "datetime", Clock)

    def fetch(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise HTTPError(url, 429, "Rate limited", {"Retry-After": retry}, None)
        return io.BytesIO(b"[]")

    monkeypatch.setattr(diagnostic, "urlopen", fetch)
    assert diagnostic.observed_summary()["races"] == []
    assert len(calls) == 2
    assert calls[0] == calls[1]
    assert sleeps == [2.1, 7, 2.1]


@pytest.mark.parametrize("retry", ["600", "invalid"])
def test_observed_rate_limit_retry_is_bounded(diagnostic, monkeypatch, retry):
    sleeps, calls = [], []
    monkeypatch.setattr(diagnostic.time, "sleep", sleeps.append)

    def fetch(url, timeout):
        calls.append(url)
        raise HTTPError(url, 429, "Rate limited", {"Retry-After": retry}, None)

    monkeypatch.setattr(diagnostic, "urlopen", fetch)
    with pytest.raises(RuntimeError):
        diagnostic.observed_summary()
    assert len(calls) == (1 if retry == "600" else 3)
    assert all(delay <= 60 for delay in sleeps)
