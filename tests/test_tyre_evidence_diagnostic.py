"""Archive evidence remains opt-in, current-season, and session-bound."""

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture
def diagnostic(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "check_tyre_evidence.py"
    spec = importlib.util.spec_from_file_location("tyre_evidence_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "normalize_timing_evidence",
                        lambda feeds: {"laps": [], "feeds_received": sorted(feeds)})
    return module


def provider(module, *, path="2026/2026-05-24_Test/2026-05-24_Race/", status="Finalised",
             info_path=None, ambiguous=False):
    meeting = {"Name": "Test Grand Prix", "Sessions": [{"Name": "Race", "Path": path}]}
    meetings = [meeting]
    if ambiguous:
        meetings.append({"Name": "Other Test Grand Prix", "Sessions": []})
    info = {"Path": path if info_path is None else info_path, "Type": "Race", "Name": "Race",
            "SessionStatus": status, "ArchiveStatus": {"Status": "Complete"}, "Key": 12,
            "StartDate": "2026-05-24T16:00:00"}
    calls = []

    def get(url, *, headers, timeout):
        calls.append(url)
        assert 0 < timeout <= 20
        assert headers["User-Agent"]
        if url == module.BASE_URL + "2026/Index.json":
            return json.dumps({"Meetings": meetings}).encode()
        if url == module.BASE_URL + path + "SessionInfo.json":
            return json.dumps(info).encode()
        assert url in {module.BASE_URL + path + name + ".jsonStream" for name in module.FEEDS}
        return b"00:00:01.000{}\n"

    return get, calls


def test_fetches_one_completed_current_session_in_memory(diagnostic):
    getter, calls = provider(diagnostic)
    report = diagnostic.collect_archive_evidence(
        "test", now=datetime(2026, 9, 18, tzinfo=timezone.utc), http_get=getter,
    )
    assert report["season"] == 2026 and report["session_key"] == 12
    assert report["evidence"]["feeds_received"] == sorted(diagnostic.FEEDS)
    assert len(calls) == 7
    assert "No thermal or wear coefficients fitted" in report["calibration"]


@pytest.mark.parametrize("path", [
    "2025/2025-05-24_Test/2025-05-24_Race/", "https://other.test/race/",
    "2026/../race/", "2026/Test/Race/?x=1", "/2026/Test/Race/",
])
def test_rejects_foreign_or_unsafe_archive_paths_before_fetch(diagnostic, path):
    getter, calls = provider(diagnostic, path=path)
    with pytest.raises(RuntimeError, match="invalid current-season"):
        diagnostic.collect_archive_evidence(
            "test", now=datetime(2026, 9, 18, tzinfo=timezone.utc), http_get=getter,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("options", [{"status": "Started"}, {"info_path": "wrong"}])
def test_rejects_incomplete_or_mismatched_session_before_loading_feeds(diagnostic, options):
    getter, calls = provider(diagnostic, **options)
    with pytest.raises(RuntimeError, match="incomplete.*identity"):
        diagnostic.collect_archive_evidence(
            "test", now=datetime(2026, 9, 18, tzinfo=timezone.utc), http_get=getter,
        )
    assert len(calls) == 2


def test_ambiguous_meeting_requires_a_unique_name(diagnostic):
    getter, calls = provider(diagnostic, ambiguous=True)
    with pytest.raises(ValueError, match="uniquely match"):
        diagnostic.collect_archive_evidence(
            "test", now=datetime(2026, 9, 18, tzinfo=timezone.utc), http_get=getter,
        )
    assert len(calls) == 1


def test_slow_custom_getter_cannot_silently_exceed_shared_budget(diagnostic, monkeypatch):
    getter, calls = provider(diagnostic)
    times = iter([0, 0, 61])
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: next(times))
    with pytest.raises(RuntimeError, match="60-second fetch budget"):
        diagnostic.collect_archive_evidence(
            "test", now=datetime(2026, 9, 18, tzinfo=timezone.utc), http_get=getter,
        )
    assert len(calls) == 1
