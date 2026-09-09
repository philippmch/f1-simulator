"""Observed calibration must use race-window evidence, including message-only flags."""

import importlib.util
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


def test_model_comparison_is_reproducible_machine_readable_and_labeled():
    path = Path(__file__).resolve().parents[1] / "examples" / "check_weather_calibration.py"
    command = [sys.executable, str(path), "--simulations", "1", "--seed", "17",
               "--race-engine", "both"]
    first = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    second = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
    summaries = json.loads(first.stdout)
    assert summaries == json.loads(second.stdout)
    assert len(summaries) == 8
    assert {row["race_engine"] for row in summaries} == {"standard", "chronological"}
    for row in summaries:
        assert row["seed"] == 17
        assert row["simulations"] == 1
        assert 0 <= row["lapped_finishers"] <= row["finishing_cars"] <= 22
        assert row["mean_winner_seconds"] > 0
        assert row["mean_pit_stops_per_entrant"] >= 0
        assert row["races_with_winner"] == 1
        if row["race_engine"] == "standard":
            assert row["lapped_finishers"] == 0


@pytest.mark.parametrize("has_finish", [True, False])
@pytest.mark.parametrize("include_stints", [True, False])
def test_observed_race_window_and_completion(monkeypatch, has_finish, include_stints):
    path = Path(__file__).resolve().parents[1] / "examples" / "check_weather_calibration.py"
    spec = importlib.util.spec_from_file_location("weather_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 6, tzinfo=timezone.utc)

    rows = {
        "sessions?year=2026&session_name=Race": [{
            "year": 2026, "session_name": "Race", "session_key": 1, "location": "Test",
            "date_start": "2026-08-01T12:00:00+00:00",
            "date_end": "2026-08-01T15:00:00+00:00",
        }],
        "race_control?session_key=1": [
            {"date": "2026-08-01T12:15:00+00:00", "flag": None,
             "message": "RED FLAG - RACE SUSPENDED"},
        ] + ([{"date": "2026-08-01T14:00:00+00:00", "flag": "CHEQUERED"}]
             if has_finish else []),
        "weather?session_key=1": [
            {"date": "2026-08-01T11:59:00+00:00", "rainfall": 1},
            {"date": "2026-08-01T12:01:00+00:00", "rainfall": 0},
            {"date": "2026-08-01T14:01:00+00:00", "rainfall": 1},
        ],
        "stints?session_key=1": [{
            "session_key": 1, "driver_number": 1, "stint_number": 1,
            "compound": "INTERMEDIATE", "lap_start": 1, "lap_end": 6,
        }],
    }
    for endpoint in ("race_control", "weather"):
        for record in rows[f"{endpoint}?session_key=1"]:
            record["session_key"] = 1
    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "urlopen", lambda url, timeout: io.BytesIO(
        json.dumps(rows[url.split("/v1/")[1]]).encode()
    ))

    if not has_finish:
        with pytest.raises(RuntimeError, match="No completed-race evidence"):
            module.observed_summary(include_stints=include_stints)
    else:
        observed = module.observed_summary(include_stints=include_stints)
        if include_stints:
            stint = observed["races"][0].pop("rain_stints")["stints"][0]
            assert stint["reported_laps"] == 6
            assert stint["tyre_age_at_end"] is None
            assert stint["end_reason"] == "unknown"
        assert observed["races"] == [{
            "session": 1, "venue": "Test", "rain_observed": False, "red_flag": True,
        }]


@pytest.fixture
def observed_feed(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "examples" / "check_weather_calibration.py"
    spec = importlib.util.spec_from_file_location("observed_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 9, tzinfo=timezone.utc)

    rows = {
        "sessions": [{"year": 2026, "session_name": "Race", "session_key": 1,
                      "location": "Test", "date_start": "2026-08-01T12:00:00Z",
                      "date_end": "2026-08-01T15:00:00Z"}],
        "race_control": [{"session_key": 1, "date": "2026-08-01T14:00:00Z",
                          "flag": "CHEQUERED", "message": None}],
        "weather": [{"session_key": 1, "date": "2026-08-01T13:00:00Z", "rainfall": 0}],
        "session_result": [{"session_key": 1, "driver_number": 7, "number_of_laps": 8,
                            "dnf": False, "dns": False, "dsq": False}],
        "stints": [{"session_key": 1, "driver_number": 7, "stint_number": 1,
                    "compound": "INTERMEDIATE", "lap_start": 1, "lap_end": 8}],
        "pit": [],
    }
    calls = []

    def fetch(url, timeout):
        path = url.split("/v1/")[1]
        calls.append(path)
        return io.BytesIO(json.dumps(rows[path.split("?")[0]]).encode())

    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "urlopen", fetch)
    return module, rows, calls


def test_strategy_and_rain_evidence_share_one_stint_request(observed_feed):
    module, _, calls = observed_feed
    result = module.observed_summary(include_stints=True, include_strategy=True)
    race = result["races"][0]
    assert race["rain_stints"]["stints"][0]["reported_laps"] == 8
    assert race["strategy"]["session_key"] == 1
    assert len(race["strategy"]["drivers"]) == 1
    assert calls.count("stints?session_key=1") == 1
    assert "session_result?session_key=1" in calls
    assert "pit?session_key=1" in calls


@pytest.mark.parametrize("endpoint", ["race_control", "weather", "stints",
                                      "session_result", "pit"])
@pytest.mark.parametrize("session_key", [2, None, True])
def test_all_observed_feeds_reject_wrong_session(observed_feed, endpoint, session_key):
    module, rows, _ = observed_feed
    if not rows[endpoint]:
        rows[endpoint] = [{}]
    rows[endpoint][0]["session_key"] = session_key
    with pytest.raises(RuntimeError, match="unexpected session"):
        module.observed_summary(include_strategy=True)


@pytest.mark.parametrize("value", [None, "0", "1", 2, -1, 0.5, [], {}])
def test_rainfall_cannot_silently_become_truthy_evidence(observed_feed, value):
    module, rows, _ = observed_feed
    rows["weather"][0]["rainfall"] = value
    with pytest.raises(RuntimeError, match="binary"):
        module.observed_summary()


@pytest.mark.parametrize("value", [None, "invalid", "2026-08-01T13:00:00"])
def test_observed_dates_require_timezone(observed_feed, value):
    module, rows, _ = observed_feed
    rows["weather"][0]["date"] = value
    with pytest.raises(RuntimeError, match="timestamp"):
        module.observed_summary()


@pytest.mark.parametrize("records", [{}, [None], [1], ["row"]])
def test_malformed_feed_fails_explicitly(observed_feed, records):
    module, rows, _ = observed_feed
    rows["weather"] = records
    with pytest.raises(RuntimeError, match="Expected OpenF1 records"):
        module.observed_summary()


def test_finish_before_start_is_not_completed_race_evidence(observed_feed):
    module, rows, _ = observed_feed
    rows["race_control"][0]["date"] = "2026-08-01T11:00:00Z"
    with pytest.raises(RuntimeError, match="No completed-race evidence"):
        module.observed_summary()


def test_repeated_session_is_not_counted_twice_and_conflicts_fail(observed_feed):
    module, rows, calls = observed_feed
    rows["sessions"].append(dict(rows["sessions"][0]))
    assert len(module.observed_summary()["races"]) == 1
    assert calls.count("weather?session_key=1") == 1
    rows["sessions"][1]["location"] = "Other"
    with pytest.raises(RuntimeError, match="Conflicting session"):
        module.observed_summary()


def test_strategy_option_implies_observations_and_single_json(observed_feed, monkeypatch, capsys):
    module, _, calls = observed_feed
    monkeypatch.setattr(sys, "argv", ["check_weather_calibration", "--observed-strategy",
                                      "--simulations", "1"])
    monkeypatch.setattr(module, "model_summary", lambda *args: [])
    module.main()
    output = json.loads(capsys.readouterr().out)
    assert output["observed"]["races"][0]["strategy"]["session_key"] == 1
    assert "rain_stints" not in output["observed"]["races"][0]
    assert output["model"] == []
    assert len(calls) == 6


def test_failed_collection_prints_no_partial_output(observed_feed, monkeypatch, capsys):
    module, rows, _ = observed_feed
    rows["session_result"] = []
    monkeypatch.setattr(sys, "argv", ["check_weather_calibration", "--observed-strategy"])
    with pytest.raises(ValueError):
        module.main()
    assert capsys.readouterr().out == ""
