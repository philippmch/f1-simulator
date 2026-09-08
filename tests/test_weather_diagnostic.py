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
