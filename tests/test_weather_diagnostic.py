"""Observed calibration must use race-window evidence, including message-only flags."""

import importlib.util
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.mark.parametrize("has_finish", [True, False])
def test_observed_race_window_and_completion(monkeypatch, has_finish):
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
    }
    monkeypatch.setattr(module, "datetime", Clock)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "urlopen", lambda url, timeout: io.BytesIO(
        json.dumps(rows[url.split("/v1/")[1]]).encode()
    ))

    if not has_finish:
        with pytest.raises(RuntimeError, match="No completed-race evidence"):
            module.observed_summary()
    else:
        assert module.observed_summary()["races"] == [{
            "session": 1, "venue": "Test", "rain_observed": False, "red_flag": True,
        }]
