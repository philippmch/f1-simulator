"""Free restart choices agree with controlled native compound alternatives."""

import runpy
from pathlib import Path

import pytest


def test_weather_restart_choices_match_executed_alternatives():
    diagnostic = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "check_weather_restart_choices.py")
    )
    rows = diagnostic["compare_restart_choices"]()
    assert len(rows) == 12
    assert {row["engine"] for row in rows} == {"standard", "chronological"}
    assert {row["case"] for row in rows} == set(diagnostic["CASES"])
    for row in rows:
        assert row["cost_vs_best_seconds"] == pytest.approx(0, abs=1e-7)
        assert row["selected"]["laps_completed"] == row["inputs"]["scheduled_laps"]
        if row["case"] in {"wetting", "drying"}:
            expected = "intermediate" if row["case"] == "wetting" else "soft"
            assert row["selected"]["restart_compound"] == expected
            assert row["selected"]["paid_stops"] == 0
