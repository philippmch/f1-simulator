"""Keep the documented exhaustive full-race strategy diagnostic executable."""

import runpy
from pathlib import Path

import pytest


def test_pit_timing_diagnostic_matches_exhaustive_full_races():
    diagnostic = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "check_pit_timing.py")
    )
    rows = diagnostic["compare_pit_timing"]()
    assert {row["tire_stress"] for row in rows} == {0.3, 0.9}
    for row in rows:
        assert row["cost_vs_best_seconds"] == pytest.approx(0.0, abs=0.1)
        assert len(row["selected"]["pit_laps"]) == 1
        assert len(set(row["selected"]["compounds"])) == 2


def test_restart_diagnostic_matches_full_race_compound_alternatives():
    diagnostic = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "check_restart_choices.py")
    )
    rows = diagnostic["compare_restart_choices"]()
    assert [row["remaining_laps"] for row in rows] == [5, 45, 50]
    for row in rows:
        assert row["cost_vs_best_seconds"] == pytest.approx(0.0, abs=0.1)
        assert row["selected"]["restart_compound"] == row["best_alternative"]["restart_compound"]
    assert rows[0]["selected"]["restart_compound"] == "soft"
    assert rows[1]["selected"]["restart_compound"] == "hard"
    assert [row["selected"]["paid_stops"] for row in rows] == [1, 1, 2]
