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
