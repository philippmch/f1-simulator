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


def test_rain_timing_matches_every_bounded_full_race_schedule():
    diagnostic = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "check_rain_pit_timing.py")
    )
    rows = diagnostic["compare_rain_pit_timing"]()
    assert len(rows) == 8
    for engine in ("standard", "chronological"):
        cases = [row for row in rows if row["race_engine"] == engine]
        assert [len(row["selected"]["pit_laps"]) for row in cases] == [1, 1, 0, 2]
        assert [row["schedules_checked"] for row in cases] == [99, 99, 99, 562]
        for row in cases:
            assert row["cost_vs_best_seconds"] == pytest.approx(0, abs=1e-8)
            assert row["selected"]["laps_completed"] == row["race_laps"]
            assert set(row["selected"]["compounds"]) == {row["compound"]}
    for standard, chronological in zip(rows[:4], rows[4:]):
        assert standard["selected"] == chronological["selected"]
