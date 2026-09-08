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


@pytest.fixture(scope="module")
def dry_schedule_diagnostic():
    return runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "examples" / "check_dry_pit_schedules.py")
    )


def test_dry_schedule_enumeration_is_complete_and_unique(dry_schedule_diagnostic):
    from collections import Counter

    schedules = list(dry_schedule_diagnostic["schedules"]())
    assert len(schedules) == len(set(schedules)) == 1092
    assert Counter(map(len, schedules)) == {1: 14, 2: 168, 3: 910}
    for schedule in schedules:
        laps, compounds = zip(*schedule)
        assert list(laps) == sorted(set(laps))
        assert all(2 <= lap <= 8 for lap in laps)
        assert set(compounds) <= {"soft", "medium", "hard"}
        assert set(compounds) != {"medium"}
    assert ((2, "soft"), (5, "soft")) in schedules  # A new set may repeat a compound.


def test_dry_policy_matches_executed_bounded_schedules(dry_schedule_diagnostic, monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Synthetic strategy diagnostic contacted the network")

    monkeypatch.setattr("socket.socket.connect", no_network)
    rows = dry_schedule_diagnostic["compare_schedules"]()
    assert len(rows) == 6
    grouped = {}
    for row in rows:
        assert row["schedules_checked"] == 1092
        assert row["gap_seconds"] == pytest.approx(0, abs=1e-8)
        for result in (row["selected"], row["best_schedule"]):
            assert result["laps_completed"] == row["case"]["laps"]
            assert result["compounds"][0] == "medium"
            assert len(set(result["compounds"])) >= 2
            assert result["paid_stops"] == len(result["pit_laps"])
        grouped.setdefault(row["case"]["name"], {})[row["engine"]] = row
    for pair in grouped.values():
        assert pair["standard"]["selected"] == pair["chronological"]["selected"]
        assert pair["standard"]["best_schedule"] == pair["chronological"]["best_schedule"]
    assert grouped["normal_lane"]["standard"]["selected"]["paid_stops"] == 1
    assert grouped["low_lane_high_wear"]["standard"]["selected"]["paid_stops"] == 2
    assert grouped["long_reference_high_wear"]["standard"]["selected"]["paid_stops"] == 3


def test_dry_explicit_schedule_accepts_compound_labels(dry_schedule_diagnostic):
    case = dry_schedule_diagnostic["CASES"][0]
    run = dry_schedule_diagnostic["run_race"]
    for engine in ("standard", "chronological"):
        result = run(case, engine, ((2, "soft"), (5, "soft")))
        assert result["pit_laps"] == [2, 5]
        assert result["compounds"] == ["medium", "soft", "soft"]
    with pytest.raises(ValueError, match="engine"):
        run(case, "typo")
