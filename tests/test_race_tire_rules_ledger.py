"""Physical race-use identity must survive paid stops, refits and interruptions."""

from copy import deepcopy

import pytest

from f1sim.simulation.tire_rules import RaceTireUsage


def run_lap(usage):
    usage.begin_running()
    usage.complete_lap()


def test_cap_is_race_only_and_cumulative_when_a_physical_set_is_refitted():
    usage = RaceTireUsage({"max_set_laps": 2, "minimum_sets": 3})
    usage.fit("used-soft")
    run_lap(usage)
    usage.fit("hard")
    run_lap(usage)
    usage.fit("used-soft")
    assert usage.remaining_laps() == 1
    run_lap(usage)
    assert usage.remaining_laps("used-soft") == 0
    assert usage.completed_laps_by_set == {"used-soft": 2, "hard": 1}
    assert usage.used_set_ids == {"used-soft", "hard"}
    assert not usage.can_fit("used-soft")
    with pytest.raises(ValueError, match="cannot run"):
        usage.begin_running()
    previous = deepcopy(usage.__dict__)
    with pytest.raises(ValueError, match="cannot fit"):
        usage.fit("used-soft")
    assert usage.__dict__ == previous


def test_free_fit_and_unused_fitting_do_not_invent_used_sets_or_completed_laps():
    usage = RaceTireUsage({"max_set_laps": 25, "minimum_sets": 3})
    usage.fit("initial")
    usage.fit("changed-before-running")
    assert usage.used_set_ids == set()
    usage.begin_running()
    usage.fit("red-flag-replacement")  # Interrupted, not completed.
    assert usage.completed_laps_by_set["changed-before-running"] == 0
    assert usage.used_set_ids == {"changed-before-running"}
    usage.begin_running()
    assert len(usage.used_set_ids) == 2
    usage.complete_lap()
    usage.fit("changed-before-running")
    run_lap(usage)
    assert len(usage.used_set_ids) == 2


def test_unlimited_pool_replacements_are_fresh_local_identities_without_a_stock_count():
    rules = {"max_set_laps": 1}
    usage = RaceTireUsage(rules)
    rules["max_set_laps"] = 100
    first = usage.fit()
    run_lap(usage)
    second = usage.fit()
    assert first != second and usage.remaining_laps() == 1
    run_lap(usage)
    assert usage.completed_laps_by_set == {first: 1, second: 1}
    assert usage.used_set_ids == {first, second}
    assert usage.can_fit(None)


def test_explicit_id_cannot_collide_with_the_next_unlimited_local_identity():
    usage = RaceTireUsage({})
    usage.fit("unlimited-fresh-1")
    run_lap(usage)
    assert usage.fit() == "unlimited-fresh-2"


def test_minimum_sets_reserves_running_laps_and_does_not_count_unrun_fittings():
    usage = RaceTireUsage({"minimum_sets": 3})
    usage.fit("A")
    assert not usage.needs_new_set(1, 3)
    run_lap(usage)
    assert usage.needs_new_set(2, 3)
    usage.fit("B")
    assert not usage.needs_new_set(2, 3)
    run_lap(usage)
    assert usage.needs_new_set(3, 3)
    usage.fit("C")
    assert not usage.needs_new_set(3, 3)
    run_lap(usage)
    assert not usage.needs_new_set(3, 3)
    assert not usage.needs_new_set(4, 3)


def test_inspecting_candidates_never_mutates_the_usage_ledger():
    usage = RaceTireUsage({"max_set_laps": 25})
    usage.fit("A")
    run_lap(usage)
    previous = deepcopy(usage.__dict__)
    assert usage.can_fit("new-candidate")
    assert usage.remaining_laps("new-candidate") == 25
    assert usage.remaining_laps() == 24
    assert usage.__dict__ == previous


def test_without_cap_completed_use_is_still_recorded_for_distinct_set_rule():
    usage = RaceTireUsage({"minimum_sets": 2})
    usage.fit("A")
    for _ in range(30):
        run_lap(usage)
    assert usage.remaining_laps() is None
    assert usage.completed_laps_by_set["A"] == 30


def test_completion_requires_started_running_and_cannot_be_credited_twice():
    usage = RaceTireUsage({"max_set_laps": 2})
    with pytest.raises(ValueError, match="fitted set"):
        usage.begin_running()
    usage.fit("A")
    with pytest.raises(ValueError, match="before running"):
        usage.complete_lap()
    run_lap(usage)
    with pytest.raises(ValueError, match="before running"):
        usage.complete_lap()
    assert usage.completed_laps_by_set["A"] == 1


@pytest.mark.parametrize("set_id", ["", " ", 2, False, []])
def test_bad_explicit_set_ids_are_rejected(set_id):
    usage = RaceTireUsage({"max_set_laps": 25})
    with pytest.raises(ValueError, match="set ID"):
        usage.fit(set_id)


@pytest.mark.parametrize("lap,total", [(0, 3), (1, 0), (True, 3), (1, 3.0)])
def test_bad_lap_horizons_are_rejected(lap, total):
    usage = RaceTireUsage({"minimum_sets": 3})
    with pytest.raises(ValueError, match="positive integers"):
        usage.needs_new_set(lap, total)
