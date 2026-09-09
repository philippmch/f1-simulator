"""Observed evidence keeps missing feeds, lap conflicts, and pit timings explicit."""

import copy
import json

import pytest

from f1sim.analysis.observed_strategy import observed_strategy_summary


def result(driver=1, laps=6, **kwargs):
    return dict(session_key=10, driver_number=driver, number_of_laps=laps,
                dnf=False, dns=False, dsq=False, **kwargs)


def stint(number=1, start=1, end=6, driver=1, compound="MEDIUM", age=0):
    return dict(session_key=10, driver_number=driver, stint_number=number,
                lap_start=start, lap_end=end, compound=compound, tyre_age_at_start=age)


def pit(**kwargs):
    return dict(session_key=10, driver_number=1, date="2026-01-01T12:00:00Z",
                lap_number=3, **kwargs)


def test_coverage_gaps_overlap_and_outside_without_large_allocations():
    summary = observed_strategy_summary([result()], [stint(end=2), stint(2, 2, 3),
                                         stint(3, 5, 10**30)], [], 10)
    coverage = summary["drivers"][0]["coverage"]
    assert coverage == {"covered_laps": 5, "known_compound_laps": 5, "missing_laps": 1,
                        "overlap_laps": 1, "outside_result_lap_exposures": 10**30 - 6,
                        "complete_lap_coverage": False}
    assert summary["drivers"][0]["stints"][-1]["lap_end"] == 10**30


def test_empty_initial_range_does_not_invalidate_positive_coverage():
    summary = observed_strategy_summary([result()], [stint(end=0, compound="INTERMEDIATE"),
                                        stint(2, 1, 3), stint(3, 4, 6)], [], 10)
    driver = summary["drivers"][0]
    assert driver["empty_range_rows"] == 1
    assert driver["coverage"]["complete_lap_coverage"] is True
    assert len(driver["stints"]) == 2
    assert all(s["end_reason"] == "unknown" for s in driver["stints"])


def test_zero_distance_retirements_and_unknown_flags_remain_in_denominators():
    rows = [result(), result(41, 0), result(2, 2)]
    rows[1].update(dns=True, dnf=None)
    rows[2].update(dnf=True, dsq=True)
    summary = observed_strategy_summary(rows, [stint(), stint(driver=41, end=1)], [], 10)
    assert summary["totals"]["result_drivers"] == 3
    assert summary["totals"]["positive_lap_drivers"] == 2
    assert summary["totals"]["dnf_true_drivers"] == 1
    assert summary["totals"]["dnf_unknown_drivers"] == 1
    zero = summary["drivers"][-1]
    assert zero["coverage"]["complete_lap_coverage"] is None
    assert zero["coverage"]["outside_result_lap_exposures"] == 1


def test_unknown_compound_age_and_invalid_range_are_separate():
    summary = observed_strategy_summary([result()], [stint(compound="UNKNOWN", age=-1),
                                         stint(2, 5, 2)], [], 10)
    driver = summary["drivers"][0]
    assert driver["coverage"]["covered_laps"] == 6
    assert driver["coverage"]["known_compound_laps"] == 0
    assert driver["coverage"]["complete_lap_coverage"] is False
    assert driver["stints"][0]["compound"] is None
    assert driver["stints"][0]["tyre_age_at_start"] is None
    assert driver["invalid_age_rows"] == driver["invalid_range_rows"] == 1


def test_empty_feeds_do_not_assert_absent_stops_or_stints():
    summary = observed_strategy_summary([result()], [], [], 10)
    assert not summary["feeds"]["stints"]["available"]
    assert not summary["feeds"]["pits"]["available"]
    assert summary["drivers"][0]["coverage"]["missing_laps"] == 6
    assert summary["drivers"][0]["pit_evidence"]["entries"] == 0


def test_duplicates_are_counted_once_and_inputs_order_independent():
    rows, stints, pits = [result(), result(2, 0)], [stint(), stint()], [pit(), pit()]
    before = copy.deepcopy((rows, stints, pits))
    summary = observed_strategy_summary(rows, stints, pits, 10)
    assert summary == observed_strategy_summary(rows[::-1], stints[::-1], pits[::-1], 10)
    assert summary["feeds"]["stints"]["exact_duplicate_rows"] == 1
    assert summary["totals"]["pit_entries"] == 1
    assert (rows, stints, pits) == before
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("feed", ["results", "stints", "pits"])
def test_conflicting_identities_fail(feed):
    inputs = {"results": [result()], "stints": [stint()], "pits": [pit()]}
    changed = dict(inputs[feed][0], other="conflicting")
    inputs[feed].append(changed)
    with pytest.raises(ValueError, match="conflicting"):
        observed_strategy_summary(inputs["results"], inputs["stints"], inputs["pits"], 10)


@pytest.mark.parametrize("bad", [True, "10", None, 11])
def test_wrong_or_invalid_session_fails(bad):
    row = stint()
    row["session_key"] = bad
    with pytest.raises(ValueError, match="session_key"):
        observed_strategy_summary([result()], [row], [], 10)


@pytest.mark.parametrize("key,value", [("driver_number", True), ("number_of_laps", -1),
                                      ("number_of_laps", 2.5), ("dnf", 1), ("dns", "false")])
def test_invalid_results_fail(key, value):
    row = result()
    row[key] = value
    with pytest.raises(ValueError):
        observed_strategy_summary([row], [], [], 10)


def test_missing_results_and_non_list_feeds_fail():
    for rows, stints in [([], []), ([result()], {}), ([result()], [1])]:
        with pytest.raises(ValueError):
            observed_strategy_summary(rows, stints, [], 10)


def test_ancillary_invalid_ids_and_unknown_drivers_are_excluded():
    broken = stint()
    broken["stint_number"] = None
    summary = observed_strategy_summary([result()], [broken, stint(driver=99)],
                                         [dict(pit(), driver_number=99),
                                          dict(pit(), date=None)], 10)
    assert summary["exclusions"]["stint_invalid_identity_rows"] == 1
    assert summary["exclusions"]["stint_unknown_driver_rows"] == 1
    assert summary["exclusions"]["pit_invalid_identity_rows"] == 1
    assert summary["exclusions"]["pit_unknown_driver_rows"] == 1


@pytest.mark.parametrize("duration", [None, float("nan"), float("inf"), -1, True, "3", 10**400])
def test_invalid_or_unknown_pit_durations_are_not_zero_or_alias_filled(duration):
    summary = observed_strategy_summary([result()], [],
                                         [pit(lane_duration=duration, stop_duration=duration,
                                              pit_duration=25)], 10)
    entry = summary["drivers"][0]["pit_entries"][0]
    assert entry["lane_duration"] is entry["stop_duration"] is None
    assert summary["totals"]["pit_entries"] == 1
    assert summary["totals"]["pit_lane_duration_count"] == 0
    assert summary["exclusions"]["pit_deprecated_duration_rows"] == 1
    json.dumps(summary, allow_nan=False)


def test_lane_and_stationary_durations_remain_separate():
    summary = observed_strategy_summary([result()], [],
                                         [pit(lane_duration=24, stop_duration=2.3,
                                              pit_duration=24)], 10)
    assert summary["totals"]["pit_lane_duration_count"] == 1
    assert summary["totals"]["pit_stop_duration_count"] == 1
    assert summary["drivers"][0]["pit_entries"][0]["stop_duration"] == 2.3


def test_equivalent_pit_instants_are_semantic_not_exact_duplicates():
    first = pit(stop_duration=2.5)
    equivalent = dict(first, date="2026-01-01T13:00:00+01:00")
    summary = observed_strategy_summary([result()], [], [first, first, equivalent], 10)
    assert summary["feeds"]["pits"]["exact_duplicate_rows"] == 1
    assert summary["feeds"]["pits"]["equivalent_instant_duplicate_rows"] == 1
    assert summary["totals"]["pit_entries"] == 1
    assert summary["drivers"][0]["pit_entries"][0]["date"] == "2026-01-01T12:00:00+00:00"


def test_conflicting_pit_durations_at_equivalent_instants_fail():
    first = pit(stop_duration=2.5)
    conflict = dict(first, date="2026-01-01T13:00:00+01:00", stop_duration=3.5)
    with pytest.raises(ValueError, match="conflicting pit identity"):
        observed_strategy_summary([result()], [], [first, conflict], 10)


def test_pit_order_uses_utc_instants_instead_of_local_timestamp_text():
    first = dict(pit(), date="2026-01-01T14:00:00+02:00")
    second = dict(pit(), date="2026-01-01T12:30:00Z")
    summary = observed_strategy_summary([result()], [], [second, first], 10)
    assert [row["date"] for row in summary["drivers"][0]["pit_entries"]] == [
        "2026-01-01T12:00:00+00:00", "2026-01-01T12:30:00+00:00",
    ]
    assert summary == observed_strategy_summary([result()], [], [first, second], 10)
