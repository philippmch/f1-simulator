"""Distance aggregates distinguish winners, finishers, and missing observations."""

import copy
import json

import numpy as np
import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.simulation.race import DriverStatus, RaceResult


def result(position, laps=None, *, retired=False, limited=False):
    return RaceResult(
        str(position), str(position), "Team", position, 100.0, 0.0, 0, 90.0,
        DriverStatus.DNF if retired else DriverStatus.FINISHED,
        laps_completed=laps, classified=True, race_time_limited=limited,
    )


def statistics(races):
    results = SimulationResults(100, "Test", {}, races, [])
    before = copy.deepcopy(results)
    summary = results.get_race_distance_statistics()
    assert results == before
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert all(type(value) in (int, float, type(None)) for value in summary.values())
    return summary


def test_standard_full_distance_and_chronological_lapped_finishers():
    summary = statistics([
        [result(1, 50), result(2, 50), result(3, 50)],
        [result(1, 50), result(2, 49), result(3, 48)],
    ])
    assert summary == {
        "recorded_races": 2,
        "races_with_winner": 2,
        "races_without_winner": 0,
        "races_with_known_winner_distance": 2,
        "mean_winner_laps": 50.0,
        "time_limited_races": 0,
        "time_limited_race_rate": 0.0,
        "finishing_cars": 6,
        "finishers_with_comparable_distance": 6,
        "lapped_finishers": 2,
        "lapped_finisher_rate": 2 / 6,
    }


def test_timed_winner_can_have_less_distance_than_classified_retirement():
    summary = statistics([
        [result(2, 40, retired=True, limited=True), result(1, 30, limited=True),
         result(3, 30, limited=True), result(4, 29, limited=True)],
        [result(1, 50), result(2, 50)],
    ])
    assert summary["mean_winner_laps"] == 40
    assert summary["time_limited_races"] == 1
    assert summary["time_limited_race_rate"] == 0.5
    assert summary["finishing_cars"] == 5
    assert summary["finishers_with_comparable_distance"] == 5
    assert summary["lapped_finishers"] == 1
    assert summary["lapped_finisher_rate"] == 0.2


@pytest.mark.parametrize("unknown", [None, 0, -1, True, False, 5.0, "5"])
def test_unknown_distance_never_invents_lapping_or_enters_denominator(unknown):
    summary = statistics([
        [result(1, unknown), result(2, 4)],
        [result(1, 10), result(2, unknown), result(3, 9)],
    ])
    assert summary["races_with_winner"] == 2
    assert summary["races_with_known_winner_distance"] == 1
    assert summary["mean_winner_laps"] == 10
    assert summary["finishing_cars"] == 5
    assert summary["finishers_with_comparable_distance"] == 2
    assert summary["lapped_finisher_rate"] == 0.5


def test_no_recorded_races_do_not_use_requested_simulation_count():
    assert statistics([]) == {
        "recorded_races": 0,
        "races_with_winner": 0,
        "races_without_winner": 0,
        "races_with_known_winner_distance": 0,
        "mean_winner_laps": None,
        "time_limited_races": 0,
        "time_limited_race_rate": None,
        "finishing_cars": 0,
        "finishers_with_comparable_distance": 0,
        "lapped_finishers": 0,
        "lapped_finisher_rate": None,
    }


def test_empty_race_and_absent_winner_are_recorded_but_not_comparable():
    summary = statistics([[], [result(1, 10, retired=True, limited=True), result(2, 9)]])
    assert summary["recorded_races"] == 2
    assert summary["races_with_winner"] == 0
    assert summary["races_without_winner"] == 2
    assert summary["mean_winner_laps"] is None
    assert summary["time_limited_race_rate"] == 0.5
    assert summary["finishing_cars"] == 1
    assert summary["finishers_with_comparable_distance"] == 0
    assert summary["lapped_finisher_rate"] is None


def test_numpy_integer_distances_produce_native_json_numbers():
    summary = statistics([[result(1, np.int64(10)), result(2, np.int64(9))]])
    assert summary["mean_winner_laps"] == 10.0
    assert summary["lapped_finisher_rate"] == 0.5
