"""Overtake summaries preserve recorded zeroes and incomplete legacy rows."""

import copy
import json

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.simulation.race import DriverStatus, RaceResult


def result(driver_id, counters=None, *, retired=False):
    counters = counters or (None, None, None)
    return RaceResult(
        driver_id, driver_id, "Team", 1, 100.0, 0.0, 0, 90.0,
        DriverStatus.DNF if retired else DriverStatus.FINISHED,
        overtake_attempts=counters[0], overtake_successes=counters[1],
        overtake_contacts=counters[2],
    )


def summarize(races):
    simulation = SimulationResults(50, "Test", {}, races, [])
    before = copy.deepcopy(simulation)
    summary = simulation.get_overtake_statistics()
    assert simulation == before
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    return summary


def test_statistics_pool_valid_rows_and_keep_legacy_inconsistent_rows_missing():
    summary = summarize([
        [result("A", (2, 1, 0)), result("B", (0, 0, 0)), result("C")],
        [result("A", (1, 0, 1)), result("B", (1, 1, 1)),
         result("D", (3, 1, 0), retired=True)],
        [result("A", (True, 0, 0))],
    ])

    assert summary["overall"] == {
        "status": "partial",
        "recorded_driver_races": 4,
        "missing_driver_races": 3,
        "attempts": 6,
        "successes": 2,
        "contacts": 1,
        "success_rate": pytest.approx(1 / 3),
        "contact_rate": pytest.approx(1 / 6),
    }
    assert summary["drivers"]["A"] == {
        "status": "partial", "recorded_driver_races": 2, "missing_driver_races": 1,
        "attempts": 3, "successes": 1, "contacts": 1,
        "success_rate": pytest.approx(1 / 3), "contact_rate": pytest.approx(1 / 3),
    }
    assert summary["drivers"]["B"] == {
        "status": "partial", "recorded_driver_races": 1, "missing_driver_races": 1,
        "attempts": 0, "successes": 0, "contacts": 0,
        "success_rate": None, "contact_rate": None,
    }
    assert summary["drivers"]["C"] == {
        "status": "not_recorded", "recorded_driver_races": 0, "missing_driver_races": 1,
        "attempts": None, "successes": None, "contacts": None,
        "success_rate": None, "contact_rate": None,
    }
    assert summary["drivers"]["D"]["recorded_driver_races"] == 1


@pytest.mark.parametrize("bad", [
    (None, 0, 0), (-1, 0, 0), (1, -1, 0), (1, 0, -1),
    (1, 2, 0), (1, 0, 2), (1.0, 0, 0), (1, False, 0), ("1", 0, 0),
])
def test_invalid_or_unknown_triplets_are_missing_not_zero(bad):
    summary = summarize([[result("A", bad)]])
    assert summary["overall"] == {
        "status": "not_recorded", "recorded_driver_races": 0,
        "missing_driver_races": 1, "attempts": None, "successes": None,
        "contacts": None, "success_rate": None, "contact_rate": None,
    }


def test_native_zero_is_recorded_but_rates_need_attempts():
    summary = summarize([[result("A", (0, 0, 0))]])
    assert summary["overall"] == {
        "status": "recorded", "recorded_driver_races": 1,
        "missing_driver_races": 0, "attempts": 0, "successes": 0,
        "contacts": 0, "success_rate": None, "contact_rate": None,
    }


def test_no_result_rows_remain_absent_and_not_recorded():
    assert summarize([]) == {
        "overall": {
            "status": "not_recorded", "recorded_driver_races": 0,
            "missing_driver_races": 0, "attempts": None, "successes": None,
            "contacts": None, "success_rate": None, "contact_rate": None,
        },
        "drivers": {},
    }
