"""Paid-stop distributions use observed entrants, including retirements."""

import copy
import json
from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult


def result(driver, stops, *, retired=False, strategy=None):
    return RaceResult(driver, driver, "Team", 1, 90, 0, stops, 90,
                      DriverStatus.DNF if retired else DriverStatus.FINISHED,
                      strategy=strategy or ["soft"])


def sample():
    races = [
        [result("A", 0, strategy=["soft", "medium"]), result("B", 1)],
        [result("A", 2, retired=True, strategy=["soft", "soft", "medium"])],
        [result("A", 4), result("B", 1, retired=True)],
        [],
    ]
    stats = {"missing": DriverStatistics("missing", "Missing", "Team")}
    return SimulationResults(100, "Test", stats, races, [])


def test_distribution_counts_paid_stops_not_stints_or_simulation_denominator():
    results = sample()
    before = copy.deepcopy(results)
    assert results.get_pit_stop_statistics() == {
        "A": {"races": 3, "average_stops": 2.0, "stop_count_distribution": {0: 1, 2: 1, 4: 1}},
        "B": {"races": 2, "average_stops": 1.0, "stop_count_distribution": {1: 2}},
    }
    assert results == before
    # Fresh output containers cannot change the source observations.
    results.get_pit_stop_statistics()["A"]["stop_count_distribution"][0] = 99
    assert results.get_pit_stop_statistics()["A"]["stop_count_distribution"][0] == 1


def test_no_observations_are_not_zero_stop_races():
    results = sample()
    results.race_results = [[], []]
    assert results.get_pit_stop_statistics() == {}


def test_json_export_preserves_counts_and_means(tmp_path):
    results = sample()
    path = Exporter(tmp_path).export_statistics_json(results)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pit_stop_statistics"] == json.loads(json.dumps(results.get_pit_stop_statistics()))
    assert "missing" not in data["pit_stop_statistics"]


def test_api_exposes_statistics_and_legacy_objects_default_empty():
    pytest.importorskip("fastapi")
    from starlette.responses import JSONResponse

    from f1sim.web.server import _summarize_scenario_results

    results = sample()
    payload = _summarize_scenario_results({"dry": results})
    assert payload["scenarios"]["dry"]["pit_stop_statistics"] == results.get_pit_stop_statistics()
    json.loads(JSONResponse(payload).body)
    legacy = SimpleNamespace(num_simulations=2, seed=None, driver_stats={},
                             race_results=[], qualifying_results=[])
    assert _summarize_scenario_results({"dry": legacy})["scenarios"]["dry"][
        "pit_stop_statistics"
    ] == {}
