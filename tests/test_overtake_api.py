"""The dashboard exposes aggregate counters and nullable race-row evidence."""

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.simulation.race import DriverStatus, RaceResult
from f1sim.web.server import _serialize_race_result, _summarize_scenario_results


def race_result(driver_id, position, counters=None):
    counters = counters or (None, None, None)
    return RaceResult(
        driver_id, driver_id, "Team", position, 100.0, 0.0, 0, 90.0,
        DriverStatus.FINISHED,
        overtake_attempts=counters[0], overtake_successes=counters[1],
        overtake_contacts=counters[2],
    )


def test_scenario_payload_includes_aggregate_and_preserves_sample_counter_nullability():
    native_zero = race_result("A", 1, (0, 0, 0))
    legacy = race_result("B", 2)
    results = SimulationResults(
        num_simulations=1,
        track_name="Test",
        driver_stats={},
        race_results=[[native_zero, legacy]],
        qualifying_results=[[]],
        seed=42,
    )

    scenario = _summarize_scenario_results({"dry": results})["scenarios"]["dry"]

    assert scenario["overtaking_statistics"] == results.get_overtake_statistics()
    assert scenario["overtaking_statistics"]["overall"] == {
        "status": "partial",
        "recorded_driver_races": 1,
        "missing_driver_races": 1,
        "attempts": 0,
        "successes": 0,
        "contacts": 0,
        "success_rate": None,
        "contact_rate": None,
    }
    sample = {row["driver_id"]: row for row in scenario["sample_race"]}
    assert (sample["A"]["overtake_attempts"], sample["A"]["overtake_successes"],
            sample["A"]["overtake_contacts"]) == (0, 0, 0)
    assert (sample["B"]["overtake_attempts"], sample["B"]["overtake_successes"],
            sample["B"]["overtake_contacts"]) == (None, None, None)


def test_direct_race_row_serializer_keeps_legacy_counters_unknown():
    row = _serialize_race_result(race_result("A", 1))
    assert row["overtake_attempts"] is None
    assert row["overtake_successes"] is None
    assert row["overtake_contacts"] is None


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_engine_zero_counters_reach_api_payload(engine):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=2,
                  base_lap_time=90)
    results = MonteCarloRunner(
        [driver], {"A": car}, track, Weather(change_probability=0), seed=74,
        race_engine=engine, starting_tires={"A": TireCompound.SOFT},
    ).run(1, parallel=False)

    scenario = _summarize_scenario_results({"dry": results})["scenarios"]["dry"]
    assert scenario["overtaking_statistics"]["overall"] == {
        "status": "recorded", "recorded_driver_races": 1, "missing_driver_races": 0,
        "attempts": 0, "successes": 0, "contacts": 0,
        "success_rate": None, "contact_rate": None,
    }
    assert (scenario["sample_race"][0]["overtake_attempts"],
            scenario["sample_race"][0]["overtake_successes"],
            scenario["sample_race"][0]["overtake_contacts"]) == (0, 0, 0)
