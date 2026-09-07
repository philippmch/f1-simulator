"""Duplicate entrants fail before mutation, randomness or worker startup."""

import copy

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.qualifying import QualifyingSimulator
from f1sim.simulation.race import RaceSimulator


def fixture():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=2, base_lap_time=90)
    return driver, {"A": car}, track, Weather(change_probability=0)


@pytest.mark.parametrize("duplicate", ["same_object", "same_id", "grid", "unknown_grid"])
def test_race_rejects_duplicates_before_reset_or_rng(duplicate):
    driver, cars, track, weather = fixture()
    driver.current_tire_laps = 17
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator.event_manager.safety_car_active = True
    drivers = [driver]
    grid = ["A"]
    if duplicate == "same_object":
        drivers.append(driver)
    elif duplicate == "same_id":
        drivers.append(driver.model_copy())
    else:
        grid = ["missing", "missing"] if duplicate == "unknown_grid" else ["A", "A"]
    before = driver.model_dump()
    rng_before = copy.deepcopy(simulator.rng.bit_generator.state)
    with pytest.raises(ValueError, match="Duplicate driver ID.*(A|missing)"):
        simulator.simulate_race(drivers, cars, track, weather, grid)
    assert driver.model_dump() == before
    assert simulator.event_manager.safety_car_active
    assert simulator.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("same_object", [False, True])
def test_qualifying_rejects_duplicates_without_rng(same_object):
    driver, cars, track, weather = fixture()
    simulator = QualifyingSimulator(np.random.default_rng(42))
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    drivers = [driver, driver if same_object else driver.model_copy()]
    with pytest.raises(ValueError, match="Duplicate driver ID 'A'"):
        simulator.simulate_qualifying(drivers, cars, track, weather)
    assert simulator.rng.bit_generator.state == before


def test_runner_constructor_rejects_before_generating_seed(monkeypatch):
    driver, cars, track, weather = fixture()

    def unexpected_rng(*args, **kwargs):
        pytest.fail("Seed generation occurred for invalid entrants")

    monkeypatch.setattr("f1sim.analysis.montecarlo.np.random.default_rng", unexpected_rng)
    with pytest.raises(ValueError, match="Duplicate driver ID 'A'"):
        MonteCarloRunner([driver, driver.model_copy()], cars, track, weather)


@pytest.mark.parametrize("parallel", [False, True])
def test_runner_revalidates_mutable_entrants_before_workers(monkeypatch, parallel):
    driver, cars, track, weather = fixture()
    drivers = [driver]
    runner = MonteCarloRunner(drivers, cars, track, weather, seed=42)
    drivers.append(driver)

    def unexpected_work(*args, **kwargs):
        pytest.fail("Simulation work started for invalid entrants")

    monkeypatch.setattr("f1sim.analysis.montecarlo.ProcessPoolExecutor", unexpected_work)
    monkeypatch.setattr("f1sim.analysis.montecarlo._run_single_simulation", unexpected_work)
    with pytest.raises(ValueError, match="Duplicate driver ID 'A'"):
        runner.run(2, parallel=parallel)


def test_unique_case_sensitive_ids_and_partial_grid_remain_supported():
    driver, cars, track, weather = fixture()
    lower = driver.model_copy(update={"id": "a"})
    missing = driver.model_copy(update={"id": "missing_car", "team_id": "missing"})
    simulator = RaceSimulator(np.random.default_rng(42))
    results = simulator.simulate_race([driver, lower, missing], cars, track, weather,
                                      ["unknown", "A", "a", "missing_car"])
    assert {result.driver_id for result in results} == {"A", "a"}
    assert simulator.simulate_race([], {}, track, weather, []) == []
    assert QualifyingSimulator().simulate_qualifying([], {}, track, weather) == []
    runner = MonteCarloRunner([], {}, track, weather, seed=42)
    assert runner.run(1, parallel=False).race_results == [[]]
