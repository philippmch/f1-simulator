"""Opening policy comparisons replay existing pit decisions without race RNG."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.opening_strategy import (
    OPENING_CANDIDATES,
    REACTION_SEEDS,
    _cached_policy_costs,
    _policy_path_cost,
    opening_policy_costs,
)
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def fixture(laps=10, rain=0.2):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90)
    weather = Weather(condition=WeatherCondition.LIGHT_RAIN, rain_intensity=rain)
    return driver, car, track, weather


@pytest.mark.parametrize("compound", OPENING_CANDIDATES)
@pytest.mark.parametrize("seed", [0, 3])
def test_projection_matches_controlled_full_race(monkeypatch, compound, seed):
    driver, car, track, weather = fixture()
    simulator = RaceSimulator(np.random.default_rng(seed))
    style = TeamStrategyArchetype.BALANCED
    expected = _policy_path_cost(driver, car, track, weather, style,
                                 simulator.strategy_tuning, simulator.strategy_profiles,
                                 compound, seed)
    monkeypatch.setattr(simulator, "_infer_team_strategy", lambda *args: style)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
    lap_time = simulator.lap_simulator.calculate_lap_time
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda **kwargs: lap_time(**kwargs, sample_variation=False))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        expected_stationary_time)
    result, = simulator.simulate_race([driver], {"A": car}, track, weather, ["A"],
                                      starting_tires={"A": compound})
    assert result.total_time == pytest.approx(expected, abs=1e-8)
    assert (any(c in ("wet", "intermediate") for c in result.strategy)
            or len(set(result.strategy)) >= 2)


def test_average_costs_and_cache_preserve_state_and_actual_rng(monkeypatch):
    driver, car, track, weather = fixture()
    simulator = RaceSimulator(np.random.default_rng(42))
    style = TeamStrategyArchetype.BALANCED
    args = (driver, car, track, weather, style,
            simulator.strategy_tuning, simulator.strategy_profiles)
    _cached_policy_costs.cache_clear()
    before = copy.deepcopy((driver, car, track, weather, simulator.rng.bit_generator.state))
    costs = opening_policy_costs(*args)
    assert _cached_policy_costs.cache_info().misses == 1
    assert costs == tuple((compound, sum(_policy_path_cost(
        driver, car, track, weather, style, simulator.strategy_tuning, simulator.strategy_profiles,
        compound, seed,
    ) for seed in REACTION_SEEDS) / 8) for compound in OPENING_CANDIDATES)
    assert opening_policy_costs(*args) is costs
    assert _cached_policy_costs.cache_info().hits == 1
    driver.current_tire_laps = 99
    assert opening_policy_costs(*args) is costs  # Transient state normalized.
    driver.current_tire_laps = 0
    assert (driver, car, track, weather, simulator.rng.bit_generator.state) == before
    simulator.strategy_tuning["pit_prob_min"] += 0.001
    opening_policy_costs(*args)
    assert _cached_policy_costs.cache_info().misses == 2
    soft = TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(update={"initial_grip": 1.06})
    monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.SOFT, soft)
    opening_policy_costs(*args)
    assert _cached_policy_costs.cache_info().misses == 3


@pytest.mark.parametrize("laps,rain", [(10, 0.2), (30, 0.2), (10, 0.3), (30, 0.3), (3, 0.2)])
def test_precautionary_selection_uses_conditional_policy_minimum(laps, rain):
    driver, car, track, weather = fixture(laps, rain)
    simulator = RaceSimulator(np.random.default_rng(42))
    style = TeamStrategyArchetype.BALANCED
    costs = opening_policy_costs(driver, car, track, weather, style,
                                 simulator.strategy_tuning, simulator.strategy_profiles)
    selected = simulator._choose_starting_compound(style, track, weather, driver, car)
    assert selected == min(costs, key=lambda pair: pair[1])[0]
    if laps >= 10:
        assert dict(costs)[selected] < dict(costs)[TireCompound.INTERMEDIATE]
    assert simulator._choose_starting_compound(style, track, weather) == TireCompound.INTERMEDIATE
    weather.rain_intensity = 0.5
    assert simulator._choose_starting_compound(style, track, weather, driver, car) == (
        TireCompound.INTERMEDIATE
    )


def test_mean_service_is_opt_in_and_default_rng_is_unchanged():
    driver, car, track, weather = fixture()
    weather = Weather()
    states = [DriverRaceState(driver.model_copy(), car, 1, tire_laps=10) for _ in range(3)]
    simulators = [RaceSimulator(np.random.default_rng(42)) for _ in range(3)]
    default = simulators[0]._execute_pit_stop(states[0], track, weather, 5)
    explicit = simulators[1]._execute_pit_stop(states[1], track, weather, 5, sample_service=True)
    assert default == explicit
    assert simulators[0].rng.random() == simulators[1].rng.random()
    before = copy.deepcopy(simulators[2].rng.bit_generator.state)
    mean = simulators[2]._execute_pit_stop(states[2], track, weather, 5, sample_service=False)
    assert mean == track.pit_lane_delta + expected_stationary_time(car)
    assert simulators[2].rng.bit_generator.state == before
