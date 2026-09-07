"""Final weather stops must have a plausible chance to recover their cost."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator


@pytest.fixture
def setup():
    simulator = RaceSimulator(np.random.default_rng(2))  # First roll below 0.7.
    state = DriverRaceState(
        Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.HARD], tire_laps=8,
        tire_compound_history=["medium", "hard"],
    )
    track = Track(id="test", name="Test", country="Test", total_laps=30,
                  base_lap_time=90, pit_lane_delta=22)
    return simulator, state, track


def test_final_mild_damp_stop_rejected_without_rng_or_state_mutation(setup):
    simulator, state, track = setup
    before = copy.deepcopy(state)
    rng_before = copy.deepcopy(simulator.rng.bit_generator.state)
    assert not simulator._should_pit(state, [state], track, 30, False,
                                     Weather(track_wetness=0.21))
    assert state == before
    assert simulator.rng.bit_generator.state == rng_before


def test_worn_set_low_cost_stop_can_still_be_taken(setup):
    simulator, state, track = setup
    state.current_tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    state.tire_laps = 30
    track.pit_lane_delta = 1
    assert simulator._should_pit(state, [state], track, 30, False,
                                Weather(track_wetness=0.3))
    expected_rng = np.random.default_rng(2)
    expected_rng.random()
    assert simulator.rng.random() == expected_rng.random()


def test_critical_mismatch_retains_unconditional_priority(setup):
    simulator, state, track = setup
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    assert simulator._should_pit(state, [state], track, 30, False,
                                Weather(track_wetness=0.8), 1000)
    assert simulator.rng.bit_generator.state == before


def test_missing_distinct_compound_preserves_existing_weather_reaction(setup):
    simulator, state, track = setup
    state.tire_compound_history = ["hard"]
    assert simulator._should_pit(state, [state], track, 30, False,
                                Weather(track_wetness=0.21), 1000)


def test_penultimate_lap_retains_existing_probability(setup):
    simulator, state, track = setup
    assert simulator._should_pit(state, [state], track, 29, False,
                                Weather(track_wetness=0.21), 1000)
    expected_rng = np.random.default_rng(2)
    expected_rng.random()
    assert simulator.rng.random() == expected_rng.random()


@pytest.mark.parametrize("flag,factor,modifier", [
    (None, 1, 1), ("safety_car_active", 0.55, 1.4), ("vsc_active", 0.75, 1.2),
])
def test_running_modifier_and_lane_discount_do_not_scale_queue(
    setup, monkeypatch, flag, factor, modifier,
):
    simulator, state, track = setup
    if flag:
        setattr(simulator.event_manager, flag, True)
    track.pit_lane_delta = 1
    calls = []

    def lap_time(*args, **kwargs):
        calls.append((args[0].current_tire_laps, kwargs))
        return 100 if args[0].current_tire_laps else 90

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap_time)
    threshold = 10 * modifier - factor - expected_stationary_time(state.car)
    weather = Weather(track_wetness=0.3)
    assert simulator._final_weather_stop_can_pay(state, track, weather, threshold - 0.01)
    assert not simulator._final_weather_stop_can_pay(state, track, weather, threshold + 0.01)
    assert [age for age, _ in calls] == [8, 0, 8, 0]
    assert all(not kwargs["sample_variation"] for _, kwargs in calls)
    assert calls[0][1]["gap_to_car_ahead"] == 0
    assert calls[1][1]["gap_to_car_ahead"] is None


def test_noise_free_lap_matches_zero_variation_and_preserves_default_sampling(setup):
    _, state, track = setup
    rng = np.random.default_rng(19)
    simulator = LapSimulator(rng)
    args = (state.driver, state.car, track, state.current_tire, Weather(), 30, 30)
    before = copy.deepcopy(rng.bit_generator.state)
    projected = simulator.calculate_lap_time(*args, sample_variation=False)
    assert rng.bit_generator.state == before

    class ZeroVariation:
        def normal(self, mean, std):
            return 0

    assert projected == LapSimulator(ZeroVariation()).calculate_lap_time(*args)
    legacy = LapSimulator(np.random.default_rng(19))
    assert simulator.calculate_lap_time(*args) == legacy.calculate_lap_time(
        *args, sample_variation=True
    )
    assert rng.random() == legacy.rng.random()
