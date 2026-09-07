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


def test_profitable_long_wet_stint_retains_existing_probability(setup):
    simulator, state, track = setup
    assert simulator._should_pit(state, [state], track, 10, False,
                                Weather(track_wetness=0.3, rain_intensity=0.3))
    expected_rng = np.random.default_rng(2)
    expected_rng.random()
    assert simulator.rng.random() == expected_rng.random()


def test_projected_critical_mismatch_prevents_cost_veto(setup):
    simulator, state, track = setup
    weather = Weather(track_wetness=0.34, rain_intensity=0.7)
    assert simulator._check_tire_weather_mismatch(state.current_tire, weather) == "suboptimal"
    assert simulator._weather_stop_can_pay(state, track, weather, 29, 10000)


def test_drying_projection_uses_shared_surface_updates_without_mutation(setup, monkeypatch):
    simulator, state, track = setup
    weather = Weather(track_wetness=0.21, rain_intensity=0, change_probability=0)
    captured = []

    def lap_time(*args, **kwargs):
        captured.append((args[4].track_wetness, args[3].compound, args[0].current_tire_laps))
        return 90

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap_time)
    assert not simulator._weather_stop_can_pay(state, track, weather, 28)
    assert sorted(set(w for w, _, _ in captured)) == pytest.approx([0.15, 0.18, 0.21])
    assert any(w == 0.21 and c == TireCompound.INTERMEDIATE and age == 0
               for w, c, age in captured)
    assert any(w < 0.2 and c == TireCompound.SOFT for w, c, _ in captured)
    assert weather.track_wetness == 0.21
    projected = weather.project_surface()
    evolved = weather.evolve(np.random.default_rng(2))
    assert projected == evolved


def test_only_current_running_gain_is_neutralized_and_future_gains_are_nonnegative(
    setup, monkeypatch,
):
    simulator, state, track = setup
    simulator.event_manager.safety_car_active = True
    track.pit_lane_delta = 1

    def lap_time(*args, **kwargs):
        # Old set gains 10s on first and last lap, loses 10s on middle lap.
        return 90 if args[0].current_tire_laps == 0 else (80 if args[5] == 29 else 100)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap_time)
    threshold = 24 - 0.55 - expected_stationary_time(state.car)
    weather = Weather(track_wetness=0.3, rain_intensity=0.3)
    assert simulator._weather_stop_can_pay(state, track, weather, 28, threshold - 0.01)
    assert not simulator._weather_stop_can_pay(state, track, weather, 28, threshold + 0.01)


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
    assert simulator._weather_stop_can_pay(state, track, weather, 30, threshold - 0.01)
    assert not simulator._weather_stop_can_pay(state, track, weather, 30, threshold + 0.01)
    assert sum(age == 8 for age, _ in calls) == 2
    assert all(age in (8, 0) for age, _ in calls)
    assert all(not kwargs["sample_variation"] for _, kwargs in calls)
    assert all(kwargs["gap_to_car_ahead"] == (0 if age else None)
               for age, kwargs in calls)


def test_intermediate_retention_window_is_included_in_optimistic_bound(setup):
    simulator, state, track = setup
    state.current_tire = TIRE_COMPOUNDS[TireCompound.WET]
    state.tire_laps = 60
    state.car.tire_degradation_factor = 1.5
    track.tire_stress = 1.0
    state.tire_compound_history = ["wet"]
    weather = Weather(track_wetness=0.75, rain_intensity=0.75)
    assert simulator._choose_weather_compound(weather) == TireCompound.WET
    assert simulator._check_tire_weather_mismatch(
        TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], weather
    ) != "critical"

    def pace(compound, age, gap):
        driver = state.driver.model_copy(update={"current_tire_laps": age})
        return simulator.lap_simulator.calculate_lap_time(
            driver, state.car, track, TIRE_COMPOUNDS[compound], weather, 30, 30,
            gap_to_car_ahead=gap, sample_variation=False,
        )

    old = pace(TireCompound.WET, 60, 0)
    wet = pace(TireCompound.WET, 0, None)
    inter = pace(TireCompound.INTERMEDIATE, 0, None)
    assert inter < wet
    midpoint_gain = old - (wet + inter) / 2
    # Choose a stop cost between the wet-only and intermediate-aware gains.
    track.pit_lane_delta = 0.1
    state.car.pit_stop_avg = 1.8
    state.car.pit_stop_std = 0
    queue = midpoint_gain - track.pit_lane_delta - expected_stationary_time(state.car)
    assert queue >= 0
    assert simulator._weather_stop_can_pay(state, track, weather, 30, queue)


def test_drying_projection_does_not_freeze_today_slick_eligibility(setup, monkeypatch):
    simulator, state, track = setup
    state.current_tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    state.tire_laps = 0
    state.tire_compound_history = ["soft", "medium"]
    assert simulator._used_slick_compounds(state) == {TireCompound.SOFT}
    assert simulator._stay_satisfies_tire_rule(state)
    weather = Weather(track_wetness=0.21, rain_intensity=0)
    track.pit_lane_delta = 1

    def pace(*args, **kwargs):
        tire, projected = args[3], args[4]
        if kwargs["gap_to_car_ahead"] == 0:  # Original set's projected path.
            return 100
        if projected.track_wetness > 0.2:
            return 99 if tire.compound == TireCompound.INTERMEDIATE else 101
        return 90 if tire.compound == TireCompound.SOFT else 100

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", pace)
    # First lap's rain set gains1s, then legal fresh soft gains10s as it dries.
    # A fixed actual-used restriction wrongly excluded soft and saw only1s.
    queue = 6 - track.pit_lane_delta - expected_stationary_time(state.car)
    assert queue > 0
    assert simulator._weather_stop_can_pay(state, track, weather, 29, queue)


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
