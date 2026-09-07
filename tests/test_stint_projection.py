"""Dry compound choices project the same fresh-set pace used by the race."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import DriverRaceState, RaceSimulator

SLICKS = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]


def setup_state():
    track = Track(
        id="test", name="Test", country="Test", total_laps=63, base_lap_time=90, tire_stress=0.6
    )
    state = DriverRaceState(
        driver=Driver(id="A", name="A", team_id="A"),
        car=Car(team_id="A", team_name="A"),
        position=1,
        current_tire=TIRE_COMPOUNDS[TireCompound.HARD].model_copy(deep=True),
        planned_pit_laps=[21, 42],
    )
    return state, track


@pytest.mark.parametrize(
    "lap,stops,plan,expected",
    [
        (18, 0, [21, 42], 24),
        (21, 0, [21, 42], 21),
        (24, 0, [21, 42], 18),
        (42, 1, [21, 42], 22),
        (45, 0, [21, 42], 19),
        (24, 0, [21, 22, 50], 26),
        (60, 3, [21, 42], 4),
        (63, 0, [], 1),
        (20, 0, [21, 80], 44),
    ],
)
def test_next_stint_consumes_current_slot_and_skips_stale_entries(lap, stops, plan, expected):
    state, track = setup_state()
    state.pit_stops = stops
    state.planned_pit_laps = plan
    assert RaceSimulator._next_stint_laps(state, track, lap) == expected


def test_short_high_stress_stint_still_prefers_soft():
    state, track = setup_state()
    track.tire_stress = 1.0
    state.planned_pit_laps = [21, 26]
    assert RaceSimulator()._choose_compound_for_next_stint(state, track, 21) == TireCompound.SOFT
    state.planned_pit_laps = [21, 51]
    assert RaceSimulator()._choose_compound_for_next_stint(state, track, 21) != TireCompound.SOFT


def test_unused_compounds_are_ranked_before_selection():
    state, track = setup_state()
    state.current_tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    state.tire_compound_history = [TireCompound.SOFT.value]
    state.planned_pit_laps = [21, 26]
    # Fastest unrestricted set is used soft. Medium is materially faster than
    # the other unused option, hard, and must win regardless of random seed.
    for seed in range(20):
        sim = RaceSimulator(rng=np.random.default_rng(seed))
        assert sim._choose_distinct_dry_compound(state, track, 21) == TireCompound.MEDIUM


def test_projection_uses_car_wear_and_driver_management_without_randomness():
    state, track = setup_state()
    sim = LapSimulator(np.random.default_rng(7))
    before = copy.deepcopy(sim.rng.bit_generator.state)
    tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    baseline = sim.projected_tire_stint_cost(state.driver, state.car, track, tire, 25)
    state.car.tire_degradation_factor = 1.3
    assert sim.projected_tire_stint_cost(state.driver, state.car, track, tire, 25) > baseline
    state.car.tire_degradation_factor = 1.0
    state.driver.tire_management = 0.1
    assert sim.projected_tire_stint_cost(state.driver, state.car, track, tire, 25) > baseline
    assert sim.rng.bit_generator.state == before
    assert state.driver.current_tire_laps == 0


def test_projected_differences_match_controlled_actual_laps():
    state, track = setup_state()
    actual, projected = {}, {}
    for compound in SLICKS:
        sim = LapSimulator(np.random.default_rng(42))
        tire = TIRE_COMPOUNDS[compound]
        actual[compound] = 0.0
        for age in range(35):
            state.driver.current_tire_laps = age
            actual[compound] += sim.calculate_lap_time(
                state.driver,
                state.car,
                track,
                tire,
                Weather(change_probability=0),
                lap_number=age + 1,
                total_laps=63,
            )
        projected[compound] = sim.projected_tire_stint_cost(
            state.driver, state.car, track, tire, 35
        )
    for compound in SLICKS:
        assert actual[compound] - actual[TireCompound.MEDIUM] == pytest.approx(
            projected[compound] - projected[TireCompound.MEDIUM], abs=1e-9
        )


def test_damp_fallback_projects_next_stop_not_current_stop(monkeypatch):
    state, track = setup_state()
    sim = RaceSimulator(rng=np.random.default_rng(9))
    horizons = []
    original = sim._rank_stint_compounds

    def record(state, track, current_lap, available):
        horizons.append(
            (current_lap, state.pit_stops, sim._next_stint_laps(state, track, current_lap))
        )
        return original(state, track, current_lap, available)

    monkeypatch.setattr(sim, "_rank_stint_compounds", record)
    monkeypatch.setattr(sim, "_plan_pit_lap_options", lambda *args, **kwargs: [[21, 42]])
    monkeypatch.setattr(
        sim, "_should_pit", lambda state, states, track, lap, *args, **kwargs: lap in [21, 42]
    )
    monkeypatch.setattr(sim.event_manager, "process_lap", lambda **kwargs: [])
    sim.simulate_race(
        [state.driver], {"A": state.car}, track,
        Weather(track_wetness=0.1, rain_intensity=0.1, change_probability=0), ["A"],
    )
    assert horizons == [(21, 0, 21), (42, 1, 22)]


def test_exhausted_ordinary_stop_budget_projects_to_finish():
    state, track = setup_state()
    track.total_laps = 30
    state.planned_pit_laps = [10, 21]
    assert RaceSimulator._next_stint_laps(state, track, 10) == 21


def test_weather_stop_consumes_ordinary_plan_slot():
    state, track = setup_state()
    sim = RaceSimulator(rng=np.random.default_rng(5))
    sim._execute_pit_stop(state, track, Weather(track_wetness=0.5), current_lap=10)
    assert state.current_tire.compound == TireCompound.INTERMEDIATE
    # The race loop records the completed stop after tyre selection.
    state.pit_stops += 1
    # A later dry stop consumes the final ordinary slot, regardless of the
    # wet stop happening before the original first scheduled stop.
    assert sim._next_stint_laps(state, track, 20) == 44
