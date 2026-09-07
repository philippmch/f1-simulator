"""Elective dry schedules share a three-stop ceiling across team styles."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.pit_strategy import plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def fixture(laps=50, stops=0):
    state = DriverRaceState(
        Driver(id="A", name="A", team_id="A", tire_management=0.4),
        Car(team_id="A", team_name="A", tire_degradation_factor=1.5), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.SOFT], tire_laps=20,
        pit_stops=stops, tire_compound_history=["medium", "soft"],
    )
    track = Track(id="test", name="Test", country="Test", total_laps=laps,
                  base_lap_time=110, pit_lane_delta=20, tire_stress=1)
    return state, track


@pytest.mark.parametrize("style", list(TeamStrategyArchetype))
@pytest.mark.parametrize("laps,stops,lap", [(50, 1, 30), (78, 2, 50)])
def test_profitable_second_and_third_stops_are_available_to_every_style(style, laps, stops, lap):
    state, track = fixture(laps, stops)
    state.strategy_archetype = style
    simulator = RaceSimulator(np.random.default_rng(8))
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    assert simulator._should_pit(state, [state], track, lap, False, Weather())
    assert state.dry_pit_proposal is not None
    assert simulator.rng.bit_generator.state == before


@pytest.mark.parametrize("style", list(TeamStrategyArchetype))
def test_available_budget_does_not_force_unprofitable_stops(style):
    state, track = fixture(50, 1)
    state.strategy_archetype = style
    state.current_tire = TIRE_COMPOUNDS[TireCompound.HARD]
    state.tire_laps = 0
    assert not RaceSimulator()._should_pit(state, [state], track, 45, False, Weather())


@pytest.mark.parametrize("past_stops,history,expected", [
    (0, ["soft"], 3), (1, ["soft", "medium"], 2),
    (2, ["wet", "soft"], 1), (4, ["soft"], 1),
])
def test_dry_budget_forwarding_preserves_weather_consumption_and_extra_compliance(
    monkeypatch, past_stops, history, expected,
):
    import f1sim.simulation.race as race_module

    state, track = fixture(50, past_stops)
    state.tire_compound_history = history
    observed = []

    def capture(*args, **kwargs):
        observed.append(args[6])
        return plan_dry_stop(*args, **kwargs)

    monkeypatch.setattr(race_module, "plan_dry_stop", capture)
    RaceSimulator()._should_pit(state, [state], track, 25, False, Weather())
    assert observed == [expected]


@pytest.mark.parametrize("history", [["soft", "medium"], ["wet", "soft"]])
def test_exhausted_compliant_budget_does_not_open_fourth_elective_stop(history):
    state, track = fixture(50, 3)
    state.tire_compound_history = history
    assert not RaceSimulator()._should_pit(state, [state], track, 25, False, Weather())


@pytest.mark.parametrize("wetness,expected", [(0, 2), (0.1, 0)])
def test_free_restart_projection_uses_dry_budget_only_in_clear_dry(monkeypatch, wetness, expected):
    import f1sim.simulation.race as race_module

    state, track = fixture(50, 1)
    observed = []

    def capture(*args, **kwargs):
        observed.append(args[6])
        return plan_dry_stop(*args, **kwargs)

    monkeypatch.setattr(race_module, "plan_dry_stop", capture)
    RaceSimulator()._choose_red_flag_tire(state, Weather(track_wetness=wetness), track, 25)
    assert observed == [expected] * 3
