"""Rain planning bypasses calendar windows without bypassing safety priorities."""

from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot


def fixture():
    sim = RaceSimulator(np.random.default_rng(42))
    state = DriverRaceState(
        Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], tire_laps=20,
        tire_compound_history=["intermediate"], pit_stops=1, pit_plan_options=[[30]],
    )
    track = Track(id="t", name="T", country="T", total_laps=50, base_lap_time=90)
    weather = Weather(condition=WeatherCondition.LIGHT_RAIN, rain_intensity=0.35,
                      track_wetness=0.35, change_probability=0)
    return sim, state, track, weather


@pytest.mark.parametrize("lap", [3, 10, 49])
def test_rain_planner_can_act_outside_windows_with_remaining_budget(monkeypatch, lap):
    sim, state, track, weather = fixture()
    calls = []

    def plan(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(should_pit=lambda: True)

    monkeypatch.setattr("f1sim.simulation.race.plan_rain_stop", plan)
    before = sim.rng.bit_generator.state
    snapshot = StrategyTrafficSnapshot(None, None, 2)
    assert sim._should_pit(state, [state], track, lap, False, weather,
                           additional_current_stop_cost=3, physical_total_laps=70,
                           traffic_snapshot=snapshot)
    args, kwargs = calls[0]
    assert args[-2:] == (lap, 3)
    multiplier = sim.lap_simulator.weather_pace_multiplier(state.driver, state.car, weather)
    assert kwargs["additional_current_stop_cost"] == pytest.approx(3 + 2 * multiplier)
    assert kwargs["physical_total_laps"] == 70
    assert sim.rng.bit_generator.state == before


def test_projected_drying_uses_compound_transition_planner(monkeypatch):
    sim, state, track, weather = fixture()
    weather.rain_intensity = 0
    weather.condition = WeatherCondition.CLOUDY
    monkeypatch.setattr("f1sim.simulation.race.plan_rain_stop",
                        lambda *a, **k: pytest.fail("Drying needs compound transitions"))
    calls = []

    def transition(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(should_pit=lambda: True, compound=TireCompound.SOFT)

    monkeypatch.setattr("f1sim.simulation.race.plan_rain_transition", transition)
    assert not sim._rain_stint_can_be_planned(state, track, weather, 10)
    assert sim._should_pit(state, [state], track, 10, False, weather)
    assert len(calls) == 1
    assert state.weather_pit_proposal == (10, TireCompound.SOFT)


def test_critical_mismatch_and_budget_keep_priority(monkeypatch):
    sim, state, track, weather = fixture()
    monkeypatch.setattr("f1sim.simulation.race.plan_rain_stop",
                        lambda *a, **k: pytest.fail("Priority path should bypass planner"))
    state.pit_stops = 4
    assert not sim._should_pit(state, [state], track, 10, False, weather)
    weather.track_wetness = 0
    weather.rain_intensity = 0
    assert sim._should_pit(state, [state], track, 10, False, weather)


def test_actual_rain_costs_choose_worn_set_stop_outside_calendar_window():
    sim, state, track, weather = fixture()
    track.pit_lane_delta = 1
    state.tire_laps = 35
    assert sim._should_pit(state, [state], track, 10, False, weather)
    state.tire_laps = 0
    assert not sim._should_pit(state, [state], track, 10, False, weather)
