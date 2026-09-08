"""A consumed wet pit schedule cannot replay its final window."""

import numpy as np
import pytest

from f1sim.analysis import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.events import EventManager
from f1sim.simulation.race import DriverRaceState, RaceSimulator


class AlwaysPit:
    def random(self):
        return 0.0


def fixture(plan, stops):
    sim = RaceSimulator(np.random.default_rng(42))
    sim.rng = AlwaysPit()
    # These unit fixtures exercise the changing-compound fallback windows.
    # Complete-race tests below construct their own unpatched runners.
    sim._rain_stint_can_be_planned = lambda *args: False
    state = DriverRaceState(
        Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
        position=1, pit_stops=stops, tire_laps=1,
        current_tire=TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
        pit_plan_options=[plan] if plan else [],
        tire_compound_history=["intermediate"],
    )
    track = Track(id="t", name="T", country="T", total_laps=50, base_lap_time=90)
    weather = Weather(condition=WeatherCondition.LIGHT_RAIN, track_wetness=0.4,
                      rain_intensity=0.4, change_probability=0)
    return sim, state, track, weather


@pytest.mark.parametrize("plan,stops", [([], 2), ([], 3), ([20], 1), ([20, 35], 2)])
def test_consumed_schedule_cannot_request_fresh_intermediates_again(plan, stops):
    sim, state, track, weather = fixture(plan, stops)
    for lap in range(35, 41):
        assert not sim._should_pit(state, [state], track, lap, False, weather)


@pytest.mark.parametrize("plan", [[], [20, 35]])
def test_second_window_remains_available_before_second_stop(plan):
    sim, state, track, weather = fixture(plan, 1)
    state.tire_laps = 18
    track.pit_lane_delta = 1
    assert sim._should_pit(state, [state], track, 35, False, weather)


def test_consumed_plan_still_allows_neutralized_opportunity_on_old_set():
    sim, state, track, weather = fixture([20, 35], 2)
    state.tire_laps = 35
    track.pit_lane_delta = 1
    sim.event_manager.safety_car_active = True
    assert sim._should_pit(state, [state], track, 40, True, weather)


def test_consumed_plan_still_allows_critical_weather_change():
    sim, state, track, weather = fixture([20, 35], 2)
    weather.track_wetness = 0
    weather.rain_intensity = 0
    assert sim._should_pit(state, [state], track, 40, False, weather)


def test_planned_stop_is_vetoed_when_fresh_set_cannot_repay_pit_loss():
    sim, state, track, weather = fixture([20, 35], 1)
    assert not sim._should_pit(state, [state], track, 35, False, weather)


def test_queue_cost_can_veto_an_otherwise_affordable_wet_stop():
    sim, state, track, weather = fixture([20, 35], 1)
    state.tire_laps = 18
    track.pit_lane_delta = 1
    assert sim._should_pit(state, [state], track, 35, False, weather)
    assert not sim._should_pit(state, [state], track, 35, False, weather,
                               additional_current_stop_cost=100)


def test_wet_cost_projection_preserves_original_fuel_distance(monkeypatch):
    sim, state, track, weather = fixture([20, 35], 1)
    calls = []

    def projection(*args, **kwargs):
        calls.append(kwargs)
        return False

    monkeypatch.setattr(sim, "_weather_stop_can_pay", projection)
    assert not sim._should_pit(state, [state], track, 35, False, weather,
                               physical_total_laps=70)
    assert calls == [{"traffic_possible": False, "physical_total_laps": 70}]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_full_fixed_rain_race_does_not_repeat_paid_windows(monkeypatch, engine):
    for method in ("_check_mechanical_failure", "_check_random_incident",
                   "_deploy_safety_measure", "_check_red_flag_conditions"):
        monkeypatch.setattr(EventManager, method, lambda *a, **kw: None)
    _, _, track, weather = fixture([], 0)
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(4)]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    result = MonteCarloRunner(drivers, cars, track, weather, seed=42,
                              race_engine=engine).run(1, parallel=False)
    for row in result.race_results[0]:
        assert row.laps_completed == 50
        assert 0 <= row.pit_stops <= 2
        assert all(b - a > 5 for a, b in zip(row.pit_laps, row.pit_laps[1:]))
