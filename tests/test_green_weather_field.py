"""Green paid forecasts use the field's weather at actual track entry."""

import pickle

import numpy as np
import pytest
from test_chronological_field_finish import FieldPhysics, field, native_path

import f1sim.simulation.race as race_module
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models._native import native_physics
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.rain_strategy import RainTransitionDecision
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext, paid_compound_candidates


def paid_entry_case(monkeypatch, wetness, delay, *, scheduled=False, follower=False):
    inputs = field(control="green", intervals=0, relative_laps=0 if follower else -1,
                   remaining=20., neutralized=False, now=6850.)
    driver, car, physical_track, observed, now = inputs
    assert observed.timeline._leader_id == ("B" if follower else "A")
    weather = Weather(track_wetness=wetness, rain_intensity=wetness if scheduled else 0.,
                      change_probability=0.)
    forecast = (WeatherForecastContext.from_schedule(
        [dict(lap=72, rain_intensity=0.)], leading_lap=71) if scheduled else None)
    calculator = LapSimulator(np.random.default_rng(4))

    def mean(self, driver, car, track, tire, surface, lap, total, **options):
        if driver.id != "A":
            return self.paces[driver.id]
        duration = calculator.calculate_lap_time(
            driver, car, track, tire, surface, lap, total, sample_variation=False, **options)
        self.entries.append((lap, surface.track_wetness, duration))
        return duration

    monkeypatch.setattr(FieldPhysics, "calculate_lap_time", mean)
    assert native_physics(driver, car, physical_track, weather)
    _, _, actual = native_path(inputs, stopped=True, delay=delay, weather=weather,
                               forecast_context=forecast)
    lap, entry_surface, duration = actual.entries[0]
    assert lap == 72
    assert entry_surface == pytest.approx(wetness - .03 * int(follower and delay >= 20.)
                                         - .03 * int(delay >= 120. if follower else delay > 120.)
                                         - .03 * int(delay >= 220.))
    stock = TireInventory.from_sets([
        dict(id="M", compound="medium", age=8), dict(id="S", compound="soft")])
    stock.fit("M")
    state = DriverRaceState(
        driver, car, 1, total_time=now, laps_completed=71,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM], tire_laps=8,
        tire_compound_history=["hard", "medium"], tire_inventory=stock,
        strategy_control_context=StrategyControlContext(observed, now, delay),
    )
    simulator = RaceSimulator(np.random.default_rng(4))
    simulator.weather_forecast_context = forecast
    # Only price the first running lap; retain the original fuel denominator.
    planning = physical_track.model_copy(update={"total_laps": 72})
    return state, simulator, planning, weather, delay + duration, entry_surface


@pytest.mark.parametrize("wetness,scheduled", [(.18, False), (.24, False), (.4, False),
                                              (.4, True)])
@pytest.mark.parametrize("delay", [20., 120., 130., 220., 230.])
def test_green_inventory_forecast_matches_native_paid_entry(monkeypatch, wetness, scheduled, delay):
    state, simulator, planning, weather, expected, _ = paid_entry_case(
        monkeypatch, wetness, delay, scheduled=scheduled)
    before = pickle.dumps((state, simulator.rng.bit_generator.state))
    result = simulator._plan_inventory(
        state, planning, weather, 72, force_stop=True, physical_total_laps=90,
        additional_current_stop_cost=delay - planning.pit_lane_delta
        - expected_stationary_time(state.car),
    )
    assert result.set_id == "S"
    assert result.pit_now_laps == 1
    assert result.pit_now_cost == pytest.approx(expected, abs=1.e-8)
    assert pickle.dumps((state, simulator.rng.bit_generator.state)) == before


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("delay", [20., 130., 230.])
def test_follower_external_clock_already_matches_native_paid_entry(monkeypatch, scheduled, delay):
    state, simulator, track, weather, expected, _ = paid_entry_case(
        monkeypatch, .4, delay, scheduled=scheduled, follower=True)
    state.strategy_control_context = None
    clock = StrategyWeatherClock((0.,), 20., 100., 18, delay,
                                 track.pit_lane_delta + expected_stationary_time(state.car))
    before = pickle.dumps((state, simulator.rng.bit_generator.state))
    result = simulator._plan_inventory(
        state, track, weather, 72, force_stop=True, physical_total_laps=90, weather_clock=clock,
        current_traffic_gaps=(None, 0. if delay == 20. else None),
        additional_current_stop_cost=delay - track.pit_lane_delta
        - expected_stationary_time(state.car))
    assert result.set_id == "S"
    assert result.pit_now_cost == pytest.approx(expected, abs=1.e-8)
    assert pickle.dumps((state, simulator.rng.bit_generator.state)) == before


@pytest.mark.parametrize("route", ["timing", "paid_fit", "reaction"])
@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize("delay", [130., 230.])
def test_unlimited_weather_routes_price_the_same_native_entry(
    monkeypatch, route, scheduled, delay,
):
    state, simulator, track, weather, _, entry_wetness = paid_entry_case(
        monkeypatch, .4, delay, scheduled=scheduled)
    state.tire_inventory = None
    entry = weather.model_copy(update={"track_wetness": entry_wetness, "rain_intensity": 0.})
    calculator = LapSimulator(np.random.default_rng(4))
    scores = {
        compound: calculator.calculate_lap_time(
            state.driver, state.car, track, TIRE_COMPOUNDS[compound], entry, 72, 90,
            sample_variation=False)
        for compound in paid_compound_candidates(weather)
    }
    expected_compound = min(scores, key=scores.get)
    captured = []
    name = "weather_stop_costs" if route == "reaction" else "plan_rain_transition"
    original = getattr(race_module, name)

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(race_module, name, record)
    options = dict(physical_total_laps=90, additional_current_stop_cost=delay
                   - track.pit_lane_delta - expected_stationary_time(state.car))
    before_rng = pickle.dumps(simulator.rng.bit_generator.state)
    if route == "timing":
        simulator._should_pit(
            state, [state], track, 72, False, weather,
            traffic_snapshot=StrategyTrafficSnapshot(None, None, 0., (None, None)),
            current_overtake_mode_allowed=False, **options)
    elif route == "paid_fit":
        assert simulator._choose_forecast_paid_compound(
            state, weather, track, 72, **options) == expected_compound
    else:
        simulator._weather_stop_can_pay(state, track, weather, 72,
                                       traffic_possible=False, **options)
    assert len(captured) == 1
    assert captured[0].pit_now_laps == 1
    assert captured[0].pit_now_cost == pytest.approx(delay + scores[expected_compound], abs=1.e-8)
    assert pickle.dumps(simulator.rng.bit_generator.state) == before_rng


def test_green_field_retained_mode_gain_is_applied_once(monkeypatch):
    state, simulator, track, weather, _, _ = paid_entry_case(monkeypatch, .18, 20.)
    state.tire_inventory = None
    state.position = 2
    decisions = []
    original = RainTransitionDecision.should_pit

    def record(decision):
        decisions.append(decision)
        return original(decision)

    monkeypatch.setattr(RainTransitionDecision, "should_pit", record)
    for energy in (0., 1.):
        state.overtake_mode_energy = energy
        simulator._should_pit(
            state, [state], track, 72, False, weather, physical_total_laps=90,
            traffic_snapshot=StrategyTrafficSnapshot(1., None, 0., (1., None)),
            current_overtake_mode_allowed=True)
    assert len(decisions) == 2
    assert decisions[0].pit_now_cost == decisions[1].pit_now_cost
    assert decisions[0].compound == decisions[1].compound
    gain = simulator._strategy_mode_gain(state, track, weather, 72, True, 1., 90)
    assert gain > 0.
    assert decisions[0].wait_cost - decisions[1].wait_cost == pytest.approx(gain)
    assert state.overtake_mode_energy == 1.


class MeanLapNoise:
    def normal(self, mean, std):
        return mean


@pytest.mark.parametrize("scenario,custom,expected", [
    ("drying", None, True), ("rising", None, True), ("scheduled", None, True),
    ("dry", None, False), ("balanced", None, False),
    ("drying", "calculator", False), ("drying", "modifier", False),
    ("drying", "aero", False), ("drying", "class_modifier", False),
])
@pytest.mark.parametrize("lane", [20., 130.])
def test_green_context_is_admitted_only_for_native_changing_weather(
    monkeypatch, scenario, custom, expected, lane,
):
    simulator = RaceSimulator(np.random.default_rng(19))
    engine = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="T", name="T", country="Test", total_laps=4, base_lap_time=90.,
                  safety_car_probability=0., pit_lane_delta=lane)
    weather = {
        "drying": Weather(track_wetness=.3, change_probability=0.),
        "rising": Weather(track_wetness=.3, rain_intensity=.4, change_probability=0.),
        "dry": Weather(change_probability=0.),
        "balanced": Weather(track_wetness=.4, rain_intensity=.4, change_probability=0.),
        "scheduled": Weather(track_wetness=.3, rain_intensity=.3, change_probability=0.),
    }[scenario]
    schedule = [dict(lap=3, rain_intensity=0.)] if scenario == "scheduled" else None
    simulator.lap_simulator.rng = MeanLapNoise()
    if custom == "calculator":
        monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lambda *a, **k: 90.)
    elif custom == "modifier":
        monkeypatch.setattr(simulator.event_manager, "get_lap_time_modifier", lambda: 2.)
    elif custom == "aero":
        monkeypatch.setattr(simulator.event_manager, "is_active_aero_allowed", lambda: False)
    elif custom == "class_modifier":
        monkeypatch.setattr(type(simulator.event_manager), "get_lap_time_modifier", lambda self: 2.)
    seen = []

    def decide(state, states, planning, lap, *args, **kwargs):
        if lap > 1:
            seen.append((state.driver.id, lap, state.strategy_control_context,
                         kwargs.get("weather_clock")))
        return False

    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(simulator.event_manager, "_check_severe_weather_red_flag", lambda *a: None)
    assert native_physics(), "The running formula itself stays native"
    engine.run(drivers, cars, track, weather, list("AB"),
               starting_tires={key: TireCompound.MEDIUM for key in "AB"},
               weather_schedule=schedule)
    assert seen
    expected = expected and lane == 130.
    assert any(context is not None for _, _, context, _ in seen) == expected
    assert all((context is not None) == (expected and clock is None)
               for _, _, context, clock in seen)
    if expected:
        assert any(clock is not None for _, _, _, clock in seen)
    assert all(state.strategy_control_context is None for state in engine.states.values())
