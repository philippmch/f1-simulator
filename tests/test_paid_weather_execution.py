"""Paid weather decisions, fitting metadata, and finish guard agree."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.rain_strategy import RainTransitionDecision
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("mode,expected,reason", [
    ("proposal", TireCompound.WET, "rain_forecast"),
    ("forced", TireCompound.WET, "forced_repair"),
    ("critical", TireCompound.WET, "critical_weather"),
    ("custom", TireCompound.INTERMEDIATE, "critical_weather"),
    ("inventory", TireCompound.INTERMEDIATE, "rain_forecast"),
])
def test_both_engines_fit_proposals_or_complete_paid_choice_with_precedence(
    monkeypatch, engine, mode, expected, reason,
):
    simulator = RaceSimulator(np.random.default_rng(7))
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="T", total_laps=12,
                  base_lap_time=90, pit_lane_delta=5)
    native_decision = simulator._should_pit

    def decide(state, states, track, lap, window, weather=None, **kwargs):
        # Isolate the lap-two paid choice; the scripted first lap deliberately
        # supplies the stated critical-soft decision snapshot for both engines.
        if lap != 2:
            return False
        if mode == "critical":
            return native_decision(state, states, track, lap, window, weather, **kwargs)
        if mode == "forced":
            state.force_pit_next_lap = True
        else:
            state.weather_pit_proposal = (lap, TireCompound.WET)
            simulator._capture_pit_decision_context(
                state, lap, "rain_forecast", RainTransitionDecision(10., 15., TireCompound.WET),
            )
        if mode == "inventory":
            state.inventory_pit_proposal = (lap, "I")
        return True

    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        expected_stationary_time)
    options = {"starting_tires": {"A": TireCompound.SOFT}, "starting_tire_ages": {"A": 5},
               "weather_schedule": [{"lap": 3, "rain_intensity": 1}]}
    if mode == "custom":
        options["pit_plans"] = {"A": [{"lap": 2, "compound": "intermediate"}]}
    if mode == "inventory":
        options["tire_inventory"] = {"A": [
            {"id": "S", "compound": "soft", "age": 5},
            {"id": "I", "compound": "intermediate", "age": 0},
            {"id": "W", "compound": "wet", "age": 0},
        ]}
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result = run([driver], {"A": car}, track, Weather(track_wetness=.5, rain_intensity=.5),
                 ["A"], **options)[0]
    assert result.pit_laps == [2]
    assert result.pit_stop_details[0]["to_compound"] == expected.value
    assert result.pit_stop_details[0]["decision_reason"] == reason
    assert result.strategy == ["soft", expected.value]
    assert result.pit_stop_details[0]["track_wetness"] == .5
    if mode == "inventory":
        assert result.pit_stop_details[0]["to_set_id"] == "I"
    if mode == "proposal":
        assert result.pit_stop_details[0]["forecast_saving_seconds"] == 5.


def state_and_simulator(*, water=.5, rain=.5):
    simulator = RaceSimulator(np.random.default_rng(7))
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="T", total_laps=12,
                  base_lap_time=90, pit_lane_delta=5)
    state = DriverRaceState(driver, car, 1, current_tire=TIRE_COMPOUNDS[TireCompound.SOFT],
                            tire_laps=5, laps_completed=1)
    simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
        [{"lap": 3, "rain_intensity": 1}], leading_lap=2,
    )
    return simulator, state, track, Weather(track_wetness=water, rain_intensity=rain)


@pytest.mark.parametrize("external", [False, True])
def test_complete_paid_chooser_prices_service_delay_before_actual_fit(external):
    simulator, state, track, weather = state_and_simulator()
    clock = (StrategyWeatherClock(tuple(index * 170. for index in range(11)),
                                  10., 90., 6, 7., 7.) if external else None)
    expected = simulator._choose_forecast_paid_compound(
        state, weather, track, 2, weather_clock=clock,
    )
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False,
                                weather_clock=clock)
    assert expected == state.current_tire.compound == TireCompound.WET
    assert state.pit_stop_details[0]["to_compound"] == "wet"


def test_forced_dry_repair_retains_safe_candidates_after_elective_budget_exhaustion():
    simulator, state, track, _ = state_and_simulator()
    state.pit_stops = 4
    state.force_pit_next_lap = True
    state.tire_compound_history = ["soft", "hard"]
    simulator._execute_pit_stop(state, track, Weather(), 2, sample_service=False)
    assert state.current_tire.compound in {
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    }
    assert state.pit_stop_details[0]["decision_reason"] == "forced_repair"


@pytest.mark.parametrize("proposal", [None, TireCompound.WET, TireCompound.INTERMEDIATE])
def test_finish_guard_replacement_choices_match_scheduled_execution(proposal):
    simulator, state, track, weather = state_and_simulator()
    engine = ChronologicalRace(simulator)
    engine.weather = weather
    if proposal is not None:
        state.weather_pit_proposal = (2, proposal)
    options = engine._finish_replacement_options(state, lap=2)
    expected = ({TireCompound.INTERMEDIATE, TireCompound.WET} if proposal is None else {proposal})
    assert {option.compound for option in options} == expected
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False)
    assert state.current_tire.compound in expected


def test_finish_guard_preserves_custom_and_physical_set_precedence():
    simulator, state, _, weather = state_and_simulator()
    engine = ChronologicalRace(simulator)
    engine.weather = weather
    state.weather_pit_proposal = (2, TireCompound.WET)
    state.pit_plan_target = TireCompound.INTERMEDIATE
    assert engine._finish_replacement_options(state, lap=2)[0].compound == TireCompound.INTERMEDIATE
    pool = TireInventory.from_sets([
        {"id": "S", "compound": "soft", "age": 5},
        {"id": "I", "compound": "intermediate", "age": 3},
        {"id": "W", "compound": "wet", "age": 1},
    ])
    simulator._initialize_inventory(state, pool, pool.sets["S"])
    state.inventory_pit_proposal = (2, "W")
    options = engine._finish_replacement_options(state, lap=2)
    assert len(options) == 1
    assert options[0].compound == TireCompound.WET and options[0].age == 1
    assert options[0].identifier == "W"


@pytest.mark.parametrize("context", [None, WeatherForecastContext(())])
def test_empty_and_no_context_honor_safe_proposal_without_rng(context):
    simulator, state, track, weather = state_and_simulator()
    simulator.weather_forecast_context = context
    state.weather_pit_proposal = (2, TireCompound.WET)
    before = deepcopy(simulator.rng.bit_generator.state)
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False)
    assert state.current_tire.compound == TireCompound.WET
    assert simulator.rng.bit_generator.state == before


def test_critical_scheduled_proposal_cannot_override_safe_paid_chooser():
    simulator, state, track, _ = state_and_simulator()
    state.weather_pit_proposal = (2, TireCompound.WET)
    simulator._execute_pit_stop(state, track, Weather(), 2, sample_service=False)
    assert state.current_tire.compound in {
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    }


@pytest.mark.parametrize("external", [False, True])
def test_ordinary_drying_chooser_and_finish_options_match_actual_fit(external):
    simulator, state, track, weather = state_and_simulator(water=.76, rain=0)
    simulator.weather_forecast_context = None
    clock = (StrategyWeatherClock(tuple(index * 170. for index in range(11)),
                                  10., 90., 6, 7., 7.) if external else None)
    before = deepcopy(simulator.rng.bit_generator.state)
    chosen = simulator._choose_forecast_paid_compound(
        state, weather, track, 2, weather_clock=clock,
    )
    assert weather.fresh_rain_compound() == TireCompound.WET
    assert chosen == TireCompound.INTERMEDIATE
    engine = ChronologicalRace(simulator)
    engine.weather = weather
    # A stale dry suggestion must not narrow the rainy paid chooser's options.
    state.dry_pit_proposal = (2, TireCompound.WET)
    options = engine._finish_replacement_options(state, lap=2)
    assert {option.compound for option in options} == {
        TireCompound.INTERMEDIATE, TireCompound.WET,
    }
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False,
                                weather_clock=clock)
    assert state.current_tire.compound == chosen
    assert simulator.rng.bit_generator.state == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_both_engines_fit_safe_intermediate_during_ordinary_drying(monkeypatch, engine):
    simulator, state, track, weather = state_and_simulator(water=.76, rain=0)
    native_decision = simulator._should_pit

    def decide(current, states, track, lap, window, weather=None, **kwargs):
        if lap != 2:
            return False
        # Supply the independently priced drying snapshot at the commitment.
        weather.track_wetness = .76
        weather.rain_intensity = 0
        return native_decision(current, states, track, lap, window, weather, **kwargs)

    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        expected_stationary_time)
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result = run([state.driver], {"A": state.car}, track, weather, ["A"],
                 starting_tires={"A": TireCompound.SOFT}, starting_tire_ages={"A": 5})[0]
    assert result.pit_laps == [2]
    detail = result.pit_stop_details[0]
    assert detail["to_compound"] == "intermediate"
    assert detail["decision_reason"] == "critical_weather"
    assert detail["track_wetness"] == .76


def test_final_planning_horizon_rejects_unused_rain_exemption_and_allows_later_correction():
    simulator, state, track, weather = state_and_simulator(water=.1, rain=.1)
    simulator.weather_forecast_context = None
    state.current_tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    state.tire_laps = state.prior_tire_laps = 0
    state.tire_compound_history = ["soft", "intermediate"]
    state.weather_pit_proposal = (3, TireCompound.SOFT)
    planning = track.model_copy(update={"total_laps": 3})
    assert simulator._automatic_weather_fit_is_eligible(state, weather, TireCompound.SOFT, 3, track)
    assert not simulator._automatic_weather_fit_is_eligible(
        state, weather, TireCompound.SOFT, 3, planning,
    )
    engine = ChronologicalRace(simulator)
    engine.weather = weather
    options = engine._finish_replacement_options(state, lap=3, planning=planning)
    assert TireCompound.SOFT not in {option.compound for option in options}
    simulator._execute_pit_stop(state, planning, weather, 3, sample_service=False,
                                physical_total_laps=track.total_laps)
    assert simulator._pit_plan_satisfies_rule(state, state.current_tire.compound)
    assert state.current_tire.compound == TireCompound.MEDIUM


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_both_engines_reject_final_proposal_crediting_an_unrun_rain_set(monkeypatch, engine):
    simulator, state, track, weather = state_and_simulator(water=.1, rain=.1)
    track.total_laps = 3

    def decide(current, states, track, lap, *args, **kwargs):
        if lap != 3:
            return False
        simulator._fit_tire(current, TireCompound.INTERMEDIATE)
        assert simulator._actually_used_compounds(current) == {TireCompound.SOFT}
        current.weather_pit_proposal = (lap, TireCompound.SOFT)
        return True

    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result = run([state.driver], {"A": state.car}, track, weather, ["A"],
                 starting_tires={"A": TireCompound.SOFT})[0]
    assert result.pit_laps == [3]
    assert result.strategy == ["soft", "medium"]
    assert result.pit_stop_details[0]["to_compound"] == "medium"
