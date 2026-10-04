"""Conditional instructions use observed control and retain their fixed deadline."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_plans import (
    current_pit_plan_instruction,
    initialize_pit_plan_state,
    parse_pit_plan_spec,
    pit_plan_may_stop,
    validate_pit_plans,
)
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator


def window(trigger="safety_car", earliest=3, deadline=6, compound="hard"):
    return {"lap": deadline, "compound": compound, "earliest_lap": earliest, "trigger": trigger}


def test_shorthand_and_json_agree_and_copy_window_inputs():
    source = {"A": [window(), {"lap": 8, "compound": "soft"}], "B": []}
    assert parse_pit_plan_spec("A=3-6@sc:hard,8:soft;B=none") == source
    normalized = validate_pit_plans(source, ["A", "B"], total_laps=10)
    normalized["A"][0]["earliest_lap"] = 4
    assert source["A"][0]["earliest_lap"] == 3
    assert parse_pit_plan_spec("A=3-6@vsc:hard")["A"] == [window("vsc")]
    assert parse_pit_plan_spec("A=3-6@neutralized:hard")["A"] == [window("neutralized")]


@pytest.mark.parametrize("changes", [
    {"earliest_lap": True}, {"earliest_lap": 3.0}, {"earliest_lap": "3"},
    {"earliest_lap": 1}, {"earliest_lap": 6}, {"earliest_lap": 7},
    {"trigger": "sc"}, {"trigger": "green"}, {"trigger": []}, {"extra": 1},
])
def test_reject_invalid_windows(changes):
    with pytest.raises(ValueError):
        validate_pit_plans({"A": [window() | changes]})


@pytest.mark.parametrize("record", [
    {"lap": 6, "compound": "hard", "earliest_lap": 3},
    {"lap": 6, "compound": "hard", "trigger": "safety_car"},
])
def test_window_options_are_an_atomic_pair(record):
    with pytest.raises(ValueError):
        validate_pit_plans({"A": [record]})


def test_windows_cannot_overlap_or_exceed_original_distance():
    with pytest.raises(ValueError, match="after"):
        validate_pit_plans({"A": [{"lap": 3, "compound": "soft"}, window()]})
    with pytest.raises(ValueError, match="exceed"):
        validate_pit_plans({"A": [window()]}, total_laps=5)


@pytest.mark.parametrize("trigger, sc, vsc, due", [
    ("safety_car", True, False, True), ("safety_car", False, True, False),
    ("vsc", False, True, True), ("vsc", True, False, False),
    ("neutralized", True, False, True), ("neutralized", False, True, True),
    ("neutralized", False, False, False),
])
def test_due_detection_is_observed_and_deadline_is_unconditional(trigger, sc, vsc, due):
    state = SimpleNamespace()
    initialize_pit_plan_state(state, [window(trigger)])
    assert current_pit_plan_instruction(state, 2, safety_car=True, vsc=True) is None
    assert bool(current_pit_plan_instruction(state, 3, safety_car=sc, vsc=vsc)) == due
    assert current_pit_plan_instruction(state, 6) == window(trigger)
    assert current_pit_plan_instruction(state, 7, safety_car=True, vsc=True) is None
    assert current_pit_plan_instruction(state, 4) is None
    assert pit_plan_may_stop(state, 4)


def run_window(monkeypatch, engine, *, trigger="safety_car", control="safety_car",
               entry=4, finite=False):
    simulator = RaceSimulator(np.random.default_rng(19))
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Window", country="Test", total_laps=8,
                  base_lap_time=90, pit_lane_delta=15)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)

    def control_lap(lap, *args, **kwargs):
        simulator.event_manager.current_lap = lap
        simulator.event_manager.safety_car_active = control == "safety_car" and lap == entry - 1
        simulator.event_manager.vsc_active = control == "vsc" and lap == entry - 1
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", control_lap)
    inventory = {"A": [
        {"id": "M", "compound": "medium", "age": 0},
        {"id": "H", "compound": "hard", "age": 0},
    ]} if finite else None
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    return execute([driver], {"A": car}, track, Weather(change_probability=0), ["A"],
                   starting_tires={"A": TireCompound.MEDIUM}, tire_inventory=inventory,
                   pit_plans={"A": [window(trigger)]})[0]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("trigger, control, entry, expected", [
    ("safety_car", "safety_car", 4, 4), ("vsc", "vsc", 4, 4),
    ("neutralized", "safety_car", 3, 3), ("neutralized", "vsc", 5, 5),
    ("safety_car", "vsc", 4, 6), ("vsc", "safety_car", 4, 6),
    ("safety_car", "safety_car", 2, 6), ("safety_car", "safety_car", 7, 6),
    ("neutralized", "green", 4, 6),
])
def test_both_engines_execute_first_observed_opportunity_or_deadline(
    monkeypatch, engine, finite, trigger, control, entry, expected,
):
    result = run_window(monkeypatch, engine, finite=finite, trigger=trigger,
                        control=control, entry=entry)
    assert result.status == DriverStatus.FINISHED
    assert result.pit_laps == [expected]
    assert result.pit_stop_details[0]["lap"] == expected
    assert result.pit_plan_history == [window(trigger) | {
        "status": "executed", "reason": "user_plan", "actual_compound": "hard",
        "actual_set_id": "H" if finite else None, "actual_lap": expected,
    }]


def test_unsafe_early_opportunity_does_not_consume_deadline_or_block_repair():
    simulator = RaceSimulator(np.random.default_rng(1))
    simulator.event_manager.safety_car_active = True
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1)
    initialize_pit_plan_state(state, [window()])
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    wet = Weather(track_wetness=.8, change_probability=0)
    before = deepcopy(state.pit_plan_history)
    # The slick request cannot be executed safely. Existing slicks need repair.
    assert simulator._custom_pit_plan_decision(state, track, wet, 3) is None
    assert state.pit_plan_history == before and state.pit_plan_index == 0
    assert state.pit_plan_override_reason is None
    # A safe current rain fit suppresses elective stops while awaiting the deadline.
    state.current_tire = state.current_tire.model_copy(update={"compound": TireCompound.WET})
    assert simulator._custom_pit_plan_decision(state, track, wet, 4) is False
    assert state.pit_plan_history == before
    assert simulator._custom_pit_plan_decision(state, track, Weather(change_probability=0), 6)


def test_chronological_window_uses_lapped_drivers_own_lap(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(9))
    drivers = [Driver(id=key, name=key, team_id=key) for key in ("FAST", "SLOW")]
    cars = {key: Car(team_id=key, team_name=key) for key in ("FAST", "SLOW")}
    track = Track(id="T", name="Lapped window", country="T", total_laps=8, base_lap_time=90)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *args, **kwargs: 90 if driver.id == "FAST" else 200)
    observed = []
    prepare = simulator._prepare_pit_plan_stop

    def prepare_with_evidence(state, track, weather, lap):
        if state.driver.id == "SLOW" and simulator.event_manager.safety_car_active:
            observed.append((lap, simulator.event_manager.current_lap))
        return prepare(state, track, weather, lap)

    def control(lap, *args, **kwargs):
        simulator.event_manager.current_lap = lap
        simulator.event_manager.safety_car_active = lap == 4
        return []

    monkeypatch.setattr(simulator, "_prepare_pit_plan_stop", prepare_with_evidence)
    monkeypatch.setattr(simulator.event_manager, "process_lap", control)
    results = ChronologicalRace(simulator).run(
        drivers, cars, track, Weather(change_probability=0), ["FAST", "SLOW"],
        starting_tires={"FAST": "medium", "SLOW": "medium"},
        pit_plans={"FAST": [{"lap": 2, "compound": "hard"}], "SLOW": [window()]},
    )
    slow = next(row for row in results if row.driver_id == "SLOW")
    assert (3, 4) in observed
    assert slow.pit_laps == [3]
    assert slow.pit_plan_history[0]["actual_lap"] == 3
    assert slow.laps_completed < track.total_laps


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_free_red_flag_fit_does_not_consume_or_trigger_window(monkeypatch, engine):
    simulator = RaceSimulator(np.random.default_rng(19))
    manager = simulator.event_manager

    def control(lap, *args, **kwargs):
        manager.current_lap = lap
        return [manager.deploy_red_flag(lap, "window test")] if lap == 2 else []

    monkeypatch.setattr(manager, "process_lap", control)
    track = Track(id="T", name="Restart", country="T", total_laps=8, base_lap_time=90)
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result, = execute([Driver(id="A", name="A", team_id="A")],
                      {"A": Car(team_id="A", team_name="A")}, track,
                      Weather(change_probability=0), ["A"], starting_tires={"A": "medium"},
                      pit_plans={"A": [window("neutralized")]})
    assert result.pit_laps == [6]
    assert result.pit_plan_history[0]["status"] == "executed"
    assert result.pit_plan_history[0]["actual_lap"] == 6


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_timed_finish_leaves_unstarted_window_not_reached(monkeypatch, engine):
    from f1sim.simulation import race_timing

    monkeypatch.setattr(race_timing, "RACING_TIME_LIMIT_SECONDS", 250)
    simulator = RaceSimulator(np.random.default_rng(19))
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    track = Track(id="T", name="Timed window", country="T", total_laps=8, base_lap_time=90)
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result, = execute([Driver(id="A", name="A", team_id="A")],
                      {"A": Car(team_id="A", team_name="A")}, track,
                      Weather(change_probability=0), ["A"], starting_tires={"A": "medium"},
                      pit_plans={"A": [window(earliest=5, deadline=7)]})
    assert result.laps_completed < 5 and result.race_time_limited
    assert result.pit_plan_history[0]["status"] == "not_reached"
    assert result.pit_plan_history[0]["actual_lap"] is None
