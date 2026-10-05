"""SC/VSC scenarios resolve observed announcements without revealing future control."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.control_schedule import (
    CONTROL_SCHEDULE_POLICY,
    parse_control_schedule_spec,
    validate_control_schedule,
    validate_control_schedule_snapshot,
)
from f1sim.simulation.events import EventManager, EventType
from f1sim.simulation.race import RaceSimulator


def entry(lap=2, control="safety_car", duration=3):
    return {"lap": lap, "control": control, "duration_laps": duration}


def track(laps=10):
    return Track(id="T", name="Control scenario", country="Test", total_laps=laps,
                 base_lap_time=90, safety_car_probability=1.)


def test_parser_normalization_and_explicit_empty_schedule():
    source = [entry(), entry(6, "vsc", 2)]
    assert parse_control_schedule_spec("2:sc:3,6:vsc:2") == source
    assert parse_control_schedule_spec("none") == []
    assert validate_control_schedule(None) is None
    assert validate_control_schedule([]) == []
    result = validate_control_schedule(source)
    result[0]["lap"] = 1
    assert source[0]["lap"] == 2


@pytest.mark.parametrize("change", [
    {"lap": True}, {"lap": 2.0}, {"lap": "2"}, {"lap": 0}, {"lap": 1001},
    {"duration_laps": False}, {"duration_laps": 3.0}, {"duration_laps": "3"},
    {"duration_laps": 0}, {"duration_laps": 7}, {"control": "sc"},
    {"control": "red_flag"}, {"control": []}, {"extra": 1},
])
def test_invalid_schedule_entries_are_rejected(change):
    with pytest.raises(ValueError):
        validate_control_schedule([entry() | change])


@pytest.mark.parametrize("source", ["", "2:sc:3,", "2.0:sc:3", "２:sc:3", "2:SC:3", "none,2:sc:3"])
def test_invalid_shorthand_is_rejected(source):
    with pytest.raises(ValueError):
        parse_control_schedule_spec(source)


def test_overlap_order_size_and_original_distance_are_strict():
    for value in ([entry(), entry(5)], [entry(6), entry()], [entry()] * 21,
                  {"lap": 2}, [entry() | {"duration_laps": 2, "other": None}]):
        with pytest.raises(ValueError):
            validate_control_schedule(value)
    with pytest.raises(ValueError, match="total_laps"):
        validate_control_schedule([entry(11)], total_laps=10)


@pytest.mark.parametrize("control", ["safety_car", "vsc"])
def test_observed_deployment_countdown_and_history_reset(monkeypatch, control):
    source = [entry(control=control)]
    manager = EventManager(np.random.default_rng(37), control_schedule=source)
    monkeypatch.setattr(manager, "_check_red_flag_conditions", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_deploy_safety_measure",
                        lambda *a, **kw: pytest.fail("random SC/VSC"))
    observed = []
    for lap in range(1, 7):
        events = manager.process_lap(lap, [], {}, track(), Weather(change_probability=0))
        observed.append((manager.safety_car_active, manager.vsc_active))
        assert len(events) == int(lap == 2)
        if events:
            assert events[0].duration_laps == 3
    active = (control == "safety_car", control == "vsc")
    assert observed == [(False, False), active, active, active, (False, False), (False, False)]
    history = manager.get_control_schedule_history()
    assert history == [source[0] | {"status": "applied", "reason": "scheduled_announcement"}]
    history[0]["lap"] = 99
    assert manager.control_schedule[0]["lap"] == 2
    assert source == [entry(control=control)]
    manager.reset()
    assert manager.get_control_schedule_history()[0]["status"] == "not_reached"
    assert manager.control_schedule_history[0]["status"] is None
    assert manager.safety_car_deployments == manager.vsc_deployments == 0


def test_explicit_empty_schedule_keeps_background_red_flag_check(monkeypatch):
    manager = EventManager(control_schedule=[])
    calls = []

    def red(lap, incidents, *args, **kwargs):
        calls.append((lap, incidents))
        return manager.deploy_red_flag(lap, "preserved red flag")

    monkeypatch.setattr(manager, "_check_red_flag_conditions", red)
    events = manager.process_lap(1, [], {}, track(), Weather(), incidents_this_lap=3)
    assert calls == [(1, 3)] and events[0].event_type == EventType.RED_FLAG
    assert manager.red_flag_active and manager.get_control_schedule_history() == []


@pytest.mark.parametrize("priority", ["already_suspended", "forced_red", "new_red", "no_survivors"])
def test_announcement_suppressed_by_higher_priority_conditions(monkeypatch, priority):
    manager = EventManager(control_schedule=[entry(lap=1)])
    drivers = []
    if priority == "already_suspended":
        manager.deploy_red_flag(0, "suspension")
    elif priority == "forced_red":
        manager.set_forced_red_flag(1)
    elif priority == "new_red":
        monkeypatch.setattr(manager, "_check_red_flag_conditions",
                            lambda lap, *a, **kw: manager.deploy_red_flag(lap, "priority"))
    else:
        drivers = [Driver(id="A", name="A", team_id="A", dnf=True)]
    manager.process_lap(1, drivers, {}, track(), Weather())
    history = manager.get_control_schedule_history()[0]
    assert history["status"] == "suppressed"
    assert history["reason"] == ("no_survivors" if priority == "no_survivors" else "red_flag")
    assert not manager.safety_car_active and manager.safety_car_deployments == 0


def test_forced_safety_car_configuration_cannot_conflict_with_schedule():
    manager = EventManager(control_schedule=[])
    with pytest.raises(ValueError, match="combined"):
        manager.set_forced_safety_car(2)
    automatic = EventManager()
    automatic.set_forced_safety_car(2)
    with pytest.raises(ValueError, match="combined"):
        automatic.set_control_schedule([])


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["safety_car", "vsc"])
def test_both_engines_take_observed_window_opportunity(monkeypatch, engine, control):
    simulator = RaceSimulator(np.random.default_rng(91), control_schedule=[entry(control=control)])
    monkeypatch.setattr(simulator.event_manager, "_check_red_flag_conditions",
                        lambda *a, **kw: None)
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    results = execute([Driver(id="A", name="A", team_id="A")],
                      {"A": Car(team_id="A", team_name="A", reliability=1.,
                                engine_reliability=1., gearbox_reliability=1.,
                                brakes_reliability=1., electrical_reliability=1.,
                                cooling_reliability=1.)},
                      track(), Weather(change_probability=0), ["A"], starting_tires={"A": "medium"},
                      pit_plans={"A": [{"lap": 6, "earliest_lap": 3,
                                       "trigger": control, "compound": "hard"}]})
    assert results[0].pit_laps == [3]
    assert results[0].pit_stop_details[0]["control"] == control
    assert results[0].pit_plan_history[0]["actual_lap"] == 3
    assert simulator.event_manager.get_control_schedule_history()[0]["status"] == "applied"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_schedule_range_preflight_does_not_reset_existing_state(engine):
    simulator = RaceSimulator(control_schedule=[entry(11)])
    simulator.event_manager.safety_car_deployments = 7
    before = deepcopy(simulator.event_manager.control_schedule_history)
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    with pytest.raises(ValueError, match="total_laps"):
        execute([], {}, track(), Weather(), [])
    assert simulator.event_manager.safety_car_deployments == 7
    assert simulator.event_manager.control_schedule_history == before


@pytest.mark.parametrize("schedule", [[], [entry()]])
def test_snapshot_policy_requires_new_schema_and_explicit_source(schedule):
    valid = {"schema_version": 11, "control_schedule": schedule,
             "control_schedule_policy": CONTROL_SCHEDULE_POLICY}
    validate_control_schedule_snapshot(valid, schedule)
    for change in ({"schema_version": 10}, {"control_schedule_policy": "unknown"},
                   {"control_schedule_policy": None}):
        with pytest.raises(ValueError):
            validate_control_schedule_snapshot(valid | change, schedule)
    with pytest.raises(ValueError):
        validate_control_schedule_snapshot({"schema_version": 11}, None)
    validate_control_schedule_snapshot({"schema_version": 10}, None)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_red_flag_priority_does_not_destroy_later_requests(engine):
    schedule = [entry(2, duration=2), entry(6, "vsc", 2)]
    simulator = RaceSimulator(np.random.default_rng(91), control_schedule=schedule)
    simulator.event_manager.set_forced_red_flag(2)
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result = execute([Driver(id="A", name="A", team_id="A")],
                     {"A": Car(team_id="A", team_name="A", reliability=1)},
                     track(8), Weather(change_probability=0), ["A"],
                     starting_tires={"A": "medium"}, pit_plans={"A": []})
    history = simulator.event_manager.get_control_schedule_history()
    assert history[0] == schedule[0] | {"status": "suppressed", "reason": "red_flag"}
    assert history[1] == schedule[1] | {"status": "applied", "reason": "scheduled_announcement"}
    assert simulator.event_manager.safety_car_deployments == 1  # Procedural resumption.
    resumption, = [event for event in simulator.event_manager.events
                   if event.event_type == EventType.SAFETY_CAR]
    assert resumption.lap == 2 and resumption.duration_laps == 1
    assert "resumption" in resumption.description
    assert simulator.event_manager.vsc_deployments == 1
    assert result[0].race_suspension_seconds > 0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_shortened_finish_keeps_original_distance_and_unreached_request(engine):
    circuit = track(20).model_copy(update={"base_lap_time": 1000.})
    schedule = [entry(2, duration=2), entry(18, "vsc", 2)]
    simulator = RaceSimulator(np.random.default_rng(91), control_schedule=schedule)
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result = execute([Driver(id="A", name="A", team_id="A")],
                     {"A": Car(team_id="A", team_name="A", reliability=1)},
                     circuit, Weather(change_probability=0), ["A"],
                     starting_tires={"A": "medium"}, pit_plans={"A": []})
    assert circuit.total_laps == 20 and result[0].race_time_limited
    assert result[0].laps_completed < 18
    history = simulator.event_manager.get_control_schedule_history()
    assert history[0]["status"] == "applied"
    assert history[1] == schedule[1] | {
        "status": "not_reached", "reason": "race_ended_before_request",
    }
