"""Focused execution coverage for explicit paid pit instructions."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def _run(
    engine, *, plan=None, inventory=None, laps=6, monkeypatch=None,
    automatic_stop=False, satisfy_rule=False,
):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(42))
    if monkeypatch is not None:
        monkeypatch.setattr(
            simulator, "_should_pit", lambda *args, **kwargs: automatic_stop,
        )
        if satisfy_rule:
            monkeypatch.setattr(simulator, "_stay_satisfies_tire_rule", lambda state: True)
        monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    kwargs = {
        "starting_tires": {"A": TireCompound.MEDIUM},
        "pit_plans": plan,
    }
    if inventory is not None:
        kwargs["tire_inventory"] = {"A": inventory}
    if engine == "standard":
        return simulator.simulate_race(
            [driver], {"A": car}, track, Weather(change_probability=0), ["A"], **kwargs,
        )[0]
    return ChronologicalRace(simulator).run(
        [driver], {"A": car}, track, Weather(change_probability=0), ["A"], **kwargs,
    )[0]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_due_instruction_commits_on_driver_own_lap_and_bypasses_elective_veto(
    engine, monkeypatch,
):
    result = _run(
        engine,
        plan={"A": [{"lap": 2, "compound": "hard"}]},
        monkeypatch=monkeypatch,
    )
    assert result.pit_laps == [2]
    assert result.pit_stop_details[0]["lap"] == 2
    assert result.pit_stop_details[0]["decision_reason"] == "user_plan"
    assert result.pit_stop_details[0]["to_compound"] == "hard"
    assert result.pit_plan_history == [{
        "lap": 2,
        "compound": "hard",
        "status": "executed",
        "reason": "user_plan",
        "actual_compound": "hard",
        "actual_set_id": None,
    }]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_explicit_empty_plan_suppresses_elective_automatic_stops(engine, monkeypatch):
    result = _run(
        engine,
        plan={"A": []},
        monkeypatch=monkeypatch,
        automatic_stop=True,
        satisfy_rule=True,
    )
    assert result.pit_laps == []
    assert result.pit_plan_history == []


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_explicit_plan_is_not_limited_to_automatic_three_stop_budget(engine, monkeypatch):
    result = _run(
        engine,
        laps=8,
        plan={"A": [
            {"lap": 2, "compound": "hard"},
            {"lap": 3, "compound": "soft"},
            {"lap": 4, "compound": "medium"},
            {"lap": 5, "compound": "hard"},
        ]},
        monkeypatch=monkeypatch,
    )
    assert result.pit_laps == [2, 3, 4, 5]
    assert result.pit_stops == 4
    assert [record["status"] for record in result.pit_plan_history] == [
        "executed", "executed", "executed", "executed",
    ]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_finite_request_uses_least_worn_set_then_input_order(engine, monkeypatch):
    result = _run(
        engine,
        plan={"A": [{"lap": 2, "compound": "hard"}]},
        inventory=[
            {"id": "M", "compound": "medium", "age": 0},
            {"id": "H-old", "compound": "hard", "age": 5},
            {"id": "H-young", "compound": "hard", "age": 2},
            {"id": "H-young-input-later", "compound": "hard", "age": 2},
        ],
        monkeypatch=monkeypatch,
    )
    assert result.pit_stop_details[0]["to_set_id"] == "H-young"
    assert result.pit_plan_history[0]["actual_set_id"] == "H-young"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_critical_requested_compound_is_skipped_without_forcing_a_stop(engine, monkeypatch):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=4,
                  base_lap_time=90, pit_lane_delta=1)
    weather = Weather(track_wetness=0.5, change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(42))
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    kwargs = {"starting_tires": {"A": TireCompound.MEDIUM},
              "pit_plans": {"A": [{"lap": 2, "compound": "soft"}]}}
    # Slicks are critical at this surface; the current medium is also critical,
    # so the test uses an intermediate opening to isolate the requested skip.
    kwargs["starting_tires"] = {"A": TireCompound.INTERMEDIATE}
    if engine == "standard":
        result = simulator.simulate_race(
            [driver], {"A": car}, track, weather, ["A"], **kwargs,
        )[0]
    else:
        result = ChronologicalRace(simulator).run(
            [driver], {"A": car}, track, weather, ["A"], **kwargs,
        )[0]
    assert result.pit_plan_history[0]["status"] == "skipped"
    assert result.pit_plan_history[0]["reason"] == "critical_requested_compound"
    assert result.pit_laps == []


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_automatic_result_keeps_nullable_plan_metadata(engine, monkeypatch):
    result = _run(engine, plan=None, monkeypatch=monkeypatch)
    assert result.pit_plan_history is None


def test_forced_repair_can_execute_the_due_requested_replacement(monkeypatch):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=5,
                  base_lap_time=90, pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(42))
    puncture = RaceEvent(EventType.PUNCTURE, 1, ["A"], forces_pit_stop=True)
    monkeypatch.setattr(
        simulator.event_manager,
        "process_lap",
        lambda **kwargs: [puncture] if kwargs["lap"] == 1 else [],
    )
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    result = simulator.simulate_race(
        [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
        starting_tires={"A": "medium"},
        tire_inventory={"A": [
            {"id": "M", "compound": "medium"},
            {"id": "H", "compound": "hard"},
        ]},
        pit_plans={"A": [{"lap": 2, "compound": "hard"}]},
    )[0]
    assert result.pit_laps == [2]
    assert result.pit_stop_details[0]["decision_reason"] == "forced_repair"
    assert result.pit_plan_history[0]["status"] == "executed"
    assert result.pit_plan_history[0]["reason"] == "forced_repair"


def test_unrelated_forced_repair_does_not_consume_a_future_instruction(monkeypatch):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=6,
                  base_lap_time=90, pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(42))
    puncture = RaceEvent(EventType.PUNCTURE, 1, ["A"], forces_pit_stop=True)
    monkeypatch.setattr(
        simulator.event_manager,
        "process_lap",
        lambda **kwargs: [puncture] if kwargs["lap"] == 1 else [],
    )
    result = simulator.simulate_race(
        [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
        starting_tires={"A": "medium"},
        tire_inventory={"A": [
            {"id": "M", "compound": "medium"},
            {"id": "H", "compound": "hard"},
            {"id": "S", "compound": "soft"},
        ]},
        pit_plans={"A": [{"lap": 4, "compound": "hard"}]},
    )[0]
    assert result.pit_laps == [2, 4]
    assert [stop["decision_reason"] for stop in result.pit_stop_details] == [
        "forced_repair", "user_plan",
    ]
    assert result.pit_plan_history[0]["status"] == "executed"
    assert result.pit_plan_history[0]["reason"] == "user_plan"


def test_forced_repair_overrides_a_runtime_unavailable_request(monkeypatch):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=5,
                  base_lap_time=90, pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(42))
    puncture = RaceEvent(EventType.PUNCTURE, 1, ["A"], forces_pit_stop=True)
    monkeypatch.setattr(
        simulator.event_manager,
        "process_lap",
        lambda **kwargs: [puncture] if kwargs["lap"] == 1 else [],
    )
    result = simulator.simulate_race(
        [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
        starting_tires={"A": "medium"},
        tire_inventory={"A": [
            {"id": "M", "compound": "medium"},
            {"id": "H", "compound": "hard"},
        ]},
        pit_plans={"A": [{"lap": 2, "compound": "medium"}]},
    )[0]
    assert result.pit_laps == [2]
    assert result.pit_stop_details[0]["to_compound"] == "hard"
    assert result.pit_plan_history[0]["status"] == "overridden"
    assert result.pit_plan_history[0]["reason"] == "forced_repair"


def test_chronological_lapped_finish_marks_unstarted_instruction_not_reached(monkeypatch):
    drivers = [
        Driver(id="FAST", name="FAST", team_id="F"),
        Driver(id="SLOW", name="SLOW", team_id="S"),
    ]
    cars = {
        "F": Car(team_id="F", team_name="F"),
        "S": Car(team_id="S", team_name="S"),
    }
    track = Track(id="T", name="Test", country="Test", total_laps=10,
                  base_lap_time=90, pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(9))
    pace = {"FAST": 90.0, "SLOW": 110.0}
    monkeypatch.setattr(
        simulator.lap_simulator,
        "calculate_lap_time",
        lambda driver, *args, **kwargs: pace[driver.id],
    )
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **kw: [])
    results = ChronologicalRace(simulator).run(
        drivers,
        cars,
        track,
        Weather(change_probability=0),
        ["FAST", "SLOW"],
        starting_tires={"FAST": "medium", "SLOW": "medium"},
        pit_plans={"SLOW": [{"lap": 10, "compound": "hard"}]},
    )
    slow = next(result for result in results if result.driver_id == "SLOW")
    assert slow.laps_completed < 10
    assert slow.pit_laps == []
    assert slow.pit_plan_history == [{
        "lap": 10,
        "compound": "hard",
        "status": "not_reached",
        "reason": "race_finished",
        "actual_compound": None,
        "actual_set_id": None,
    }]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_final_lap_request_cannot_replace_unrun_red_flag_set_with_same_slick(
    engine, monkeypatch,
):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=4,
                  base_lap_time=90, pit_lane_delta=1)
    simulator = RaceSimulator(np.random.default_rng(19))
    red_flag = RaceEvent(EventType.RED_FLAG, 3, [])

    def events(*args, **kwargs):
        lap = kwargs.get("lap", args[0] if args else None)
        return [red_flag] if lap == 3 else []

    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        simulator, "_choose_red_flag_tire", lambda *args, **kwargs: TireCompound.MEDIUM,
    )
    execute = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator, red_flag_pause_seconds=0).run)
    result = execute(
        [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
        starting_tires={"A": "soft"},
        pit_plans={"A": [{"lap": 4, "compound": "soft"}]},
    )[0]

    assert result.pit_laps == []
    assert result.strategy == ["soft", "medium"]
    assert result.pit_plan_history == [{
        "lap": 4,
        "compound": "soft",
        "status": "overridden",
        "reason": "compound_requirement",
        "actual_compound": None,
        "actual_set_id": None,
    }]
