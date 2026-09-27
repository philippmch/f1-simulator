"""Native timed finishes with free red-flag fits and finite tire stock."""

import numpy as np
import pytest

import f1sim.simulation.race_timing as race_timing
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType
from f1sim.simulation.race import DriverStatus, RaceSimulator
from f1sim.simulation.race_timing import RaceFinishClock, RaceFinishTimeline


def _inventory(case):
    if case == "alternate_dry":
        return ([{"id": "S", "compound": "soft", "age": 0},
                 {"id": "H", "compound": "hard", "age": 0}],
                TireCompound.SOFT, Weather(change_probability=0))
    if case == "same_dry_only":
        return ([{"id": "S", "compound": "soft", "age": 0},
                 {"id": "S2", "compound": "soft", "age": 0}],
                TireCompound.SOFT, Weather(change_probability=0))
    if case == "wet_alternative":
        return ([{"id": "S", "compound": "soft", "age": 0},
                 {"id": "W", "compound": "wet", "age": 0}],
                TireCompound.SOFT,
                Weather(track_wetness=0.4, rain_intensity=0.3, change_probability=0))
    if case == "wet_opening":
        return ([{"id": "I", "compound": "intermediate", "age": 0}],
                TireCompound.INTERMEDIATE,
                Weather(track_wetness=0.4, rain_intensity=0.3, change_probability=0))
    raise ValueError(case)


def _run_inventory_case(monkeypatch, *, engine_name, red_lap, inventory_case, pause=0):
    records, opening, weather = _inventory(inventory_case)
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="timed-inventory", name="Timed inventory", country="Test",
                  total_laps=8, base_lap_time=10, pit_lane_delta=3)
    simulator = RaceSimulator(np.random.default_rng(20260927), red_flag_pause_seconds=pause)
    manager = simulator.event_manager
    manager.set_forced_red_flag(red_lap)
    monkeypatch.setattr(manager, "_deploy_safety_measure", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(manager, "_check_random_incident", lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *args, **kwargs: 10.0)
    monkeypatch.setattr(race_timing, "RACING_TIME_LIMIT_SECONDS", 25.0)

    decisions = []
    original_should_pit = simulator._should_pit

    def observe_pit_decision(state, states, planning_track, lap, *args, **kwargs):
        should_pit = original_should_pit(
            state, states, planning_track, lap, *args, **kwargs,
        )
        decisions.append({
            "lap": lap,
            "should_pit": bool(should_pit),
            "reason": (state.pit_decision_context or {}).get("decision_reason"),
            "proposal": state.inventory_pit_proposal,
            "current_set": state.tire_inventory.current_set_id,
            "tire_laps": state.tire_laps,
            "actually_used": {compound.value
                              for compound in simulator._actually_used_compounds(state)},
        })
        return should_pit

    monkeypatch.setattr(simulator, "_should_pit", observe_pit_decision)

    clock_marks = []
    refits = []
    if engine_name == "standard":
        clock_type = RaceFinishClock
        observe_name = "observe_leader_crossing"

        def observe_crossing(clock, completed_lap, crossing_time, *args, **kwargs):
            final_lap = original_observe(
                clock, completed_lap, crossing_time, *args, **kwargs,
            )
            clock_marks.append({
                "lap": completed_lap,
                "time": crossing_time,
                "final_lap": final_lap,
                "announced": clock.time_limit_announced,
                "limit": clock.time_limit_seconds,
            })
            return final_lap
    elif engine_name == "chronological":
        clock_type = RaceFinishTimeline
        observe_name = "observe_crossing"

        def observe_crossing(timeline, driver_id, completed_lap, crossing_time,
                             *args, **kwargs):
            crossing = original_observe(
                timeline, driver_id, completed_lap, crossing_time, *args, **kwargs,
            )
            clock_marks.append({
                "lap": completed_lap,
                "time": crossing_time,
                "final_lap": timeline.final_lap,
                "announced": timeline.time_limit_announced,
                "limit": timeline.time_limit_seconds,
            })
            return crossing
    else:
        raise ValueError(engine_name)

    original_observe = getattr(clock_type, observe_name)
    monkeypatch.setattr(clock_type, observe_name, observe_crossing)
    original_refit = simulator._refit_inventory_free

    def observe_refit(state, planning_track, refit_weather, current_lap, **kwargs):
        used_before = {compound.value
                       for compound in simulator._actually_used_compounds(state)}
        success = original_refit(
            state, planning_track, refit_weather, current_lap, **kwargs,
        )
        refits.append({
            "lap": current_lap,
            "success": bool(success),
            "used_before": used_before,
            "current_set": state.tire_inventory.current_set_id,
            "fit": dict(state.tire_set_history[-1]) if state.tire_set_history else None,
            "clock": dict(clock_marks[-1]) if clock_marks else None,
        })
        return success

    monkeypatch.setattr(simulator, "_refit_inventory_free", observe_refit)

    engine = None
    if engine_name == "chronological":
        engine = ChronologicalRace(simulator, red_flag_pause_seconds=pause)
        results = engine.run(
            [driver], {"A": car}, track, weather, ["A"],
            starting_tires={"A": opening}, starting_tire_ages={"A": 0},
            tire_inventory={"A": records},
        )
        suspensions = engine.suspensions
    else:
        results = simulator.simulate_race(
            [driver], {"A": car}, track, weather, ["A"],
            starting_tires={"A": opening}, starting_tire_ages={"A": 0},
            tire_inventory={"A": records},
        )
        suspensions = simulator.suspensions

    return {
        "result": results[0],
        "refits": refits,
        "decisions": decisions,
        "clock_marks": clock_marks,
        "red_flags": [event for event in manager.events
                      if event.event_type == EventType.RED_FLAG],
        "suspensions": suspensions,
        "initial_inventory": records,
    }


def _assert_inventory_conservation(run):
    result = run["result"]
    ages = {record["id"]: record["age"] for record in run["initial_inventory"]}
    for stint in result.tire_set_history:
        assert stint["set_id"] in ages
        assert stint["age_at_fit"] == ages[stint["set_id"]]
        assert stint["age_at_end"] == stint["age_at_fit"] + stint["laps_used"]
        ages[stint["set_id"]] = stint["age_at_end"]
    assert {item["id"]: item["age"] for item in result.tire_inventory} == ages
    assert sum(item["current"] for item in result.tire_inventory) == 1
    assert sum(stint["laps_used"] for stint in result.tire_set_history) == result.laps_completed
    paid_stints = sum(stint["kind"] == "paid" for stint in result.tire_set_history)
    assert result.pit_stops == paid_stints == len(result.pit_laps)


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
@pytest.mark.parametrize(("red_lap", "pause", "laps_on_hard", "announced", "limit"), [
    (1, 0, 3, False, 25.0),
    (3, 20, 1, True, 45.0),
])
def test_free_distinct_dry_fit_is_used_without_a_paid_correction(
    monkeypatch, engine_name, red_lap, pause, laps_on_hard, announced, limit,
):
    run = _run_inventory_case(
        monkeypatch, engine_name=engine_name, red_lap=red_lap,
        inventory_case="alternate_dry", pause=pause,
    )
    result = run["result"]
    assert len(run["refits"]) == 1
    (refit,) = run["refits"]
    assert refit["success"] and refit["current_set"] == "H"
    assert refit["used_before"] == {"soft"}  # The free fit has not run yet.
    assert refit["fit"]["kind"] == "red_flag" and refit["fit"]["laps_used"] == 0
    assert refit["clock"]["lap"] == red_lap
    assert refit["clock"]["announced"] is announced
    assert refit["clock"]["final_lap"] == (8 if red_lap == 1 else 4)
    assert refit["clock"]["limit"] == 25.0
    restart = next(decision for decision in run["decisions"]
                   if decision["lap"] == red_lap + 1)
    assert restart["current_set"] == "H" and restart["tire_laps"] == 0
    assert restart["actually_used"] == {"soft"}
    assert not restart["should_pit"]
    assert result.status == DriverStatus.FINISHED
    assert result.laps_completed == 4 and result.race_time_limited
    assert result.pit_stops == 0 and result.pit_laps == []
    assert [(stint["set_id"], stint["laps_used"]) for stint in result.tire_set_history] == [
        ("S", red_lap), ("H", laps_on_hard),
    ]
    assert result.tire_set_history[-1]["laps_used"] > 0
    assert result.race_suspension_seconds == pause
    assert run["clock_marks"][-1]["time"] == pytest.approx(40.0 + pause)
    if pause:
        assert run["suspensions"][0][1] - run["suspensions"][0][0] == pause
        assert run["clock_marks"][-1]["final_lap"] == 4
        assert run["clock_marks"][-1]["limit"] == limit
        assert run["clock_marks"][-1]["announced"]
    assert not any(decision["should_pit"] for decision in run["decisions"])
    assert len(run["red_flags"]) == 1 and run["red_flags"][0].lap == red_lap
    _assert_inventory_conservation(run)


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
def test_duplicate_dry_free_fit_does_not_meet_rule_or_reach_paid_service(
    monkeypatch, engine_name,
):
    run = _run_inventory_case(
        monkeypatch, engine_name=engine_name, red_lap=1,
        inventory_case="same_dry_only",
    )
    result = run["result"]
    (refit,) = run["refits"]
    assert refit["success"] and refit["current_set"] == "S2"
    assert refit["used_before"] == {"soft"}
    assert refit["fit"]["kind"] == "red_flag" and refit["fit"]["laps_used"] == 0
    assert next(row for row in run["decisions"] if row["lap"] == 2)["actually_used"] == {
        "soft",
    }
    mandatory = next(row for row in run["decisions"] if row["lap"] == 4)
    assert mandatory["should_pit"] and mandatory["reason"] == "compound_requirement"
    assert tuple(mandatory["proposal"]) == (4, None)
    assert result.status == DriverStatus.DNF and result.laps_completed == 3
    assert result.dnf_reason == "No suitable replacement tyre set available"
    assert result.pit_stops == 0 and result.pit_laps == [] and result.pit_stop_details == []
    assert [(stint["set_id"], stint["laps_used"]) for stint in result.tire_set_history] == [
        ("S", 1), ("S2", 2),
    ]
    _assert_inventory_conservation(run)


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
def test_safe_wet_free_fit_is_used_without_redundant_stop(monkeypatch, engine_name):
    run = _run_inventory_case(
        monkeypatch, engine_name=engine_name, red_lap=1,
        inventory_case="wet_alternative",
    )
    result = run["result"]
    (refit,) = run["refits"]
    assert refit["success"] and refit["current_set"] == "W"
    assert refit["used_before"] == {"soft"}
    assert result.status == DriverStatus.FINISHED and result.race_time_limited
    assert result.laps_completed == 4 and result.pit_stops == 0
    assert [(stint["set_id"], stint["laps_used"]) for stint in result.tire_set_history] == [
        ("S", 1), ("W", 3),
    ]
    assert not any(decision["should_pit"] for decision in run["decisions"])
    _assert_inventory_conservation(run)


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
def test_red_flag_at_timed_finish_records_event_without_refit_or_suspension(
    monkeypatch, engine_name,
):
    run = _run_inventory_case(
        monkeypatch, engine_name=engine_name, red_lap=4,
        inventory_case="wet_opening", pause=20,
    )
    result = run["result"]
    assert [(event.lap, event.event_type) for event in run["red_flags"]] == [
        (4, EventType.RED_FLAG),
    ]
    assert run["clock_marks"][-1]["lap"] == 4
    assert run["clock_marks"][-1]["announced"]
    assert run["refits"] == [] and run["suspensions"] == []
    assert result.status == DriverStatus.FINISHED and result.race_time_limited
    assert result.laps_completed == 4 and result.race_suspension_seconds == 0
    assert result.pit_stops == 0
    _assert_inventory_conservation(run)
