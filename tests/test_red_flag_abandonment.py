"""Abandonment classifies an earlier completed race, without later fit/stop credit."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.control_schedule import parse_control_schedule_spec, validate_control_schedule
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.race_points import race_scoring_context


def run_case(monkeypatch, engine, *, after=4, total_laps=10, schedule=None, inventory=None,
             pit_plans=None, failure=None, pace=None, starting=None, weather=None):
    schedule = (schedule if schedule is not None else
                [{"lap": after, "control": "red_flag", "action": "abandon"}])
    simulator = RaceSimulator(np.random.default_rng(91), control_schedule=schedule,
                              red_flag_pause_seconds=100)
    control = simulator.event_manager
    for name in ("_check_random_incident", "_check_red_flag_conditions"):
        monkeypatch.setattr(control, name, lambda *a, **kw: None)

    def mechanical(driver, car, track, lap, weather):
        if failure == (driver.id, lap):
            driver.dnf = True
            driver.dnf_reason = "mechanical"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(control, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *a, **kw: 0)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (False, False))
    samples, evolves, fits = [], [], []

    def physics(driver, car, track, tire, tire_laps=None, lap_number=None, weather=None, **kwargs):
        samples.append((driver.id, lap_number))
        return (pace or {"A": 90., "B": 110.})[driver.id]

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng:
                        evolves.append(True) or self.model_copy(deep=True))
    fit = simulator._fit_tire

    def record_fit(state, compound):
        fits.append((state.driver.id, state.laps_completed, compound.value))
        return fit(state, compound)

    monkeypatch.setattr(simulator, "_fit_tire", record_fit)
    runners = {"standard": simulator.simulate_race,
               "chronological": ChronologicalRace(simulator, red_flag_pause_seconds=100).run}
    drivers = [Driver(id=key, name=key, team_id="T", consistency=1.) for key in ("A", "B")]
    results = runners[engine](
        drivers, {"T": Car(team_id="T", team_name="T", reliability=1.)},
        Track(id="T", name="T", country="Synthetic", total_laps=total_laps,
              base_lap_time=90., safety_car_probability=0.),
        weather or Weather(change_probability=0.),
        ["A", "B"], starting_tires=starting or {key: "medium" for key in ("A", "B")},
        tire_inventory=inventory, pit_plans=pit_plans,
    )
    return results, simulator, samples, evolves, fits


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("after", [4, 6])
def test_countback_uses_previous_lap_and_each_followers_own_crossing(monkeypatch, engine, after):
    results, simulator, samples, evolves, fits = run_case(monkeypatch, engine, after=after)
    countback = after - 1
    assert [(r.driver_id, r.laps_completed, r.total_time, r.status.value) for r in results] == [
        ("A", countback, countback * 90. + 30., "finished"),
        ("B", countback, countback * 110. + 30., "finished"),
    ]
    evidence = results[0].race_abandonment
    assert evidence.signal_leader_lap == after + 1
    assert evidence.countback_lap == countback
    assert evidence.countback_time == countback * 90.
    assert evidence.signal_time == after * 90.
    assert all(r.race_abandonment == evidence for r in results)
    assert len(evolves) == after - 1 and fits == []
    assert max(lap for _, lap in samples) <= after
    assert simulator.event_manager.safety_car_deployments == 0
    assert not simulator.event_manager.red_flag_active
    assert not any(r.race_time_limited for r in results)
    assert race_scoring_context(results)["winner_laps"] == countback


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_countback_drops_later_paid_stop_inventory_use_and_instruction(monkeypatch, engine):
    inventory = {key: [{"id": key + "M", "compound": "medium", "age": 0},
                       {"id": key + "H", "compound": "hard", "age": 0}]
                 for key in ("A", "B")}
    plans = {"A": [{"lap": 3, "compound": "hard"}],
             "B": [{"lap": 4, "compound": "hard"}]}
    results, _, _, _, _ = run_case(monkeypatch, engine, inventory=inventory, pit_plans=plans)
    a, b = results
    assert a.pit_laps == [3] and a.pit_stops == 1
    assert b.pit_laps == [] and b.pit_stops == 0 and b.pit_stop_details == []
    assert b.strategy == ["medium"]
    assert b.pit_plan_history[0]["status"] == "not_reached"
    assert b.pit_plan_history[0]["reason"] == "race_abandoned"
    assert a.pit_plan_history[0]["status"] == "executed"
    for result in results:
        assert sum(fit["laps_used"] for fit in result.tire_set_history) == result.laps_completed
    assert next(item for item in b.tire_inventory if item["id"] == "BH")["age"] == 0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("after,points,winner_laps", [(1, 0, None), (2, 0, 1), (3, 6, 2)])
def test_early_abandonment_has_no_fictional_distance_or_green_credit(
    monkeypatch, engine, after, points, winner_laps,
):
    results, _, _, _, _ = run_case(monkeypatch, engine, after=after)
    assert results[0].points_awarded == points
    assert results[0].laps_completed == after - 1
    context = race_scoring_context(results)
    assert context is not None and context["winner_laps"] == winner_laps
    if after == 1:
        assert all(r.status.value == "no_result" and not r.classified for r in results)
        assert all(r.total_time == r.fastest_lap == 0 for r in results)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_green_pair_completed_after_countback_cannot_qualify_points(monkeypatch, engine):
    schedule = [{"lap": 1, "control": "safety_car", "duration_laps": 1},
                {"lap": 4, "control": "red_flag", "action": "abandon"}]
    results, _, _, _, _ = run_case(monkeypatch, engine, schedule=schedule)
    assert results[0].laps_completed == 3
    assert not results[0].race_points_context.has_two_green_laps
    assert all(r.points_awarded == 0 for r in results)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("failed_lap,expected_status,expected_laps", [(2, "dnf", 1),
                                                                     (4, "finished", 3)])
def test_retirement_status_is_taken_from_the_countback_finish(
    monkeypatch, engine, failed_lap, expected_status, expected_laps,
):
    results, simulator, _, _, _ = run_case(monkeypatch, engine, failure=("B", failed_lap))
    b = next(r for r in results if r.driver_id == "B")
    assert b.status.value == expected_status and b.laps_completed == expected_laps
    assert (b.dnf_reason is None) == (expected_status == "finished")
    assert any(e.event_type == EventType.MECHANICAL_FAILURE for e in simulator.event_manager.events)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_a_later_request_remains_unreached_and_never_restarts_the_field(monkeypatch, engine):
    schedule = [{"lap": 4, "control": "red_flag", "action": "abandon"},
                {"lap": 7, "control": "safety_car", "duration_laps": 1}]
    _, simulator, _, _, fits = run_case(monkeypatch, engine, schedule=schedule)
    history = simulator.event_manager.get_control_schedule_history()
    assert [r["status"] for r in history] == ["applied", "not_reached"]
    assert simulator.event_manager.safety_car_deployments == 0 and fits == []


def test_cli_red_flag_requests_are_explicit_about_resumption():
    expected = [{"lap": 4, "control": "red_flag", "action": "resume"},
                {"lap": 7, "control": "red_flag", "action": "abandon"}]
    assert parse_control_schedule_spec("4:red:resume,7:red:abandon") == expected
    assert validate_control_schedule(expected) == expected


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_scheduled_resumption_retains_the_native_one_lap_sc_procedure(monkeypatch, engine):
    schedule = [{"lap": 2, "control": "red_flag", "action": "resume"}]
    results, simulator, _, _, _ = run_case(monkeypatch, engine, schedule=schedule)
    assert simulator.event_manager.safety_car_deployments == 1
    assert results[0].laps_completed == 10
    assert all(r.race_abandonment is None for r in results)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_abandonment_after_a_resumption_retains_only_the_historical_wait(monkeypatch, engine):
    schedule = [{"lap": 2, "control": "red_flag", "action": "resume"},
                {"lap": 5, "control": "red_flag", "action": "abandon"}]
    results, simulator, _, _, _ = run_case(monkeypatch, engine, schedule=schedule)
    assert results[0].laps_completed == results[0].race_abandonment.countback_lap == 4
    assert results[0].race_abandonment.countback_time > 4 * 90.
    assert results[0].race_suspension_seconds > 100.
    assert simulator.event_manager.safety_car_deployments == 1
    assert [row["status"] for row in simulator.event_manager.get_control_schedule_history()] == [
        "applied", "applied",
    ]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_finish_signal_precedes_an_after_crossing_abandonment_request(monkeypatch, engine):
    results, simulator, _, _, _ = run_case(monkeypatch, engine, after=10)
    assert results[0].laps_completed == 10
    assert all(r.race_abandonment is None for r in results)
    assert simulator.event_manager.get_control_schedule_history()[0]["reason"] == "race_finished"
    assert simulator.event_manager.red_flag_deployments == 0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_a_timed_chequered_flag_precedes_the_requested_abandonment(monkeypatch, engine):
    results, simulator, _, _, _ = run_case(
        monkeypatch, engine, after=4, pace={"A": 2500., "B": 2600.},
    )
    assert results[0].laps_completed == 4 and results[0].race_time_limited
    assert all(row.race_abandonment is None for row in results)
    assert simulator.event_manager.get_control_schedule_history()[0]["reason"] == "race_finished"
    assert simulator.event_manager.red_flag_deployments == 0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_suppressed_terminal_red_keeps_the_existing_sc_and_followers_finish(monkeypatch, engine):
    schedule = [{"lap": 1, "control": "safety_car", "duration_laps": 6}]
    baseline, original, _, _, _ = run_case(monkeypatch, engine, total_laps=6, schedule=schedule)
    results, simulator, _, _, _ = run_case(
        monkeypatch, engine, total_laps=6,
        schedule=schedule + [{"lap": 6, "control": "red_flag", "action": "abandon"}],
    )
    assert results == baseline
    assert simulator.event_manager.safety_car_active == original.event_manager.safety_car_active
    assert simulator.event_manager.safety_car_laps_remaining == (
        original.event_manager.safety_car_laps_remaining
    )
    assert simulator.event_manager.get_control_schedule_history()[1]["reason"] == "race_finished"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_tyre_requirement_uses_actual_running_after_countback_and_can_change_the_winner(
    monkeypatch, engine,
):
    results, _, _, _, _ = run_case(
        monkeypatch, engine, pace={"A": 90., "B": 95.},
        pit_plans={"B": [{"lap": 4, "compound": "hard"}]},
    )
    b, a = results
    assert (b.driver_id, b.total_time, b.points_awarded) == ("B", 285., 13)
    assert (a.driver_id, a.total_time, a.points_awarded) == ("A", 300., 10)
    assert b.pit_stops == 0 and b.strategy == ["medium"]
    assert b.abandonment_tire_rule.used_compounds == ("hard", "medium")
    assert b.abandonment_tire_rule.penalty_seconds == 0
    assert a.abandonment_tire_rule.used_compounds == ("medium",)
    assert a.abandonment_tire_rule.penalty_seconds == 30
    assert b.gap_to_leader == 0 and a.gap_to_leader == 15.


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_a_fitted_second_compound_never_run_before_retirement_cannot_waive_the_penalty(
    monkeypatch, engine,
):
    results, _, _, _, _ = run_case(
        monkeypatch, engine, failure=("A", 4),
        pit_plans={"A": [{"lap": 4, "compound": "hard"}]},
    )
    a = next(row for row in results if row.driver_id == "A")
    assert a.status.value == "finished" and a.laps_completed == 3
    assert a.abandonment_tire_rule.used_compounds == ("medium",)
    assert a.abandonment_tire_rule.penalty_seconds == 30


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_running_intermediate_tyres_waives_the_dry_compound_penalty(monkeypatch, engine):
    results, _, _, _, _ = run_case(
        monkeypatch, engine, starting={key: "intermediate" for key in ("A", "B")},
        weather=Weather(rain_intensity=.5, track_wetness=.6, change_probability=0.),
    )
    assert [(row.driver_id, row.total_time) for row in results] == [("A", 270.), ("B", 330.)]
    assert all(row.abandonment_tire_rule.reason == "wet_tyre_used" for row in results)
    assert all(row.abandonment_tire_rule.penalty_seconds == 0 for row in results)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_abandonment_supersedes_an_active_requested_safety_car(monkeypatch, engine):
    schedule = [{"lap": 1, "control": "safety_car", "duration_laps": 5},
                {"lap": 3, "control": "red_flag", "action": "abandon"}]
    results, simulator, _, _, _ = run_case(monkeypatch, engine, schedule=schedule)
    assert results[0].race_abandonment.countback_lap == 2
    assert not simulator.event_manager.safety_car_active
    assert simulator.event_manager.safety_car_deployments == 1
    assert [row["status"] for row in simulator.event_manager.get_control_schedule_history()] == [
        "applied", "applied",
    ]
