"""Red-flag resumption is a counted SC lap before the first green restart."""

from copy import deepcopy

import numpy as np
import pytest

import f1sim.analysis.montecarlo as montecarlo
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventManager, EventType
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.strategy_neutralization import observed_control_intervals


def quiet(monkeypatch, control):
    for name in ("_check_mechanical_failure", "_check_random_incident",
                 "_check_red_flag_conditions"):
        monkeypatch.setattr(control, name, lambda *a, **kw: None)


def track(laps=8):
    return Track(id="T", name="Test", country="Synthetic", total_laps=laps,
                 base_lap_time=90, safety_car_probability=0)


def controlled(monkeypatch, engine, *, laps=4, red=(1,), schedule=(), pause=100,
               warmup=0, inventory=None):
    simulator = RaceSimulator(np.random.default_rng(71), red_flag_pause_seconds=pause,
                              control_schedule=list(schedule), tire_warmup={"hard": warmup})
    control = simulator.event_manager
    quiet(monkeypatch, control)
    control.set_forced_red_flag(list(red))
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(simulator, "_choose_red_flag_tire", lambda *a, **kw: TireCompound.HARD)
    samples, recharges, forecasts = [], [], []

    def physics(*args, **kwargs):
        lap = args[5] if args else kwargs["lap_number"]
        tire = args[3] if args else kwargs["tire"]
        samples.append((lap, control.safety_car_active, kwargs["active_aero_enabled"],
                        control.is_overtake_mode_allowed(lap, Weather()), tire.compound))
        return 90.

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    recharge = simulator._recharge_overtake_mode_energy

    def record_recharge(states, neutralized=False):
        recharges.append(neutralized)
        recharge(states, neutralized=neutralized)

    monkeypatch.setattr(simulator, "_recharge_overtake_mode_energy", record_recharge)
    fit = simulator._fit_red_flag_tires

    def record_fit(states, weather, planning, lap, **kwargs):
        forecasts.append(planning.total_laps)
        return fit(states, weather, planning, lap, **kwargs)

    if engine == "standard":
        monkeypatch.setattr(simulator, "_fit_red_flag_tires", record_fit)
        runner = simulator
        run = simulator.simulate_race
    else:
        runner = ChronologicalRace(simulator, red_flag_pause_seconds=pause)
        fit_set = runner._fit_red_flag_set

        def record_fit_set(state, planning, *args):
            forecasts.append(planning.total_laps)
            return fit_set(state, planning, *args)

        monkeypatch.setattr(runner, "_fit_red_flag_set", record_fit_set)
        run = runner.run
    (result,) = run([Driver(id="A", name="A", team_id="T")],
                   {"T": Car(team_id="T", team_name="T", reliability=1)}, track(laps),
                   Weather(change_probability=0), ["A"], starting_tires={"A": "soft"},
                   tire_inventory=inventory)
    return result, simulator, runner, samples, recharges, forecasts


def test_resumption_has_one_observed_sc_lap_and_a_subsequent_restart_delay(monkeypatch):
    control = EventManager(np.random.default_rng(71), control_schedule=[])
    quiet(monkeypatch, control)
    control.set_forced_red_flag(2)
    red, = control.process_lap(2, [], {}, track(), Weather())
    before_rng = deepcopy(control.rng.bit_generator.state)
    control.end_red_flag()
    assert not control.red_flag_active and control.safety_car_active
    assert observed_control_intervals(control) == 1
    assert control.get_lap_time_modifier() == 1.4
    assert not control.is_active_aero_allowed() and not control.is_overtake_mode_allowed(3)
    assert control.red_flag_restart_lap_number == 3
    assert control.safety_car_deployments == 1
    resumption, = [event for event in control.events if event.event_type == EventType.SAFETY_CAR]
    assert resumption.lap == red.lap == 2 and resumption.duration_laps == 1
    assert resumption.announced_after_crossing
    assert "resumption" in resumption.description.lower()
    control.process_lap(3, [], {}, track(), Weather())
    assert not control.safety_car_active and observed_control_intervals(control) == 0
    assert not control.is_overtake_mode_allowed(3)  # Completed SC snapshot remains frozen.
    assert control.is_restart_lap(4) and not control.is_overtake_mode_allowed(4)
    assert control.is_active_aero_allowed()
    control.process_lap(4, [], {}, track(), Weather())
    assert control.is_overtake_mode_allowed(5)
    assert control.rng.bit_generator.state == before_rng


def test_repeated_end_does_not_manufacture_another_resumption(monkeypatch):
    control = EventManager(np.random.default_rng(71), control_schedule=[])
    quiet(monkeypatch, control)
    control.set_forced_red_flag(2)
    control.process_lap(2, [], {}, track(), Weather())
    control.end_red_flag()
    control.process_lap(3, [], {}, track(), Weather())
    before = (list(control.events), control.safety_car_deployments)
    control.end_red_flag()
    assert not control.safety_car_active
    assert (control.events, control.safety_car_deployments) == before


def test_direct_suspension_does_not_reuse_an_older_green_mode_snapshot(monkeypatch):
    control = EventManager(np.random.default_rng(71), control_schedule=[])
    quiet(monkeypatch, control)
    control.process_lap(3, [], {}, track(), Weather())
    assert control.is_overtake_mode_allowed(3)
    control.deploy_red_flag(4, "direct controller")
    assert not control.is_overtake_mode_allowed(4)
    control.end_red_flag()
    assert not control.is_overtake_mode_allowed(5)
    resumption, = control.events
    assert resumption.lap == 4 and control.red_flag_restart_lap_number == 5


def test_new_suspension_supersedes_resumption_and_reset_clears_it(monkeypatch):
    control = EventManager(np.random.default_rng(71), control_schedule=[])
    quiet(monkeypatch, control)
    control.set_forced_red_flag([2, 3])
    control.process_lap(2, [], {}, track(), Weather())
    control.end_red_flag()
    events = control.process_lap(3, [], {}, track(), Weather())
    assert [event.event_type for event in events] == [EventType.RED_FLAG]
    assert control.red_flag_active and not control.safety_car_active
    assert control.sc_restart_lap_number is None
    control.end_red_flag()
    assert control.safety_car_active and control.safety_car_deployments == 2
    assert control.red_flag_restart_lap_number == 4
    control.reset()
    assert not control.safety_car_active and control.safety_car_deployments == 0
    assert control.events == [] and control.red_flag_restart_lap_number is None


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("laps,points", [(2, 0), (3, 0), (4, 25)])
def test_native_resumption_counts_distance_but_never_green_credit(
    monkeypatch, engine, laps, points,
):
    result, simulator, runner, samples, recharges, _ = controlled(
        monkeypatch, engine, laps=laps,
    )
    assert result.laps_completed == laps and result.classified
    assert result.points_awarded == points
    assert result.total_time == pytest.approx(100 + 90 * laps + 36)
    assert result.race_suspension_seconds == 100
    assert result.fastest_lap == 90 and result.pit_stops == 0
    assert [row[1] for row in samples] == [False, True] + [False] * (laps - 2)
    assert [row[2] for row in samples] == [True, False] + [True] * (laps - 2)
    assert [row[3] for row in samples] == [False] * min(laps, 3) + [True] * (laps - 3)
    assert recharges == [False, True] + [False] * (laps - 2)
    assert simulator.event_manager.safety_car_deployments == 1
    assert [event.event_type for event in simulator.event_manager.events] == [
        EventType.RED_FLAG, EventType.SAFETY_CAR,
    ]
    assert runner.suspensions == [(90, 190, ("A",))]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_resumption_pace_is_used_by_timed_finish_and_free_fit_horizon(monkeypatch, engine):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 200.)
    result, _, _, _, _, forecasts = controlled(monkeypatch, engine, laps=10)
    # The 100-second suspension extends expiry to 300. The counted SC lap
    # ends at 316, announces the final lap, and green lap three ends at 406.
    assert forecasts == [3]
    assert result.laps_completed == 3 and result.race_time_limited
    assert result.total_time == pytest.approx(406)
    assert result.points_awarded == 0  # No complete green pair either side of suspension.


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_resumption_fit_fee_is_unscaled_and_run_only_once(monkeypatch, engine):
    result, _, _, _, _, _ = controlled(monkeypatch, engine, warmup=30)
    assert result.total_time == pytest.approx(100 + 90 * 4 + 36 + 30)
    assert result.pit_stops == 0
    assert result.strategy == ["soft", "hard"]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_final_red_flag_never_creates_sc_resumption(monkeypatch, engine):
    result, simulator, runner, _, _, forecasts = controlled(
        monkeypatch, engine, laps=2, red=(2,),
    )
    assert result.laps_completed == 2 and result.total_time == 180
    assert runner.suspensions == [] and forecasts == []
    assert [event.event_type for event in simulator.event_manager.events] == [EventType.RED_FLAG]
    assert simulator.event_manager.safety_car_deployments == 0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_scheduled_controls_observe_procedural_resumption(monkeypatch, engine):
    schedule = [dict(lap=2, control="vsc", duration_laps=1),
                dict(lap=4, control="safety_car", duration_laps=1)]
    result, simulator, _, samples, _, _ = controlled(
        monkeypatch, engine, laps=5, schedule=schedule,
    )
    control = simulator.event_manager
    assert control.get_control_schedule_history() == [
        schedule[0] | {"status": "suppressed", "reason": "existing_neutralization"},
        schedule[1] | {"status": "applied", "reason": "scheduled_announcement"},
    ]
    assert [row[1] for row in samples] == [False, True, False, False, True]
    assert control.safety_car_deployments == 2 and control.vsc_deployments == 0
    assert result.points_awarded == 25  # Green laps three and four already earned eligibility.


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_resumption_lap_consumes_finite_set_life(monkeypatch, engine):
    inventory = {"A": [dict(id="S", compound="soft", age=0, remaining_laps=1),
                       dict(id="H", compound="hard", age=4, remaining_laps=3)]}
    result, _, _, samples, _, _ = controlled(monkeypatch, engine, inventory=inventory)
    assert result.laps_completed == 4 and result.pit_stops == 0
    assert [row[4] for row in samples] == [TireCompound.SOFT] + [TireCompound.HARD] * 3
    assert {row["id"]: (row["age"], row["remaining_laps"])
            for row in result.tire_inventory} == {"S": (1, 0), "H": (7, 0)}
    assert [(row["set_id"], row["laps_used"]) for row in result.tire_set_history] == [
        ("S", 1), ("H", 3),
    ]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_worker_and_statistics_include_resumption_without_fabricating_requests(monkeypatch, engine):
    actual_simulator = montecarlo.RaceSimulator

    def simulator(**kwargs):
        instance = actual_simulator(**kwargs)
        quiet(monkeypatch, instance.event_manager)
        instance.event_manager.set_forced_red_flag(1)
        monkeypatch.setattr(instance, "_should_pit", lambda *a, **kw: False)
        monkeypatch.setattr(instance, "_choose_red_flag_tire", lambda *a, **kw: TireCompound.HARD)
        return instance

    monkeypatch.setattr(montecarlo, "RaceSimulator", simulator)
    results = montecarlo.MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")],
        {"T": Car(team_id="T", team_name="T", reliability=1)}, track(3),
        Weather(change_probability=0), seed=71, race_engine=engine,
        starting_tires={"A": "soft"}, control_schedule=[],
    ).run(2, parallel=False)
    assert results.event_stats.safety_car_count == results.event_stats.red_flag_count == 2
    assert results.event_stats.vsc_count == 0
    assert results.get_event_rates()["avg_safety_cars"] == 1
    assert results.control_schedule_histories == [[], []]
    assert all(context["has_two_green_laps"] is False for context in results.race_points_contexts)
    assert all(row.points_awarded == 0 for race in results.race_results for row in race)
