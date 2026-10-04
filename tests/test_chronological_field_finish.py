"""Frozen field finish forecasts agree with the native chronological scheduler."""

import heapq
from copy import deepcopy
from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import numpy as np
import pytest
from test_custom_pit_replacements import snapshot

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import (
    ChronologicalFinishCar,
    ChronologicalFinishContext,
    evaluate_chronological_finish_protection,
)
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap, _PitServiceRecord
from f1sim.simulation.finish_strategy import ReplacementOption
from f1sim.simulation.pit_plans import initialize_pit_plan_state
from f1sim.simulation.pit_service import expected_remaining_service
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.race_timing import RaceFinishTimeline


class FieldPhysics:
    def __init__(self, paces):
        self.paces = paces
        self.entries = []

    def calculate_lap_time(self, driver, car, track, tire, weather, lap, total, **kwargs):
        self.entries.append((driver.id, lap, total, weather.track_wetness,
                             kwargs.get("active_aero_enabled"), driver.current_tire_laps))
        if driver.id != "A":
            return self.paces[driver.id]
        return 95. if tire.compound == TireCompound.SOFT else 99.


def ledger_signature(timeline):
    return deepcopy(timeline._clock.__dict__), dict(timeline.states), (
        timeline._leader_id, timeline.winner_id, timeline._last_observation_time)


def field(*, control="vsc", relative_laps=-1, remaining=50., neutralized=False,
          off_track=False, fee=0., now=7061., scheduled=90, intervals=1):
    completed = {"A": 71, "B": 71 + relative_laps}
    last = {"A": now, "B": now - 100.}
    observations = [(lap * last[key] / laps, lap, key)
                    for key, laps in completed.items() for lap in range(1, laps + 1)]
    timeline = RaceFinishTimeline(scheduled, completed)
    for time, lap, key in sorted(observations, key=lambda row: (row[0], -row[1], row[2])):
        leading = lap > max(row.completed_laps for row in timeline.states.values())
        timeline.observe_crossing(key, lap, time, is_leader=leading)
    row = ChronologicalFinishCar("B", completed["B"], 100., now + remaining,
                                 None if off_track else last["B"], neutralized, 0, fee)
    context = ChronologicalFinishContext(
        "A", timeline, ("A",) if off_track else ("B", "A"), (row,), 99.,
        1.4 if control == "sc" else 1.2 if control == "vsc" else 1., control == "sc",
        control_intervals=intervals)
    track = Track(id="timed", name="Timed", country="Test", total_laps=scheduled,
                  base_lap_time=100., pit_lane_delta=10.)
    driver, car = Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A")
    return driver, car, track, context, now


def native_path(inputs, *, stopped, delay=10., weather=None, warmup=None, pending_fit=False,
                replacement_age=0, forecast_context=None):
    """Use real running/crossing/exit methods; only future draws/policies are fixed."""
    driver, car, track, context, now = inputs
    warmup = warmup or {}
    simulator = RaceSimulator(np.random.default_rng(25), tire_warmup=warmup)
    simulator.weather_forecast_context = forecast_context
    engine = ChronologicalRace(simulator)
    engine.track = track.model_copy(deep=True)
    engine.weather = (weather or Weather(change_probability=0.)).model_copy(deep=True)
    engine.timeline = deepcopy(context.timeline)
    engine.order = list(context.order)
    engine.states = {}
    for key, row in context.timeline.states.items():
        engine.states[key] = DriverRaceState(
            driver.model_copy(deep=True) if key == "A" else Driver(id=key, name=key, team_id=key),
            car.model_copy(deep=True) if key == "A" else Car(team_id=key, team_name=key), 1,
            laps_completed=row.completed_laps,
            total_time=row.last_crossing_time or 0.,
            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM if key == "A" else TireCompound.HARD],
            tire_laps=8, tire_compound_history=["hard", "medium"],
        )
        engine.states[key].driver.current_tire_laps = 8
        if row.retired:
            engine.states[key].status = DriverStatus.DNF
        elif row.finish_time is not None:
            engine.states[key].status = DriverStatus.FINISHED
    engine.pending = {}
    engine.queue = []
    engine.serial = count(max((row.event_order for row in context.rivals), default=-1) + 1)
    engine.running_paces = {"A": context.own_pace,
                            **{row.identifier: row.free_running for row in context.rivals}}
    engine.control_intervals = max(row.completed_laps for row in context.timeline.states.values())
    engine.leader_id = context.timeline._leader_id
    engine.regrouping = False
    engine.green_streak, engine.has_two_green, engine.incidents = 0, False, 0
    engine.box_releases, engine.expected_box_releases, engine.fastest = {}, {}, {}
    engine.free_refits, engine.red_waiting, engine.pit_service_records = set(), set(), []
    control = simulator.event_manager
    control.safety_car_active = context.safety_car and context.control_intervals > 0
    control.vsc_active = not context.safety_car and context.control_intervals > 0
    control.safety_car_laps_remaining = control.vsc_laps_remaining = context.control_intervals
    physics = FieldPhysics(engine.running_paces)
    simulator.lap_simulator.calculate_lap_time = physics.calculate_lap_time
    simulator._should_pit = lambda *a, **k: False
    simulator._deploy_overtake_mode_if_eligible = lambda *a, **k: False
    simulator.overtaking_model.attempt_overtake = lambda *a, **k: (True, False)
    control._check_mechanical_failure = lambda *a, **k: None
    control._check_random_incident = lambda *a, **k: None

    def advance_control(*args, **kwargs):
        if control.safety_car_active:
            control.safety_car_laps_remaining -= 1
            control.safety_car_active = control.safety_car_laps_remaining > 0
        if control.vsc_active:
            control.vsc_laps_remaining -= 1
            control.vsc_active = control.vsc_laps_remaining > 0
        return []

    control.process_lap = advance_control
    if forecast_context is None:
        simulator._advance_race_weather = lambda surface: surface.project_surface()
    for row in context.rivals:
        state = engine.states[row.identifier]
        if row.committed_fit is not None:
            fit = row.committed_fit
            state.driver = fit.driver.model_copy(deep=True)
            state.car = fit.car.model_copy(deep=True)
            state.current_tire = fit.tire.model_copy(deep=True)
            state.tire_laps = state.driver.current_tire_laps = fit.age
        if row.running_start is None:
            state.fit_lap_pending = row.fitting_cost > 0.
            simulator.tire_warmup[state.current_tire.compound.value] = row.fitting_cost
        engine.pending[row.identifier] = _PendingLap(
            row.completed_laps + 1, state.total_time, row.ready, engine.weather,
            row.neutralized, state.current_tire, state.tire_laps, row.free_running,
            on_track=row.running_start is not None, running_start=row.running_start,
            paid_stop=row.committed_fit is not None)
        heapq.heappush(engine.queue, (row.ready, -row.completed_laps - 1, row.event_order,
                                      "cross" if row.running_start is not None else "exit",
                                      row.identifier, 0))
    own = engine.states["A"]
    own.fit_lap_pending = pending_fit
    if stopped:
        own.current_tire, own.tire_laps = TIRE_COMPOUNDS[TireCompound.SOFT], replacement_age
        own.driver.current_tire_laps, own.fit_lap_pending = replacement_age, True
        engine.order.remove("A")
    pending = _PendingLap(own.laps_completed + 1, now, now + delay if stopped else now,
                          engine.weather, True, own.current_tire, own.tire_laps, 0.,
                          on_track=not stopped, paid_stop=stopped)
    engine.pending["A"] = pending
    if not stopped:
        engine._begin_running(own, pending, now)
    engine._enqueue("A", pending.ready, "exit" if stopped else "cross")
    while engine.queue:
        time, _, _, kind, key, generation = heapq.heappop(engine.queue)
        pending = engine.pending.get(key)
        if pending is None or generation != pending.generation:
            continue
        if kind == "exit":
            engine.order.append(key)
            engine._begin_running(engine.states[key], pending, time)
            engine._enqueue(key, pending.ready, "cross")
        elif engine._resolve_crossing(key, time):
            if not engine.timeline.can_start_next_lap("A"):
                distance = own.laps_completed - context.timeline.states["A"].completed_laps
                return distance, time, physics
            if engine.timeline.can_start_next_lap(key):
                engine._start_lap(engine.states[key], time)
    pytest.fail("native scheduler did not finish the candidate")


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("relative_laps", [-1, 0, 1])
@pytest.mark.parametrize("neutralized", [False, True])
@pytest.mark.parametrize("remaining", [5., 100., 250.])
@pytest.mark.parametrize("delay", [10., 200.])
def test_field_clock_matches_native_pending_barriers_and_pit_exits(
    control, relative_laps, neutralized, remaining, delay,
):
    inputs = field(control=control, relative_laps=relative_laps,
                   neutralized=neutralized, remaining=remaining)
    driver, car, track, context, now = inputs
    before = (driver.model_copy(deep=True), car.model_copy(deep=True),
              track.model_copy(deep=True), ledger_signature(context.timeline))
    retained = native_path(inputs, stopped=False, delay=delay)
    stopped = native_path(inputs, stopped=True, delay=delay)
    physics = FieldPhysics({"B": 100.})
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), now, context,
        expected_lane_loss=delay, replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=physics)
    assert result.retained_laps == retained[0]
    assert result.stop_laps == stopped[0]
    assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    assert result.veto == (retained[0] > stopped[0])
    assert before == (driver, car, track, ledger_signature(context.timeline))
    assert all(row[2] == 90 for row in physics.entries)


@pytest.mark.parametrize("control", ["vsc", "sc"])
def test_lapped_physical_predecessor_prevents_a_false_distance_veto(control):
    inputs = field(control=control, relative_laps=-1, remaining=250., neutralized=True)
    driver, car, track, context, now = inputs
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), now, context,
        expected_lane_loss=10., replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=FieldPhysics({"B": 100.}))
    assert result.retained_laps == result.stop_laps == 2
    assert result.retained_crossing_time > 7400.
    assert not result.veto


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("remaining", [5., 120., 220.])
@pytest.mark.parametrize("delay", [10., 200.])
@pytest.mark.parametrize("replacement_age", [0, 7])
def test_expected_service_fits_and_entry_weather_match_native_execution(
    control, remaining, delay, replacement_age,
):
    inputs = field(control=control, relative_laps=0, remaining=remaining,
                   off_track=True, fee=7.)
    driver, car, track, context, now = inputs
    weather = Weather(track_wetness=.01, rain_intensity=.07, change_probability=0.)
    warmup = {"medium": 3., "soft": 5., "hard": 7.}
    retained = native_path(inputs, stopped=False, delay=delay, weather=weather,
                           warmup=warmup, pending_fit=True)
    stopped = native_path(inputs, stopped=True, delay=delay, weather=weather,
                          warmup=warmup, replacement_age=replacement_age)
    physics = FieldPhysics({"B": 100.})
    before = weather.model_copy(deep=True), ledger_signature(context.timeline)
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, weather, now, context,
        expected_lane_loss=delay,
        replacements=(ReplacementOption(TireCompound.SOFT, replacement_age, "used"),),
        tire_warmup=warmup, current_fit_pending=True, lap_simulator=physics)
    assert result.retained_laps == retained[0]
    assert result.stop_laps == stopped[0]
    assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    stop_entry = next(row for row in stopped[2].entries if row[0] == "A")
    retained_entries = [row for row in retained[2].entries if row[0] == "A"]
    assert physics.entries == [stop_entry, *retained_entries]
    assert stop_entry[-1] == replacement_age
    assert before == (weather, ledger_signature(context.timeline))


def snapshot_engine(*, control="vsc", sampled_end=7100., relative_laps=-1):
    driver, car, track, context, now = field(control=control, off_track=True, fee=7.,
                                          relative_laps=relative_laps)
    simulator = RaceSimulator(np.random.default_rng(24), tire_warmup={"hard": 7., "soft": 5.})
    engine = ChronologicalRace(simulator)
    engine.track, engine.weather = track, Weather()
    engine.timeline = deepcopy(context.timeline)
    engine.order = list(context.order)
    engine.states = {}
    for key, row in engine.timeline.states.items():
        engine.states[key] = DriverRaceState(
            Driver(id=key, name=key, team_id=key), Car(team_id=key, team_name=key), 1,
            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM if key == "A" else TireCompound.HARD],
            laps_completed=row.completed_laps, total_time=row.last_crossing_time,
            tire_laps=8, tire_compound_history=["hard", "medium"],
        )
    rival = engine.states["B"]
    rival.fit_lap_pending = True
    engine.pending = {"B": _PendingLap(rival.laps_completed + 1, now - 1., sampled_end + 10.,
                                        Weather(), True, rival.current_tire, 8, 0.,
                                        on_track=False, expected_exit=now + 5.)}
    engine.pit_service_records = [_PitServiceRecord(
        "B", "B", rival.laps_completed + 1, now - 1., now - 1., sampled_end, rival.car, 10.)]
    engine.queue = [(sampled_end + 10., -rival.laps_completed - 1, 37, "exit", "B", 0)]
    engine.running_paces = {"A": 99., "B": 100.}
    engine.control_intervals = max(row.laps_completed for row in engine.states.values())
    simulator.event_manager.safety_car_active = control == "sc"
    simulator.event_manager.vsc_active = control == "vsc"
    engine.states["A"].dry_pit_proposal = (72, TireCompound.SOFT)
    return engine, engine.states["A"], now


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("sampled_end", [7063., 7300.])
def test_snapshot_uses_conditional_service_and_does_not_advance_live_state(
    monkeypatch, control, sampled_end,
):
    engine, state, now = snapshot_engine(control=control, sampled_end=sampled_end)
    simulator = engine.simulator
    before = ([snapshot(row, simulator) for row in engine.states.values()],
              deepcopy(engine.pending), deepcopy(engine.queue), ledger_signature(engine.timeline),
              engine.weather.model_copy(deep=True))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *a, **k: pytest.fail("forecast called the live sampler"))
    context = engine._chronological_finish_context(state, now)
    assert context is not None
    assert context.rivals[0].ready == pytest.approx(
        now + expected_remaining_service(engine.states["B"].car, 1.) + 10.)
    assert context.rivals[0].running_start is None
    assert context.rivals[0].event_order == 37
    assert context.rivals[0].fitting_cost == 7.
    assert context.timeline is not engine.timeline
    engine._protect_elective_finish_distance(state, engine.track, now, 0., None)
    assert before == ([snapshot(row, simulator) for row in engine.states.values()],
                      engine.pending, engine.queue, ledger_signature(engine.timeline),
                      engine.weather)


@pytest.mark.parametrize("reason", ["missing_pending", "missing_event", "missing_pace",
                                    "repair", "critical", "instruction", "restart"])
def test_unresolved_commitment_cannot_create_a_field_finish_veto(monkeypatch, reason):
    engine, state, now = snapshot_engine()
    if reason == "missing_pending":
        engine.pending.clear()
    elif reason == "missing_event":
        engine.queue.clear()
    elif reason == "missing_pace":
        engine.running_paces["B"] = float("nan")
    elif reason == "repair":
        engine.states["B"].force_pit_next_lap = True
    elif reason == "critical":
        engine.states["B"].current_tire = TIRE_COMPOUNDS[TireCompound.WET]
    elif reason == "instruction":
        initialize_pit_plan_state(engine.states["B"], [dict(lap=72, compound="soft")])
    from f1sim.simulation import chronological_race as module
    monkeypatch.setattr(module, "evaluate_chronological_finish_protection",
                        lambda *a, **k: pytest.fail("unresolved field was priced"))
    assert not engine._protect_elective_finish_distance(
        state, engine.track, now, 0., None, restart=reason == "restart")


@pytest.mark.parametrize("reason", ["duplicate_order", "missing_order", "duplicate_rival",
                                    "missing_ledger", "stale_event", "invalid_own_pace",
                                    "missing_clock", "suspended", "future_observation"])
def test_invalid_field_context_leaves_stop_available(reason):
    driver, car, track, context, now = field()
    if reason == "duplicate_order":
        context = replace(context, order=("B", "A", "A"))
    elif reason == "missing_order":
        context = replace(context, order=("A",))
    elif reason == "duplicate_rival":
        context = replace(context, rivals=context.rivals * 2)
    elif reason == "missing_ledger":
        context = replace(context, timeline=SimpleNamespace())
    elif reason == "stale_event":
        context = replace(context, rivals=(replace(context.rivals[0], ready=now - 1.),))
    elif reason == "invalid_own_pace":
        context = replace(context, own_pace=float("nan"))
    elif reason == "missing_clock":
        context.timeline._clock = None
    elif reason == "suspended":
        context.timeline.begin_suspension(now)
    else:
        context.timeline._last_observation_time = now + 1.
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), now, context)
    assert not result.veto and result.reason.startswith("invalid")


@pytest.mark.parametrize("control", ["vsc", "sc"])
def test_announced_finish_follows_lapped_successor_after_retirement(control):
    driver, car, track, _, _ = field()
    timeline = RaceFinishTimeline(90, "ABZ")
    past = [(lap * last / count, lap, key)
            for key, count, last in (("A", 69, 7100.), ("B", 70, 7190.), ("Z", 71, 7200.1))
            for lap in range(1, count + 1)]
    for time, lap, key in sorted(past, key=lambda row: (row[0], -row[1], row[2])):
        leading = lap > max(row.completed_laps for row in timeline.states.values())
        timeline.observe_crossing(key, lap, time, is_leader=leading)
    timeline.retire("Z", 7202.)
    timeline.observe_crossing("A", 70, 7300.)
    rival = ChronologicalFinishCar("B", 70, 100., 7400., 7190., True, 0)
    context = ChronologicalFinishContext("A", timeline, ("B", "A"), (rival,), 99.,
                                         1.4 if control == "sc" else 1.2, control == "sc")
    inputs = driver, car, track, context, 7300.
    retained, stopped = native_path(inputs, stopped=False), native_path(inputs, stopped=True)
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), 7300., context,
        expected_lane_loss=10., replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=FieldPhysics({"B": 100.}))
    assert timeline.time_limit_announced and timeline.final_lap == 72
    assert result.retained_laps == retained[0] == result.stop_laps == stopped[0] == 1
    assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    assert not result.veto


@pytest.mark.parametrize("control", ["vsc", "sc"])
def test_scheduled_cap_skips_unnecessary_retained_physics(control):
    inputs = field(control=control, scheduled=72)
    driver, car, track, context, now = inputs
    stopped = native_path(inputs, stopped=True)
    physics = FieldPhysics({"B": 100.})
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), now, context,
        expected_lane_loss=10., replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=physics)
    assert result.stop_laps == stopped[0] == 1
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    assert result.retained_laps is None and not result.veto
    assert len(physics.entries) == 1 and physics.entries[0][2] == 72


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("offset", [-.001, 0., .001])
def test_expiry_ties_and_fitting_cost_follow_authoritative_finish_phase(control, offset):
    inputs = field(control=control, off_track=True, remaining=1000.)
    driver, car, track, context, now = inputs
    fee = 5.
    delay = 7200. - now - 95. * context.modifier - fee + offset
    stopped = native_path(inputs, stopped=True, delay=delay, warmup={"soft": fee})
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), now, context,
        expected_lane_loss=delay, tire_warmup={"soft": fee},
        replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=FieldPhysics({"B": 100.}))
    assert result.stop_laps == stopped[0] == (3 if offset < 0. else 2)
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("intervals", [0, 2, 4])
@pytest.mark.parametrize("relative_laps", [-1, 1])
@pytest.mark.parametrize("remaining", [5., 250.])
@pytest.mark.parametrize("off_track", [False, True])
def test_known_control_duration_and_old_pending_restrictions_follow_native_clock(
    control, intervals, relative_laps, remaining, off_track,
):
    inputs = field(control="green" if intervals == 0 else control,
                   intervals=intervals, relative_laps=relative_laps,
                   remaining=remaining, neutralized=True, off_track=off_track)
    driver, car, track, context, now = inputs
    retained, stopped = native_path(inputs, stopped=False), native_path(inputs, stopped=True)
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), now, context,
        expected_lane_loss=10., replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=FieldPhysics({"B": 100.}))
    assert result.retained_laps == retained[0]
    assert result.stop_laps == stopped[0]
    assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    assert result.veto == (retained[0] > stopped[0])


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("intervals", [2, 4])
@pytest.mark.parametrize("delay", [10., 100.])
def test_single_car_uses_remaining_control_instead_of_an_early_green_forecast(
    control, intervals, delay,
):
    driver, car, track, context, now = field(control=control, intervals=intervals, now=6800.)
    timeline = RaceFinishTimeline(90, ("A",))
    for lap in range(1, 72):
        timeline.observe_crossing("A", lap, lap * now / 71, is_leader=True)
    context = replace(context, timeline=timeline, order=("A",), rivals=())
    inputs = driver, car, track, context, now
    retained = native_path(inputs, stopped=False, delay=delay)
    stopped = native_path(inputs, stopped=True, delay=delay)
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, Weather(), now, context,
        expected_lane_loss=delay, replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=FieldPhysics({}))
    assert result.retained_laps == retained[0]
    assert result.stop_laps == stopped[0]
    assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    assert result.veto == (retained[0] > stopped[0])


@pytest.mark.parametrize("remaining", [np.int64(2), np.int64(4)])
@pytest.mark.parametrize("control", ["vsc", "sc"])
def test_native_duration_counter_is_frozen_without_consuming_it(control, remaining):
    engine, state, now = snapshot_engine(control=control)
    setattr(engine.simulator.event_manager,
            "safety_car_laps_remaining" if control == "sc" else "vsc_laps_remaining", remaining)
    context = engine._chronological_finish_context(state, now)
    assert context.control_intervals == int(remaining)
    assert type(context.control_intervals) is int
    assert engine._field_finish_required(engine._neutralized_finish_intervals())
    assert getattr(engine.simulator.event_manager,
                   "safety_car_laps_remaining" if control == "sc" else "vsc_laps_remaining") == (
                       remaining)


def test_green_decision_keeps_lapped_predecessors_old_neutralized_barrier(monkeypatch):
    engine, state, now = snapshot_engine()
    engine.simulator.event_manager.vsc_active = False
    engine.track.pit_lane_delta = 60.
    engine.order = ["B", "A"]
    pending = engine.pending["B"]
    pending.on_track, pending.running_start, pending.running = True, now - 100., 100.
    pending.ready = now + 250.
    engine.queue = [(pending.ready, -pending.lap, 37, "cross", "B", 0)]
    from f1sim.simulation.lap import LapSimulator
    physics = FieldPhysics({"B": 100.})
    monkeypatch.setattr(LapSimulator, "calculate_lap_time",
                        lambda self, *a, **k: physics.calculate_lap_time(*a, **k))
    assert engine._field_finish_required(engine._neutralized_finish_intervals())
    assert not engine._protect_elective_finish_distance(state, engine.track, now, 0., None)
    # The old independent first crossing would incorrectly claim an extra lap.
    monkeypatch.setattr(engine, "_field_finish_required", lambda intervals: False)
    assert engine._protect_elective_finish_distance(state, engine.track, now, 0., None)
