"""Reusable control forecasts agree with native service and crossing events."""

import heapq
from copy import deepcopy

import numpy as np
import pytest
from test_chronological_field_finish import field, ledger_signature, native_path

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import ObservedChronologicalField
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap
from f1sim.simulation.controlled_dry_strategy import _field_key
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.strategy_control_clock import StrategyControlContext


def signature(projection):
    return (ledger_signature(projection.timeline), projection.order.copy(),
            deepcopy(projection.pending), projection.queue.copy(), projection.free_paces.copy(),
            projection.now, projection.updates, projection.events.copy(), projection.entered)


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("intervals", [2, 4])
@pytest.mark.parametrize("relative_laps", [-1, 1])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("service", [0., 200.])
@pytest.mark.parametrize("first_stop", [False, True])
def test_repeated_paid_entries_and_fitting_laps_match_native_scheduler(
    monkeypatch, control, intervals, relative_laps, off_track, service, first_stop,
):
    inputs = field(control=control, intervals=intervals, relative_laps=relative_laps,
                   remaining=120., neutralized=True, off_track=off_track,
                   fee=7. if off_track else 0., now=6850.)
    _, _, _, context, now = inputs
    initial_lap = context.timeline.states["A"].completed_laps + 1
    warmup = {"medium": 3., "soft": 5., "hard": 7.}
    crossings, updates, entries = [], [], []
    start = ChronologicalRace._start_lap
    resolve = ChronologicalRace._resolve_crossing
    after = ChronologicalRace._after_leader_crossing

    def start_with_later_stops(engine, state, time, **kwargs):
        offset = state.laps_completed + 1 - initial_lap
        if state.driver.id != "A" or offset not in {1, 3}:
            return start(engine, state, time, **kwargs)
        # Commit a deterministic paid visit; native pit-exit/running/crossing
        # methods still resolve control, physical order, fitting and the flag.
        compound = TireCompound.MEDIUM if offset == 3 else TireCompound.SOFT
        state.current_tire = TIRE_COMPOUNDS[compound]
        state.tire_laps = state.driver.current_tire_laps = 0
        state.fit_lap_pending = True
        lane = engine.track.pit_lane_delta * engine.simulator._pit_lane_factor()
        engine.order.remove("A")
        pending = _PendingLap(
            state.laps_completed + 1, time, time + lane + service, engine.weather,
            True, state.current_tire, 0, 0., on_track=False, paid_stop=True)
        engine.pending["A"] = pending
        engine._enqueue("A", pending.ready, "exit")

    def crossing(engine, key, time):
        completed = resolve(engine, key, time)
        if completed and key == "A":
            crossings.append(time)
        return completed

    def leader(engine, time, red):
        if engine.timeline.chequered_time is None:
            updates.append(time)
        return after(engine, time, red)

    begin = ChronologicalRace._begin_running

    def capture_entry(engine, state, pending, time):
        result = begin(engine, state, pending, time)
        if state.driver.id == "A":
            entries.append((time, pending.active_aero_enabled))
        return result

    monkeypatch.setattr(ChronologicalRace, "_start_lap", start_with_later_stops)
    monkeypatch.setattr(ChronologicalRace, "_resolve_crossing", crossing)
    monkeypatch.setattr(ChronologicalRace, "_after_leader_crossing", leader)
    monkeypatch.setattr(ChronologicalRace, "_begin_running", capture_entry)
    native_path(inputs, stopped=first_stop, delay=service, warmup=warmup, pending_fit=True)

    before = ledger_signature(context.timeline), context.rivals
    projection = ObservedChronologicalField(context, now)
    projected_entries, projected_crossings = [], []
    compound = TireCompound.SOFT if first_stop else TireCompound.MEDIUM
    offset = 0
    while not projection.finished:
        fitted = first_stop if offset == 0 else offset in {1, 3}
        if offset in {1, 3}:
            compound = TireCompound.MEDIUM if offset == 3 else TireCompound.SOFT
        lane = inputs[2].pit_lane_delta * (
            .55 if projection.controlled and control == "sc" else
            .75 if projection.controlled else 1.)
        stop_delay = (service if offset == 0 else service + lane) if fitted else None
        projection.enter(stop_delay)
        projected_entries.append((projection.now, not projection.controlled))
        fee = warmup[compound.value] if fitted or offset == 0 else 0.
        projection.cross(95. if compound == TireCompound.SOFT else 99., fee)
        projected_crossings.append(projection.now)
        offset += 1
    assert projected_crossings == pytest.approx(crossings, abs=1.e-8)
    assert [row[0] for row in projected_entries] == pytest.approx(
        [row[0] for row in entries], abs=1.e-8)
    assert [row[1] for row in projected_entries] == [row[1] for row in entries]
    assert [event.time for event in projection.events if event.leading and event.flag_time is None
            ] == pytest.approx(updates, abs=1.e-8)
    assert projection.updates == len(updates)
    assert before == (ledger_signature(context.timeline), context.rivals)


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("off_track", [False, True])
def test_projection_branches_share_no_mutable_ledger_or_pending_events(control, off_track):
    *_, context, now = field(control=control, intervals=4, off_track=off_track,
                            remaining=5., neutralized=True, fee=7. if off_track else 0.)
    root = ObservedChronologicalField(context, now)
    before = signature(root)
    retained, stopped = root.fork(), root.fork()
    for branch, delay in ((retained, None), (stopped, 200.)):
        branch.enter(delay)
        entered = branch.fork()
        entered_before = signature(entered)
        branch.cross(99., 3.)
        assert signature(entered) == entered_before
        assert signature(root) == before
    assert retained.now < stopped.now
    assert retained.timeline is not stopped.timeline
    assert root.timeline is not context.timeline


def test_zero_service_is_a_paid_rejoin_and_does_not_consume_fitting_before_entry():
    *_, context, now = field(control="sc", intervals=2, remaining=100., now=6850.)
    root = ObservedChronologicalField(context, now)
    branch = root.fork()
    branch.enter(0.)
    assert branch.now == now and branch.updates == 0
    assert signature(root) != signature(branch)
    branch.cross(99., 20.)
    assert branch.now > now + 99.
    assert root.now == now and root.updates == 0


@pytest.mark.parametrize("delay", [-1., float("nan"), float("inf"), True])
def test_invalid_service_does_not_change_projection(delay):
    *_, context, now = field(intervals=4)
    projection = ObservedChronologicalField(context, now)
    before = signature(projection)
    with pytest.raises(ValueError):
        projection.enter(delay)
    assert signature(projection) == before


def test_copied_context_compares_observations_and_detects_ledger_changes():
    *_, context, now = field(intervals=4)
    original = StrategyControlContext(context, now, 20.)
    copied = deepcopy(original)
    assert original == copied
    assert copied.field.timeline is not original.field.timeline
    copied.field.timeline._clock.final_lap -= 1
    assert original != copied


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("stopped", [False, True])
def test_equivalent_event_counters_and_stale_events_preserve_complete_crossings(control, stopped):
    *_, context, now = field(control=control, intervals=4, relative_laps=-1,
                            remaining=120., neutralized=True, now=6850.)
    original = ObservedChronologicalField(context, now)
    shifted = original.fork()
    shifted.serial += 1000
    for row in shifted.pending.values():
        row.generation += 10
    shifted.queue = [(time, lap, serial + 1000, kind, key, generation + 10)
                     for time, lap, serial, kind, key, generation in shifted.queue]
    shifted.queue.append((0., -1, 0, "cross", "B", -1))
    heapq.heapify(shifted.queue)
    assert _field_key(original) == _field_key(shifted)
    changed = original.fork()
    changed.pending["B"].ready += .5
    assert _field_key(original) != _field_key(changed)
    for offset in range(3):
        for branch in (original, shifted):
            branch.enter(200. if stopped and offset in {0, 2} else None)
            branch.cross(99., 3. if offset in {0, 2} else 0.)
        assert ledger_signature(original.timeline) == ledger_signature(shifted.timeline)
        assert original.events == shifted.events
        assert original.order == shifted.order and original.now == shifted.now
        assert _field_key(original) == _field_key(shifted)
        if original.finished:
            break


def test_extra_mutable_ledger_attributes_are_copied_between_branches():
    *_, context, now = field(intervals=4)
    context.timeline.notes = ["original"]
    context.timeline._clock.notes = ["original"]
    observation = context.timeline.states["B"]
    object.__setattr__(observation, "notes", ["original"])
    original = ObservedChronologicalField(context, now)
    copied = original.fork()
    for owner in (copied.timeline, copied.timeline._clock, copied.timeline.states["B"]):
        owner.notes.append("copied")
    for ledger in (original.timeline, context.timeline):
        assert ledger.notes == ledger._clock.notes == ledger.states["B"].notes == ["original"]


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("intervals", [2, 4])
@pytest.mark.parametrize("lane", [2., 20.])
@pytest.mark.parametrize("fitting_fee", [0., 5.])
def test_standard_field_matches_completed_native_queue_with_later_paid_stop(
    monkeypatch, control, intervals, lane, fitting_fee,
):
    simulator = RaceSimulator(np.random.default_rng(32), tire_warmup={"soft": fitting_fee,
                                                                   "medium": 3.})
    controller = simulator.event_manager
    captured, starts, crossings = {}, {}, {}
    execute = RaceSimulator._execute_pit_stop
    batch = simulator._process_pit_stops

    def update(lap, *args, **kwargs):
        remaining = max(0, 2 + intervals - lap)
        active = 2 <= lap < 2 + intervals
        controller.safety_car_active = active and control == "sc"
        controller.vsc_active = active and control == "vsc"
        controller.safety_car_laps_remaining = remaining if controller.safety_car_active else 0
        controller.vsc_laps_remaining = remaining if controller.vsc_active else 0
        return []

    def stops(state, states, track, lap, *args, **kwargs):
        if state.driver.id == "A" and lap in {3, 5}:
            state.force_pit_next_lap = True
            return True
        return False

    def replacement(state, track, lap, *args, **kwargs):
        if lap == 3:
            captured["context"] = state.strategy_control_context
        return TireCompound.SOFT if lap == 3 else TireCompound.MEDIUM

    def mean_stop(owner, *args, **kwargs):
        kwargs["sample_service"] = False
        return execute(owner, *args, **kwargs)

    def physics(owner, driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        assert total_laps == 9
        return 100. if driver.id == "B" else 95. if tire.compound == TireCompound.SOFT else 99.

    def capture_batch(states, frozen, track, weather, lap, **kwargs):
        own = next(state for state in states if state.driver.id == "A")
        if lap > 3:
            crossings[lap - 1] = own.total_time
        result = batch(states, frozen, track, weather, lap, **kwargs)
        starts[lap] = own.total_time
        return result

    monkeypatch.setattr(controller, "process_lap", update)
    monkeypatch.setattr(controller, "_check_mechanical_failure", lambda *a, **k: None)
    monkeypatch.setattr(controller, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(simulator, "_should_pit", stops)
    monkeypatch.setattr(simulator, "_choose_committed_dry_compound", replacement)
    monkeypatch.setattr(simulator, "_deploy_overtake_mode_if_eligible", lambda *a, **k: False)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **k: (True, False))
    monkeypatch.setattr(simulator, "_process_pit_stops", capture_batch)
    monkeypatch.setattr(RaceSimulator, "_execute_pit_stop", mean_stop)
    monkeypatch.setattr(LapSimulator, "calculate_lap_time", physics)
    simulator.simulate_race(
        [Driver(id=key, name=key, team_id=key) for key in ("A", "B")],
        {key: Car(team_id=key, team_name=key) for key in ("A", "B")},
        Track(id="T", name="T", country="Test", total_laps=9, base_lap_time=100.,
              pit_lane_delta=lane), Weather(change_probability=0.), ["A", "B"],
        starting_tires={"A": TireCompound.HARD, "B": TireCompound.HARD})

    context = captured["context"]
    assert context is not None
    projection = context.new_field()
    for offset in range(intervals):
        lap = 3 + offset
        projection.enter(context.current_stop_delay if lap in {3, 5} else None)
        assert projection.now == pytest.approx(starts[lap], abs=1.e-8)
        compound = TireCompound.SOFT if lap < 5 else TireCompound.MEDIUM
        fee = simulator.tire_warmup.get(compound.value, 0.) if lap in {3, 5} else 0.
        projection.cross(95. if compound == TireCompound.SOFT else 99., fee)
        assert projection.now == pytest.approx(crossings[lap], abs=1.e-8)
    assert not projection.projection_required
