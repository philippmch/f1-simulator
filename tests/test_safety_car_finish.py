"""Whole SC first-lap fields establish each branch's conditional finish clock."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_custom_pit_replacements import snapshot
from test_leading_finish_context import inputs
from test_leading_finish_strategy import BoundPhysics, execute, history, models

from f1sim.models import TireCompound, Weather
from f1sim.models._native import shared_forecast_available
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import neutralization
from f1sim.simulation.finish_strategy import (
    LeadingFinishContext,
    ReplacementOption,
    RivalFinishForecast,
    SafetyCarFinishBranch,
    SafetyCarFinishCar,
    SafetyCarFinishField,
    evaluate_finish_protection,
)
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_plans import initialize_pit_plan_state
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.tire_inventory import TireInventory


def crossings(rows, modifier):
    """Independent queue equation, then the real model's fitting barrier.

    A follower lies between its free crossing and nominal crossing, approaches
    its predecessor plus one second, and cannot cross that predecessor. Raw
    running is resolved before any fitting fee; fitted clocks preserve order.
    """
    queue_pace = rows[0][2] * modifier
    result, raw_ahead, fitted_ahead = {}, None, None
    for identifier, entry, free, fee in rows:
        nominal = max(free, queue_pace)
        raw = entry + nominal if raw_ahead is None else max(
            raw_ahead, entry + free, min(entry + nominal, raw_ahead + 1.))
        fitted = raw + fee
        if fitted_ahead is not None:
            fitted = max(fitted, fitted_ahead)
        result[identifier] = fitted
        raw_ahead, fitted_ahead = raw, fitted
    return result


def test_replaced_queue_law_bypasses_native_forecast_caches(monkeypatch):
    branch = SafetyCarFinishBranch((None, SafetyCarFinishCar("B", 1., 90.)))
    native_crossings = branch.project(0., 90., 0., 1.4)
    assert shared_forecast_available()
    original = neutralization.safety_car_running_time
    monkeypatch.setattr(neutralization, "safety_car_running_time",
                        lambda *args, **kwargs: original(*args, **kwargs) + 2.)
    assert not shared_forecast_available()
    assert branch.project(0., 90., 0., 1.4) != native_crossings


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("pending_fit", [False, True])
@pytest.mark.parametrize("queue_delay", [0., 5.])
def test_complete_snapshot_prices_mean_rival_fits_and_both_queue_orders(
    monkeypatch, index, finite, committed, pending_fit, queue_delay,
):
    simulator, frozen, track = inputs()
    simulator.event_manager.safety_car_active = True
    simulator.tire_warmup = {"hard": 100., "soft": 9., "medium": 4.}
    for state, clock in zip(frozen, (7100., 7101., 7106.)):
        state.total_time = clock
    own = frozen[index]
    rival = next(row for row in frozen if row is not own)
    rival.current_tire = TIRE_COMPOUNDS[TireCompound.HARD]
    rival.fit_lap_pending = pending_fit
    if finite:
        pool = TireInventory.from_sets([
            dict(id="H", compound="hard", age=8), dict(id="S", compound="soft", age=7),
        ])
        simulator._initialize_inventory(rival, pool, pool.sets["H"])
        rival.inventory_pit_proposal = (72, "S")
    else:
        rival.dry_pit_proposal = (72, TireCompound.SOFT)
    active = deepcopy(frozen)
    # A private sampled clock differs between active and lap-start views.
    # Only the known expected loss below may enter the field projection.
    active_rival = next(row for row in active if row.driver.id == rival.driver.id)
    active_rival.total_time += 999.
    losses = {rival.driver.id: 12.} if committed else {}
    before = ([snapshot(state, simulator) for state in frozen + active], deepcopy(track))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *a, **k: pytest.fail("live sampler used by a finish forecast"))
    field = simulator._standard_safety_car_finish_field(
        own, frozen, active, track, Weather(), 72, queue_delay, losses, 90,
    )
    assert field is not None
    for stopped, branch in ((False, field.retained), (True, field.stopped)):
        rows = deepcopy(frozen)
        candidate = rows[index]
        pitting = []
        for row in rows:
            if row.driver.id in losses:
                row.total_time += losses[row.driver.id]
                row.current_tire = TIRE_COMPOUNDS[TireCompound.SOFT]
                row.tire_laps = 7 if finite else 0
                pitting.append(row)
                row.fit_lap_pending = True
        if stopped:
            candidate.total_time = (candidate.total_time + track.pit_lane_delta * .55
                                    + expected_stationary_time(candidate.car) + queue_delay)
            candidate.current_tire = TIRE_COMPOUNDS[TireCompound.SOFT]
            candidate.tire_laps, candidate.fit_lap_pending = 3, True
            pitting.append(candidate)
        simulator._handle_pit_batch_position_changes(pitting, rows)
        ordered = sorted(rows, key=lambda row: row.position)
        observations = []
        for row in ordered:
            row.driver.current_tire_laps = row.tire_laps
            gap = simulator._get_gap_to_car_ahead(row, rows)
            free = LapSimulator().calculate_lap_time(
                row.driver, row.car, track, row.current_tire, Weather(), 72, 90,
                sample_variation=False, active_aero_enabled=False, gap_to_car_ahead=gap,
            )
            fee = simulator.tire_warmup.get(row.current_tire.compound.value, 0.) if (
                row.fit_lap_pending) else 0.
            observations.append((row.driver.id, row.total_time, free, fee))
        assert [None if row is None else row.identifier for row in branch.rows] == [
            None if row is candidate else row.driver.id for row in ordered]
        own_values = next(row for row in observations if row[0] == candidate.driver.id)
        projected, rivals = branch.project(*own_values[1:], 1.4)
        expected = crossings(observations, 1.4)
        assert projected == pytest.approx(expected.pop(candidate.driver.id), abs=1.e-8)
        assert rivals == pytest.approx(expected, abs=1.e-8)
        assert branch.gap_at(candidate.total_time) == simulator._get_gap_to_car_ahead(
            candidate, rows)
    assert before == ([snapshot(state, simulator) for state in frozen + active], track)


@pytest.mark.parametrize("rival_free", [95., 100.])
@pytest.mark.parametrize("fitting", [0., 100.])
def test_branch_queue_clock_agrees_with_executed_leading_announcement(rival_free, fitting):
    now, fee, maximum = 7061., 10., 90
    rival = SafetyCarFinishCar("B", now + 1., rival_free, fitting)
    field = SafetyCarFinishField(SafetyCarFinishBranch((None, rival)),
                                 SafetyCarFinishBranch((rival, None)))
    before = deepcopy(field)
    observed = RivalFinishForecast(71, now + 1. + 96. * 1.4 + fitting, 96., "B")
    context = LeadingFinishContext(7200., rivals=(observed,), lockstep=True, safety_car_field=field)
    outcomes = []
    for stopped, free in ((False, 99.), (True, 95.)):
        candidate = ("A", now + (fee if stopped else 0.), free, 0.)
        other = ("B", rival.entry_time, rival.free_running, rival.fitting_cost)
        first = crossings([other, candidate] if stopped else [candidate, other], 1.4)
        rivals = (replace(observed, next_crossing_time=first["B"]),)
        past = history(now, rivals, maximum)
        outcomes.append(execute(past, now, rivals, maximum, fee if stopped else 0., free, True,
                                first_crossing=first["A"]))
    driver, car, track = models(maximum)
    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, 72, Weather(),
        now, None, maximum, lap_simulator=BoundPhysics(), expected_lane_loss=fee,
        replacements=(ReplacementOption(TireCompound.SOFT),), leading_finish_context=context,
        current_lap_time_modifier=1.4, active_aero_enabled=False,
    )
    assert result.retained_laps == outcomes[0][0]
    assert result.stop_laps == outcomes[1][0]
    assert result.retained_crossing_time == pytest.approx(outcomes[0][1], abs=1.e-8)
    assert result.stop_crossing_time == pytest.approx(outcomes[1][1], abs=1.e-8)
    assert result.veto == (outcomes[0][0] > outcomes[1][0])
    assert field == before


@pytest.mark.parametrize("reason", ["repair", "critical", "instruction", "missing_frozen",
                                    "unresolved_fit", "unavailable_mean", "merge_extension"])
def test_unresolved_field_decisions_do_not_guess_a_finish_queue(monkeypatch, reason):
    simulator, states, track = inputs()
    simulator.event_manager.safety_car_active = True
    frozen, losses, weather = deepcopy(states), {}, Weather()
    if reason == "repair":
        states[1].force_pit_next_lap = True
    elif reason == "critical":
        states[1].current_tire = TIRE_COMPOUNDS[TireCompound.WET]
    elif reason == "instruction":
        initialize_pit_plan_state(states[1], [dict(lap=72, compound="soft")])
    elif reason == "missing_frozen":
        frozen = frozen[:2]
    elif reason == "unresolved_fit":
        losses["B"] = 12.
    elif reason == "unavailable_mean":
        monkeypatch.setattr(LapSimulator, "calculate_lap_time",
                            lambda *a, **k: float("nan"))
    else:
        monkeypatch.setattr(simulator, "_handle_pit_batch_position_changes",
                            lambda *a, **k: pytest.fail("unresolved merge extension was called"))
    before = [snapshot(row, simulator) for row in states + frozen]
    assert simulator._standard_safety_car_finish_field(
        states[0], frozen, states, track, weather, 72, 0., losses, 90,
    ) is None
    assert before == [snapshot(row, simulator) for row in states + frozen]


@pytest.mark.parametrize("values", [
    dict(identifier="", entry_time=0., free_running=90.),
    dict(identifier="B", entry_time=-1., free_running=90.),
    dict(identifier="B", entry_time=0., free_running=True),
    dict(identifier="B", entry_time=0., free_running=float("inf")),
    dict(identifier="B", entry_time=0., free_running=90., fitting_cost=-1.),
])
def test_invalid_mean_field_observations_are_rejected(values):
    with pytest.raises(ValueError):
        SafetyCarFinishCar(**values)


@pytest.mark.parametrize("reason", ["missing_rival", "duplicate_rival", "malformed_rival",
                                    "unhashable_identifier", "chronological"])
def test_inconsistent_leading_context_cannot_veto(reason):
    rival = SafetyCarFinishCar("B", 100., 100.)
    branch = SafetyCarFinishBranch((None, rival))
    context = LeadingFinishContext(7200., lockstep=True,
                                   rivals=(RivalFinishForecast(0, 100., 100., "B"),),
                                   safety_car_field=SafetyCarFinishField(branch, branch))
    if reason == "missing_rival":
        context = replace(context, rivals=())
    elif reason == "duplicate_rival":
        context = replace(context, rivals=context.rivals * 2)
    elif reason == "malformed_rival":
        context = replace(context, rivals=(object(),))
    elif reason == "unhashable_identifier":
        context = replace(context, rivals=(replace(context.rivals[0], identifier=[]),))
    else:
        context = replace(context, lockstep=False)
    driver, car, track = models()
    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, 1, Weather(),
        0., None, 90, leading_finish_context=context,
    )
    assert not result.veto and result.reason.startswith("invalid")
