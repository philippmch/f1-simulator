"""Mean dry tyre costs through the observed control prefix and green suffix."""

from dataclasses import dataclass
from math import inf, isfinite

import numpy as np

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models._native import forecast_json, register_forecast_helpers, shared_forecast_available
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import (
    SLICKS,
    DryPitDecision,
    _floor_plan,
    _floor_tables,
    _full_row,
    _ScaledDryWeather,
    expected_stationary_time,
)
from f1sim.simulation.strategy_control_clock import ObservedStandardField, StrategyControlContext


@dataclass(frozen=True, slots=True)
class _Cost:
    laps: int
    seconds: float

    @property
    def rank(self):
        return self.laps, -self.seconds


def _field_key(field):
    if type(field) is ObservedStandardField:
        # Without an external car, absolute time cannot change lockstep
        # running, pit discounts or fuel. Reuse equivalent future states.
        return (field.lap, field.intervals_left,
                field.now if len(field.rows) > 1 else None, tuple(field.rows),
                field.pre_stop_order, field.pitting)
    # Absolute scheduler counters and stale events cannot change subsequent
    # crossings. Preserve the relative priority of valid events, including
    # exact-time ties, so physically equivalent histories share one suffix.
    queue = tuple((time, distance, kind, identifier)
                  for time, distance, _, kind, identifier, generation in sorted(field.queue)
                  if identifier in field.pending
                  and field.pending[identifier].generation == generation)
    pending = tuple((key, tuple(value for name, value in vars(row).items()
                               if name != "generation")) for key, row in field.pending.items())
    return (field.now, field.intervals_left, field.updates, tuple(field.order),
            tuple(vars(field.timeline._clock).items()), tuple(field.timeline.states.items()),
            pending, queue, tuple(field.free_paces.items()))


def plan_controlled_dry_stop(
    driver, car, track, tire, age, lap, budget, mask, physical, scale, warmup_profile,
    current_fit_pending, context,
):
    """Price known controls, expected service and fitting before green futures.

    This decision-local search advances private field branches only while
    observed control or its pending no-passing restriction remains. Rivals
    hold observed pace and make no future choices. The established green
    suffix still uses its fixed own-lap horizon and clean-air assumptions.
    A prefix that reaches the actual flag is ranked by completed distance
    before elapsed time. No model, service sampler or race ledger is mutated.
    """
    if type(context) is not StrategyControlContext:
        raise ValueError("control_context must be a StrategyControlContext")
    horizon = track.total_laps - lap + 1
    projection = driver.model_copy(deep=True)
    package = car.model_copy(deep=True)
    tire = tire.model_copy(deep=True)
    fresh = {compound: TIRE_COMPOUNDS[compound].model_copy(deep=True) for compound in SLICKS}
    surface = _ScaledDryWeather(pace_scale=scale)
    physics = LapSimulator(np.random.default_rng(0))
    warmup = dict(warmup_profile)
    service = expected_stationary_time(package)
    root = context.new_field()
    if ((type(root) is ObservedStandardField and (root.lap != lap or root.now != context.now))
            or (type(root) is not ObservedStandardField
                and (root.timeline.states[root.identifier].completed_laps + 1 != lap
                     or root.timeline._clock.scheduled_laps != physical))):
        raise ValueError("control_context must match the current lap and physical race distance")
    native = shared_forecast_available()
    memo, green, running, green_rows = {}, {}, {}, {}
    green_tables = None
    invalid = _Cost(-1, inf)

    def lap_cost(set_tire, set_age, offset, field):
        gap = field.gap_ahead(
            field.free_paces[field.identifier] * field.running_modifier
            if type(field) is not ObservedStandardField else 1.)
        key = id(set_tire), set_age, offset, field.controlled, gap
        if native and key in running:
            return running[key]
        clean = projection if native else projection.model_copy(deep=True)
        clean.current_tire_laps = set_age
        value = physics.calculate_lap_time(
            clean, package if native else package.model_copy(deep=True),
            track if native else track.model_copy(deep=True),
            set_tire if native else set_tire.model_copy(deep=True),
            surface if native else surface.model_copy(deep=True), lap + offset, physical,
            active_aero_enabled=not field.controlled,
            gap_to_car_ahead=gap, sample_variation=False)
        if native:
            running[key] = value
        return value

    def green_cost(set_tire, set_age, offset, left, used):
        nonlocal green_tables
        if not native:
            decision = _floor_plan(
                projection, package, track, set_tire, set_age, lap + offset,
                left, used, physical, scale, True, 1., 1., 0.,
                warmup_profile=warmup_profile)
            return min(decision.wait_cost, decision.pit_now_cost)
        if green_tables is None:
            clean = projection.model_copy(deep=True)
            clean.reset_race_state()
            clean.id = clean.name = clean.team_id = "projection"
            prepared = package.model_copy(deep=True)
            prepared.team_id = prepared.team_name = "projection"
            models = forecast_json(clean), forecast_json(prepared), forecast_json(track)
            sets = tuple(forecast_json(TIRE_COMPOUNDS[compound]) for compound in SLICKS)
            green_tables = _floor_tables(models, sets, physical, scale, warmup_profile)
        costs, prefixes = green_tables
        number = lap + offset
        key = id(set_tire), set_age, offset
        if key not in green_rows:
            green_rows[key] = np.cumsum(_full_row(
                projection, package, track, set_tire, set_age, number, physical, scale))
        old = green_rows[key]
        wait_mask = (used | (1 << SLICKS.index(set_tire.compound))
                     if set_tire.compound in SLICKS else used)
        best = float(old[-1]) if wait_mask.bit_count() >= 2 else inf
        if left and number < track.total_laps:
            best = min(best, float(np.min(
                old[:-1] + costs[left, wait_mask, number + 1:track.total_laps + 1])))
        if left:
            for index in range(3):
                next_mask = used | (1 << index)
                prefix = prefixes[number, index]
                fitted = float(prefix[-1]) if next_mask.bit_count() >= 2 else inf
                if left > 1 and number < track.total_laps:
                    fitted = min(fitted, float(np.min(
                        prefix[:-1] + costs[left - 1, next_mask,
                                           number + 1:track.total_laps + 1])))
                # Keep the absolute floor planner's arithmetic order. At this
                # green entry its fresh first lap is exactly prefix[0].
                fitted = (fitted + float(prefix[0]) - float(prefix[0])
                          + track.pit_lane_delta + service)
                best = min(best, fitted)
        return best

    def future(field, set_tire, set_age, offset, left, used):
        cancellation_checkpoint()
        if offset == horizon or field.finished:
            return _Cost(0, 0.) if used.bit_count() >= 2 else invalid
        if not field.projection_required:
            key = id(set_tire), set_age, offset, left, used
            if not native or key not in green:
                cost = green_cost(set_tire, set_age, offset, left, used)
                green[key] = _Cost(horizon - offset, cost) if isfinite(cost) else invalid
            return green[key]
        key = (_field_key(field), id(set_tire), set_age, offset, left, used) if native else None
        if native and key in memo:
            return memo[key]
        best = action(field, set_tire, set_age, offset, left, used, fitted=False)
        if left:
            for compound in SLICKS:
                candidate = action(field, fresh[compound], 0, offset, left,
                                   used, fitted=True)
                if candidate.rank > best.rank:
                    best = candidate
        if native:
            memo[key] = best
        return best

    def action(field, set_tire, set_age, offset, left, used, *, fitted, first=False):
        cancellation_checkpoint()
        branch = field.fork()
        stopped = fitted or first and context.paid_fit
        if stopped:
            factor = .55 if field.controlled and field.safety_car else (
                .75 if field.controlled else 1.)
            delay = (context.current_stop_delay if first else
                     track.pit_lane_delta * factor + service)
        else:
            delay = None
        branch.enter(delay)
        fee = warmup.get(set_tire.compound.value, 0.) if (
            fitted or first and current_fit_pending) else 0.
        branch.cross(lap_cost(set_tire, set_age, offset, branch), fee)
        next_used = used
        if set_tire.compound in SLICKS:
            next_used |= 1 << SLICKS.index(set_tire.compound)
        suffix = future(branch, set_tire, set_age + 1, offset + 1,
                        left - int(fitted), next_used)
        if suffix.laps < 0:
            return invalid
        return _Cost(1 + suffix.laps, branch.now - field.now + suffix.seconds)

    retained = action(root, tire, age, 0, budget, mask, fitted=False, first=True)
    stopped, selected = invalid, None
    if budget and not context.paid_fit:
        for compound in SLICKS:
            candidate = action(root, fresh[compound], 0, 0, budget,
                               mask, fitted=True, first=True)
            if candidate.rank > stopped.rank:
                stopped, selected = candidate, compound
    common = context.current_stop_delay if context.paid_fit else 0.
    return DryPitDecision(
        stopped.seconds - common, retained.seconds - common, selected,
        stopped.laps, retained.laps,
    )


register_forecast_helpers(globals(), (
    "plan_controlled_dry_stop", "_field_key", "_floor_plan", "_floor_tables", "_full_row",
    "forecast_json", "expected_stationary_time",
    "ObservedStandardField", "StrategyControlContext", "_ScaledDryWeather", "_Cost",
    "shared_forecast_available",
))
register_forecast_helpers(vars(_Cost), ("rank",))
