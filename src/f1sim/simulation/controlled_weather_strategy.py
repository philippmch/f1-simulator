"""Weather and physical tyre choices through an observed racing field."""

from bisect import bisect_right
from copy import copy
from dataclasses import dataclass
from functools import lru_cache
from math import inf, isfinite, nextafter

import numpy as np

from f1sim.cancellation import cancellation_checkpoint
from f1sim.models._native import native_physics, register_forecast_helpers
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.chronological_finish import native_finish_timeline
from f1sim.simulation.lap import LapSimulator, minimum_lap_time
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_control_clock import (
    ObservedStandardField,
    ProjectedControlCost,
    StrategyControlContext,
    observed_control_key,
)
from f1sim.simulation.strategy_lap import (
    control_lap_scope,
    install_control_wear_bound,
    isolated_strategy_lap,
)
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import (
    exchange_tire_slots,
    tire_set_slot,
    tire_slot_usable,
)
from f1sim.simulation.warmup import tire_warmup_seconds
from f1sim.simulation.weather_schedule import paid_compound_candidates, project_next_surface


@dataclass(frozen=True, slots=True)
class ControlledWeatherDecision:
    pit: ProjectedControlCost
    wait: ProjectedControlCost
    compound: TireCompound | None
    set_id: str | None


def usable_weather_control(context, weather_clock):
    """Custom weather-clock dispatch keeps the existing forecast semantics."""
    if context is not None and type(context) is not StrategyControlContext:
        raise ValueError("control_context must be a StrategyControlContext")
    return (context is not None and not context.paid_fit
            and (weather_clock is None or type(weather_clock) is StrategyWeatherClock))


def _green_weather_crossings(field, horizon):
    """Hold independent green crossings without rebuilding follower ledgers.

    The caller has verified native dispatch and cleared every neutralized lap.
    Each car's next committed crossing is retained, including a pending service
    exit and fitting fee. Subsequent crossings use repeated addition, matching
    the event scheduler's rounding. Only a new leading distance can update the
    weather or finish clock; follower crossings cannot change these signals.
    """
    clock = copy(field.timeline._clock)
    identifier, origin = field.identifier, field.now
    pace = field.free_paces[identifier]
    streams = [(field.timeline.states[identifier].completed_laps + 1,
                origin + pace, pace)]
    for key, row in field.pending.items():
        if key == identifier:
            continue
        ready = row.ready
        if row.running_start is None:
            ready = (ready + row.free_running) + row.fitting_cost
        streams.append((row.lap, ready, row.free_running))
    reset = (field.timeline._leader_id is not None
             and field.timeline.states[field.timeline._leader_id].retired)
    times = []
    distance = field.active_distance
    while clock.winner_time is None:
        cancellation_checkpoint()
        distance += 1
        candidates = []
        for lap, ready, free in streams:
            while lap < distance:
                crossing = ready + free
                if not isfinite(crossing) or crossing <= ready:
                    raise ValueError("invalid projected running crossing")
                lap, ready = lap + 1, crossing
            candidates.append((lap, ready, free))
        streams = candidates
        crossing = min(ready for _, ready, _ in streams)
        clock.observe_leader_crossing(distance, crossing, allow_leadership_reset=reset)
        reset = False
        if clock.winner_time is None:
            times.append(crossing)
    starts, counts = [], []
    now = origin
    for _ in range(horizon):
        cancellation_checkpoint()
        starts.append(now - origin)
        counts.append(bisect_right(times, now))
        crossing = now + pace
        if not isfinite(crossing) or crossing <= now:
            raise ValueError("invalid projected running crossing")
        now = crossing
    return starts, counts, tuple(time - origin for time in times)


def green_weather_forecast(field, horizon, stop_delay, *, native=False):
    """Rebase a held green field onto its actual post-control observations.

    A projected external leader supplies paid-stop-aware update events. When
    the candidate leads, its own running crossings supply the weather cadence,
    as in the existing green planner. No flag produces an extra update.
    """
    if type(field) is ObservedStandardField:
        return None, None
    if (native and not field.entered and not field.projection_required
            and (field._native_crossings or native_finish_timeline(field.timeline))):
        starts, counts, times = _green_weather_crossings(field, horizon)
        identifier = field.identifier
        leader = min(field.order,
                     key=lambda key: -field.timeline.states[key].completed_laps)
        if leader == identifier:
            return tuple(counts), None
        return None, StrategyWeatherClock(
            tuple(starts), times[0] if times else 0., field.free_paces[leader], len(times),
            stop_delay, stop_delay, update_offsets=times,
        )
    origin, updates = field.now, field.updates
    projected = field.fork()
    projected._record_events = True
    first_event = len(projected.events)
    identifier = field.identifier
    leader = min(field.order,
                 key=lambda key: -field.timeline.states[key].completed_laps)
    own_pace = field.free_paces[identifier]
    starts, counts = [], []
    distance = field.timeline._clock.scheduled_laps
    remaining = distance - field.timeline.states[identifier].completed_laps
    for index in range(max(horizon, remaining)):
        cancellation_checkpoint()
        if projected.finished and index >= horizon:
            break
        if index < horizon:
            starts.append(projected.now - origin)
            counts.append(projected.updates - updates)
        if projected.finished:
            # The suffix retains its existing fixed own-lap horizon. Weather
            # is frozen after the conditional flag, including padded entries.
            projected.now += own_pace
        else:
            projected.enter()
            projected.cross(own_pace)
    if leader == identifier:
        return tuple(counts), None
    times = tuple(event.time - origin for event in projected.events[first_event:]
                  if event.leading and event.flag_time is None)
    pace = field.free_paces[leader]
    clock = StrategyWeatherClock(
        tuple(starts), times[0] if times else 0., pace, len(times), stop_delay, stop_delay,
        update_offsets=times,
    )
    return None, clock


def _green_weather_clock_key(clock, warmup):
    """Collapse equivalent short native suffixes by every reachable update.

    With no fitting delays, a suffix has at most one paid visit per own lap.
    Green planners only observe these update counts and the total update cap.
    Preserve the actual clock for long horizons and all other clock shapes.
    """
    if (clock is None or warmup or len(clock.lap_start_offsets) > 12
            or clock.update_offsets is None or clock.current_running_times is not None
            or clock.current_stop_delay != clock.future_stop_delay):
        return clock
    tolerance = clock.update_interval * 1.e-12
    counts = tuple(tuple(
        0 if offset == paid == 0 else bisect_right(
            clock.update_offsets, start + paid * clock.future_stop_delay + tolerance)
        for paid in range(offset + 2)
    ) for offset, start in enumerate(clock.lap_start_offsets))
    return clock.max_updates, counts


@control_lap_scope
def plan_controlled_weather(
    driver, car, track, weather, current_tire, tire_age, current_lap, remaining_stops,
    *, control_context, physical_total_laps, used_compounds=(), remaining_dry_stops=None,
    remaining_damp_stops=None, tire_warmup=None, current_fit_pending=False,
    forecast_context=None, inventory=None, force_stop=False, require_compound_rule=True,
    same_compound=False, retained_weather_bound=False, traffic_possible=True,
):
    """Price the known field prefix, preserving the existing green policy.

    Rivals hold their observed free pace or a known paid outlap's mean pace
    and make no future decisions. The
    candidate's own entries, surfaces, fitting fees and crossings remain
    branch-dependent. Removed physical sets retain their completed wear.
    """
    if type(control_context) is not StrategyControlContext or control_context.paid_fit:
        raise ValueError("control_context must describe a pending weather decision")
    horizon = track.total_laps - current_lap + 1
    root = control_context.new_field()
    if ((type(root) is ObservedStandardField
         and (root.lap != current_lap or root.now != control_context.now))
            or (type(root) is not ObservedStandardField and (
                root.timeline.states[root.identifier].completed_laps + 1 != current_lap
                or root.timeline._clock.scheduled_laps != physical_total_laps))):
        raise ValueError("control_context must match the current lap and physical distance")
    driver, car, track = (model.model_copy(deep=True) for model in (driver, car, track))
    simulator = LapSimulator(np.random.default_rng(0))
    native = native_physics(driver, car, track, weather, current_tire)
    if type(root) is not ObservedStandardField:
        native = native and native_finish_timeline(root.timeline)
    if native:
        driver.reset_race_state()
        if type(root) is not ObservedStandardField:
            root._native_crossings = True
            # Costs consume the ledger and cumulative updates. The compact
            # green clock needs no history of earlier follower crossings.
            root._record_events = False
    prepared = (simulator.prepare_deterministic_lap_time(driver, car, track, physical_total_laps)
                if native else None)
    service = expected_stationary_time(car)
    bits = {compound.value: 1 << index if index < 3 else 8
            for index, compound in enumerate(TireCompound)}
    used = 0
    for compound in used_compounds:
        used |= bits[TireCompound(compound).value]
    surfaces = [weather.model_copy(deep=True)]
    memo, lap_costs, suffixes = {}, {}, {}
    invalid = ProjectedControlCost(-1, inf)
    physical = inventory is not None
    retired = ProjectedControlCost(0, 0., False) if physical else invalid
    current_expiry = -1
    if physical and inventory.current_set_id in inventory.sets:
        current_set = inventory.sets[inventory.current_set_id]
        if current_set.remaining_laps is not None:
            inventory.current_remaining_laps(tire_age)
        current_expiry = tire_set_slot(current_set)[2]
    current_usable = (not physical or inventory.current_set_id in inventory.sets
                      and inventory.current_set_id not in inventory.unavailable_ids
                      and tire_slot_usable(tire_age, current_expiry))
    stock = (tuple(inventory.replacements()) if physical else None)
    pool = (tuple(sorted(tire_set_slot(item) for item in stock))
            if physical else None)

    def surface(count):
        while len(surfaces) <= count:
            cancellation_checkpoint()
            surfaces.append(project_next_surface(surfaces[-1], forecast_context,
                                                 len(surfaces) - 1))
        return surfaces[count]

    if native and physical:
        original_ages = tuple((c, age) for c, age, _ in pool) + (
                               ((current_tire.compound.value, tire_age),)
                                if current_usable else ())

        def stock_bound():
            from f1sim.simulation.inventory_strategy import _conserved_wear_lower_bounds

            # Ignore already-used stock slots, all stops, weather chronology
            # and compulsory compound changes. Every real suffix still uses
            # each original physical set/age slot at most once, so this common
            # relaxation stays optimistic after any prefix history.
            if type(root) is ObservedStandardField:
                count = horizon
            else:
                count = physical_total_laps - root.active_distance
            floor = minimum_lap_time(track)
            if count > 1000:
                # Very large custom schedules need not materialize a weather
                # path merely to obtain an optimistic pruning bound.
                lower = [0.] * (horizon + 1)
                for index in range(horizon - 1, -1, -1):
                    cancellation_checkpoint()
                    lower[index] = nextafter(floor + lower[index + 1], -inf)
                return tuple(lower)
            possible = [surface(index) for index in range(count + 1)]
            scales = [simulator.weather_pace_multiplier(driver, car, value) for value in possible]
            easiest = min(range(len(possible)), key=scales.__getitem__)
            base = track.base_lap_time
            car_delta = car.pace_delta_seconds(base) + simulator._track_car_delta(car, track, base)
            fixed = (base + car_delta) + (1. - driver.skill_rating) * base * .03
            minimum_tire = min((simulator.tire_pace_contribution(
                driver, car, track, TIRE_COMPOUNDS[TireCompound(c)], age + elapsed)
                for c, age in set(original_ages) for elapsed in range(horizon + 1)), default=inf)
            aero = simulator._active_aero_gain(car, track)
            positive = nextafter(fixed + nextafter(minimum_tire - aero, -inf), -inf) >= 0.
            penalties = {c: tuple(simulator._tire_weather_mismatch(
                TIRE_COMPOUNDS[TireCompound(c)], value) for value in possible)
                for c, _ in original_ages}

            @lru_cache(maxsize=None)
            def lower_running(offset, compound, age):
                if not positive:
                    return floor
                mean = prepared(TIRE_COMPOUNDS[TireCompound(compound)], possible[easiest],
                                current_lap + offset, age, None, True)
                # Native raw running is nonnegative. Independently minimize
                # weather scaling and mismatch; floor clipping cannot raise
                # this relaxation above an actual lap. Round subtraction and
                # addition downward before applying the same absolute floor.
                row = penalties[compound]
                return max(floor, nextafter(nextafter(nextafter(mean, -inf)
                                            - row[easiest], -inf) + min(row), -inf))

            critical = {compound.value: (False,) * horizon for compound in TireCompound}
            return _conserved_wear_lower_bounds(horizon, original_ages, critical, lower_running)

        install_control_wear_bound(
            driver, car, track, physical_total_laps, current_lap, stock_bound)

    def legal(mask):
        return not require_compound_rule or bool(mask & 8) or (mask & 7).bit_count() >= 2

    def reduced(value):
        return None if value is None else max(0, value - 1)

    def completed(mask, compound):
        result = mask | bits[compound]
        return 8 if native and result & 8 else result

    def state_key(state):
        if physical and not same_compound and not retained_weather_bound:
            from f1sim.simulation.inventory_strategy import _expired_inventory_state

            state = _expired_inventory_state(state)
        offset, compound, age, available, left, dry, damp, mask, retained, expiry = state
        room = horizon - offset
        # Only one fit can precede each remaining own lap. Surplus allowances
        # and already satisfied rule histories cannot change any suffix edge.
        credit = (0 if not require_compound_rule else
                  8 if mask & 8 else 7 if legal(mask) else mask)
        return (offset, compound, age, available, min(left, room),
                None if dry is None else min(dry, room),
                None if damp is None else min(damp, room),
                credit, retained, expiry)

    def allowed(state, before, target):
        _, compound, age, _, left, dry, damp, mask, _, expiry = state
        if not tire_slot_usable(age, expiry):
            return True
        if same_compound:
            return left > 0
        if before.tire_mismatch(TireCompound(compound)) == "critical":
            return True
        limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
        return (left > 0 and (compound in ("intermediate", "wet")
                             or before.track_wetness > .3 or limit is None or limit > 0)
                or not legal(mask) and not mask & bits[target])

    def choices(state, before, retention):
        _, compound, _, available, _, _, _, _, _, _ = state
        if retention and before.tire_mismatch(TireCompound(compound)) != "critical":
            return ()
        candidates = ((TireCompound(compound),) if same_compound else
                      paid_compound_candidates(before, forecast_context))
        if retention:
            required = before.fresh_rain_compound()
            candidates = ((required,) if required is not None else tuple(
                value for value in candidates if value.value in ("soft", "medium", "hard")))
        if physical:
            previous, result = None, []
            for index, item in enumerate(available):
                if (item != previous and tire_slot_usable(item[1], item[2])
                        and TireCompound(item[0]) in candidates):
                    result.append((item[0], item[1], index, item[2]))
                previous = item
            return result
        return tuple((value.value, 0, None, -1) for value in candidates)

    def running(field, state):
        offset, compound, age, _, _, _, _, _, retained, _expiry = state
        before = surface(field.updates)
        gap = field.gap_ahead(1. if type(field) is ObservedStandardField else
                              field.free_paces[field.identifier] * field.running_modifier)
        aero = not field.controlled
        key = offset, compound, age, retained, field.updates, gap, aero
        if native and key in lap_costs:
            return lap_costs[key]
        tire = current_tire if retained else TIRE_COMPOUNDS[TireCompound(compound)]
        if prepared is not None:
            value = prepared(tire, before, current_lap + offset, age, gap, aero)
        else:
            value = isolated_strategy_lap(
                simulator, driver, car, track, tire, before,
                current_lap + offset, physical_total_laps, tire_age=age,
                active_aero_enabled=aero, gap_to_car_ahead=gap)
        if native:
            lap_costs[key] = value
        return value

    def green(field, state, retention):
        from f1sim.simulation.inventory_strategy import plan_inventory_strategy
        from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
        from f1sim.simulation.tire_inventory import TireInventory
        from f1sim.simulation.weather_strategy import weather_stop_costs

        offset, compound, age, available, left, dry, damp, mask, retained, expiry = state
        before = surface(field.updates)
        if (native and forecast_context is None
                and before.track_wetness == before.rain_intensity):
            # An external clock cannot alter a native constant surface.
            intervals, clock = None, None
        else:
            intervals, clock = green_weather_forecast(field, horizon - offset,
                                                     track.pit_lane_delta + service,
                                                     native=native)
        context = None if forecast_context is None else forecast_context.advanced(field.updates)
        key = (state_key(state) if native else state, field.updates, intervals,
               _green_weather_clock_key(clock, tire_warmup) if native else clock, retention)
        if native and key in suffixes:
            return suffixes[key]
        used_compounds = {value for value in TireCompound if mask & bits[value.value]}
        tire = current_tire if retained else TIRE_COMPOUNDS[TireCompound(compound)]
        common = dict(physical_total_laps=physical_total_laps, weather_intervals=intervals,
                      weather_clock=clock, tire_warmup=tire_warmup, forecast_context=context)
        if physical:
            from f1sim.simulation.tire_inventory import TireSet

            pool = TireInventory([
                TireSet("current", TireCompound(compound), age,
                        None if expiry < 0 else expiry - age),
                *(TireSet(f"future-{index}", TireCompound(value), wear,
                          None if end < 0 else end - wear)
                  for index, (value, wear, end) in enumerate(available)),
            ])
            # The last accepted crossing may have exhausted the active set.
            # Keep that physical state so the suffix must replace it.
            pool.current_set_id = "current"
            result = plan_inventory_strategy(
                driver, car, track, before, pool, current_lap + offset,
                tire_age=age, remaining_stops=left, remaining_dry_stops=dry,
                remaining_damp_stops=damp, used_compounds=used_compounds,
                require_compound_rule=require_compound_rule, **common)
            # No admissible next action ends this suffix before another lap;
            # it must not invalidate crossings already accepted under control.
            result = max((result.continuation(True), result.continuation(False), retired),
                         key=lambda value: value.rank)
        elif retention:
            rivals = (len(field.rows) > 1 if type(field) is ObservedStandardField
                      else len(field.free_paces) > 1)
            value = weather_stop_costs(driver, car, track, before, tire, age,
                                       current_lap + offset,
                                       traffic_possible=traffic_possible and rivals,
                                       **common).stay_cost
        elif same_compound:
            result = plan_rain_stop(driver, car, track, before, tire, age,
                                    current_lap + offset, left, **common)
            value = min(result.pit_now_cost, result.wait_cost)
        else:
            result = plan_rain_transition(
                driver, car, track, before, tire, age, current_lap + offset, left,
                remaining_dry_stops=dry, remaining_damp_stops=damp,
                used_compounds=used_compounds if require_compound_rule else {TireCompound.WET},
                **common)
            value = min(result.pit_now_cost, result.wait_cost)
        if not physical:
            result = (ProjectedControlCost(horizon - offset, value)
                      if isfinite(value) else invalid)
        if native:
            suffixes[key] = result
        return result

    def action(field, state, fitted=False, first=False, retention=False):
        cancellation_checkpoint()
        offset, compound, age, available, left, dry, damp, mask, retained, expiry = state
        branch = field.fork()
        factor = .55 if field.controlled and field.safety_car else .75 if field.controlled else 1.
        delay = (control_context.current_stop_delay if first else
                 track.pit_lane_delta * factor + service)
        branch.enter(delay if fitted else None)
        fee = (tire_warmup_seconds(tire_warmup, compound) if tire_warmup
               and (fitted or first and current_fit_pending) else 0.)
        branch.cross(running(branch, state), fee)
        if physical and (offset + 1 == horizon or branch.finished) \
                and not legal(completed(mask, compound)):
            return retired
        child = (offset + 1, compound, age + 1, available, max(0, left - int(fitted)),
                 reduced(dry) if fitted else dry, reduced(damp) if fitted else damp,
                 completed(mask, compound), retained, expiry)
        suffix = future(branch, child, retention)
        return suffix.prepend_lap(branch.now - field.now)

    def future(field, state, retention):
        cancellation_checkpoint()
        offset, compound, age, available, _, _, _, mask, _, expiry = state
        if offset == horizon or field.finished:
            return ProjectedControlCost(0, 0.) if legal(mask) else retired
        if not field.projection_required and (not retention or
                surface(field.updates).tire_mismatch(TireCompound(compound)) != "critical"):
            return green(field, state, retention)
        key = observed_control_key(field), state_key(state) if native else state, retention
        if native and key in memo:
            return memo[key]
        before = surface(field.updates)
        best = (action(field, state, retention=retention)
                if tire_slot_usable(age, expiry) and (same_compound or
                    before.tire_mismatch(TireCompound(compound)) != "critical")
                else retired)
        for target, wear, index, target_expiry in choices(state, before, retention):
            if not retention and not allowed(state, before, target):
                continue
            replacement_pool = (exchange_tire_slots(available, index, (compound, age, expiry))
                                if physical else None)
            replacement = (offset, target, wear, replacement_pool,
                           *state[4:8], False, target_expiry)
            option = action(field, replacement, fitted=True, retention=retention)
            if option.rank > best.rank:
                best = option
        if native:
            memo[key] = best
        return best

    initial = (0, current_tire.compound.value, tire_age, pool, remaining_stops,
               remaining_dry_stops, remaining_damp_stops, used, True, current_expiry)
    before = surface(0)
    wait = invalid
    if current_usable and not force_stop and (same_compound or
            before.tire_mismatch(current_tire.compound) != "critical"):
        wait = action(root, initial, first=True, retention=retained_weather_bound)
    best, selected, set_id = invalid, None, None
    first_choices = (tuple((item.compound.value, item.age, item.id, tire_set_slot(item)[2])
                          for item in stock)
                     if physical else choices(initial, before, False))
    for target, age, identity, expiry in first_choices:
        cancellation_checkpoint()
        if (not same_compound and TireCompound(target) not in
                paid_compound_candidates(before, forecast_context)):
            continue
        if not force_stop and current_usable and not allowed(initial, before, target):
            continue
        available = None
        if physical:
            index = pool.index((target, age, expiry))
            available = pool[:index] + pool[index + 1:]
            if current_usable:
                available = tuple(sorted(available + ((current_tire.compound.value, tire_age,
                                                      current_expiry),)))
        state = (0, target, age, available, remaining_stops,
                 remaining_dry_stops, remaining_damp_stops, used, False, expiry)
        option = action(root, state, fitted=True, first=True)
        if option.laps > 0 and option.rank > best.rank:
            best, selected, set_id = option, TireCompound(target), identity if physical else None
    return ControlledWeatherDecision(best, wait, selected, set_id)


register_forecast_helpers(globals(), (
    "ControlledWeatherDecision", "green_weather_forecast", "_green_weather_crossings",
    "_green_weather_clock_key",
    "plan_controlled_weather",
    "usable_weather_control",
    "native_physics", "project_next_surface", "paid_compound_candidates",
    "ObservedStandardField", "ProjectedControlCost", "StrategyControlContext",
    "observed_control_key", "StrategyWeatherClock",
    "isolated_strategy_lap", "control_lap_scope",
    "install_control_wear_bound", "minimum_lap_time",
    "native_finish_timeline",
    "exchange_tire_slots", "tire_set_slot", "tire_slot_usable",
))
