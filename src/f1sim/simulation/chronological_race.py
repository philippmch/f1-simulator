"""Chronological race execution with individual car timing and lapped finishes.

Each car owns one pending lap. Whole-lap physics freezes weather/control when
running starts, after any paid service; red flags collect the field before a
shared restart. There is no mid-lap sector redistribution. Circular crossing order
requires a sampled pass or compliant blue-flag yield before a faster car can
cross a physical predecessor.
"""

import heapq
from copy import deepcopy
from dataclasses import dataclass, field, replace
from itertools import count
from math import ceil, floor, isfinite
from numbers import Real

from f1sim.cancellation import raise_if_cancelled
from f1sim.models._native import native_physics, register_forecast_helpers
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import (
    ChronologicalFinishCar,
    ChronologicalFinishContext,
    CommittedRivalFit,
    ObservedChronologicalField,
    evaluate_chronological_finish_protection,
    project_observed_chronological_clock,
)
from f1sim.simulation.custom_pit_strategy import CustomPitFinishContext
from f1sim.simulation.events import EventManager, EventType, RaceEvent
from f1sim.simulation.execution import validate_starting_tire_ages, validate_starting_tires
from f1sim.simulation.finish_strategy import (
    LeadingFinishContext,
    RivalFinishForecast,
    evaluate_finish_protection,
)
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.neutralization import safety_car_running_time
from f1sim.simulation.pit_plans import (
    finalize_pit_plan,
    initialize_pit_plan_state,
    override_pit_plan_instruction,
    pit_plan_may_stop,
    validate_pit_plans,
)
from f1sim.simulation.pit_service import expected_remaining_service
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceResult
from f1sim.simulation.race_points import points_for_classification
from f1sim.simulation.race_timing import (
    RaceFinishTimeline,
    forecast_final_lap,
    forecast_running_duration,
)
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.strategy_neutralization import (
    SafetyCarBranch,
    StrategySafetyCarSnapshot,
    observed_control_intervals,
)
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import projected_surfaces
from f1sim.simulation.tire_inventory import validate_tire_inventory
from f1sim.simulation.validation import validate_unique_ids
from f1sim.simulation.weather_schedule import (
    WeatherForecastContext,
    validate_weather_schedule,
)

_GREEN_CONTROL_METHODS = (
    ("get_lap_time_modifier", EventManager.get_lap_time_modifier),
    ("is_active_aero_allowed", EventManager.is_active_aero_allowed),
)


@dataclass
class _PendingLap:
    lap: int
    start: float
    ready: float
    weather: object
    neutralized: bool
    tire: object
    tire_age: int
    running: float
    generation: int = 0
    mechanical_checked: bool = False
    attempted: set[str] = field(default_factory=set)
    running_start: float | None = None
    on_track: bool = True
    paid_stop: bool = False
    lap_time_modifier: float = 1.0
    active_aero_enabled: bool = True
    mode_allowed: bool = False
    mode_active: bool = False
    restart_boost: bool = False
    detected_gap: float | None = None
    safety_car: bool = False
    sc_queue_pace: float | None = None
    expected_exit: float | None = None


@dataclass
class _PitServiceRecord:
    """One committed stop's observable service lifecycle.

    The sampled start/end timestamps are retained for phase detection only.
    Forecasts deliberately ignore a timestamp that lies in the future and
    replace it with the conditional remaining-service expectation.
    """

    driver_id: str
    team_id: str
    lap: int
    arrival_time: float
    service_start: float
    service_end: float
    car: object
    committed_lane_loss: float


class ChronologicalRace:
    """Race engine with absolute crossings, stops, and per-car flags."""

    def __init__(self, simulator, *, red_flag_pause_seconds=600.0):
        if (isinstance(red_flag_pause_seconds, bool)
                or not isinstance(red_flag_pause_seconds, Real)
                or not isfinite(red_flag_pause_seconds) or red_flag_pause_seconds < 0):
            raise ValueError("red_flag_pause_seconds must be finite and nonnegative")
        self.simulator = simulator
        self.red_flag_pause_seconds = float(red_flag_pause_seconds)
        self.crossings: list[tuple[str, int, float]] = []
        self.pit_exits: list[tuple[str, int, float]] = []
        self.suspensions: list[tuple[float, float, tuple[str, ...]]] = []
        self.order: list[str] = []
        self._overtake_restart_waiting: set[str] = set()

    def run(self, drivers, cars, track, weather, starting_grid, *, starting_tires=None,
            starting_tire_ages=None, tire_inventory=None, pit_plans=None, weather_schedule=None):
        schedule = validate_weather_schedule(weather_schedule, total_laps=track.total_laps)
        driver_ids = tuple(driver.id for driver in drivers)
        normalized_pit_plans = validate_pit_plans(
            pit_plans,
            driver_ids=driver_ids,
            total_laps=track.total_laps,
            tire_inventory=tire_inventory,
        )
        validate_unique_ids(driver_ids, "driver")
        validate_unique_ids(starting_grid, "starting grid")
        starting_tires = validate_starting_tires(starting_tires, (d.id for d in drivers))
        ages = validate_starting_tire_ages(starting_tire_ages, starting_tires,
                                           (d.id for d in drivers))
        inventories = validate_tire_inventory(tire_inventory, starting_tires, ages,
                                              (d.id for d in drivers))
        self.simulator.weather_forecast_context = (
            WeatherForecastContext.from_schedule(schedule) if schedule else None)
        self.track = track
        self.weather = weather.model_copy(deep=True)
        self.simulator.event_manager.reset()
        self.simulator.weather_history = []
        for driver in drivers:
            driver.reset_race_state()
        lookup = {driver.id: driver for driver in drivers}
        self.states = {}
        for driver_id in starting_grid:
            driver = lookup.get(driver_id)
            if driver is None or driver.team_id not in cars:
                continue
            car = cars[driver.team_id]
            style = self.simulator._infer_team_strategy(car, track)
            pit_plan = normalized_pit_plans.get(driver_id)
            inventory = selected_set = None
            if driver_id in inventories:
                inventory, selected_set = self.simulator._inventory_opening_set(
                    driver, car, track, self.weather, style, inventories[driver_id],
                    starting_tires.get(driver_id), ages.get(driver_id, 0),
                    **({"pit_plan": pit_plan} if pit_plan is not None else {}),
                )
                compound = selected_set.compound
            elif driver_id in starting_tires:
                compound = starting_tires[driver_id]
            else:
                compound = self.simulator._choose_starting_compound(
                    style, track, self.weather, driver, car,
                    **({"pit_plan": pit_plan} if pit_plan is not None else {}),
                )
            driver.current_tire_laps = ages.get(driver_id, 0)
            plans = self.simulator._plan_pit_lap_options(style, track)
            self.states[driver_id] = DriverRaceState(
                driver, car, len(self.states) + 1,
                current_tire=TIRE_COMPOUNDS[compound].model_copy(deep=True),
                tire_laps=ages.get(driver_id, 0), prior_tire_laps=ages.get(driver_id, 0),
                strategy_archetype=style, planned_pit_laps=plans[0], pit_plan_options=plans,
            )
            initialize_pit_plan_state(self.states[driver_id], pit_plan)
            if inventory is not None:
                self.simulator._initialize_inventory(
                    self.states[driver_id], inventory, selected_set,
                )
        self.timeline = RaceFinishTimeline(track.total_laps, self.states)
        self.order = list(self.states)
        self.pending = {}
        self.queue = []
        self.serial = count()
        self.box_releases = {}
        self.expected_box_releases = {}
        self.pit_service_records: list[_PitServiceRecord] = []
        self.fastest = {}
        self.running_paces = {}
        self.free_refits = set()
        self.regrouping = False
        self.red_waiting = set()
        self.resumption_order = []
        self.red_flag_start = None
        self.leader_id = None
        self.incidents = 0
        self.control_intervals = 0
        self._overtake_restart_waiting.clear()
        self.green_streak = 0
        self.has_two_green = False
        self.crossings.clear()
        self.pit_exits.clear()
        self.suspensions.clear()
        if self.states:
            self.simulator._record_weather(1, self.weather)
        for state in self.states.values():
            self._start_lap(state, 0.0)
        while self.queue:
            raise_if_cancelled()
            now, _, _, kind, driver_id, generation = heapq.heappop(self.queue)
            pending = self.pending.get(driver_id)
            if pending is None or generation != pending.generation:
                continue
            if kind == "exit":
                if self.regrouping:
                    self.red_waiting.add(driver_id)  # Pit exit remains closed.
                else:
                    self.order.append(driver_id)  # Pit exit is immediately after the line.
                    self.pit_exits.append((driver_id, pending.lap, now))
                    self._begin_running(self.states[driver_id], pending, now)
                    self._enqueue(driver_id, pending.ready, "cross")
            elif self._resolve_crossing(driver_id, now):
                state = self.states[driver_id]
                if self.timeline.can_start_next_lap(driver_id):
                    if self.regrouping:
                        self.red_waiting.add(driver_id)
                        self.order.remove(driver_id)
                    else:
                        self._start_lap(state, now)
            if self.regrouping:
                self._resume_if_collected(now)
        return self._results()

    def _fit_red_flag_set(
        self, state, planning, weather_intervals=None,
        weather_clock: StrategyWeatherClock | None = None,
    ):
        if state.tire_inventory is not None:
            self.simulator._refit_inventory_free(
                state, planning, self.weather, state.laps_completed,
                physical_total_laps=self.track.total_laps, weather_intervals=weather_intervals,
                weather_clock=weather_clock,
            )
            self.free_refits.remove(state.driver.id)
            return
        compound = self.simulator._choose_red_flag_tire(
            state, self.weather, planning, state.laps_completed,
            physical_total_laps=self.track.total_laps,
            **({"weather_intervals": weather_intervals} if weather_intervals is not None else {}),
            **({"weather_clock": weather_clock} if weather_clock is not None else {}),
        )
        self.simulator._fit_tire(state, compound)
        state.force_pit_next_lap = False
        state.pit_decision_context = None
        state.dry_pit_proposal = None
        state.weather_pit_proposal = None
        self.free_refits.remove(state.driver.id)

    def _red_flag_order(self):
        """Freeze known on-track order without undoing a completed passing move.

        Cars in service retain their last recorded rank slots; among on-track
        cars, completed distance and circular order supersede old line clocks.
        """
        ranked = sorted((state for state in self.states.values()
                         if state.status == DriverStatus.RACING), key=lambda state: state.position)
        physical = {driver_id: index for index, driver_id in enumerate(self.order)}
        on_track = iter(sorted(
            (state for state in ranked if state.driver.id in physical),
            key=lambda state: (-state.laps_completed, physical[state.driver.id]),
        ))
        return [next(on_track).driver.id if state.driver.id in physical else state.driver.id
                for state in ranked]

    def _resume_if_collected(self, now):
        active = {key for key, state in self.states.items() if state.status == DriverStatus.RACING}
        if not active <= self.red_waiting:
            return
        resume = now + self.red_flag_pause_seconds
        self.timeline.end_suspension(resume)
        self.simulator.event_manager.end_red_flag()
        self.regrouping = False
        self.order = [key for key in self.resumption_order if key in active]
        self.suspensions.append((self.red_flag_start, resume, tuple(self.order)))
        self.red_waiting.clear()
        # Collection waits for all paid services; free restart fits reserve no box.
        self.expected_box_releases.clear()
        self.pit_service_records.clear()
        # Freeze all restart forecasts before fitting or releasing any car.
        # Collected services are finished; their old expected exits and the
        # order in which new running is sampled cannot anchor these forecasts.
        restart_plans = {}
        forecast_leader = self._forecast_leader()
        for driver_id in self.order:
            state = self.states[driver_id]
            state.strategy_finish_context = (
                CustomPitFinishContext(resume, self.timeline.time_limit_seconds,
                                       self.timeline.time_limit_announced)
                if state.pit_plan is not None and state is forecast_leader else None
            )
            planning = self._planning_track(state, resume, restart=True)
            cadence = self._weather_intervals(state, resume, planning, restart=True)
            restart_plans[driver_id] = (
                planning, cadence,
                (self._strategy_weather_clock(state, resume, planning, 0.0, restart=True)
                 if cadence is not None else None),
            )
        for driver_id in list(self.order):
            state = self.states[driver_id]
            # Suspension fits belong to the collected field. Complete them all
            # before a new paid stop changes its leader or running consumes RNG.
            self._fit_red_flag_set(state, *restart_plans[driver_id])
            if state.status != DriverStatus.RACING:
                self._retire(driver_id, resume, state.dnf_reason)
        leading_context = (self._leading_finish_context(forecast_leader, resume, restart=True)
                           if forecast_leader is not None
                           and forecast_leader.status == DriverStatus.RACING else None)
        for state in self.states.values():
            state.strategy_leading_finish_context = (
                leading_context if state is forecast_leader else None
            )
        for driver_id in list(self.order):
            state = self.states[driver_id]
            pending = self.pending.get(driver_id)
            if pending is None:
                self._start_lap(state, resume, restart_planning=restart_plans.get(driver_id))
                continue
            # This paid stop completed service while the exit was closed. Its
            # running has never been sampled. Fit the shared restart set and
            # release the existing lap without charging/sampling another stop.
            pending.tire = state.current_tire.model_copy(deep=True)
            pending.tire_age = state.tire_laps
            pending.generation += 1
            self.pit_exits.append((driver_id, pending.lap, resume))
            self._begin_running(state, pending, resume)
            self._enqueue(driver_id, pending.ready, "cross")

    def _enqueue(self, driver_id, time, kind):
        # At an exact-time crossing tie the leading distance receives the
        # flag first. Serial order alone can start a lapped car's extra lap.
        heapq.heappush(self.queue, (time, -self.pending[driver_id].lap,
                                   next(self.serial), kind, driver_id,
                                   self.pending[driver_id].generation))

    def _planning_track(self, state, now, *, restart=False, _observed=...):
        """Forecast an own-lap finish horizon without altering the actual flag.

        Use observed free running pace (excluding stops, incidents and blocking),
        through the observed remaining control intervals. The leading
        pending lap's absolute readiness anchors the forecast. Same-distance
        cars still on track precede cars in service; a lap-ahead pitter remains
        ahead. Unfinished pit service uses its expected exit, never its future
        sampled completion. The deadline
        includes elapsed suspension time; running after those intervals is green.
        Future weather, interruptions and elective stops are unknown.
        """
        horizon = self.timeline.final_lap
        observed = (_observed if _observed is not ... else
                    None if restart else self._observed_control_projection(now))
        if observed is not None and observed.identifier == state.driver.id:
            horizon = min(horizon, state.laps_completed + len(observed.own_crossings))
            return self.track.model_copy(update={"total_laps": horizon})
        own_pace = self.running_paces.get(state.driver.id)
        flag_time = self._projected_flag_time(now, restart=restart)
        if (own_pace is not None and isfinite(own_pace) and own_pace > 0
                and flag_time is not None and isfinite(flag_time)):
            modifier = self.simulator.event_manager.get_lap_time_modifier()
            intervals = self._neutralized_finish_intervals()
            if intervals is not None and intervals > 1:
                clock = self._weather_projection_clock(now, restart=restart)
                starts = self._projected_lap_starts(
                    own_pace, now, self.track.total_laps - state.laps_completed + 1, clock)
                remaining = next((offset for offset, time in enumerate(starts[1:], 1)
                                  if now + time >= flag_time - own_pace * 1.e-12),
                                 self.track.total_laps - state.laps_completed)
            else:
                remaining = max(1, 1 + ceil(
                    (flag_time - now - own_pace * modifier) / own_pace - 1e-12,
                ))
            horizon = min(self.timeline.final_lap, state.laps_completed + remaining)
        return self.track.model_copy(update={"total_laps": horizon})

    def _forecast_leader(self):
        """Select the observed active leader consistently across strategy forecasts."""
        active = [other for other in self.states.values() if other.status == DriverStatus.RACING]
        if active:
            physical_order = {driver_id: index for index, driver_id in enumerate(self.order)}
            return min(active, key=lambda other: (
                -other.laps_completed,
                other.driver.id not in physical_order,
                physical_order.get(other.driver.id, other.position),
            ))
        return None

    def _projected_flag_time(self, now, *, restart=False):
        """Expected leading finish crossing using the same horizon assumptions."""
        if self.timeline.chequered_time is not None:
            return self.timeline.chequered_time
        observed = None if restart else self._observed_control_projection(now)
        if observed is not None:
            return observed.flag_time
        leader = self._forecast_leader()
        if leader is not None:
            pending = None if restart else self.pending.get(leader.driver.id)
            leader_pace = self.running_paces.get(leader.driver.id)
            modifier = self.simulator.event_manager.get_lap_time_modifier()
            intervals = self._neutralized_finish_intervals()
            future_control = 1 if intervals is None else intervals
            if pending is not None:
                flag_time = pending.ready
                if leader_pace is None:
                    leader_pace = pending.running or None
                if not pending.on_track and leader_pace is not None:
                    expected_exit = self._pending_service_exit(leader.driver.id, pending, now)
                    flag_time = max(now, expected_exit or now)
                    flag_time += leader_pace * modifier
                    flag_time += self._pending_fit_cost(leader.driver.id)
                anchor_lap = pending.lap
                future_control = max(0, future_control - 1)
                next_modifier = modifier if future_control else 1.
            else:
                flag_time = now
                anchor_lap = leader.laps_completed
                next_modifier = modifier
            if leader_pace is not None:
                if self.timeline.time_limit_announced:
                    # A lapped successor can receive the flag below the old
                    # leader's announced lap number. Its next crossing wins.
                    laps_left = 0 if pending is not None else 1
                else:
                    resumed_after_suspension = (
                        pending is None and self.timeline.total_suspension_seconds > 0
                    )
                    crossing_time = leader.total_time if resumed_after_suspension else flag_time
                    projected_final = forecast_final_lap(
                        self.timeline.final_lap, anchor_lap, crossing_time, leader_pace,
                        self.timeline.time_limit_seconds, next_modifier,
                        **({"controlled_laps": future_control} if future_control > 1 else {}),
                        **({"next_lap_start_time": now}
                           if resumed_after_suspension else {}),
                        **({"time_limit_announced": False}
                           if pending is None and crossing_time >= self.timeline.time_limit_seconds
                           else {}),
                    )
                    laps_left = max(0, projected_final - anchor_lap)
                flag_time += forecast_running_duration(
                    leader_pace, laps_left, next_modifier, future_control)
                return flag_time
        return None

    def _pending_fit_cost(self, driver_id):
        """Read a committed fit's one-time cost without consuming live state."""
        state = self.states[driver_id]
        if not state.fit_lap_pending:
            return 0.0
        return self.simulator.tire_warmup.get(state.current_tire.compound.value, 0.0)

    @staticmethod
    def _remaining_service(car, elapsed):
        """Return the conditional mean service still outstanding."""
        return expected_remaining_service(car, elapsed)

    def _team_service_records(self, team_id, records=None):
        source = (getattr(self, "pit_service_records", ())
                  if records is None else records)
        return [record for record in source if record.team_id == team_id]

    def _team_release_forecast(self, team_id, now, *, records=None):
        """Estimate a team's box release using only service visible at ``now``."""
        supplied = records is not None
        team_records = [record for record in self._team_service_records(team_id, records)
                        if record.arrival_time <= now]
        if not team_records:
            if supplied:
                return now
            return max(now, getattr(self, "expected_box_releases", {}).get(team_id, now))
        release = now
        for record in team_records:
            if record.service_start <= now:
                if record.service_end <= now:
                    release = max(release, record.service_end)
                else:
                    elapsed = max(0.0, now - record.service_start)
                    release = max(release, now + self._remaining_service(
                        record.car, elapsed,
                    ))
            else:
                # The future sampled queue/start timestamp is private.  A
                # queued commitment consumes the ordinary expected service.
                release = max(release, record.arrival_time)
                release += expected_stationary_time(record.car)
        return release

    def _record_pit_service(self, state, lap, arrival_time, committed_lane_loss):
        """Capture an actual stop lifecycle after execution has sampled it."""
        details = state.pit_stop_details[-1] if state.pit_stop_details else None
        if not isinstance(details, dict):
            return
        try:
            queue_time = float(details["queue_time"])
            service_time = float(details["service_time"])
            queue_time = max(0.0, queue_time)
            service_time = max(0.0, service_time)
            service_start = float(arrival_time) + queue_time
            service_end = service_start + service_time
            lane_loss = float(committed_lane_loss)
        except (KeyError, TypeError, ValueError, OverflowError):
            return
        if any(not isfinite(value) for value in (
            arrival_time, service_start, service_end, lane_loss,
        )) or lane_loss < 0:
            return
        self.pit_service_records.append(_PitServiceRecord(
            state.driver.id,
            state.car.team_id,
            lap,
            float(arrival_time),
            service_start,
            service_end,
            state.car.model_copy(deep=True),
            lane_loss,
        ))

    def _pending_service_exit(self, driver_id, pending, now):
        """Forecast one pending paid lap's track entry from observable service."""
        if (isinstance(now, bool) or not isinstance(now, Real)
                or not isfinite(now)):
            return pending.expected_exit
        all_records = list(getattr(self, "pit_service_records", ()))
        target_indexes = [index for index, record in enumerate(all_records)
                          if record.driver_id == driver_id and record.lap == pending.lap]
        if not target_indexes:
            return pending.expected_exit
        target_index = target_indexes[-1]
        record = all_records[target_index]
        if record.arrival_time > now:
            return pending.expected_exit
        if record.service_start <= now:
            if record.service_end <= now:
                completion = record.service_end
            else:
                elapsed = max(0.0, now - record.service_start)
                completion = now + self._remaining_service(record.car, elapsed)
        else:
            # Lifecycle records are appended in reservation order.  Use the
            # ordered prefix before this stop so simultaneous team arrivals
            # retain FIFO order even when their sampled future timestamps do
            # not reveal that order.
            prior = [item for item in all_records[:target_index]
                     if item.team_id == record.team_id]
            release = self._team_release_forecast(
                record.team_id, now, records=prior,
            )
            completion = max(now, release) + expected_stationary_time(record.car)
        return completion + record.committed_lane_loss

    def _weather_projection_clock(self, now, *, restart=False, _observed=...):
        """Return the frozen leading update clock used by strategy forecasts.

        The tuple is ``(first_update_time, observed_leader_pace,
        available_updates)``.  Keeping this calculation shared means a finish
        forecast at an expected pit exit sees precisely the same persistent
        rainfall cadence as the existing own-lap ``weather_intervals`` path.
        """
        leader = self._forecast_leader()
        if leader is None or not isfinite(now):
            return None
        observed = (_observed if _observed is not ... else
                    None if restart else self._observed_control_projection(now))
        if observed is not None:
            first = (observed.leading_updates[0] if observed.leading_updates else
                     max(now, observed.flag_time))
            pace = self.running_paces.get(leader.driver.id)
            if pace is None and leader.driver.id in self.pending:
                pace = self.pending[leader.driver.id].running or None
            if pace is not None and isfinite(pace) and pace > 0:
                return first, pace, len(observed.leading_updates)
        pending = None if restart else self.pending.get(leader.driver.id)
        leader_pace = self.running_paces.get(leader.driver.id)
        if leader_pace is None and pending is not None:
            leader_pace = pending.running or None
        if (leader_pace is None or not isfinite(leader_pace) or leader_pace <= 0):
            return None
        modifier = self.simulator.event_manager.get_lap_time_modifier()
        if not isfinite(modifier) or modifier <= 0:
            return None
        if pending is None:
            first_update = now + leader_pace * modifier
        elif pending.on_track:
            first_update = max(now, pending.ready)
        else:
            expected_exit = self._pending_service_exit(leader.driver.id, pending, now)
            first_update = max(now, expected_exit or now)
            first_update += leader_pace * modifier
            first_update += self._pending_fit_cost(leader.driver.id)
        flag_time = self._projected_flag_time(now, restart=restart)
        if (flag_time is None or not isfinite(flag_time)
                or not isfinite(first_update)):
            return None
        # Include updates before an equal-time crossing, but never evolve the
        # surface at the chequered crossing itself.
        remaining = self._neutralized_finish_intervals()
        controlled = min(self.track.total_laps, max(0, (remaining or 1) - 1))
        controlled_span = controlled * leader_pace * modifier
        span = flag_time - first_update
        if span <= controlled_span:
            available = max(0, ceil(span / (leader_pace * modifier) - 1.e-12))
        else:
            available = controlled + max(0, ceil(
                (span - controlled_span) / leader_pace - 1.e-12))
        return first_update, leader_pace, available

    def _observed_control_projection(self, now):
        intervals = self._neutralized_finish_intervals()
        if (intervals is None or type(getattr(self, "pending", None)) is not dict
                or intervals <= 1 and not self._has_committed_rival_fit()):
            return None
        candidates = [state for key, state in self.states.items()
                      if state.status == DriverStatus.RACING and key not in self.pending]
        if len(candidates) != 1:
            return None
        state = candidates[0]
        context = self._chronological_finish_context(state, now)
        if context is None:
            return None
        return project_observed_chronological_clock(
            context, now, own_fitting_cost=self._pending_fit_cost(state.driver.id))

    def _weather_update_times(self, clock, *, now=None, _observed=...):
        """Explicit leading weather events; the flag itself is excluded."""
        first, pace, available = clock
        observed = (_observed if _observed is not ... else
                    self._observed_control_projection(now) if now is not None else None)
        if observed is not None:
            return observed.leading_updates
        remaining = self._neutralized_finish_intervals()
        future_control = min(self.track.total_laps, max(0, (remaining or 1) - 1))
        modifier = self.simulator.event_manager.get_lap_time_modifier()
        return tuple(first + forecast_running_duration(pace, offset, modifier, future_control)
                     for offset in range(available))

    def _projected_lap_starts(self, pace, now, horizon, clock, *, _observed=...):
        """Nominal own starts through observed leading control intervals."""
        modifier = self.simulator.event_manager.get_lap_time_modifier()
        remaining = self._neutralized_finish_intervals()
        if clock is None:
            return (0., *(pace * (modifier + offset - 1) for offset in range(1, horizon)))
        observed = (_observed if _observed is not ... else self._observed_control_projection(now))
        if observed is not None and len(observed.own_crossings) >= horizon - 1:
            return (0., *(time - now for time in observed.own_crossings[:horizon - 1]))
        if remaining is None or remaining <= 1:
            return (0., *(pace * (modifier + offset - 1) for offset in range(1, horizon)))
        first, leader_pace, _ = clock
        end = first + min(self.track.total_laps, remaining - 1) * leader_pace * modifier
        starts = [0.]
        for offset in range(1, horizon):
            entry = now + starts[-1]
            running_modifier = modifier if offset == 1 or entry < end - 1.e-10 else 1.
            starts.append(starts[-1] + pace * running_modifier)
        return tuple(starts)

    def _weather_updates_at(self, time, clock, *, now=None):
        first, pace, available = clock
        remaining = self._neutralized_finish_intervals()
        observed = self._observed_control_projection(now) if now is not None else None
        if observed is None and (remaining is None or remaining <= 1):
            return min(available, max(0, floor((time - first) / pace + 1.e-12) + 1))
        times = self._weather_update_times(clock, now=now, _observed=observed)
        return sum(value <= time + pace * 1.e-12 for value in times)

    def _weather_intervals(self, state, now, planning, *, restart=False, _observed=...):
        """Map projected leading weather updates onto future own-lap starts.

        Extrapolate observed free pace through the known remaining control
        intervals and then green running, as in the finish forecast. A leader
        already in service uses its expected exit. Later stops, incidents and
        pace changes are unknown. The winner's crossing produces no update.
        """
        own_pace = self.running_paces.get(state.driver.id)
        observed = (_observed if _observed is not ... else
                    None if restart else self._observed_control_projection(now))
        clock = self._weather_projection_clock(now, restart=restart, _observed=observed)
        if (own_pace is None or not isfinite(own_pace) or own_pace <= 0
                or clock is None):
            return None
        intervals = [0]
        starts = self._projected_lap_starts(
            own_pace, now, planning.total_laps - state.laps_completed, clock, _observed=observed)
        remaining = self._neutralized_finish_intervals()
        times = (self._weather_update_times(clock, now=now, _observed=observed)
                 if observed is not None or remaining is not None and remaining > 1 else None)
        for start in starts[1:]:
            if times is None:
                intervals.append(self._weather_updates_at(now + start, clock))
            else:
                intervals.append(sum(value <= now + start + clock[1] * 1.e-12 for value in times))
        return tuple(intervals)

    def _strategy_weather_clock(self, state, now, planning, queue_delay, *, restart=False,
                                _observed=...):
        """Build a paid-stop-aware clock only from an external observed leader."""
        leader = self._forecast_leader()
        if leader is None or leader.driver.id == state.driver.id:
            return None
        own_pace = self.running_paces.get(state.driver.id)
        observed = (_observed if _observed is not ... else
                    None if restart else self._observed_control_projection(now))
        projection = self._weather_projection_clock(now, restart=restart, _observed=observed)
        modifier = self.simulator.event_manager.get_lap_time_modifier()
        if (own_pace is None or not isfinite(own_pace) or own_pace <= 0
                or projection is None or not isfinite(modifier) or modifier <= 0):
            return None
        # A constant surface has no branch-dependent weather state to price.
        # Keeping the legacy interval/cache path here also avoids turning a
        # forced stop in steady dry or steady-rain conditions into a new
        # weather-transition branch.
        if (self.weather.track_wetness == self.weather.rain_intensity
                and not self.simulator._has_weather_schedule()):
            return None
        first_update, leader_pace, available = projection
        horizon = planning.total_laps - state.laps_completed
        if horizon < 1 or not isfinite(first_update - now):
            return None
        offsets = self._projected_lap_starts(
            own_pace, now, horizon, projection, _observed=observed)
        if any(not isfinite(value) for value in offsets):
            return None
        service = expected_stationary_time(state.car)
        current_delay = (
            service + self.track.pit_lane_delta * self.simulator._pit_lane_factor()
            + queue_delay
        )
        future_delay = service + self.track.pit_lane_delta
        if (not isfinite(current_delay) or current_delay < 0
                or not isfinite(future_delay) or future_delay < 0):
            return None
        try:
            # _start_lap has just frozen this decision's queue on the state.
            # Restart forecasts must not reuse an earlier lap's observation.
            safety_car = state.strategy_safety_car_snapshot
            running_times = (tuple(branch.running_time(own_pace, modifier)
                                   for branch in (safety_car.retained, safety_car.stopped))
                             if safety_car is not None and not restart
                             and self.simulator.event_manager.safety_car_active
                             and not self.simulator.event_manager.red_flag_active else None)
            if observed is not None and len(offsets) > 1:
                fee = self._pending_fit_cost(state.driver.id)
                offsets = (0., *(value - fee for value in offsets[1:]))
                if running_times is not None:
                    running_times = (offsets[1], running_times[1])
            return StrategyWeatherClock(
                tuple(offsets), first_update - now, leader_pace, available,
                current_delay, future_delay,
                current_running_times=running_times,
                update_offsets=(tuple(value - now for value in
                                      self._weather_update_times(
                                          projection, now=now, _observed=observed))
                                if observed is not None
                                or (self._neutralized_finish_intervals() or 0) > 1 else None),
            )
        except ValueError:
            return None

    def _projected_surface_at(self, absolute_time, *, now, restart=False):
        """Return a frozen persistent-rain surface at an absolute entry time."""
        clock = self._weather_projection_clock(now, restart=restart)
        if clock is None or not isfinite(absolute_time):
            return None
        updates = self._weather_updates_at(absolute_time, clock, now=now)
        return projected_surfaces(self.weather, 2, (0, updates),
                                  **self.simulator._forecast_options())[-1]

    @staticmethod
    def _clear_one_lap_pit_proposals(state):
        """Drop native planner proposals when a finish guard vetoes a stop."""
        state.dry_pit_proposal = None
        state.weather_pit_proposal = None
        state.inventory_pit_proposal = None
        state.pit_decision_context = None

    def _neutralized_finish_intervals(self):
        """Read known remaining control intervals without evolving race control."""
        return observed_control_intervals(self.simulator.event_manager)

    def _can_project_green_weather(self, state, weather, *, weather_clock=None):
        """A paid green lap can hand leading weather updates to another car."""
        if weather_clock is not None and not self._has_committed_rival_fit():
            return False
        active = [other for other in self.states.values() if other.status == DriverStatus.RACING]
        control = self.simulator.event_manager
        return (state.pit_plan is None and len(active) > 1
                and self._neutralized_finish_intervals() == 0
                and not (control.safety_car_active or control.vsc_active or control.red_flag_active)
                and all(getattr(getattr(control, name), "__func__", None) is method
                        for name, method in _GREEN_CONTROL_METHODS)
                and native_physics(state.driver, state.car, self.track, weather, state.current_tire)
                and type(self.simulator.lap_simulator) is LapSimulator
                and getattr(self.simulator.lap_simulator.calculate_lap_time, "__func__", None)
                is LapSimulator.calculate_lap_time
                and (self.simulator.weather_forecast_context is not None
                     or weather.project_surface().track_wetness != weather.track_wetness)
                and not any(other.pit_plan is not None
                            and other.pit_plan_index < len(other.pit_plan) for other in active))

    def _green_service_can_advance_weather(self, state, context, now, stop_delay):
        """Skip field search when no held rival can lead before paid entry."""
        entry = now + stop_delay
        for other in context.rivals:
            if other.committed_fit is not None:
                # Removing the candidate changes the fitted rival's entry
                # traffic even when its first crossing follows our pit exit.
                return True
            crossing = other.ready
            pace = other.free_running
            if other.running_start is None:
                crossing += pace + other.fitting_cost
            for _ in range(max(0, state.laps_completed - other.completed_laps)):
                crossing += pace
            if crossing <= entry + 1.e-10:
                return True
        return False

    def _field_finish_required(self, intervals):
        if intervals > 1:
            return True
        active = {key for key, row in self.states.items() if row.status == DriverStatus.RACING}
        return len(active) > 1 and (intervals > 0 or any(
            pending.neutralized for key, pending in getattr(self, "pending", {}).items()
            if key in active and pending.on_track) or self._has_committed_rival_fit())

    def _finish_protection_skip_reason(self, state, *, restart=False, now=None):
        """Return why the bounded elective-stop guard must remain inactive."""
        control = self.simulator.event_manager
        if state.force_pit_next_lap:
            return "forced stop"
        if control.red_flag_active:
            return "race suspended"
        intervals = self._neutralized_finish_intervals()
        if intervals is None:
            return "unavailable control duration"
        neutralized_field = self._field_finish_required(intervals)
        if now is None or not isfinite(now):
            return "invalid current time"
        if self.weather.tire_mismatch(state.current_tire.compound) == "critical":
            return "current tire is critical"
        inventory = state.tire_inventory
        if inventory is not None:
            if inventory.current_set_id in inventory.unavailable_ids:
                return "current inventory set unavailable"
            remaining = inventory.current_remaining_laps(state.tire_laps)
            if remaining is not None and remaining < self.track.total_laps - state.laps_completed:
                # The retained-distance guard assumes no further compulsory
                # service. Leave finite-lifetime paths to the inventory policy.
                return "current inventory set has usage limit"
            if not self.simulator._stay_satisfies_tire_rule(state):
                return "compound rule unresolved"
        elif not self.simulator._stay_satisfies_tire_rule(state):
            return "compound rule unresolved"
        if not neutralized_field and self._weather_projection_clock(now, restart=restart) is None:
            return "missing weather forecast"
        return None

    def _has_committed_rival_fit(self):
        return any(self._can_project_committed_fit(key, pending)
                   for key, pending in getattr(self, "pending", {}).items())

    def _can_project_committed_fit(self, key, pending):
        """Only a known native paid replacement can supply a new mean pace."""
        other = self.states[key]
        control = self.simulator.event_manager
        if (other.status != DriverStatus.RACING or pending.on_track or not pending.paid_stop
                or not hasattr(self, "weather")
                or type(self.simulator.lap_simulator) is not LapSimulator
                or getattr(self.simulator.lap_simulator.calculate_lap_time, "__func__", None)
                is not LapSimulator.calculate_lap_time
                or not all(getattr(getattr(control, name), "__func__", None) is method
                           for name, method in _GREEN_CONTROL_METHODS)
                or pending.tire_age != other.tire_laps
                or pending.tire != other.current_tire
                or not native_physics(other.driver, other.car, self.track, pending.tire,
                                      self.weather)):
            return False
        return True

    def _committed_rival_fit(self, key, pending):
        """Capture an unrun paid fit only when native entry physics is known."""
        if not self._can_project_committed_fit(key, pending):
            return None
        other = self.states[key]
        return CommittedRivalFit(
            other.driver.model_copy(deep=True), other.car.model_copy(deep=True),
            self.track.model_copy(deep=True), pending.tire.model_copy(deep=True),
            pending.tire_age, self.weather.model_copy(deep=True),
            self.simulator.weather_forecast_context,
        )

    def _strategy_projection_options(self, state, now):
        """Share one native snapshot while assembling this decision's views.

        This local value expires before the policy runs. Custom view/control
        dispatch and free refits retain their individual forecast calls.
        """
        control = self.simulator.event_manager
        if (type(self) is not ChronologicalRace or type(control) is not EventManager
                or type(self).__getattribute__ is not object.__getattribute__
                or type(control).__getattribute__ is not object.__getattribute__
                or getattr(type(self), "__getattr__", None) is not None
                or getattr(type(control), "__getattr__", None) is not None
                or state.pit_plan is not None or state.driver.id in self.free_refits
                or not all(getattr(getattr(self, name), "__func__", None) is method
                           for name, method in _STRATEGY_VIEW_METHODS)
                or not self._has_committed_rival_fit()
                or not native_physics(state.driver, state.car, self.track, self.weather,
                                      state.current_tire)):
            return {}
        observed = self._observed_control_projection(now)
        return {} if observed is None else {"_observed": observed}

    def _chronological_finish_context(self, state, now, *, restart=False):
        """Freeze observable pending events without sampled future service."""
        pending_laps = getattr(self, "pending", None)
        if (restart or getattr(self, "regrouping", False) or type(pending_laps) is not dict
                or state.driver.id in pending_laps):
            return None
        intervals = self._neutralized_finish_intervals()
        if intervals is None:
            return None
        active = {key: row for key, row in self.states.items()
                  if row.status == DriverStatus.RACING}
        if state.driver.id not in active:
            return None
        events = {}
        for _, _, serial, kind, key, generation in getattr(self, "queue", ()):
            pending = self.pending.get(key)
            if (pending is not None and generation == pending.generation
                    and kind == ("cross" if pending.on_track else "exit")):
                events[key] = min(serial, events.get(key, serial))
        rivals = []
        try:
            for key, other in active.items():
                if other is state:
                    continue
                pending = self.pending.get(key)
                if (pending is None or pending.lap != other.laps_completed + 1
                        or key not in events or other.force_pit_next_lap
                        or self.weather.tire_mismatch(other.current_tire.compound) == "critical"
                        or pit_plan_may_stop(other, pending.lap + 1)):
                    return None
                pace = (pending.running if pending.on_track else self.running_paces.get(key))
                ready = (pending.ready if pending.on_track else
                         self._pending_service_exit(key, pending, now))
                rivals.append(ChronologicalFinishCar(
                    key, other.laps_completed, pace, max(now, ready),
                    pending.running_start if pending.on_track else None,
                    pending.neutralized, events[key],
                    0. if pending.on_track else self._pending_fit_cost(key),
                    self._committed_rival_fit(key, pending),
                ))
            return ChronologicalFinishContext(
                state.driver.id, deepcopy(self.timeline), tuple(self.order), tuple(rivals),
                self.running_paces.get(state.driver.id),
                self.simulator.event_manager.get_lap_time_modifier(),
                self.simulator.event_manager.safety_car_active,
                self.simulator.weather_forecast_context,
                control_intervals=intervals,
            )
        except (TypeError, ValueError, OverflowError, AttributeError, KeyError):
            return None

    def _protect_neutralized_field_finish(self, state, planning, now, delay, traffic, *,
                                        restart=False):
        context = self._chronological_finish_context(state, now, restart=restart)
        if context is None:
            return False
        replacements = self._finish_replacement_options(
            state, lap=state.laps_completed + 1, planning=planning)
        result = evaluate_chronological_finish_protection(
            state.driver, state.car, self.track, state.current_tire, state.tire_laps,
            self.weather, now, context,
            expected_lane_loss=self.track.pit_lane_delta * self.simulator._pit_lane_factor(),
            expected_service_time=expected_stationary_time(state.car),
            expected_queue_delay=delay, replacements=replacements,
            tire_warmup=self.simulator.tire_warmup, current_fit_pending=state.fit_lap_pending,
            current_overtake_mode_active=self.simulator._strategy_overtake_mode_active(
                state, self.track, self.control_intervals + 1, self.weather,
                traffic.gap_ahead if traffic is not None else None,
                mode_allowed=self._overtake_mode_allowed(self.weather)),
        )
        return result.veto

    def _finish_replacement_options(self, state, *, lap, planning=None):
        """Use a known native replacement, or retain an optimistic option set."""
        return self.simulator._finish_replacement_options(
            state, self.weather, lap=lap, planning=planning,
        )

    def _leading_finish_context(self, state, now, *, restart=False):
        """Freeze rival crossings and the timed signal before a leading stop."""
        timeline = getattr(self, "timeline", None)
        if timeline is None or timeline.chequered_time is not None:
            return None
        rivals = []
        modifier = self.simulator.event_manager.get_lap_time_modifier()
        for other in self.states.values():
            if other is state or other.status != DriverStatus.RACING:
                continue
            pending = None if restart else self.pending.get(other.driver.id)
            pace = self.running_paces.get(other.driver.id)
            if pace is None and pending is not None:
                pace = pending.running or None
            if pace is None or not isfinite(pace) or pace <= 0:
                return None
            if pending is not None and pending.on_track:
                first = max(now, pending.ready)
            else:
                expected_exit = (self._pending_service_exit(other.driver.id, pending, now)
                                 if pending is not None else now)
                if expected_exit is None:
                    return None
                first = max(now, expected_exit) + pace * modifier
                first += self._pending_fit_cost(other.driver.id)
            if not isfinite(first):
                return None
            rivals.append(RivalFinishForecast(other.laps_completed, first, pace,
                                             other.driver.id))
        return LeadingFinishContext(timeline.time_limit_seconds, timeline.time_limit_announced,
                                    tuple(rivals), self.simulator.weather_forecast_context)

    def _protect_elective_finish_distance(
        self, state, planning, now, delay, traffic, *, restart=False,
    ):
        """Return whether a native elective stop should be vetoed."""
        reason = self._finish_protection_skip_reason(state, restart=restart, now=now)
        if reason is not None:
            return False
        if self._field_finish_required(self._neutralized_finish_intervals()):
            return self._protect_neutralized_field_finish(
                state, planning, now, delay, traffic, restart=restart)
        leading_context = state.strategy_leading_finish_context if restart else None
        if self._forecast_leader() is state:
            if not restart:
                leading_context = self._leading_finish_context(state, now)
            if leading_context is None:
                return False
        flag_time = (None if leading_context is not None
                     else self._projected_flag_time(now, restart=restart))
        if leading_context is None and (flag_time is None or not isfinite(flag_time)):
            return False
        physical_total = self.track.total_laps
        # ``planning`` is an observed-pace strategy horizon.  The protection
        # must retain the actual scheduled cap so a fast bound cannot be
        # hidden by that estimate before it reaches the projected flag.
        max_lap = self.track.total_laps
        if (not isinstance(max_lap, int) or max_lap < state.laps_completed + 1
                or not isinstance(physical_total, int) or physical_total < max_lap):
            return False
        replacements = self._finish_replacement_options(
            state, lap=state.laps_completed + 1, planning=planning,
        )
        # An empty finite pool is decided by native preparation/retirement;
        # there is no optimistic stop path to compare here.
        if replacements is not None and not replacements:
            return False
        modifier = self.simulator.event_manager.get_lap_time_modifier()
        gap = traffic.gap_ahead if traffic is not None else None
        result = evaluate_finish_protection(
            state.driver,
            state.car,
            self.track,
            state.current_tire,
            state.tire_laps,
            state.laps_completed + 1,
            self.weather,
            now,
            flag_time,
            max_lap,
            # Forecast work must not call the live execution simulator: test
            # instrumentation and its RNG belong solely to actual laps.
            lap_simulator=LapSimulator(),
            physical_total_laps=physical_total,
            expected_lane_loss=(self.track.pit_lane_delta
                                * self.simulator._pit_lane_factor()),
            expected_service_time=expected_stationary_time(state.car),
            expected_queue_delay=delay,
            current_lap_time_modifier=modifier,
            active_aero_enabled=self.simulator.event_manager.is_active_aero_allowed(),
            observed_gap=gap,
            current_overtake_mode_active=self.simulator._strategy_overtake_mode_active(
                state, self.track, self.control_intervals + 1, self.weather, gap,
                mode_allowed=self._overtake_mode_allowed(self.weather),
            ),
            replacements=replacements,
            projected_surface_at=lambda absolute: self._projected_surface_at(
                absolute, now=now, restart=restart,
            ),
            **({"tire_warmup": self.simulator.tire_warmup,
                "current_fit_pending": state.fit_lap_pending}
               if self.simulator.tire_warmup else {}),
            **({"leading_finish_context": leading_context}
               if leading_context is not None else {}),
        )
        return result.veto

    def _start_lap(self, state, now, *, restart_planning=None):
        driver_id = state.driver.id
        if not self.timeline.can_start_next_lap(driver_id):
            return
        lap = state.laps_completed + 1
        control = self.simulator.event_manager
        projection_options = (self._strategy_projection_options(state, now)
                              if restart_planning is None else {})
        if restart_planning is None:
            state.strategy_leading_finish_context = None
            state.strategy_finish_context = (
                CustomPitFinishContext(now, self.timeline.time_limit_seconds,
                                       self.timeline.time_limit_announced)
                if state.pit_plan is not None and state is self._forecast_leader() else None
            )
            planning = self._planning_track(state, now, **projection_options)
            cadence, weather_clock = None, None
        else:
            # Keep the leader clock frozen with this restart's horizon and
            # weather path, even if an earlier released car has entered service.
            planning = restart_planning[0]
            cadence = restart_planning[1] if len(restart_planning) > 1 else None
            weather_clock = restart_planning[2] if len(restart_planning) > 2 else None
        if restart_planning is None:
            cadence = self._weather_intervals(state, now, planning, **projection_options)
        if driver_id in self.free_refits:
            self._fit_red_flag_set(state, planning, cadence, weather_clock)
            if state.status != DriverStatus.RACING:
                self._retire(driver_id, now, state.dnf_reason)
                return
        # Completed service is observable; its future sampled duration is not.
        team = state.car.team_id
        if self.box_releases.get(team, now) <= now:
            self.expected_box_releases.pop(state.car.team_id, None)
        forecast_release = self._team_release_forecast(team, now)
        delay = max(0.0, forecast_release - now)
        active = [other for other in self.states.values() if other.status == DriverStatus.RACING]
        traffic = self._strategy_traffic(state, now, delay, **projection_options)
        safety_car = self._safety_car_strategy_snapshot(state, now, delay)
        if safety_car is not None:
            traffic = replace(traffic, current_traffic_gaps=safety_car.traffic_gaps,
                              safety_car=safety_car)
        state.strategy_safety_car_snapshot = traffic.safety_car
        restart = restart_planning is not None
        state.strategy_control_context = None
        if weather_clock is None and cadence is not None:
            weather_clock = self._strategy_weather_clock(
                state, now, planning, delay, restart=restart, **projection_options,
            )
        if (not restart and (self.simulator._can_project_dry_control(
                state, planning, self.weather, lap)
                or self._can_project_green_weather(state, self.weather,
                                                   weather_clock=weather_clock))):
            context = self._chronological_finish_context(state, now)
            if (context is not None and isinstance(context.own_pace, Real)
                    and not isinstance(context.own_pace, bool)
                    and isfinite(context.own_pace) and context.own_pace > 0 and not any(
                pit_plan_may_stop(
                    self.states[row.identifier], row.completed_laps + offset + 1)
                for row in context.rivals
                for offset in range(1, min(context.control_intervals,
                                           planning.total_laps - lap + 1) + 1)
            )):
                stop_delay = (self.track.pit_lane_delta * self.simulator._pit_lane_factor()
                              + expected_stationary_time(state.car) + delay)
                if (context.control_intervals > 0 or self._green_service_can_advance_weather(
                        state, context, now, stop_delay)):
                    state.strategy_control_context = StrategyControlContext(
                        context, now, stop_delay)
        forced_repair = state.force_pit_next_lap
        if forced_repair:
            # Forced execution bypasses policy; stale elective context must
            # not survive to the forced record.
            state.pit_decision_context = None
        custom_stop = self.simulator._custom_pit_plan_decision(
            state, planning, self.weather, lap,
        )
        if custom_stop is True:
            stop = True
        elif custom_stop is False:
            stop = False
        else:
            stop = forced_repair or self.simulator._should_pit(
                state, active, planning, lap, control.is_pit_window_open(), self.weather,
                additional_current_stop_cost=delay, physical_total_laps=self.track.total_laps,
                traffic_snapshot=traffic,
                weather_intervals=cadence,
                weather_clock=weather_clock,
                current_overtake_mode_allowed=self._overtake_mode_allowed(self.weather),
            )
        if (
            stop
            and custom_stop is not True
            and not state.force_pit_next_lap
            and self._protect_elective_finish_distance(
            state, planning, now, delay, traffic, restart=restart,
            )
        ):
            self._clear_one_lap_pit_proposals(state)
            stop = False
        if not stop and state.pit_plan_override_reason is not None:
            override_pit_plan_instruction(
                state, state.pit_plan_override_reason,
            )
        loss = 0.0
        expected_exit = None
        if stop:
            if state.tire_inventory is not None and not self.simulator._prepare_inventory_pit(
                state, planning, self.weather, lap, physical_total_laps=self.track.total_laps,
                weather_intervals=cadence, current_traffic_gaps=traffic.current_traffic_gaps,
                additional_current_stop_cost=delay,
                weather_clock=weather_clock,
            ):
                state.strategy_control_context = None
                self._retire(driver_id, now, state.dnf_reason)
                return
            expected_service = expected_stationary_time(state.car)
            self.expected_box_releases[state.car.team_id] = now + delay + expected_service
            expected_exit = (now + delay + expected_service
                             + self.track.pit_lane_delta * self.simulator._pit_lane_factor())
            loss = self.simulator._execute_pit_stop(
                state, planning, self.weather, lap, pit_box_releases=self.box_releases,
                arrival_time=now,
                physical_total_laps=self.track.total_laps,
                weather_intervals=cadence,
                weather_clock=weather_clock,
                current_traffic_gaps=traffic.current_traffic_gaps,
                additional_current_stop_cost=delay,
            )
            if state.status != DriverStatus.RACING:
                state.strategy_control_context = None
                self._retire(driver_id, now, state.dnf_reason)
                return
            state.pit_stops += 1
            state.pit_laps.append(lap)
            self.simulator._commit_pit_plan_if_due(
                state,
                overridden=state.pit_plan_override_reason is not None,
            )
            state.force_pit_next_lap = False
            self._record_pit_service(state, lap, now, expected_exit - (
                now + delay + expected_service
            ))
            # Keep the old public ledger for diagnostics and synthetic callers;
            # future decisions use the lifecycle records above when present.
            self.expected_box_releases[team] = self._team_release_forecast(team, now)
            if driver_id in self.order:
                self.order.remove(driver_id)
        snapshot = self.weather.model_copy(deep=True)
        state.strategy_control_context = None
        neutralized = not control.is_active_aero_allowed()
        interval = self.control_intervals + 1
        pending = _PendingLap(
            lap, state.total_time, now + loss, snapshot, neutralized,
            state.current_tire.model_copy(deep=True), state.tire_laps, 0.0,
            on_track=not stop, paid_stop=stop,
            lap_time_modifier=control.get_lap_time_modifier(),
            active_aero_enabled=control.is_active_aero_allowed(),
            mode_allowed=self._overtake_mode_allowed(snapshot),
            restart_boost=control.is_restart_lap(interval),
            safety_car=control.safety_car_active,
            expected_exit=expected_exit,
        )
        self.pending[driver_id] = pending
        state.overtake_mode_active_lap = False
        state.overtake_mode_detected_gap = None
        if not stop:
            self._begin_running(state, pending, now)
        self._enqueue(driver_id, pending.ready, "exit" if stop else "cross")

    def _queue_leader_id(self):
        """Anchor the on-track SC queue by distance, then circular track order."""
        racing = [driver_id for driver_id in self.order
                  if self.states[driver_id].status == DriverStatus.RACING]
        if not racing:
            return self.leader_id
        return min(racing, key=lambda driver_id: -self.states[driver_id].laps_completed)

    def _projected_progress(self, driver_id, when, exit_lap, *, now=None, flag_time=None):
        """Forecast a rival's fractional position without simulating new events.

        Keep committed running and expected pit delay, then extrapolate observed free pace.
        Future tyre/weather changes, stops and battle delays are unknown. This
        projection prices one rejoin; it never changes actual crossing order.
        """
        pending = self.pending.get(driver_id)
        if pending is None or self.states[driver_id].status != DriverStatus.RACING:
            return None
        free_pace = (pending.running
                     or self.running_paces.get(driver_id, self.track.base_lap_time))
        pace = free_pace * self.simulator.event_manager.get_lap_time_modifier()
        first_pace = free_pace * pending.lap_time_modifier if pending.on_track else pace
        if not isfinite(pace) or pace <= 0 or not isfinite(first_pace) or first_pace <= 0:
            return None
        if pending.on_track:
            start, ready = pending.running_start, pending.ready
        else:
            expected_exit = self._pending_service_exit(driver_id, pending, now)
            start = expected_exit if expected_exit is not None else pending.ready
            if now is not None:
                start = max(now, start)
            ready = start + first_pace + self._pending_fit_cost(driver_id)
            if when == start and pending.lap < exit_lap:
                return None  # Our higher-distance exit has priority at this tie.
        if start is None or when < start or ready <= start:
            return None
        if when < ready:
            return (when - start) / (ready - start)
        if when == ready and pending.lap < exit_lap:
            return 1.0  # Our exit precedes this lower-distance crossing.
        # A car already under the chequered flag exits at its next crossing;
        # never project it back into traffic for another lap.
        if self.timeline.chequered_time is not None:
            return None
        elapsed = when - ready
        completed = pending.lap + int(elapsed // pace)
        progress = (elapsed % pace) / pace
        final_lap = self.timeline.final_lap
        if flag_time is None and now is not None:
            flag_time = self._projected_flag_time(now)
        if flag_time is not None:
            # A lapped rival remains on track until its own first crossing at
            # or after the leader's projected flag, not merely until that flag.
            final_lap = min(final_lap, pending.lap + max(
                0, ceil((flag_time - ready) / pace - 1e-12),
            ))
        if (elapsed > 0 and progress == 0 and completed <= exit_lap
                and completed <= final_lap):
            # Future crossings are enqueued after this candidate pit exit;
            # its serial priority also wins when their lap distances tie.
            return 1.0
        if completed >= final_lap:
            return None
        return progress

    def _strategy_traffic(self, state, now, queue_delay, *, _observed=...):
        """Snapshot physical gaps and expected rejoin cost at this own-lap start.

        The circular successor supplies the space behind, including lapped
        traffic. Race rank and old completed-crossing clocks are not positions.
        Service and unfinished queue delay are expected; no RNG is used.
        """
        driver_id = state.driver.id
        modifier = self.simulator.event_manager.get_lap_time_modifier()
        pace = self.running_paces.get(driver_id, self.track.base_lap_time) * modifier
        ahead = self._physical_gap_ahead(driver_id, now, pace)
        behind = None
        if driver_id in self.order and len(self.order) > 1:
            index = self.order.index(driver_id)
            follower_id = self.order[(index + 1) % len(self.order)]
            follower = self.pending.get(follower_id)
            if follower is not None and follower.on_track:
                behind = max(0.0, follower.ready - now)
        cost = 0.0
        traffic_gaps = None
        if self.simulator.event_manager.is_active_aero_allowed():
            exit_time = (now + self.track.pit_lane_delta * self.simulator._pit_lane_factor()
                         + expected_stationary_time(state.car) + queue_delay)
            flag_time = (_observed.flag_time if _observed is not ... and _observed is not None
                         else self._projected_flag_time(now))
            progress = self._fitted_rejoin_progress(state, now, exit_time)
            if progress is None:
                progress = [value for other_id in self.pending if other_id != driver_id
                            and (value := self._projected_progress(
                                other_id, exit_time, state.laps_completed + 1,
                                now=now, flag_time=flag_time,
                            )) is not None]
            rejoin_gap = min(progress) * pace if progress else None
            traffic_gaps = (ahead, rejoin_gap)
            traffic = self.simulator.lap_simulator.traffic_pace_contribution
            cost = traffic(rejoin_gap) - traffic(ahead)
        return StrategyTrafficSnapshot(ahead, behind, cost, traffic_gaps)

    def _fitted_rejoin_progress(self, state, now, exit_time):
        """Advance known fitted rivals while the candidate awaits paid entry."""
        if not self._has_committed_rival_fit():
            return None
        context = self._chronological_finish_context(state, now)
        if context is None or not any(row.committed_fit is not None for row in context.rivals):
            return None
        try:
            field = ObservedChronologicalField(context, now)
            field.enter(exit_time - now)
            return [min(1., (exit_time - row.running_start) / (row.ready - row.running_start))
                    for key, row in field.pending.items()
                    if key != state.driver.id and row.running_start is not None]
        except (TypeError, ValueError, OverflowError, AttributeError, KeyError):
            return None

    def _safety_car_strategy_snapshot(self, state, now, queue_delay):
        """Hold the observed on-track queue through expected service.

        Project only unfinished running already visible at this decision.
        If a rival crosses or is in service before expected rejoin, its next
        start/queue order is unresolved and the uniform forecast is retained.
        Actual sampled future service and hypothetical rival choices are never
        read. Race control/weather are conditional on the current snapshot.
        """
        control = self.simulator.event_manager
        if not control.safety_car_active or control.red_flag_active:
            return None
        identifier = state.driver.id
        others = [key for key in self.order if key != identifier
                  and self.states[key].status == DriverStatus.RACING]
        pace = self.running_paces.get(identifier)
        if not others or pace is None or not isfinite(pace) or pace <= 0:
            return None
        if any(other.status == DriverStatus.RACING and key != identifier and key not in others
               for key, other in self.states.items()):
            return None
        modifier = control.get_lap_time_modifier()
        exit_time = (now + self.track.pit_lane_delta * self.simulator._pit_lane_factor()
                     + expected_stationary_time(state.car) + queue_delay)
        for key in others:
            pending = self.pending.get(key)
            if (pending is None or not pending.on_track or pending.running_start is None
                    or not isfinite(pending.ready) or pending.ready <= exit_time
                    or pending.running_start > now or pending.ready <= pending.running_start
                    or not isfinite(pending.running) or pending.running <= 0):
                return None

        def branch(stopped):
            order = others + [identifier] if stopped else list(self.order)
            if identifier not in order:
                return None
            leader = min(order, key=lambda key: -self.states[key].laps_completed)
            queue = (None if leader == identifier else
                     self.pending[leader].running * modifier)
            index = order.index(identifier)
            if index == 0:
                return SafetyCarBranch(queue)
            ahead = self.pending[order[index - 1]]
            entry = exit_time if stopped else now
            progress = min(1., (entry - ahead.running_start) / (ahead.ready - ahead.running_start))
            return SafetyCarBranch(queue, progress * pace * modifier, ahead_progress=progress,
                                   blocked_until=ahead.ready + 1.e-9 - entry)

        retained, stopped = branch(False), branch(True)
        return (StrategySafetyCarSnapshot(retained, stopped)
                if retained is not None and stopped is not None else None)

    def _physical_gap_ahead(self, driver_id, now, reference_pace=None):
        """Time-equivalent forward distance to the circular physical predecessor.

        Pending running is interpolated at the line/pit-exit snapshot. A car's
        race lap count and its old crossing clock are irrelevant to this gap.
        The initial grid leader has no predecessor; after each crossing, the
        newly starting car is at the circular tail. Cars in the pit lane are
        absent from that order and cannot impose dirty air or detection.
        """
        if driver_id not in self.order or not isfinite(now):
            return None
        index = self.order.index(driver_id)
        if index == 0:
            return None
        ahead_id = self.order[index - 1]
        ahead = self.states[ahead_id]
        pending = self.pending.get(ahead_id)
        if (ahead.status != DriverStatus.RACING or pending is None
                or not pending.on_track or pending.running_start is None):
            return None
        duration = pending.ready - pending.running_start
        elapsed = now - pending.running_start
        if (not isfinite(duration) or duration <= 0 or not isfinite(elapsed) or elapsed < 0):
            return None
        # Whole-lap interpolation includes any already-known delay. Clamping
        # retains the physical predecessor if its ready crossing awaits a pass.
        progress = min(1.0, elapsed / duration)
        own_pending = self.pending.get(driver_id)
        modifier = own_pending.lap_time_modifier if own_pending is not None else 1.0
        pace = (reference_pace if reference_pace is not None else
                self.running_paces.get(driver_id, self.track.base_lap_time) * modifier)
        return progress * pace if isfinite(pace) and pace > 0 else None

    def _overtake_mode_allowed(self, weather):
        """A shared restart lap cannot replace an unfinished car's line crossing."""
        return (not self._overtake_restart_waiting
                and self.simulator.event_manager.is_overtake_mode_allowed(
                    self.control_intervals + 1, weather))

    def _capture_running_conditions(self, pending):
        """Freeze running conditions at track entry, after any paid service."""
        control = self.simulator.event_manager
        pending.weather = self.weather.model_copy(deep=True)
        pending.neutralized = not control.is_active_aero_allowed()
        pending.lap_time_modifier = control.get_lap_time_modifier()
        pending.active_aero_enabled = control.is_active_aero_allowed()
        interval = self.control_intervals + 1
        pending.mode_allowed = self._overtake_mode_allowed(pending.weather)
        pending.restart_boost = control.is_restart_lap(interval)
        pending.safety_car = control.safety_car_active
        pending.sc_queue_pace = None
        if pending.safety_car:
            leader_id = self._queue_leader_id()
            leader_pending = self.pending.get(leader_id)
            leader_pace = (leader_pending.running if leader_pending is not None
                           and leader_pending.running > 0 else self.running_paces.get(leader_id))
            if leader_pace is not None:
                pending.sc_queue_pace = leader_pace * pending.lap_time_modifier

    def _begin_running(self, state, pending, now):
        """Sample exactly once, using traffic at actual track entry after service."""
        self._capture_running_conditions(pending)
        pending.on_track = True
        pending.running_start = now
        gap = self._physical_gap_ahead(state.driver.id, now)
        pending.detected_gap = gap
        pending.mode_active = (not pending.paid_stop
                               and self.simulator._deploy_overtake_mode_if_eligible(
                                   state, self.track, gap, pending.mode_allowed,
                               ))
        state.overtake_mode_active_lap = pending.mode_active
        free_running = self.simulator.lap_simulator.calculate_lap_time(
            state.driver, state.car, self.track, pending.tire, pending.weather,
            pending.lap, self.track.total_laps, gap_to_car_ahead=gap,
            active_aero_enabled=pending.active_aero_enabled,
            overtake_mode_active=pending.mode_active,
        )
        pending.running = free_running
        running = free_running * pending.lap_time_modifier
        if pending.safety_car:
            if state.driver.id == self._queue_leader_id():
                # The anchor sets queue pace; it never chases a lapped predecessor.
                pending.sc_queue_pace = running
            else:
                nominal = max(free_running, pending.sc_queue_pace or running)
                queue_gap = self._physical_gap_ahead(state.driver.id, now, nominal)
                running = safety_car_running_time(free_running, nominal, queue_gap)
        fit_cost = (self.simulator._consume_tire_warmup(state)
                    if self.simulator.tire_warmup else 0.0)
        # The fitting sensitivity is elapsed running time after control has
        # set this lap's pace. It delays the next crossing, never pit exit.
        pending.ready = now + running + fit_cost

    def _retire(self, driver_id, now, reason):
        self.timeline.retire(driver_id, now)
        state = self.states[driver_id]
        state.status = DriverStatus.DNF
        state.dnf_reason = reason
        state.driver.dnf = True
        state.driver.dnf_reason = reason
        self.pending.pop(driver_id, None)
        self._overtake_restart_waiting.discard(driver_id)
        self.free_refits.discard(driver_id)
        self.red_waiting.discard(driver_id)
        if driver_id in self.order:
            self.order.remove(driver_id)

    def _delay(self, driver_id, loss):
        pending = self.pending.get(driver_id)
        if pending is not None:
            pending.ready += loss
            pending.generation += 1
            self._enqueue(driver_id, pending.ready, "cross")

    def _resolve_crossing(self, driver_id, now):
        state = self.states[driver_id]
        pending = self.pending[driver_id]
        if not pending.mechanical_checked:
            event = self.simulator.event_manager._check_mechanical_failure(
                state.driver, state.car, self.track, pending.lap, pending.weather,
            )
            pending.mechanical_checked = True
            if event:
                self.simulator.event_manager.events.append(event)
                self._retire(driver_id, now, state.driver.dnf_reason)
                return False
            if not pending.neutralized:
                incident = self.simulator.event_manager._check_random_incident(
                    [state.driver], self.track, pending.weather, pending.lap,
                    exposure_drivers=[active.driver for active in self.states.values()
                                      if active.status == DriverStatus.RACING],
                )
                if incident:
                    self.simulator.event_manager.events.append(incident)
                    self.incidents += 1
                    if state.driver.dnf:
                        self._retire(driver_id, now, state.driver.dnf_reason)
                        return False
                    if incident.forces_pit_stop:
                        state.force_pit_next_lap = True
                        self.simulator._damage_inventory_tire(state)
                    if incident.time_loss_seconds:
                        self._delay(driver_id, incident.time_loss_seconds)
                        return False
        while self.order and self.order[0] != driver_id:
            index = self.order.index(driver_id)
            defender_id = self.order[index - 1]
            defender = self.states[defender_id]
            defender_lap = self.pending[defender_id]
            success = incident = False
            control = self.simulator.event_manager
            passing_allowed = not (self.regrouping or pending.neutralized
                                   or defender_lap.neutralized or control.safety_car_active
                                   or control.vsc_active or control.red_flag_active)
            if (passing_allowed
                    and pending.lap > defender_lap.lap):
                # This encounter puts the physical predecessor another lap
                # down. Model compliant yielding at the lap-level catch point;
                # no defensive battle or extra passing/incident draw is needed.
                # A same-lap attack or unlapping attempt still needs a pass.
                self.order[index - 1], self.order[index] = driver_id, defender_id
                continue
            if (passing_allowed
                    and defender_id not in pending.attempted):
                pending.attempted.add(defender_id)
                # Earlier unconstrained readiness means the car has caught its
                # predecessor somewhere in this lap. At that contact the gap
                # is zero; readiness itself never grants an automatic pass.
                advantage = self.simulator.lap_simulator.tire_weather_pace_contribution(
                    defender.driver, defender.car, self.track, defender_lap.tire,
                    defender_lap.tire_age + 1, pending.weather,
                ) - self.simulator.lap_simulator.tire_weather_pace_contribution(
                    state.driver, state.car, self.track, pending.tire,
                    pending.tire_age + 1, pending.weather,
                )
                success, incident = self.simulator.overtaking_model.attempt_overtake(
                    state.driver, state.car, defender.driver, defender.car, self.track, 0.0,
                    overtake_mode_active=pending.mode_active,
                    detected_gap=pending.detected_gap,
                    restart_boost=pending.restart_boost,
                    is_wet=pending.weather.is_wet(), tire_pace_advantage_seconds=advantage,
                )
                state.overtake_attempts += 1
                state.overtake_successes += int(success)
                state.overtake_contacts += int(incident)
            if success:
                self.order[index - 1], self.order[index] = driver_id, defender_id
                continue
            if incident:
                collision_rng = self.simulator._driver_rng(driver_id, "collision_loss")
                losses = {driver_id: float(collision_rng.uniform(1, 3)),
                          defender_id: float(collision_rng.uniform(0.5, 2))}
                self.simulator.event_manager.events.append(RaceEvent(
                    EventType.COLLISION, pending.lap, [driver_id, defender_id],
                    description="Contact during chronological passing attempt",
                    applied_time_losses=losses,
                ))
                self.incidents += 1
                self._delay(defender_id, losses[defender_id])
                pending.ready += losses[driver_id]
            pending.ready = max(pending.ready, defender_lap.ready + 1e-9)
            pending.generation += 1
            self._enqueue(driver_id, pending.ready, "cross")
            return False
        active_distance = max(other.laps_completed for other in self.states.values()
                              if other.status == DriverStatus.RACING)
        leading = (self.timeline.chequered_time is None
                   and pending.lap > active_distance)
        # This lap-level model uses accepted on-track crossings; pit-lane
        # Control Line geometry is not resolved. Clear the previous barrier
        # before control processing can arm a new one at this crossing.
        self._overtake_restart_waiting.discard(driver_id)
        red = self._leader_interval(pending) if leading else False
        crossing = self.timeline.observe_crossing(driver_id, pending.lap, now, is_leader=leading)
        state.laps_completed = pending.lap
        state.total_time = now
        state.last_lap_time = now - pending.start
        state.tire_laps = pending.tire_age + 1
        state.driver.current_tire_laps = state.tire_laps
        self.running_paces[driver_id] = pending.running
        self.fastest[driver_id] = min(
            self.fastest.get(driver_id, float("inf")), state.last_lap_time,
        )
        self.crossings.append((driver_id, pending.lap, now))
        self.simulator._recharge_overtake_mode_energy([state], neutralized=pending.neutralized)
        del self.pending[driver_id]
        self.order.remove(driver_id)
        if crossing.finish_time is not None:
            state.status = DriverStatus.FINISHED
            self._overtake_restart_waiting.discard(driver_id)
        else:
            self.order.append(driver_id)
        self._positions()
        if leading:
            self.leader_id = driver_id
            self._after_leader_crossing(now, red)
        return True

    def _positions(self):
        ordered = sorted(self.states.values(), key=lambda state: (
            -state.laps_completed, state.total_time, state.position,
        ))
        for position, state in enumerate(ordered, 1):
            state.position = position

    def _leader_interval(self, pending):
        control = self.simulator.event_manager
        was_safety_car = control.safety_car_active
        # Leadership can pass to a lapped survivor. Race-control cadence still
        # advances once per leading interval rather than replaying its old laps.
        self.control_intervals += 1
        events = control.process_lap(self.control_intervals, [], {}, self.track, pending.weather,
                                     incidents_this_lap=self.incidents)
        if control.safety_car_active or control.vsc_active or control.red_flag_active:
            # Disable activation immediately even on a green lap still running
            # when the signal changes. Its sampled crossing and already spent
            # energy remain committed; a later pass cannot reuse this burst.
            for driver_id, running in self.pending.items():
                running.mode_active = False
                self.states[driver_id].overtake_mode_active_lap = False
        if control.safety_car_active or control.red_flag_active:
            self._overtake_restart_waiting.clear()
        elif was_safety_car:
            # SC return is observed at this shared crossing. Every continuing
            # car must supply a subsequent crossing, including cars in service
            # and cars on a lower own lap. A red-flag collection supersedes it.
            self._overtake_restart_waiting = {
                driver_id for driver_id, state in self.states.items()
                if state.status == DriverStatus.RACING
            }
        self.incidents = 0
        neutral = pending.neutralized or any(event.event_type in (
            EventType.SAFETY_CAR, EventType.VIRTUAL_SAFETY_CAR, EventType.RED_FLAG,
        ) for event in events)
        self.green_streak = 0 if neutral else self.green_streak + 1
        self.has_two_green |= self.green_streak >= 2
        return any(event.event_type == EventType.RED_FLAG for event in events)

    def _after_leader_crossing(self, now, red):
        if red and self.timeline.chequered_time is None:
            self.timeline.begin_suspension(now)
            self.regrouping = True
            self.red_flag_start = now
            self.resumption_order = self._red_flag_order()
            self.free_refits.update(self.resumption_order)
        if (self.timeline.chequered_time is None
                and any(state.status == DriverStatus.RACING for state in self.states.values())):
            self.weather = self.simulator._advance_race_weather(self.weather)
            # Shared leading intervals, not an individual car's completed distance.
            self.simulator._record_weather(self.control_intervals + 1, self.weather)

    def _results(self):
        winner = self.states.get(self.timeline.winner_id)
        winner_laps = winner.laps_completed if winner else 0
        ordered = sorted(self.states.values(), key=lambda state: (
            state.driver.id != self.timeline.winner_id,
            -state.laps_completed, state.total_time, state.position,
        ))
        results = []
        for position, state in enumerate(ordered, 1):
            classified = winner is not None and state.laps_completed > 0 and (
                state.laps_completed >= winner_laps * 9 // 10
            )
            results.append(RaceResult(
                state.driver.id, state.driver.name, state.car.team_name, position,
                state.total_time,
                state.total_time - winner.total_time
                if winner and state.status == DriverStatus.FINISHED else 0,
                state.pit_stops, self.fastest.get(state.driver.id, 0), state.status,
                dnf_reason=state.dnf_reason, strategy=list(state.tire_compound_history),
                laps_completed=state.laps_completed, classified=classified,
                pit_laps=list(state.pit_laps),
                pit_stop_details=[dict(stop) for stop in state.pit_stop_details],
                race_time_limited=(winner is not None
                                   and self.timeline.time_limit_announced
                                   and winner_laps < self.track.total_laps),
                points_awarded=points_for_classification(
                    position, classified, winner_laps, self.track.total_laps, self.has_two_green,
                ),
                race_suspension_seconds=self.timeline.total_suspension_seconds,
                pit_plan_history=finalize_pit_plan(
                    state,
                    "retired" if state.status == DriverStatus.DNF else "race_finished",
                ),
                overtake_attempts=state.overtake_attempts,
                overtake_successes=state.overtake_successes,
                overtake_contacts=state.overtake_contacts,
                **self.simulator._inventory_result_fields(state),
            ))
        return results


def simulate_chronological_race(simulator, drivers, cars, track, weather, starting_grid,
                                *, starting_tires=None, starting_tire_ages=None,
                                red_flag_pause_seconds=600.0, tire_inventory=None,
                                pit_plans=None, weather_schedule=None):
    """Run chronological car timing using the supplied simulator's physics."""
    schedule = validate_weather_schedule(weather_schedule, total_laps=track.total_laps)
    return ChronologicalRace(simulator, red_flag_pause_seconds=red_flag_pause_seconds).run(
        drivers, cars, track, weather, starting_grid, starting_tires=starting_tires,
        starting_tire_ages=starting_tire_ages,
        tire_inventory=tire_inventory,
        pit_plans=pit_plans,
        **({"weather_schedule": schedule} if schedule else {}),
    )


_STRATEGY_VIEW_METHODS = tuple((name, getattr(ChronologicalRace, name)) for name in (
    "_planning_track", "_weather_intervals", "_strategy_traffic", "_strategy_weather_clock",
    "_weather_projection_clock", "_observed_control_projection", "_projected_lap_starts",
    "_weather_update_times", "_weather_updates_at", "_projected_flag_time",
    "_chronological_finish_context", "_forecast_leader", "_pending_service_exit",
    "_pending_fit_cost", "_physical_gap_ahead", "_team_release_forecast",
    "_safety_car_strategy_snapshot", "_has_committed_rival_fit", "_can_project_committed_fit",
    "_committed_rival_fit",
))


register_forecast_helpers(globals(), (
    "ChronologicalFinishCar", "ChronologicalFinishContext", "CommittedRivalFit",
    "ObservedChronologicalField",
    "evaluate_chronological_finish_protection",
    "project_observed_chronological_clock",
    "observed_control_intervals", "forecast_running_duration",
    "StrategyControlContext",
    "native_physics",
    "_STRATEGY_VIEW_METHODS",
    "pit_plan_may_stop",
))
register_forecast_helpers(vars(ChronologicalRace), (
    "_chronological_finish_context", "_protect_neutralized_field_finish",
    "_neutralized_finish_intervals", "_field_finish_required",
    "_weather_update_times", "_projected_lap_starts", "_weather_updates_at",
    "_observed_control_projection",
    "_can_project_green_weather",
    "_green_service_can_advance_weather",
    "_has_committed_rival_fit", "_can_project_committed_fit", "_committed_rival_fit",
    "_fitted_rejoin_progress",
    "_strategy_projection_options",
))
