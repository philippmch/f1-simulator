"""Experimental chronological race execution; production still uses RaceSimulator.

Each car owns one pending lap. Whole-lap physics freezes weather/control when
that lap starts; interventions affect subsequently started laps. No suspension
wall time or mid-lap sector redistribution is invented. Circular crossing order
requires a sampled pass or compliant blue-flag yield before a faster car can
cross a physical predecessor.
"""

import heapq
from dataclasses import dataclass, field
from itertools import count
from math import ceil, isfinite

from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.neutralization import safety_car_running_time
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceResult
from f1sim.simulation.race_points import points_for_classification
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot
from f1sim.simulation.validation import validate_unique_ids


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


class ChronologicalRace:
    """Runnable opt-in engine with absolute crossings, stops, and per-car flags."""

    def __init__(self, simulator):
        self.simulator = simulator
        self.crossings: list[tuple[str, int, float]] = []
        self.pit_exits: list[tuple[str, int, float]] = []

    def run(self, drivers, cars, track, weather, starting_grid, *, starting_tires=None):
        validate_unique_ids([driver.id for driver in drivers], "driver")
        validate_unique_ids(starting_grid, "starting grid")
        self.track = track
        self.weather = weather.model_copy(deep=True)
        self.simulator.event_manager.reset()
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
            compound = ((starting_tires or {}).get(driver_id)
                        or self.simulator._choose_starting_compound(style, track, self.weather,
                                                                   driver, car))
            plans = self.simulator._plan_pit_lap_options(style, track)
            self.states[driver_id] = DriverRaceState(
                driver, car, len(self.states) + 1,
                current_tire=TIRE_COMPOUNDS[compound].model_copy(deep=True),
                strategy_archetype=style, planned_pit_laps=plans[0], pit_plan_options=plans,
            )
        self.timeline = RaceFinishTimeline(track.total_laps, self.states)
        self.order = list(self.states)
        self.pending = {}
        self.queue = []
        self.serial = count()
        self.box_releases = {}
        self.fastest = {}
        self.running_paces = {}
        self.free_refits = set()
        self.leader_id = None
        self.incidents = 0
        self.control_intervals = 0
        self.green_streak = 0
        self.has_two_green = False
        self.crossings.clear()
        self.pit_exits.clear()
        for state in self.states.values():
            self._start_lap(state, 0.0)
        while self.queue:
            now, _, _, kind, driver_id, generation = heapq.heappop(self.queue)
            pending = self.pending.get(driver_id)
            if pending is None or generation != pending.generation:
                continue
            if kind == "exit":
                self.order.append(driver_id)  # Pit exit is immediately after the line.
                self.pit_exits.append((driver_id, pending.lap, now))
                self._begin_running(self.states[driver_id], pending, now)
                self._enqueue(driver_id, pending.ready, "cross")
                continue
            if self._resolve_crossing(driver_id, now):
                state = self.states[driver_id]
                if self.timeline.can_start_next_lap(driver_id):
                    self._start_lap(state, now)
        return self._results()

    def _enqueue(self, driver_id, time, kind):
        # At an exact-time crossing tie the leading distance receives the
        # flag first. Serial order alone can start a lapped car's extra lap.
        heapq.heappush(self.queue, (time, -self.pending[driver_id].lap,
                                   next(self.serial), kind, driver_id,
                                   self.pending[driver_id].generation))

    def _planning_track(self, state, now):
        """Forecast an own-lap finish horizon without altering the actual flag.

        Use observed free running pace (excluding stops, incidents and blocking),
        adjusted for the current control multiplier. The leading pending lap's
        absolute readiness anchors the forecast. Future weather, interruptions
        and elective stops are unknown; this is a strategy estimate only.
        """
        horizon = self.timeline.final_lap
        own_pace = self.running_paces.get(state.driver.id)
        active = [other for other in self.states.values() if other.status == DriverStatus.RACING]
        if own_pace is not None and active:
            leader = min(active, key=lambda other: (
                -other.laps_completed, other.total_time, other.position,
            ))
            pending = self.pending.get(leader.driver.id)
            leader_pace = self.running_paces.get(leader.driver.id)
            modifier = self.simulator.event_manager.get_lap_time_modifier()
            if pending is not None:
                flag_time = pending.ready
                laps_left = max(0, self.timeline.final_lap - pending.lap)
                if leader_pace is None:
                    leader_pace = pending.running or None
                if not pending.on_track and leader_pace is not None:
                    flag_time += leader_pace * pending.lap_time_modifier
            else:
                flag_time = now
                laps_left = max(0, self.timeline.final_lap - leader.laps_completed)
            if leader_pace is not None:
                flag_time += laps_left * leader_pace * modifier
                remaining = max(1, ceil((flag_time - now) / (own_pace * modifier) - 1e-12))
                horizon = min(self.timeline.final_lap, state.laps_completed + remaining)
        return self.track.model_copy(update={"total_laps": horizon})

    def _start_lap(self, state, now):
        driver_id = state.driver.id
        if not self.timeline.can_start_next_lap(driver_id):
            return
        lap = state.laps_completed + 1
        control = self.simulator.event_manager
        planning = self._planning_track(state, now)
        if driver_id in self.free_refits:
            compound = self.simulator._choose_red_flag_tire(
                state, self.weather, planning, state.laps_completed,
            )
            self.simulator._fit_tire(state, compound)
            state.force_pit_next_lap = False
            self.free_refits.remove(driver_id)
        delay = max(0.0, self.box_releases.get(state.car.team_id, now) - now)
        active = [other for other in self.states.values() if other.status == DriverStatus.RACING]
        stop = state.force_pit_next_lap or self.simulator._should_pit(
            state, active, planning, lap, control.is_pit_window_open(), self.weather,
            additional_current_stop_cost=delay, physical_total_laps=self.track.total_laps,
            traffic_snapshot=self._strategy_traffic(state, now, delay),
        )
        loss = 0.0
        if stop:
            loss = self.simulator._execute_pit_stop(
                state, planning, self.weather, lap, pit_box_releases=self.box_releases,
                arrival_time=now,
            )
            state.pit_stops += 1
            state.pit_laps.append(lap)
            state.force_pit_next_lap = False
            if driver_id in self.order:
                self.order.remove(driver_id)
        snapshot = self.weather.model_copy(deep=True)
        neutralized = not control.is_active_aero_allowed()
        interval = self.control_intervals + 1
        pending = _PendingLap(
            lap, now, now + loss, snapshot, neutralized,
            state.current_tire.model_copy(deep=True), state.tire_laps, 0.0,
            on_track=not stop, paid_stop=stop,
            lap_time_modifier=control.get_lap_time_modifier(),
            active_aero_enabled=control.is_active_aero_allowed(),
            mode_allowed=control.is_overtake_mode_allowed(interval, snapshot),
            restart_boost=control.is_restart_lap(interval),
            safety_car=control.safety_car_active,
        )
        if pending.safety_car:
            leader_id = self._queue_leader_id()
            leader_pending = self.pending.get(leader_id)
            leader_pace = (leader_pending.running if leader_pending is not None
                           and leader_pending.running > 0 else self.running_paces.get(leader_id))
            if leader_pace is not None:
                pending.sc_queue_pace = leader_pace * pending.lap_time_modifier
        self.pending[driver_id] = pending
        state.overtake_mode_active_lap = False
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

    def _projected_progress(self, driver_id, when, exit_lap):
        """Forecast a rival's fractional position without simulating new events.

        Keep committed running/pit delay, then extrapolate observed free pace.
        Future tyre/weather changes, stops and battle delays are unknown. This
        projection prices one rejoin; it never changes actual crossing order.
        """
        pending = self.pending.get(driver_id)
        if pending is None or self.states[driver_id].status != DriverStatus.RACING:
            return None
        free_pace = (pending.running
                     or self.running_paces.get(driver_id, self.track.base_lap_time))
        pace = free_pace * self.simulator.event_manager.get_lap_time_modifier()
        first_pace = free_pace * pending.lap_time_modifier
        if not isfinite(pace) or pace <= 0 or not isfinite(first_pace) or first_pace <= 0:
            return None
        if pending.on_track:
            start, ready = pending.running_start, pending.ready
        else:
            start, ready = pending.ready, pending.ready + first_pace
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
        if (elapsed > 0 and progress == 0 and completed <= exit_lap
                and completed <= self.timeline.final_lap):
            # Future crossings are enqueued after this candidate pit exit;
            # its serial priority also wins when their lap distances tie.
            return 1.0
        if completed >= self.timeline.final_lap:
            return None
        return progress

    def _strategy_traffic(self, state, now, queue_delay):
        """Snapshot physical gaps and expected rejoin cost at this own-lap start.

        The circular successor supplies the space behind, including lapped
        traffic. Race rank and old completed-crossing clocks are not positions.
        Service is expected, existing queue delay is known, and no RNG is used.
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
        if self.simulator.event_manager.is_active_aero_allowed():
            exit_time = (now + self.track.pit_lane_delta * self.simulator._pit_lane_factor()
                         + expected_stationary_time(state.car) + queue_delay)
            progress = [value for other_id in self.pending if other_id != driver_id
                        and (value := self._projected_progress(
                            other_id, exit_time, state.laps_completed + 1,
                        )) is not None]
            rejoin_gap = min(progress) * pace if progress else None
            traffic = self.simulator.lap_simulator.traffic_pace_contribution
            cost = traffic(rejoin_gap) - traffic(ahead)
        return StrategyTrafficSnapshot(ahead, behind, cost)

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

    def _begin_running(self, state, pending, now):
        """Sample exactly once, using traffic at actual track entry after service."""
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
        pending.ready = now + running

    def _retire(self, driver_id, now, reason):
        self.timeline.retire(driver_id, now)
        state = self.states[driver_id]
        state.status = DriverStatus.DNF
        state.dnf_reason = reason
        state.driver.dnf = True
        state.driver.dnf_reason = reason
        self.pending.pop(driver_id, None)
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
                )
                if incident:
                    self.simulator.event_manager.events.append(incident)
                    self.incidents += 1
                    if state.driver.dnf:
                        self._retire(driver_id, now, state.driver.dnf_reason)
                        return False
                    if incident.forces_pit_stop:
                        state.force_pit_next_lap = True
                    if incident.time_loss_seconds:
                        self._delay(driver_id, incident.time_loss_seconds)
                        return False
        while self.order and self.order[0] != driver_id:
            index = self.order.index(driver_id)
            defender_id = self.order[index - 1]
            defender = self.states[defender_id]
            defender_lap = self.pending[defender_id]
            success = incident = False
            if (not pending.neutralized and not defender_lap.neutralized
                    and pending.lap > defender_lap.lap):
                # This encounter puts the physical predecessor another lap
                # down. Model compliant yielding at the lap-level catch point;
                # no defensive battle or extra passing/incident draw is needed.
                # A same-lap attack or unlapping attempt still needs a pass.
                self.order[index - 1], self.order[index] = driver_id, defender_id
                continue
            if (not pending.neutralized and not defender_lap.neutralized
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
                    overtake_mode_active=(pending.mode_active and pending.detected_gap is not None
                                          and pending.detected_gap
                                          <= self.track.overtake_mode_detection_gap),
                    restart_boost=pending.restart_boost,
                    is_wet=pending.weather.is_wet(), tire_pace_advantage_seconds=advantage,
                )
            if success:
                self.order[index - 1], self.order[index] = driver_id, defender_id
                continue
            if incident:
                losses = {driver_id: float(self.simulator.rng.uniform(1, 3)),
                          defender_id: float(self.simulator.rng.uniform(0.5, 2))}
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
        previous_retired = (self.leader_id is not None
                            and self.states[self.leader_id].status == DriverStatus.DNF)
        leading = (self.timeline.chequered_time is None
                   and (pending.lap > active_distance
                        or (previous_retired and pending.lap >= active_distance)))
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
        else:
            self.order.append(driver_id)
        self._positions()
        if leading:
            self.leader_id = driver_id
            self._leader_interval(state, now, pending)
        return True

    def _positions(self):
        ordered = sorted(self.states.values(), key=lambda state: (
            -state.laps_completed, state.total_time, state.position,
        ))
        for position, state in enumerate(ordered, 1):
            state.position = position

    def _leader_interval(self, leader, now, pending):
        control = self.simulator.event_manager
        # Leadership can pass to a lapped survivor. Race-control cadence still
        # advances once per leading interval rather than replaying its old laps.
        self.control_intervals += 1
        events = control.process_lap(self.control_intervals, [], {}, self.track, pending.weather,
                                     incidents_this_lap=self.incidents)
        self.incidents = 0
        neutral = pending.neutralized or any(event.event_type in (
            EventType.SAFETY_CAR, EventType.VIRTUAL_SAFETY_CAR, EventType.RED_FLAG,
        ) for event in events)
        self.green_streak = 0 if neutral else self.green_streak + 1
        self.has_two_green |= self.green_streak >= 2
        red = any(event.event_type == EventType.RED_FLAG for event in events)
        if red:
            self.free_refits.update(state.driver.id for state in self.states.values()
                                    if state.status == DriverStatus.RACING)
            control.end_red_flag()  # No invented wall-time pause or clock rewind.
        if (self.timeline.chequered_time is None
                and any(state.status == DriverStatus.RACING for state in self.states.values())):
            self.weather = self.weather.evolve(self.simulator.rng)

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
                state.total_time, state.total_time - winner.total_time if winner else 0,
                state.pit_stops, self.fastest.get(state.driver.id, 0), state.status,
                dnf_reason=state.dnf_reason, strategy=list(state.tire_compound_history),
                laps_completed=state.laps_completed, classified=classified,
                pit_laps=list(state.pit_laps),
                race_time_limited=(winner is not None
                                   and self.timeline.final_lap < self.track.total_laps),
                points_awarded=points_for_classification(
                    position, classified, winner_laps, self.track.total_laps, self.has_two_green,
                ),
            ))
        return results


def simulate_chronological_race(simulator, drivers, cars, track, weather, starting_grid,
                                *, starting_tires=None):
    """Run the experimental engine explicitly; production dispatch is unchanged."""
    return ChronologicalRace(simulator).run(drivers, cars, track, weather, starting_grid,
                                             starting_tires=starting_tires)
