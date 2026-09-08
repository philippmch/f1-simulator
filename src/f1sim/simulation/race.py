"""Race simulation engine."""

from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.events import EventManager, EventType, RaceEvent
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.opening_strategy import dry_opening_policy_costs, opening_policy_costs
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.pit_strategy import expected_stationary_time, plan_dry_stop
from f1sim.simulation.race_points import points_for_classification
from f1sim.simulation.race_timing import RaceFinishClock, forecast_final_lap
from f1sim.simulation.rain_strategy import plan_rain_stop
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot
from f1sim.simulation.validation import validate_unique_ids
from f1sim.simulation.weather_strategy import weather_stop_costs


class DriverStatus(str, Enum):
    """Driver race status."""

    RACING = "racing"
    FINISHED = "finished"
    DNF = "dnf"


class TeamStrategyArchetype(str, Enum):
    """High-level team race strategy behavior."""

    AGGRESSIVE = "aggressive"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@dataclass
class DriverRaceState:
    """Tracks a driver's state during the race."""

    driver: Driver
    car: Car
    position: int
    total_time: float = 0.0
    current_tire: Tire = field(default_factory=lambda: TIRE_COMPOUNDS[TireCompound.MEDIUM])
    tire_laps: int = 0
    pit_stops: int = 0
    pit_laps: list[int] = field(default_factory=list)
    last_lap_time: float = 0.0
    status: DriverStatus = DriverStatus.RACING
    dnf_reason: str | None = None
    strategy_archetype: TeamStrategyArchetype = TeamStrategyArchetype.BALANCED
    planned_pit_laps: list[int] = field(default_factory=list)
    pit_plan_options: list[list[int]] = field(default_factory=list)
    active_pit_plan_index: int = 0
    # Actual compounds used, in stint order.  This is deliberately state on
    # the race rather than a post-hoc strategy guess.
    tire_compound_history: list[str] = field(default_factory=list)
    force_pit_next_lap: bool = False
    # One-lap proposal; execution rechecks weather before honoring it.
    dry_pit_proposal: tuple[int, TireCompound] | None = None
    # 2026 Overtake Mode energy store.  The store is bounded and deliberately
    # kept on each race state so a driver cannot deploy on every lap forever.
    overtake_mode_energy: float = 1.0
    overtake_mode_deployments: int = 0
    overtake_mode_active_lap: bool = False
    laps_completed: int = 0
    last_crossing_position: int | None = None
    pit_stop_details: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Seed tyre history from the driver's actual starting set."""
        if not self.tire_compound_history:
            self.tire_compound_history.append(self.current_tire.compound.value)


@dataclass
class RaceResult:
    """Final race result for a driver."""

    driver_id: str
    driver_name: str
    team: str
    position: int
    total_time: float
    gap_to_leader: float
    pit_stops: int
    fastest_lap: float
    status: DriverStatus
    dnf_reason: str | None = None
    strategy: list[str] = field(default_factory=list)  # List of compounds used
    laps_completed: int | None = None
    classified: bool | None = None
    pit_laps: list[int] | None = None
    race_time_limited: bool = False
    points_awarded: int | None = None
    pit_stop_details: list[dict] | None = None


def result_is_classified(result: RaceResult) -> bool:
    """Classification eligibility, preserving legacy finished-result callers."""
    classified = getattr(result, "classified", None)
    if isinstance(classified, bool):
        return classified
    return getattr(result.status, "value", result.status) == DriverStatus.FINISHED.value


@dataclass
class RaceState:
    """Complete race state."""

    lap: int
    total_laps: int
    weather: Weather
    driver_states: list[DriverRaceState]
    events: list[RaceEvent]
    safety_car_active: bool = False
    fastest_lap: float = float("inf")
    fastest_lap_driver: str | None = None


class RaceSimulator:
    """Simulates a full F1 race."""

    # A deployment is a meaningful burst rather than a free per-lap bonus.
    # Green running recharges gradually; neutralized laps recharge faster
    # while the field is constrained and no deployment can be made.
    OVERTAKE_MODE_DEPLOYMENT_COST = 0.35
    OVERTAKE_MODE_GREEN_RECHARGE = 0.04
    OVERTAKE_MODE_NEUTRAL_RECHARGE = 0.12

    def __init__(
        self,
        rng: np.random.Generator | None = None,
        strategy_tuning: dict[str, float] | None = None,
        strategy_profiles: dict[str, dict[str, float]] | None = None,
    ):
        """Initialize race simulator.

        Args:
            rng: Random number generator
            strategy_tuning: Optional strategy threshold overrides
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.weather_history: list[dict] = []
        self.lap_simulator = LapSimulator(rng=self.rng)
        self.overtaking_model = OvertakingModel(rng=self.rng)
        self.event_manager = EventManager(rng=self.rng)
        self.strategy_tuning = {
            "conservative_switch_gap": 2.0,
            "conservative_switch_race_progress": 0.4,
            "undercut_min_gap": 0.4,
            "undercut_max_gap": 2.2,
            "aggressive_traffic_gap": 1.2,
            "pit_prob_min": 0.03,
            "pit_prob_max": 0.92,
        }
        if strategy_tuning:
            self.strategy_tuning.update(strategy_tuning)

        self.strategy_profiles: dict[TeamStrategyArchetype, dict[str, float]] = {
            TeamStrategyArchetype.AGGRESSIVE: {
                "medium_prob": 0.65,
                "sprint_soft_prob": 0.95,
                "long_stint_threshold": 18,
            },
            TeamStrategyArchetype.BALANCED: {
                "medium_prob": 0.8,
                "sprint_soft_prob": 0.85,
                "long_stint_threshold": 20,
            },
            TeamStrategyArchetype.CONSERVATIVE: {
                "medium_prob": 0.9,
                "sprint_soft_prob": 0.7,
                "long_stint_threshold": 22,
            },
        }
        if strategy_profiles:
            for key, profile in strategy_profiles.items():
                try:
                    archetype = TeamStrategyArchetype(key)
                except ValueError:
                    continue
                self.strategy_profiles[archetype].update(profile)

    def _record_weather(self, lap: int, weather: Weather) -> None:
        """Record the shared race weather without retaining mutable model references."""
        self.weather_history.append({
            "lap": int(lap), "condition": weather.condition.value,
            "rain_intensity": float(weather.rain_intensity),
            "track_wetness": float(weather.track_wetness),
        })

    def simulate_race(
        self,
        drivers: list[Driver],
        cars: dict[str, Car],
        track: Track,
        weather: Weather,
        starting_grid: list[str],
        starting_tires: dict[str, TireCompound] | None = None,
    ) -> list[RaceResult]:
        """Simulate a complete race.

        Args:
            drivers: All drivers
            cars: Dictionary of cars by team_id
            track: Circuit
            weather: Initial weather
            starting_grid: Driver IDs in starting order
            starting_tires: Optional starting tire compounds per driver

        Returns:
            List of RaceResult sorted by finishing position
        """
        validate_unique_ids((driver.id for driver in drivers), "drivers")
        validate_unique_ids(starting_grid, "starting_grid")
        # Reset mutable driver state as well as event state.  Monte Carlo
        # workers may intentionally reuse model instances between simulations.
        for driver in drivers:
            driver.reset_race_state()

        # Reset event manager
        self.event_manager.reset()
        self.weather_history = []

        # Keep the caller's supplied weather as the immutable lap-one
        # snapshot.  Weather evolution is applied only after a lap has been
        # simulated, so every driver and race-control decision on lap one
        # sees exactly the conditions the caller provided.
        current_weather = weather.model_copy(deep=True)

        # Initialize driver states based on starting grid
        driver_map = {d.id: d for d in drivers}
        states: list[DriverRaceState] = []

        for pos, driver_id in enumerate(starting_grid, 1):
            driver = driver_map.get(driver_id)
            if driver is None:
                continue

            car = cars.get(driver.team_id)
            if car is None:
                continue

            strategy = self._infer_team_strategy(car, track)

            # Starting tyres are a strategy decision, not a hidden grid-
            # position rule.  Explicit caller overrides remain authoritative;
            # otherwise choose from the seeded RNG using the team strategy,
            # track stress and initial weather.
            if starting_tires and driver_id in starting_tires:
                tire_compound = starting_tires[driver_id]
            else:
                tire_compound = self._choose_starting_compound(
                    strategy,
                    track,
                    current_weather,
                    driver,
                    car,
                )

            pit_plans = self._plan_pit_lap_options(strategy, track)
            states.append(
                DriverRaceState(
                    driver=driver,
                    car=car,
                    position=pos,
                    current_tire=TIRE_COMPOUNDS[tire_compound].model_copy(deep=True),
                    strategy_archetype=strategy,
                    planned_pit_laps=pit_plans[0],
                    pit_plan_options=pit_plans,
                    active_pit_plan_index=0,
                )
            )

        # An empty or unusable grid has no racing laps or race-control events.
        if not states:
            return []

        # Track fastest laps
        fastest_laps: dict[str, float] = {}

        # Simulate each lap
        finish_clock = RaceFinishClock(track.total_laps)
        final_lap = finish_clock.final_lap
        observed_running_pace: dict[str, float] = {}
        consecutive_green_laps = 0
        has_two_green_laps = False
        for lap in range(1, track.total_laps + 1):
            self._record_weather(lap, current_weather)
            leader = min((state for state in states if state.status == DriverStatus.RACING),
                         key=lambda state: state.position)
            planning_final_lap = forecast_final_lap(
                final_lap, lap - 1, leader.total_time,
                observed_running_pace.get(leader.driver.id), finish_clock.time_limit_seconds,
                self.event_manager.get_lap_time_modifier(),
            )
            planning_track = (track if planning_final_lap == track.total_laps else
                              track.model_copy(update={"total_laps": planning_final_lap}))
            # Snapshot race-control state once.  SC/VSC/red-flag transitions
            # are resolved after this lap's running; using immutable values
            # prevents a neutralization ending during process_lap from
            # changing mode eligibility for only part of the lap.
            lap_active_aero_enabled = self.event_manager.is_active_aero_allowed()
            lap_overtake_mode_allowed = self.event_manager.is_overtake_mode_allowed(
                lap,
                current_weather,
            )
            lap_started_neutralized = not lap_active_aero_enabled
            lap_restart = self.event_manager.is_restart_lap(lap)

            fastest_laps_before_lap = dict(fastest_laps)
            # Timing and track positions must describe the same completed
            # lap for every driver. Shallow copies freeze these scalar fields
            # while the live states accumulate this lap's running and pit loss.
            lap_start_states = [replace(state) for state in states]
            lap_start_times = {state.driver.id: state.total_time for state in lap_start_states}
            lap_start_gaps = {
                state.driver.id: self._get_gap_to_car_ahead(state, lap_start_states)
                for state in lap_start_states if state.status == DriverStatus.RACING
            }

            # Simulate lap for each driver
            lap_times: dict[str, float] = {}
            incidents_this_lap = 0
            drivers_pitting: list[DriverRaceState] = []
            material_penalty_ids: set[str] = set()

            # Decide every stop before sampling service or running laps. Both
            # pit-box queues use frozen lap-start clocks as arrival proxies.
            drivers_pitting = self._process_pit_stops(
                states, lap_start_states, planning_track, current_weather, lap,
                **({"physical_total_laps": track.total_laps}
                   if planning_final_lap < track.total_laps else {}),
            )
            pitting_ids = {state.driver.id for state in drivers_pitting}
            pit_lap_losses = {
                state.driver.id: state.total_time - lap_start_times[state.driver.id]
                for state in drivers_pitting
            }
            traffic_gaps = lap_start_gaps
            if pitting_ids:
                # Merge only copies: every car sees the same rejoin traffic,
                # while detection and the actual completed-lap merge retain
                # their existing timing. This also frees followers of pitters.
                traffic_states = [replace(state) for state in states]
                self._handle_pit_batch_position_changes(
                    [state for state in traffic_states if state.driver.id in pitting_ids],
                    traffic_states,
                )
                traffic_gaps = {
                    state.driver.id: self._get_gap_to_car_ahead(state, traffic_states)
                    for state in traffic_states if state.status == DriverStatus.RACING
                }
            for state in states:
                if state.status != DriverStatus.RACING:
                    continue
                state.overtake_mode_active_lap = False
                gap_ahead = lap_start_gaps[state.driver.id]
                should_pit = state.driver.id in pitting_ids

                # Calculate lap time
                if not should_pit:
                    state.overtake_mode_active_lap = (
                        self._deploy_overtake_mode_if_eligible(
                            state,
                            track,
                            gap_ahead,
                            lap_overtake_mode_allowed,
                        )
                    )

                lap_time = self.lap_simulator.calculate_lap_time(
                    driver=state.driver,
                    car=state.car,
                    track=track,
                    tire=state.current_tire,
                    weather=current_weather,
                    lap_number=lap,
                    total_laps=track.total_laps,
                    gap_to_car_ahead=traffic_gaps[state.driver.id],
                    active_aero_enabled=lap_active_aero_enabled,
                    overtake_mode_active=state.overtake_mode_active_lap,
                )

                observed_running_pace[state.driver.id] = lap_time

                # Apply safety car modifier
                lap_time *= self.event_manager.get_lap_time_modifier()

                state.total_time += lap_time
                # Pit loss already entered total_time before running this lap.
                # Include it in the recorded lap too, without charging it twice
                # or multiplying stationary/queue time by the SC lap modifier.
                state.last_lap_time = lap_time + pit_lap_losses.get(state.driver.id, 0.0)
                state.tire_laps += 1
                state.driver.current_tire_laps = state.tire_laps

                lap_times[state.driver.id] = lap_time

            # Resolve the entire pit batch by actual post-stop clocks on every
            # track. Pit-lane position changes do not require on-track passing.
            self._handle_pit_batch_position_changes(drivers_pitting, states)

            # Overtakes happen on the racing lap before race-control events
            # are resolved.  Their incidents therefore feed SC/VSC/red-flag
            # decisions for this same lap.
            if (
                not self.event_manager.safety_car_active
                and not self.event_manager.vsc_active
                and not self.event_manager.red_flag_active
            ):
                overtake_incidents = self._process_overtakes(
                    states,
                    track,
                    current_weather,
                    restart_lap=lap_restart,
                    lap=lap,
                    overtake_mode_allowed=lap_overtake_mode_allowed,
                )
                incidents_this_lap += overtake_incidents

            # Resolve random/mechanical incidents and race-control events
            # after all on-track incidents have been collected.
            active_drivers = [s.driver for s in states if s.status == DriverStatus.RACING]
            lap_events = self.event_manager.process_lap(
                lap=lap,
                drivers=active_drivers,
                cars=cars,
                track=track,
                weather=current_weather,
                incidents_this_lap=incidents_this_lap,
            )

            # Update driver states based on events
            for event in lap_events:
                for driver_id in event.drivers_involved:
                    for state in states:
                        if state.driver.id != driver_id:
                            continue

                        # Events are sampled after pace calculation but describe
                        # incidents during this lap. Survivors carry their time
                        # loss; a retirement does not complete this crossing.
                        if event.time_loss_seconds > 0.0:
                            state.total_time += event.time_loss_seconds
                            state.last_lap_time += event.time_loss_seconds
                            material_penalty_ids.add(driver_id)

                        if event.forces_pit_stop and state.status == DriverStatus.RACING:
                            state.force_pit_next_lap = True

                        if state.driver.dnf:
                            state.status = DriverStatus.DNF
                            state.dnf_reason = state.driver.dnf_reason

            # Restore a retired car's last completed crossing. Neither the
            # sampled failure lap nor its service/incident time is race distance.
            for state, before in zip(states, lap_start_states):
                if state.status == DriverStatus.DNF and before.status == DriverStatus.RACING:
                    state.total_time = before.total_time
                    state.last_lap_time = before.last_lap_time

            # A post-lap incident can make the car that was physically ahead
            # slower on elapsed time than a car behind it.  Reclassify only
            # drivers carrying a material penalty; ordinary on-track
            # overtakes remain position-authoritative.
            if material_penalty_ids:
                self._reorder_positions_after_material_penalties(
                    states,
                    material_penalty_ids,
                )

            # Lap pace is unconstrained until passing and incident outcomes
            # establish the physical order. A car that remains behind must
            # spend any excess pace waiting, rather than banking a faster
            # cumulative clock for a later lap or the final classification.
            self._reconcile_racing_times(states)

            # Check if safety car was just deployed - bunch up the field
            sc_deployed_this_lap = any(e.event_type == EventType.SAFETY_CAR for e in lap_events)
            if sc_deployed_this_lap:
                # Classify any same-lap incident consequences while elapsed
                # times still contain their penalties.  Bunching then resets
                # clean gaps without erasing the victim's position loss.
                self._classify_positions_before_neutralization(
                    states,
                    material_penalty_ids,
                )
                self.event_manager.bunch_field(states)

            # Check if red flag was just deployed
            red_flag_deployed_this_lap = any(e.event_type == EventType.RED_FLAG for e in lap_events)
            if red_flag_deployed_this_lap:
                # Handle red flag: bunch field and allow tire changes
                self._classify_positions_before_neutralization(
                    states,
                    material_penalty_ids,
                )
                self._handle_red_flag_stop(states, current_weather, track, lap, defer_tire_fit=True)

            # Commit fastest laps only after all on-track incidents and race
            # control consequences for this lap have been applied.  In
            # particular, an incident lap cannot retain the clean pre-penalty
            # value that was measured before the event was detected.
            for state in states:
                if state.driver.id in lap_times and state.status == DriverStatus.RACING:
                    fastest_laps[state.driver.id] = min(
                        fastest_laps_before_lap.get(state.driver.id, float("inf")),
                        state.last_lap_time,
                    )

            # Update positions
            self._update_positions(states)
            for state in states:
                if state.status == DriverStatus.RACING:
                    state.laps_completed += 1
                    state.last_crossing_position = state.position

            # Overtake Mode is unavailable while neutralized, but the store
            # can recharge.  Include a lap that started under neutralization
            # even if the event countdown expires during process_lap.
            self._recharge_overtake_mode_energy(
                states,
                neutralized=(
                    lap_started_neutralized
                    or self.event_manager.safety_car_active
                    or self.event_manager.vsc_active
                    or self.event_manager.red_flag_active
                    or red_flag_deployed_this_lap
                ),
            )

            # Preserve the final retirement lap's consequences, then stop.
            # Later scheduled laps cannot produce events with no running cars.
            if not any(state.status == DriverStatus.RACING for state in states):
                break

            # A deployment during the lap disqualifies the whole lap, even
            # when the flag has already ended by this point (notably red flags).
            lap_was_neutralized = (
                lap_started_neutralized
                or self.event_manager.safety_car_active
                or self.event_manager.vsc_active
                or self.event_manager.red_flag_active
                or any(event.event_type in (
                    EventType.SAFETY_CAR, EventType.VIRTUAL_SAFETY_CAR, EventType.RED_FLAG,
                ) for event in lap_events)
            )
            consecutive_green_laps = 0 if lap_was_neutralized else consecutive_green_laps + 1
            has_two_green_laps |= consecutive_green_laps >= 2

            leader = min((state for state in states if state.status == DriverStatus.RACING),
                         key=lambda state: state.position)
            final_lap = finish_clock.observe_leader_crossing(lap, leader.total_time)
            if lap >= final_lap:
                break

            # The initial weather snapshot was used unchanged on lap one.
            # Evolve only when another lap will actually consume the result.
            if lap < final_lap:
                current_weather = current_weather.evolve(self.rng)
                if red_flag_deployed_this_lap:
                    restart_final_lap = forecast_final_lap(
                        final_lap, lap, leader.total_time,
                        observed_running_pace.get(leader.driver.id),
                        finish_clock.time_limit_seconds,
                    )
                    restart_track = (track if restart_final_lap == track.total_laps else
                                     track.model_copy(update={"total_laps": restart_final_lap}))
                    self._fit_red_flag_tires(
                        states, current_weather, restart_track, lap,
                        **({"physical_total_laps": track.total_laps}
                           if restart_final_lap < track.total_laps else {}),
                    )

        # Mark finished drivers
        for state in states:
            if state.status == DriverStatus.RACING:
                state.status = DriverStatus.FINISHED

        # Build results - use track position (state.position), not total time
        # DNF drivers go after finishers
        sorted_states = sorted(states, key=lambda s: (s.status == DriverStatus.DNF, s.position))

        # Find leader time for gap calculation
        leader_time = (
            sorted_states[0].total_time
            if sorted_states and sorted_states[0].status == DriverStatus.FINISHED
            else 0
        )

        winner_laps = (
            sorted_states[0].laps_completed
            if sorted_states and sorted_states[0].status == DriverStatus.FINISHED
            else None
        )
        # Integer arithmetic implements the 90% threshold rounded down.
        classification_minimum = winner_laps * 9 // 10 if winner_laps is not None else None
        results = []
        for state in sorted_states:
            # Emit the compounds actually used by this state, including
            # weather/red-flag changes and incident-forced stops.
            strategy = list(state.tire_compound_history)
            classified = (
                classification_minimum is not None
                and state.laps_completed > 0
                and state.laps_completed >= classification_minimum
            )

            results.append(
                RaceResult(
                    driver_id=state.driver.id,
                    driver_name=state.driver.name,
                    team=state.car.team_name,
                    position=state.position,
                    total_time=state.total_time,
                    gap_to_leader=state.total_time - leader_time
                    if state.status == DriverStatus.FINISHED
                    else 0,
                    pit_stops=state.pit_stops,
                    fastest_lap=fastest_laps.get(state.driver.id, 0),
                    status=state.status,
                    dnf_reason=state.dnf_reason,
                    strategy=strategy,
                    laps_completed=state.laps_completed,
                    pit_laps=list(state.pit_laps),
                    pit_stop_details=[dict(stop) for stop in state.pit_stop_details],
                    race_time_limited=winner_laps is not None and final_lap < track.total_laps,
                    classified=classified,
                    points_awarded=points_for_classification(
                        state.position, classified, winner_laps or 0, track.total_laps,
                        has_two_green_laps,
                    ),
                )
            )

        return results

    def _infer_team_strategy(
        self,
        car: Car,
        track: Track,
    ) -> TeamStrategyArchetype:
        """Infer default team strategy archetype from car/track profile."""
        # Fragile or high-deg packages trend conservative.
        if car.reliability < 0.9 or car.tire_degradation_factor > 1.08:
            return TeamStrategyArchetype.CONSERVATIVE

        # Fast cars at difficult overtaking tracks tend to undercut aggressively.
        if car.base_pace > 0.82 and track.overtake_difficulty > 0.55:
            return TeamStrategyArchetype.AGGRESSIVE

        return TeamStrategyArchetype.BALANCED

    def _choose_starting_compound(
        self,
        strategy: TeamStrategyArchetype,
        track: Track,
        weather: Weather,
        driver: Driver | None = None,
        car: Car | None = None,
    ) -> TireCompound:
        """Choose a plausible opening tyre set for the current conditions.

        Current regulations allow drivers to choose their starting compound;
        qualifying position no longer determines a hidden medium/soft split.
        Wet starts are deterministic because using the wrong tyre is unsafe.
        Dry starts use seeded probabilities shaped by team strategy, tyre
        stress, race length and overtaking difficulty. With driver/car context,
        complete-race costs restrict those probabilities to optimal opening sets.
        """
        weather_compound = self._choose_weather_compound(weather)
        if weather_compound is not None:
            return weather_compound
        if (
            (weather.rain_intensity >= 0.2
             or weather.condition in {
                 WeatherCondition.LIGHT_RAIN,
                 WeatherCondition.HEAVY_RAIN,
             })
            and self._check_tire_weather_mismatch(
                TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], weather,
            ) != "critical"
        ):
            if driver is not None and car is not None:
                costs = opening_policy_costs(
                    driver, car, track, weather, strategy,
                    self.strategy_tuning, self.strategy_profiles,
                )
                return min(costs, key=lambda candidate: candidate[1])[0]
            return TireCompound.INTERMEDIATE

        # A condition label alone must not fit a set that our own mismatch
        # rule would immediately replace at a paid stop before lap one.
        # Weights represent realistic dry-grid variation rather than a
        # position-based assignment.  Aggressive teams bias toward a short
        # soft opening stint; conservative teams protect the long race.
        strategy_weights = {
            TeamStrategyArchetype.AGGRESSIVE: np.array([0.48, 0.42, 0.10]),
            TeamStrategyArchetype.BALANCED: np.array([0.35, 0.50, 0.15]),
            TeamStrategyArchetype.CONSERVATIVE: np.array([0.20, 0.55, 0.25]),
        }
        weights = strategy_weights[strategy].astype(float, copy=True)

        # Tyre stress makes the durable compounds more attractive.  Shorter
        # races and difficult-to-pass tracks reward track position and hence
        # a little more fresh soft-tyre grip.
        stress = float(np.clip(track.tire_stress, 0.0, 1.0))
        stress_shift = (stress - 0.5) * 0.24
        weights[0] -= stress_shift
        weights[2] += stress_shift
        if track.total_laps < 45:
            weights[0] += 0.05
            weights[1] -= 0.03
            weights[2] -= 0.02
        if track.overtake_difficulty > 0.7:
            weights[0] += 0.04
            weights[1] -= 0.03
            weights[2] -= 0.01

        weights = np.clip(weights, 0.02, None)
        if (driver is not None and car is not None and track.total_laps > 1
                and weather.track_wetness < 0.08 and weather.rain_intensity < 0.15):
            scores = dry_opening_policy_costs(
                driver, car, track, weather, strategy,
                self.strategy_tuning, self.strategy_profiles,
            )
            best = min(score for _, score in scores)
            if np.isfinite(best.mean_time):
                # Compare the actual policy's completed distance first: a
                # shorter timed race must not win merely by ending sooner.
                weights *= np.asarray([
                    score.negative_mean_laps == best.negative_mean_laps
                    and score.mean_time <= best.mean_time + 1e-9
                    for _, score in scores
                ])
        weights /= weights.sum()
        # NumPy converts Enum objects to truncated unicode labels when it
        # builds an object array (for example ``"TireCo"``).  Sample the
        # stable string values instead, then reconstruct the enum.
        compounds = [
            TireCompound.SOFT.value,
            TireCompound.MEDIUM.value,
            TireCompound.HARD.value,
        ]
        selected = self.rng.choice(compounds, p=weights)
        if isinstance(selected, TireCompound):
            return selected
        return TireCompound(str(selected))

    def _should_switch_conservative_to_balanced(
        self,
        state: DriverRaceState,
        lap: int,
        track: Track,
        gap_ahead: float | None,
    ) -> bool:
        """Decide whether conservative strategy should switch to balanced."""
        if state.strategy_archetype != TeamStrategyArchetype.CONSERVATIVE:
            return False
        if state.position <= 6:
            return False
        if not self._has_strategy_traffic(
            gap_ahead, lap, self.strategy_tuning["conservative_switch_gap"],
        ):
            return False

        threshold = self.strategy_tuning["conservative_switch_race_progress"]
        return lap > int(track.total_laps * threshold)

    def _has_strategy_traffic(
        self, gap_ahead: float | None, lap: int, maximum_gap: float = 2.0,
    ) -> bool:
        """Close green-running traffic, excluding temporary restart bunching."""
        return (
            gap_ahead is not None
            and 0.0 <= gap_ahead <= maximum_gap
            and self.lap_simulator.traffic_pace_contribution(gap_ahead) > 0.0
            and not self.event_manager.safety_car_active
            and not self.event_manager.vsc_active
            and not self.event_manager.red_flag_active
            and not self.event_manager.is_restart_lap(lap)
        )

    def _plan_pit_lap_options(
        self,
        strategy: TeamStrategyArchetype,
        track: Track,
    ) -> list[list[int]]:
        """Generate multiple pit-plan options by strategy archetype."""
        laps = track.total_laps
        if strategy == TeamStrategyArchetype.AGGRESSIVE:
            return [
                [int(laps * 0.28), int(laps * 0.55), int(laps * 0.78)],
                [int(laps * 0.32), int(laps * 0.62)],
            ]
        if strategy == TeamStrategyArchetype.CONSERVATIVE:
            return [
                [int(laps * 0.45), int(laps * 0.78)],
                [int(laps * 0.52)],
            ]
        return [
            [int(laps * 0.35), int(laps * 0.7)],
            [int(laps * 0.42), int(laps * 0.78)],
        ]

    def _select_active_pit_plan(
        self,
        state: DriverRaceState,
        weather: Weather | None,
        lap: int,
        gap_ahead: float | None,
        *,
        track: Track,
    ) -> list[int]:
        """Select active pit plan based on race context."""
        if not state.pit_plan_options:
            return []

        # In wet/changing conditions, prefer conservative fallback plan.
        if weather is not None and weather.track_wetness > 0.3:
            state.active_pit_plan_index = min(1, len(state.pit_plan_options) - 1)
        # If stuck in traffic in race second half, prefer aggressive plan.
        elif (
            self._has_strategy_traffic(gap_ahead, lap)
            and lap > track.total_laps / 2
            and state.strategy_archetype != TeamStrategyArchetype.CONSERVATIVE
        ):
            state.active_pit_plan_index = 0

        state.active_pit_plan_index = min(
            state.active_pit_plan_index,
            len(state.pit_plan_options) - 1,
        )
        return state.pit_plan_options[state.active_pit_plan_index]

    def _process_pit_stops(
        self,
        states: list[DriverRaceState],
        lap_start_states: list[DriverRaceState],
        track: Track,
        weather: Weather,
        lap: int,
        physical_total_laps: int | None = None,
    ) -> list[DriverRaceState]:
        """Reserve then serve each constructor's box in frozen arrival order.

        A lap-start race clock approximates arrival at the shared box. Lane
        transit is outside box occupancy. Reservations use expected service;
        random service is sampled only after all decisions are committed.
        Local queues cannot leak across laps or simulator reuse, including
        race-control clock compression and free suspension tyre changes.
        """
        arrivals = {state.driver.id: (state.total_time, state.position)
                    for state in lap_start_states}
        ordered = sorted(
            (state for state in states if state.status == DriverStatus.RACING),
            key=lambda state: (*arrivals[state.driver.id], state.driver.id),
        )
        expected_releases: dict[str, float] = {}
        pitting = []
        for state in ordered:
            arrival = arrivals[state.driver.id][0]
            team = state.car.team_id
            delay = max(0.0, expected_releases.get(team, arrival) - arrival)
            if state.force_pit_next_lap or self._should_pit(
                state, lap_start_states, track, lap,
                self.event_manager.is_pit_window_open(), weather=weather,
                additional_current_stop_cost=delay,
                **({"physical_total_laps": physical_total_laps}
                   if physical_total_laps is not None else {}),
            ):
                pitting.append(state)
                expected_releases[team] = arrival + delay + expected_stationary_time(state.car)

        actual_releases: dict[str, float] = {}
        for state in pitting:
            state.total_time += self._execute_pit_stop(
                state, track, weather, current_lap=lap,
                pit_box_releases=actual_releases,
                arrival_time=arrivals[state.driver.id][0],
                **({"physical_total_laps": physical_total_laps}
                   if physical_total_laps is not None else {}),
            )
            state.pit_stops += 1
            state.pit_laps.append(lap)
            state.force_pit_next_lap = False
        return pitting

    def _should_pit(
        self,
        state: DriverRaceState,
        all_states: list[DriverRaceState],
        track: Track,
        lap: int,
        pit_window_open: bool,
        weather: Weather | None = None,
        additional_current_stop_cost: float = 0.0,
        physical_total_laps: int | None = None,
        traffic_snapshot: StrategyTrafficSnapshot | None = None,
    ) -> bool:
        """Decide if driver should pit this lap."""
        state.dry_pit_proposal = None
        # CRITICAL: Force pit if tires are completely wrong for conditions
        if weather is not None:
            tire_mismatch = self._check_tire_weather_mismatch(state.current_tire, weather)
            if tire_mismatch == "critical":
                return True  # Must pit immediately
            elif tire_mismatch == "suboptimal":
                dry_rule_satisfied = self._stay_satisfies_tire_rule(state)
                if (
                    dry_rule_satisfied
                    and not self._weather_stop_can_pay(
                        state, track, weather, lap, additional_current_stop_cost,
                        traffic_possible=any(
                            other.status == DriverStatus.RACING
                            and other.driver.id != state.driver.id for other in all_states
                        ),
                        **({"physical_total_laps": physical_total_laps}
                           if physical_total_laps is not None else {}),
                    )
                ):
                    return False
                if self.rng.random() < 0.7:
                    return True  # Should pit soon

        if self.event_manager.red_flag_active:
            # Suspension tyre changes are handled separately without a lane loss.
            return False

        strategy = state.strategy_archetype

        gap_ahead = (self._get_gap_to_car_ahead(state, all_states)
                     if traffic_snapshot is None else traffic_snapshot.gap_ahead)
        gap_behind = (self._get_gap_to_car_behind(state, all_states)
                      if traffic_snapshot is None else traffic_snapshot.gap_behind)

        # Mid-race strategy switching trigger (conservative => balanced)
        # when following closely outside the top positions in green running.
        if self._should_switch_conservative_to_balanced(state, lap, track, gap_ahead):
            strategy = TeamStrategyArchetype.BALANCED
            state.strategy_archetype = strategy

        strategy_bias = {
            TeamStrategyArchetype.AGGRESSIVE: 0.12,
            TeamStrategyArchetype.BALANCED: 0.0,
            TeamStrategyArchetype.CONSERVATIVE: -0.1,
        }[strategy]

        clearly_dry = weather is None or (
            weather.track_wetness < 0.08 and weather.rain_intensity < 0.15
        )
        dry_planning = clearly_dry and state.current_tire.compound in {
            TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
        }
        # Dry optimization chooses how many stops pay for themselves; style
        # must not exclude a faster legal schedule before it is evaluated.
        max_stops = (self._dry_stop_budget(state, track) if dry_planning
                     else self._ordinary_stop_budget(state, track))
        if weather is not None and weather.track_wetness > 0.3:
            max_stops = max(max_stops, 4)  # Allow more stops in changing conditions

        # The dry-race regulation is about two distinct slick compounds,
        # not simply a stop count.  Keep one additional stop available when
        # an earlier stop repeated the same compound, so a short race cannot
        # exhaust its ordinary stop budget before satisfying the rule.
        dry_rule_required = not self._stay_satisfies_tire_rule(state)
        if dry_rule_required:
            max_stops = max(max_stops, state.pit_stops + 1)
        if state.pit_stops >= max_stops:
            return False

        # Stops precede running, so the final lap can still use a distinct set.
        # Let earlier dry decisions compare costs instead of forcing a slower
        # penultimate-lap correction.
        if (
            lap >= max(2, track.total_laps)
            and dry_rule_required
        ):
            return True

        # Elective stops need at least one lap on the starting set. Weather
        # and mandatory safeguards above still take priority.
        if lap <= 1:
            return False

        if dry_planning:
            tire_multiplier = (self.lap_simulator.weather_pace_multiplier(
                state.driver, state.car, weather,
            ) if weather is not None else 1.0)
            traffic_cost = 0.0
            if not (
                self.event_manager.safety_car_active or self.event_manager.vsc_active
                or self.event_manager.red_flag_active
            ):
                traffic_cost = (
                    self._pit_rejoin_traffic_cost(
                        state, all_states, track, additional_current_stop_cost,
                    ) if traffic_snapshot is None else traffic_snapshot.rejoin_traffic_cost
                ) * tire_multiplier
            decision = plan_dry_stop(
                state.driver, state.car, track, state.current_tire, state.tire_laps,
                track.total_laps - lap + 1,
                min(3, max(0, max_stops - state.pit_stops)),
                self._used_slick_compounds(state), self._has_used_wet_compound(state),
                self._pit_lane_factor(), additional_current_stop_cost + traffic_cost,
                current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
                tire_pace_multiplier=tire_multiplier,
                physical_total_laps=physical_total_laps,
                active_aero_enabled=self.event_manager.is_active_aero_allowed(),
            )
            timing_bias = {
                TeamStrategyArchetype.AGGRESSIVE: 0.1,
                TeamStrategyArchetype.BALANCED: 0.0,
                TeamStrategyArchetype.CONSERVATIVE: -0.1,
            }[strategy]
            if decision.should_pit(timing_bias):
                state.dry_pit_proposal = (lap, decision.compound)
                return True
            return False

        if weather is not None and self._rain_stint_can_be_planned(state, track, weather, lap):
            traffic_cost = 0.0
            if not (self.event_manager.safety_car_active or self.event_manager.vsc_active):
                traffic_cost = (
                    self._pit_rejoin_traffic_cost(
                        state, all_states, track, additional_current_stop_cost,
                    ) if traffic_snapshot is None else traffic_snapshot.rejoin_traffic_cost
                ) * self.lap_simulator.weather_pace_multiplier(state.driver, state.car, weather)
            return plan_rain_stop(
                state.driver, state.car, track, weather, state.current_tire,
                state.tire_laps, lap, max_stops - state.pit_stops,
                pit_lane_factor=self._pit_lane_factor(),
                additional_current_stop_cost=additional_current_stop_cost + traffic_cost,
                current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
                active_aero_enabled=self.event_manager.is_active_aero_allowed(),
                physical_total_laps=physical_total_laps,
            ).should_pit()

        # Changing compound requirements retain the reactive fallback windows.
        if lap <= 5 or lap >= track.total_laps - 5:
            return False

        def wet_stop_can_pay() -> bool:
            # A window proposes a stop; even an optimistic fresh-set plan
            # must recover its paid loss before we accept that proposal.
            if weather is None:
                return True  # Legacy callers supplied no surface to project.
            return self._weather_stop_can_pay(
                state, track, weather, lap, additional_current_stop_cost,
                traffic_possible=any(
                    other.status == DriverStatus.RACING and other.driver.id != state.driver.id
                    for other in all_states
                ),
                **({"physical_total_laps": physical_total_laps}
                   if physical_total_laps is not None else {}),
            )

        # Pit window opportunity (under SC/VSC) - usually strong strategic value.
        if pit_window_open and state.tire_laps > 10 and state.pit_stops < max_stops:
            # If we have a free stop window (big gap behind), almost always take it.
            if gap_behind is not None and gap_behind > track.pit_lane_delta * 0.85:
                return wet_stop_can_pay()
            window_prob = np.clip(0.85 + strategy_bias, 0.55, 0.98)
            if self.rng.random() < window_prob:
                return wet_stop_can_pay()

        # Prefer explicit planned pit laps when available for current stint.
        active_plan = self._select_active_pit_plan(state, weather, lap, gap_ahead, track=track)
        state.planned_pit_laps = active_plan
        planned_lap = None
        if active_plan:
            if state.pit_stops >= len(active_plan):
                return False
            planned_lap = active_plan[state.pit_stops]
        elif state.pit_stops >= min(2, max_stops):
            # The generic schedule has at most two windows. A larger wet
            # stop allowance permits reactive stops, not repeated visits
            # to the already consumed second window. Weather and SC/VSC
            # opportunities have been evaluated above.
            return False

        # Calculate optimal pit windows for 1-stop or 2-stop strategy
        if planned_lap is not None:
            window_start = planned_lap - 3
            window_end = planned_lap + 4
        elif max_stops == 1:
            # Single stop around lap 40-60% of race
            optimal_lap = int(track.total_laps * 0.5)
            window_start = optimal_lap - 5
            window_end = optimal_lap + 5
        else:
            # Two stops: first around 33%, second around 66%
            if state.pit_stops == 0:
                optimal_lap = int(track.total_laps * 0.35)
            else:
                optimal_lap = int(track.total_laps * 0.7)
            window_start = optimal_lap - 3
            window_end = optimal_lap + 5

        # Position-based reluctance at hard-to-pass tracks
        position_reluctance = 0.0
        if track.overtake_difficulty > 0.7:
            position_reluctance = max(0, (10 - state.position) * 0.03) * track.overtake_difficulty

        if window_start <= lap <= window_end:
            laps_into_window = lap - window_start
            base_prob = 0.15 + laps_into_window * 0.1  # 15% to 65% over window
            adjusted_prob = max(0.05, base_prob - position_reluctance + strategy_bias)

            if planned_lap is not None:
                distance_to_plan = abs(lap - planned_lap)
                if distance_to_plan == 0:
                    adjusted_prob += 0.2
                elif distance_to_plan <= 1:
                    adjusted_prob += 0.1

            # Undercut trigger: close to car ahead and hard to overtake.
            if (
                gap_ahead is not None
                and self.strategy_tuning["undercut_min_gap"]
                <= gap_ahead
                <= self.strategy_tuning["undercut_max_gap"]
                and track.overtake_difficulty > 0.45
                and state.tire_laps >= 12
            ):
                adjusted_prob += 0.2

            # Overcut trigger: enough clean air behind to extend stint.
            if (
                gap_behind is not None
                and gap_behind > track.pit_lane_delta * 0.7
                and state.current_tire.grip_at_lap(state.tire_laps) > 0.68
            ):
                adjusted_prob -= 0.12

            # Dynamic mid-race strategy reaction to changing weather.
            if weather is not None and weather.track_wetness > 0.2:
                if strategy == TeamStrategyArchetype.CONSERVATIVE:
                    adjusted_prob += 0.12
                elif strategy == TeamStrategyArchetype.AGGRESSIVE:
                    adjusted_prob += 0.04

            # Push aggressive teams to attack earlier when stuck in traffic.
            if (
                strategy == TeamStrategyArchetype.AGGRESSIVE
                and gap_ahead is not None
                and gap_ahead < self.strategy_tuning["aggressive_traffic_gap"]
                and state.position > 1
            ):
                adjusted_prob += 0.08

            if self.rng.random() < np.clip(
                adjusted_prob,
                self.strategy_tuning["pit_prob_min"],
                self.strategy_tuning["pit_prob_max"],
            ):
                return wet_stop_can_pay()

        return False

    @staticmethod
    def _rain_stint_can_be_planned(
        state: DriverRaceState, track: Track, weather: Weather, lap: int,
    ) -> bool:
        compound = state.current_tire.compound
        if compound not in {TireCompound.INTERMEDIATE, TireCompound.WET}:
            return False
        surface = weather
        for _ in range(track.total_laps - lap + 1):
            if surface.fresh_rain_compound() != compound:
                return False
            surface = surface.project_surface()
        return True

    @staticmethod
    def _actually_used_compounds(state: DriverRaceState) -> set[TireCompound]:
        """Exclude only the current unrun fitted set, preserving earlier stints."""
        history = state.tire_compound_history
        if (state.tire_laps == 0 and history
                and history[-1] == state.current_tire.compound.value):
            history = history[:-1]
        used = set()
        for compound in history:
            try:
                used.add(TireCompound(compound))
            except (TypeError, ValueError):
                continue
        if state.tire_laps > 0:
            used.add(state.current_tire.compound)
        return used

    @classmethod
    def _has_used_wet_compound(cls, state: DriverRaceState) -> bool:
        return bool(cls._actually_used_compounds(state) & {
            TireCompound.INTERMEDIATE, TireCompound.WET,
        })

    @classmethod
    def _used_slick_compounds(cls, state: DriverRaceState) -> set[TireCompound]:
        return cls._actually_used_compounds(state) & {
            TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
        }

    @classmethod
    def _stay_satisfies_tire_rule(cls, state: DriverRaceState) -> bool:
        """A stay-out decision will run the fitted set on the upcoming lap."""
        prospective = cls._actually_used_compounds(state) | {state.current_tire.compound}
        return bool(prospective & {TireCompound.INTERMEDIATE, TireCompound.WET}) or len(
            prospective & {TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD}
        ) >= 2

    @staticmethod
    def _fit_tire(state: DriverRaceState, compound: TireCompound) -> None:
        """Replace an unrun fitting instead of recording a fictitious stint."""
        if (state.tire_laps == 0 and state.tire_compound_history
                and state.tire_compound_history[-1] == state.current_tire.compound.value):
            state.tire_compound_history[-1] = compound.value
        else:
            state.tire_compound_history.append(compound.value)
        state.current_tire = TIRE_COMPOUNDS[compound].model_copy(deep=True)
        state.tire_laps = 0
        state.driver.current_tire_laps = 0

    def _choose_distinct_dry_compound(
        self,
        state: DriverRaceState,
        track: Track,
        current_lap: int,
        weather: Weather | None = None,
        *, physical_total_laps: int | None = None,
    ) -> TireCompound:
        """Choose a new slick compound while the dry-use rule is open."""
        slick_compounds = [
            TireCompound.SOFT,
            TireCompound.MEDIUM,
            TireCompound.HARD,
        ]
        used = self._used_slick_compounds(state)
        available = [compound for compound in slick_compounds if compound not in used]

        return self._rank_stint_compounds(
            state, track, current_lap, available or slick_compounds,
            weather=weather, physical_total_laps=physical_total_laps,
        )

    @staticmethod
    def _dry_stop_budget(state: DriverRaceState, track: Track) -> int:
        """Maximum elective paid dry stops; the optimizer may use fewer."""
        return 3

    @staticmethod
    def _ordinary_stop_budget(state: DriverRaceState, track: Track) -> int:
        return (2 if track.total_laps > 50 else 1) + (
            1 if state.strategy_archetype == TeamStrategyArchetype.AGGRESSIVE else 0
        )

    @classmethod
    def _next_stint_laps(cls, state: DriverRaceState, track: Track, current_lap: int) -> int:
        """Fresh tyres run this lap; the current stop consumes its plan slot.

        Weather stops consume the same slot as any other stop. Ignore expired
        entries after that slot, including when a scheduled stop happened late.
        """
        laps_to_finish = max(1, track.total_laps - current_lap + 1)
        if state.pit_stops + 1 >= cls._ordinary_stop_budget(state, track):
            return laps_to_finish
        for pit_lap in state.planned_pit_laps[state.pit_stops + 1:]:
            if current_lap < pit_lap <= track.total_laps:
                return pit_lap - current_lap
        return laps_to_finish

    def _choose_compound_for_next_stint(
        self, state: DriverRaceState, track: Track, current_lap: int,
        weather: Weather | None = None, *, physical_total_laps: int | None = None,
    ) -> TireCompound:
        """Rank fresh slicks using the same tyre pace as the actual race."""
        return self._rank_stint_compounds(
            state, track, current_lap,
            [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
            weather=weather, physical_total_laps=physical_total_laps,
        )

    def _rank_stint_compounds(
        self, state: DriverRaceState, track: Track, current_lap: int,
        available: list[TireCompound],
        *, weather: Weather | None = None, physical_total_laps: int | None = None,
    ) -> TireCompound:
        target_stint = self._next_stint_laps(state, track, current_lap)
        costs = {
            compound: self.lap_simulator.projected_stint_lap_cost(
                state.driver, state.car, track, TIRE_COMPOUNDS[compound], target_stint,
                current_lap, weather if weather is not None else Weather(),
                physical_total_laps=physical_total_laps,
                current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
                active_aero_enabled=self.event_manager.is_active_aero_allowed(),
            ) for compound in available
        }
        fastest = min(available, key=costs.__getitem__)
        preferred = self._preferred_stint_compound(state, target_stint)
        # Model tolerance: team style may sacrifice at most 0.05 seconds per
        # projected lap. Materially slower sets cannot win by random fallback.
        if preferred in costs and costs[preferred] <= costs[fastest] + 0.05 * target_stint:
            return preferred
        return fastest

    def _preferred_stint_compound(
        self, state: DriverRaceState, target_stint: int
    ) -> TireCompound:
        """Team style proposes a set, subject to the projected pace bound."""
        current = state.current_tire.compound
        profile = self.strategy_profiles[state.strategy_archetype]
        long_stint_threshold = int(profile["long_stint_threshold"])
        medium_prob = float(profile["medium_prob"])
        sprint_soft_prob = float(profile["sprint_soft_prob"])

        # Long stint => harder compounds.
        if target_stint >= long_stint_threshold:
            return TireCompound.HARD if current != TireCompound.HARD else TireCompound.MEDIUM

        # Medium stint => medium baseline.
        if target_stint >= 12:
            if current == TireCompound.HARD:
                return TireCompound.MEDIUM
            return TireCompound.MEDIUM if self.rng.random() < medium_prob else TireCompound.SOFT

        # Short sprint stint => soft bias.
        if current == TireCompound.HARD:
            return TireCompound.SOFT
        return TireCompound.SOFT if self.rng.random() < sprint_soft_prob else TireCompound.MEDIUM

    def _pit_lane_factor(self) -> float:
        """Relative lane loss shared by execution and current-stop planning."""
        if self.event_manager.safety_car_active:
            return 0.55
        if self.event_manager.vsc_active:
            return 0.75
        return 1.0

    def _choose_committed_dry_compound(
        self, state: DriverRaceState, track: Track, current_lap: int,
        weather: Weather | None = None,
        *, physical_total_laps: int | None = None,
    ) -> TireCompound:
        """Price a chosen paid stop's fresh set and all remaining dry stints.

        This stop has not entered pit_stops yet. Its common service/lane loss
        cancels between compounds; only subsequent stops consume the remaining
        budget. A set must run this lap before another stop is possible.
        """
        used = self._used_slick_compounds(state)
        wet_exemption = self._has_used_wet_compound(state)
        candidates = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
        future_budget = min(3, max(0, self._dry_stop_budget(state, track) - state.pit_stops - 1))

        def remaining_cost(compound: TireCompound) -> float:
            return plan_dry_stop(
                state.driver, state.car, track, TIRE_COMPOUNDS[compound], 0,
                track.total_laps - current_lap + 1, future_budget,
                used | {compound}, wet_exemption,
                current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
                physical_total_laps=physical_total_laps,
                active_aero_enabled=self.event_manager.is_active_aero_allowed(),
                tire_pace_multiplier=(self.lap_simulator.weather_pace_multiplier(
                    state.driver, state.car, weather,
                ) if weather is not None else 1.0),
            ).wait_cost

        return min(candidates, key=remaining_cost)

    def _execute_pit_stop(
        self,
        state: DriverRaceState,
        track: Track,
        weather: Weather,
        current_lap: int,
        *,
        pit_box_releases: dict[str, float] | None = None,
        arrival_time: float | None = None,
        sample_service: bool = True,
        physical_total_laps: int | None = None,
    ) -> float:
        """Execute pit stop and return total time lost.

        Args:
            state: Driver state
            track: Current track
            weather: Current weather

        Returns:
            Time lost in seconds
        """
        # Pit lane time + stationary time.  Under a full safety car the field
        # is travelling much more slowly, so the relative pit-lane loss is
        # materially smaller; VSC provides a moderate reduction.  Stationary
        # service remains unchanged and is sampled per team/car in actual races.
        pit_lane_time = track.pit_lane_delta * self._pit_lane_factor()
        stationary_time = (self.lap_simulator.calculate_pit_stop_time(state.car)
                           if sample_service else expected_stationary_time(state.car))
        queue_time = 0.0
        if pit_box_releases is not None:
            arrival = state.total_time if arrival_time is None else arrival_time
            team = state.car.team_id
            queue_time = max(0.0, pit_box_releases.get(team, arrival) - arrival)
            pit_box_releases[team] = arrival + queue_time + stationary_time

        # Choose new tire compound
        weather_compound = self._choose_weather_compound(weather)
        proposal = state.dry_pit_proposal
        state.dry_pit_proposal = None
        if weather_compound is not None:
            new_compound = weather_compound
        elif (
            proposal is not None and proposal[0] == current_lap
            and weather.track_wetness < 0.08 and weather.rain_intensity < 0.15
        ):
            new_compound = proposal[1]
        elif weather.track_wetness < 0.08 and weather.rain_intensity < 0.15:
            new_compound = self._choose_committed_dry_compound(
                state, track, current_lap, weather,
                **({"physical_total_laps": physical_total_laps}
                   if physical_total_laps is not None else {}),
            )
        elif len(self._used_slick_compounds(state)) < 2 and not self._has_used_wet_compound(state):
            # A dry stop must add a new slick compound until the two-compound
            # requirement is satisfied.  In particular, do not let a
            # strategy callback repeating the current compound count as a
            # distinct set.
            new_compound = self._choose_distinct_dry_compound(
                state,
                track,
                current_lap,
                weather, physical_total_laps=physical_total_laps,
            )
        else:
            new_compound = self._choose_compound_for_next_stint(
                state,
                track,
                current_lap,
                weather, physical_total_laps=physical_total_laps,
            )

        total_loss = pit_lane_time + stationary_time + queue_time
        state.pit_stop_details.append({
            "lap": int(current_lap),
            "from_compound": state.current_tire.compound.value,
            "to_compound": new_compound.value,
            "tire_age": int(state.tire_laps),
            "condition": weather.condition.value,
            "rain_intensity": float(weather.rain_intensity),
            "track_wetness": float(weather.track_wetness),
            "control": ("safety_car" if self.event_manager.safety_car_active
                        else "vsc" if self.event_manager.vsc_active else "green"),
            "lane_loss": float(pit_lane_time),
            "service_time": float(stationary_time),
            "queue_time": float(queue_time),
            "total_loss": float(total_loss),
        })
        self._fit_tire(state, new_compound)

        return total_loss

    def _pit_rejoin_traffic_cost(
        self, state: DriverRaceState, all_states: list[DriverRaceState],
        track: Track, queue_delay: float = 0.0,
    ) -> float:
        """One green lap's expected dirty-air difference versus staying out.

        Use frozen clocks and expected own service only. Other cars are assumed
        to stay out; this is not a prediction of future traffic or their stops.
        """
        own = next(other for other in all_states if other.driver.id == state.driver.id)
        stay_gap = self._get_gap_to_car_ahead(own, all_states)
        rejoin_clock = (
            own.total_time + track.pit_lane_delta
            + expected_stationary_time(state.car) + queue_delay
        )
        rejoin_key = (rejoin_clock, own.position)
        # Match the batch merge's insertion before the first physical car
        # with a larger clock key, without copying whole driver states for
        # every candidate stop. Two scalar scans are independent of list order.
        rivals = [other for other in all_states
                  if other.status == DriverStatus.RACING and other.driver.id != own.driver.id]
        insertion_position = min(
            (other.position for other in rivals
             if rejoin_key < (other.total_time, other.position)),
            default=float("inf"),
        )
        ahead = max(
            (other for other in rivals if other.position < insertion_position),
            key=lambda other: other.position, default=None,
        )
        rejoin_gap = None if ahead is None else max(0.0, rejoin_clock - ahead.total_time)
        return (
            self.lap_simulator.traffic_pace_contribution(rejoin_gap)
            - self.lap_simulator.traffic_pace_contribution(stay_gap)
        )

    def _get_gap_to_car_ahead(
        self,
        state: DriverRaceState,
        all_states: list[DriverRaceState],
    ) -> float | None:
        """Get time gap to car ahead."""
        if state.position == 1:
            return None

        for other in all_states:
            if other.position == state.position - 1 and other.status == DriverStatus.RACING:
                # A time-penalised car can transiently be faster in elapsed
                # time than the car it is physically chasing.  Treat that as
                # zero racing gap until classification resolves the ordering,
                # rather than inflating it with an absolute-value shortcut.
                return max(0.0, state.total_time - other.total_time)

        return None

    def _get_gap_to_car_behind(
        self,
        state: DriverRaceState,
        all_states: list[DriverRaceState],
    ) -> float | None:
        """Get time gap to car behind."""
        for other in all_states:
            if other.position == state.position + 1 and other.status == DriverStatus.RACING:
                return max(0.0, other.total_time - state.total_time)

        return None

    def _deploy_overtake_mode_if_eligible(
        self,
        state: DriverRaceState,
        track: Track,
        gap_ahead: float | None,
        mode_allowed: bool,
    ) -> bool:
        """Consume one Overtake Mode burst when a driver is eligible.

        The FIA detection-gap rule is represented at lap resolution: a car
        must be within the track-configured gap at the start of its lap, the
        event manager must allow deployment, and the bounded store must hold
        enough energy for a full burst.  No random draw is used here, making
        the energy timeline deterministic for a fixed simulation seed.
        """
        if not mode_allowed or gap_ahead is None:
            return False
        if gap_ahead > track.overtake_mode_detection_gap:
            return False

        available = float(np.clip(state.overtake_mode_energy, 0.0, 1.0))
        cost = self.OVERTAKE_MODE_DEPLOYMENT_COST
        if available + 1e-12 < cost:
            state.overtake_mode_energy = available
            return False

        state.overtake_mode_energy = float(np.clip(available - cost, 0.0, 1.0))
        state.overtake_mode_deployments += 1
        return True

    def _recharge_overtake_mode_energy(
        self,
        states: list[DriverRaceState],
        neutralized: bool = False,
    ) -> None:
        """Recharge each racing car's Overtake Mode store by a bounded step."""
        recharge = (
            self.OVERTAKE_MODE_NEUTRAL_RECHARGE
            if neutralized
            else self.OVERTAKE_MODE_GREEN_RECHARGE
        )
        for state in states:
            if state.status != DriverStatus.RACING:
                continue
            state.overtake_mode_energy = float(
                np.clip(state.overtake_mode_energy + recharge, 0.0, 1.0)
            )
            state.overtake_mode_active_lap = False

    def _process_overtakes(
        self,
        states: list[DriverRaceState],
        track: Track,
        weather: Weather,
        restart_lap: bool = False,
        lap: int | None = None,
        overtake_mode_allowed: bool | None = None,
    ) -> int:
        """Process overtaking opportunities. Returns number of incidents.

        Args:
            states: Driver race states
            track: Current track
            weather: Current weather
            restart_lap: If True, this is a restart after SC (more overtaking)
            lap: Current race lap, used to enforce Overtake Mode activation
            overtake_mode_allowed: Immutable mode snapshot for this lap
        """
        incidents = 0
        material_penalty_ids: set[str] = set()
        racing_states = [s for s in states if s.status == DriverStatus.RACING]

        # Sort by position to check if faster cars are stuck behind slower ones
        racing_states.sort(key=lambda s: s.position)

        # On restart laps, process more potential battles (cars are bunched)
        max_battles = len(racing_states) if restart_lap else len(racing_states)

        for i in range(1, min(max_battles, len(racing_states))):
            # Car behind (higher position number)
            attacker = racing_states[i]
            # Car ahead (lower position number)
            defender = racing_states[i - 1]

            # Gap: positive means attacker is behind in time (normal)
            # Negative means attacker has caught up and is faster
            gap = attacker.total_time - defender.total_time

            # On restart, everyone is within ~1s, so always check
            # Normal racing: skip if gap is too large
            if not restart_lap and gap > 1.5:
                continue

            # Gap for overtake purposes (how close they are)
            overtake_gap = max(0.0, gap)  # If negative, they're right on their tail

            # On restart laps, drivers are more aggressive
            if restart_lap:
                # Always attempt on restart (everyone is close)
                pass
            elif not self.overtaking_model.should_attempt_overtake(
                attacker.driver,
                defender.driver,
                overtake_gap,
                track.total_laps if lap is None else max(1, track.total_laps - lap + 1),
                attacker.position,
            ):
                continue

            # Compare tyre-only seconds at each driver's current weather-scaled
            # condition after this lap's running, including mismatch grip.
            tire_advantage = self.lap_simulator.tire_weather_pace_contribution(
                defender.driver, defender.car, track,
                defender.current_tire, defender.tire_laps, weather,
            ) - self.lap_simulator.tire_weather_pace_contribution(
                attacker.driver, attacker.car, track,
                attacker.current_tire, attacker.tire_laps, weather,
            )

            # Attempt overtake - boosted probability on restart
            success, incident = self.overtaking_model.attempt_overtake(
                attacker=attacker.driver,
                attacker_car=attacker.car,
                defender=defender.driver,
                defender_car=defender.car,
                track=track,
                gap=overtake_gap,
                overtake_mode_active=(
                    attacker.overtake_mode_active_lap
                    and (
                        self.event_manager.is_overtake_mode_allowed(lap, weather)
                        if overtake_mode_allowed is None
                        else overtake_mode_allowed
                    )
                ),
                is_wet=weather.is_wet(),
                restart_boost=restart_lap,  # Extra chance on restart
                tire_pace_advantage_seconds=tire_advantage,
            )

            if success:
                # Swap positions
                attacker.position, defender.position = defender.position, attacker.position
                # Later battles must use the new immediate neighbour. Keeping
                # the old list would let a following car skip the displaced car.
                racing_states[i - 1], racing_states[i] = attacker, defender

            if incident:
                incidents += 1
                # Small time loss for both drivers
                attacker_loss = float(self.rng.uniform(1, 3))
                defender_loss = float(self.rng.uniform(0.5, 2))
                attacker.total_time += attacker_loss
                defender.total_time += defender_loss
                attacker.last_lap_time += attacker_loss
                defender.last_lap_time += defender_loss
                self.event_manager.events.append(RaceEvent(
                    event_type=EventType.COLLISION,
                    # Direct helper calls can omit the lap; zero means unknown.
                    lap=lap if lap is not None else 0,
                    drivers_involved=[attacker.driver.id, defender.driver.id],
                    description="Contact during an overtaking attempt",
                    applied_time_losses={
                        attacker.driver.id: attacker_loss,
                        defender.driver.id: defender_loss,
                    },
                ))
                material_penalty_ids.update((attacker.driver.id, defender.driver.id))

        if material_penalty_ids:
            self._reorder_positions_after_material_penalties(states, material_penalty_ids)

        return incidents

    def _handle_pit_batch_position_changes(
        self,
        pitting_drivers: list[DriverRaceState],
        all_states: list[DriverRaceState],
    ) -> None:
        """Merge cars leaving the pits into the unchanged on-track queue.

        Actual elapsed clocks include each car's own service and pit-lane loss.
        Resolve simultaneous stops together, using pre-stop positions to break
        ties. Cars that stayed out retain their relative physical order until
        the overtaking model resolves their battles.
        """
        if not pitting_drivers:
            return
        pitting_ids = {state.driver.id for state in pitting_drivers}
        racing = sorted(
            (state for state in all_states if state.status == DriverStatus.RACING),
            key=lambda state: state.position,
        )
        pitting = sorted(
            (state for state in racing if state.driver.id in pitting_ids),
            key=lambda state: (state.total_time, state.position),
        )
        staying_out = [state for state in racing if state.driver.id not in pitting_ids]
        ordered: list[DriverRaceState] = []
        pit_index = 0
        for on_track in staying_out:
            while pit_index < len(pitting) and (
                pitting[pit_index].total_time, pitting[pit_index].position
            ) < (on_track.total_time, on_track.position):
                ordered.append(pitting[pit_index])
                pit_index += 1
            ordered.append(on_track)
        ordered.extend(pitting[pit_index:])
        for position, state in enumerate(ordered, 1):
            state.position = position

    def _reconcile_racing_times(self, states: list[DriverRaceState]) -> None:
        """Charge blocked running to the lap without changing track order.

        A successful pass can also put a car with a slightly slower provisional
        clock ahead. The displaced car then waits behind that new leader.
        Equal clocks represent a gap below this lap-level model's resolution;
        physical position remains the tie-breaker. Retired cars do not constrain
        running cars, and this correction never removes elapsed race time.
        """
        racing = sorted(
            (state for state in states if state.status == DriverStatus.RACING),
            key=lambda state: state.position,
        )
        for ahead, behind in zip(racing, racing[1:]):
            blocked_time = max(0.0, ahead.total_time - behind.total_time)
            behind.total_time = max(behind.total_time, ahead.total_time)
            behind.last_lap_time += blocked_time

    def _normalize_positions(
        self,
        states: list[DriverRaceState],
        preferred_order: dict[str, int] | None = None,
    ) -> None:
        """Restore one unique position per car without sorting by race time.

        ``preferred_order`` supplies the authoritative order immediately
        before a batch operation such as simultaneous pit stops. It is used
        only to resolve temporary equal-position ties; ordinary position
        changes remain authoritative.
        """
        preferred_order = preferred_order or {}
        fallback_order = {id(state): index for index, state in enumerate(states)}

        def order_key(state: DriverRaceState) -> tuple[int, int]:
            return (
                state.position,
                preferred_order.get(
                    state.driver.id,
                    fallback_order[id(state)],
                ),
            )

        racing = sorted(
            (state for state in states if state.status == DriverStatus.RACING),
            key=order_key,
        )
        dnf = sorted(
            (state for state in states if state.status == DriverStatus.DNF),
            key=lambda state: (
                -state.laps_completed,
                state.last_crossing_position
                if state.last_crossing_position is not None else order_key(state)[0],
                order_key(state)[1],
            ),
        )
        for position, state in enumerate(racing, 1):
            state.position = position
        for position, state in enumerate(dnf, len(racing) + 1):
            state.position = position

    def _update_positions(self, states: list[DriverRaceState]) -> None:
        """Update positions, respecting track position (no auto-sort by time).

        Positions only change through:
        1. Successful overtakes (handled in _process_overtakes)
        2. Pit stops (handled in _handle_pit_position_changes)
        3. DNFs (moved to back)
        """
        self._normalize_positions(states)

    def _classify_positions_before_neutralization(
        self,
        states: list[DriverRaceState],
        penalized_driver_ids: set[str] | None = None,
    ) -> None:
        """Apply explicit penalty position loss before a field gap reset.

        A safety-car or red-flag bunching pass intentionally replaces the
        absolute time gaps with a compact queue.  Track position is therefore
        authoritative for clean laps, including a valid overtake whose
        cumulative clock happens to be slower.  Only drivers carrying an
        explicit material incident/time penalty may be reclassified before
        the gap reset.

        ``None`` is retained as a safe no-op for callers that only need to
        request a neutralization classification without penalty metadata.
        """
        if penalized_driver_ids:
            self._reorder_positions_after_material_penalties(
                states,
                penalized_driver_ids,
            )

    def _reorder_positions_after_material_penalties(
        self,
        states: list[DriverRaceState],
        penalized_driver_ids: set[str],
    ) -> None:
        """Move penalized racing cars behind any faster elapsed-time rivals.

        Track position remains authoritative during normal racing so a clean
        overtake is not undone merely because cumulative lap clocks differ by
        a fraction.  An explicit incident/time penalty is different: once its
        material consequence is applied, a slower elapsed-time car cannot
        remain ahead of a faster car.  Only penalized cars are allowed to
        cross position boundaries here; the rest of the on-track order is
        preserved.
        """
        racing = [state for state in states if state.status == DriverStatus.RACING]
        if len(racing) < 2:
            return
        if len({state.position for state in racing}) != len(racing):
            # Defensive repair for external/direct callers. The race loop
            # already normalizes every pit batch, but this method must never
            # spin if it receives a malformed equal-position state.
            self._normalize_positions(states)
            racing = [state for state in states if state.status == DriverStatus.RACING]

        for driver_id in sorted(penalized_driver_ids):
            affected = next((state for state in racing if state.driver.id == driver_id), None)
            if affected is None:
                continue

            # A penalty adds time, so the affected car can only fall back.
            # Locate by identity rather than Pydantic/dataclass equality, then
            # compute the destination in one bounded pass. This cannot loop
            # indefinitely even if malformed state reaches the helper.
            ordered = sorted(racing, key=lambda state: state.position)
            index = next(
                (position for position, state in enumerate(ordered) if state is affected),
                None,
            )
            if index is None:
                continue
            destination = index
            for candidate in ordered[index + 1 :]:
                if affected.total_time <= candidate.total_time + 1e-9:
                    break
                destination += 1
            if destination == index:
                continue

            position_slots = [
                state.position for state in ordered[index : destination + 1]
            ]
            for state, position in zip(
                ordered[index + 1 : destination + 1],
                position_slots,
            ):
                state.position = position
            affected.position = position_slots[-1]

    def _handle_red_flag_stop(
        self,
        states: list[DriverRaceState],
        weather: Weather,
        track: Track,
        current_lap: int,
        *,
        defer_tire_fit: bool = False,
    ) -> None:
        """Handle red flag stoppage.

        During a red flag:
        - All cars return to pit lane
        - Teams can change tires and make limited repairs
        - Gaps are reset for restart

        Args:
            states: Driver race states
            weather: Current weather conditions
            track: Race distance and tyre physics
            current_lap: Lap completed before suspension
            defer_tire_fit: Live races fit after the existing restart weather update
        """
        # Bunch the field - gaps are reset on red flag
        self.event_manager.bunch_field(states)

        if not defer_tire_fit:
            self._fit_red_flag_tires(states, weather, track, current_lap)

        # Keep race-control ordering unchanged; no stopped duration is modeled.
        self.event_manager.end_red_flag()

    def _fit_red_flag_tires(
        self, states: list[DriverRaceState], weather: Weather, track: Track, current_lap: int,
        *, physical_total_laps: int | None = None,
    ) -> None:
        """Fit free sets using weather at the restart, before next-lap decisions."""
        # All drivers can change tires during red flag (free tire change)
        for state in states:
            if state.status != DriverStatus.RACING or current_lap >= track.total_laps:
                continue

            # Choose optimal tire for current conditions
            new_compound = self._choose_red_flag_tire(
                state, weather, track, current_lap,
                **({"physical_total_laps": physical_total_laps}
                   if physical_total_laps is not None else {}),
            )
            self._fit_tire(state, new_compound)
            # The sole modeled forced-stop cause is a puncture. A free fresh
            # set resolves it without charging another stop on the restart.
            state.force_pit_next_lap = False
            state.dry_pit_proposal = None

    def _choose_red_flag_tire(
        self, state: DriverRaceState, weather: Weather, track: Track, current_lap: int,
        *, physical_total_laps: int | None = None,
    ) -> TireCompound:
        """Price a free set from the next lap, including future paid dry stops."""
        remaining_laps = track.total_laps - current_lap
        if remaining_laps <= 0:
            return state.current_tire.compound
        weather_compound = self._choose_weather_compound(weather)
        if weather_compound is not None:
            return weather_compound

        used = self._used_slick_compounds(state)
        wet_exemption = self._has_used_wet_compound(state)
        budget_limit = (
            self._dry_stop_budget(state, track)
            if weather.track_wetness < 0.08 and weather.rain_intensity < 0.15
            else self._ordinary_stop_budget(state, track)
        )
        remaining_stops = min(3, max(0, budget_limit - state.pit_stops))

        def finish_cost(compound: TireCompound) -> float:
            prospective_used = used | {compound}
            budget = remaining_stops
            if not wet_exemption and len(prospective_used) < 2:
                # As in ordinary planning, retain a correction stop if all
                # paid stops repeated a compound. A distinct free set can
                # satisfy that rule without consuming a paid-stop slot.
                budget = max(budget, 1)
            return plan_dry_stop(
                state.driver, state.car, track, TIRE_COMPOUNDS[compound], 0,
                remaining_laps, budget, prospective_used, wet_exemption,
                physical_total_laps=physical_total_laps,
                tire_pace_multiplier=self.lap_simulator.weather_pace_multiplier(
                    state.driver, state.car, weather,
                ),
            ).wait_cost

        return min(
            (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD),
            key=finish_cost,
        )

    def _weather_stop_can_pay(
        self, state: DriverRaceState, track: Track, weather: Weather, current_lap: int,
        additional_current_stop_cost: float = 0.0, *, traffic_possible: bool = True,
        physical_total_laps: int | None = None,
    ) -> bool:
        """Compare an optimistic paid-refit plan with retaining while safe."""
        costs = weather_stop_costs(
            state.driver, state.car, track, weather, state.current_tire,
            state.tire_laps, current_lap,
            pit_lane_factor=self._pit_lane_factor(),
            additional_current_stop_cost=additional_current_stop_cost,
            current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
            active_aero_enabled=self.event_manager.is_active_aero_allowed(),
            traffic_possible=traffic_possible,
            physical_total_laps=physical_total_laps,
        )
        return costs.pit_now_cost < costs.stay_cost

    @staticmethod
    def _choose_weather_compound(weather: Weather) -> TireCompound | None:
        """Choose a fresh rain tyre, or leave slick choice to the stint strategy.

        Use the slick mismatch crossover for fresh sets so a weather stop
        cannot immediately fit another unsuitable slick. Existing rain tyres
        retain their wider drying windows in the mismatch check, avoiding
        unnecessary stops when conditions hover around a crossover.
        """
        return weather.fresh_rain_compound()

    def _check_tire_weather_mismatch(self, tire: Tire, weather: Weather) -> str:
        """Check if tires match current weather conditions.

        Returns:
            "ok" - tires are appropriate
            "suboptimal" - tires are wrong but survivable
            "critical" - must pit immediately
        """
        return weather.tire_mismatch(tire.compound)
