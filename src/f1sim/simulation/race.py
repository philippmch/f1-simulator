"""Race simulation engine."""

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum
from math import isfinite
from numbers import Real

import numpy as np

from f1sim.cancellation import raise_if_cancelled
from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.events import EventManager, EventType, RaceEvent
from f1sim.simulation.execution import validate_starting_tire_ages, validate_starting_tires
from f1sim.simulation.inventory_race import (
    InventoryStrategyMixin,
    _timed_stop_budget_envelope,
)
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.neutralization import safety_car_running_time
from f1sim.simulation.opening_strategy import dry_opening_policy_costs, opening_policy_costs
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.pit_plans import (
    commit_pit_plan_service,
    current_pit_plan_instruction,
    finalize_pit_plan,
    initialize_pit_plan_state,
    override_pit_plan_instruction,
    skip_pit_plan_instruction,
    validate_pit_plans,
)
from f1sim.simulation.pit_strategy import expected_stationary_time, plan_dry_stop
from f1sim.simulation.race_points import points_for_classification
from f1sim.simulation.race_timing import RaceFinishClock, forecast_final_lap
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.randomness import MechanicalRngFactory
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import projected_surfaces
from f1sim.simulation.tire_inventory import TireInventory, validate_tire_inventory
from f1sim.simulation.validation import validate_unique_ids
from f1sim.simulation.warmup import tire_warmup_seconds, validate_tire_warmup
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
    fit_lap_pending: bool = False
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
    weather_pit_proposal: tuple[int, TireCompound] | None = None
    prior_tire_laps: int = 0
    tire_inventory: TireInventory | None = None
    tire_set_history: list[dict] = field(default_factory=list)
    inventory_pit_proposal: tuple[int, str | None] | None = None
    # One-lap planner metadata consumed by the next paid stop.  This remains
    # ephemeral so a veto, free refit, or direct execution cannot inherit an
    # old proposal and present it as a new policy decision.
    pit_decision_context: dict[str, object] | None = None
    # Explicit user instructions are separate from the automatic strategy
    # slots above.  The target fields are transient and cleared on commit.
    pit_plan: list[dict] | None = None
    pit_plan_history: list[dict] | None = None
    pit_plan_index: int = 0
    pit_plan_target: TireCompound | None = None
    pit_plan_target_set_id: str | None = None
    pit_plan_reason: str | None = None
    pit_plan_override_reason: str | None = None

    def __post_init__(self) -> None:
        """Seed tyre history from the driver's actual starting set."""
        if not self.tire_compound_history:
            self.tire_compound_history.append(self.current_tire.compound.value)


@dataclass(slots=True)
class _PitTrafficProjectionRow:
    """The small mutable view needed by native pit-traffic merge helpers."""

    driver: Driver
    position: int
    total_time: float
    status: DriverStatus


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
    tire_set_history: list[dict] | None = None
    tire_inventory: list[dict] | None = None
    race_suspension_seconds: float | None = None
    pit_plan_history: list[dict] | None = None


def get_race_suspension_seconds(results: Iterable[RaceResult]) -> float | None:
    """Return one race's shared suspension duration when it is fully recorded.

    A race result row carries the same global suspension duration for every
    driver in that race.  Legacy rows may omit the field, and malformed or
    inconsistent rows must remain unknown rather than being interpreted as a
    zero-duration race.
    """
    try:
        rows = list(results)
    except (TypeError, ValueError):
        return None
    if not rows:
        return None

    def finite_nonnegative_real(value) -> tuple[Real, float] | None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return None
        try:
            converted = float(value)
        except (OverflowError, TypeError, ValueError):
            return None
        if not isfinite(converted) or converted < 0:
            return None
        # Keep the public representation canonical for negative zero.
        return value, (0.0 if converted == 0 else converted)

    first = finite_nonnegative_real(getattr(rows[0], "race_suspension_seconds", None))
    if first is None:
        return None
    shared_original, shared = first
    for row in rows[1:]:
        validated = finite_nonnegative_real(getattr(row, "race_suspension_seconds", None))
        if validated is None or validated[0] != shared_original:
            return None
    return shared


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


class RaceSimulator(InventoryStrategyMixin):
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
        *,
        weather_rng: np.random.Generator | None = None,
        mechanical_rng_factory: MechanicalRngFactory | None = None,
        red_flag_pause_seconds: float = 600.0,
        tire_warmup: dict[str, float] | None = None,
    ):
        """Initialize race simulator.

        Args:
            rng: Random number generator
            strategy_tuning: Optional strategy threshold overrides
            weather_rng: Independent weather stream; omitted callers share rng
            mechanical_rng_factory: Optional per-driver/per-lap mechanical stream factory
            red_flag_pause_seconds: Suspension pause after field collection
            tire_warmup: Optional absolute cost on the first running lap after a fit
        """
        if (isinstance(red_flag_pause_seconds, bool)
                or not isinstance(red_flag_pause_seconds, Real)
                or not isfinite(red_flag_pause_seconds) or red_flag_pause_seconds < 0):
            raise ValueError("red_flag_pause_seconds must be finite and nonnegative")
        self.rng = rng if rng is not None else np.random.default_rng()
        self.weather_rng = weather_rng if weather_rng is not None else self.rng
        self.red_flag_pause_seconds = float(red_flag_pause_seconds)
        self.tire_warmup = validate_tire_warmup(tire_warmup)
        self.suspensions: list[tuple[float, float, tuple[str, ...]]] = []
        self.weather_history: list[dict] = []
        self.lap_simulator = LapSimulator(rng=self.rng)
        self.overtaking_model = OvertakingModel(rng=self.rng)
        event_manager_kwargs = {"rng": self.rng}
        if mechanical_rng_factory is not None:
            event_manager_kwargs["mechanical_rng_factory"] = mechanical_rng_factory
        self.event_manager = EventManager(**event_manager_kwargs)
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

    @staticmethod
    def _decision_forecast_saving(decision) -> float | None:
        """Return a finite planner saving when the decision exposes both costs."""
        wait_cost = getattr(decision, "wait_cost", None)
        pit_cost = getattr(decision, "pit_now_cost", None)
        if (isinstance(wait_cost, bool) or not isinstance(wait_cost, Real)
                or isinstance(pit_cost, bool) or not isinstance(pit_cost, Real)):
            return None
        try:
            saving = float(wait_cost) - float(pit_cost)
        except (OverflowError, TypeError, ValueError):
            return None
        return saving if isfinite(saving) else None

    def _capture_pit_decision_context(
        self, state: DriverRaceState, lap: int, reason: str, decision=None,
    ) -> None:
        """Attach metadata to one accepted policy decision until execution."""
        state.pit_decision_context = {
            "lap": int(lap),
            "decision_reason": reason,
            "forecast_saving_seconds": self._decision_forecast_saving(decision),
        }

    def simulate_race(
        self,
        drivers: list[Driver],
        cars: dict[str, Car],
        track: Track,
        weather: Weather,
        starting_grid: list[str],
        starting_tires: dict[str, TireCompound] | None = None,
        starting_tire_ages: dict[str, int] | None = None,
        tire_inventory: dict[str, list[dict]] | None = None,
        pit_plans: dict[str, list[dict]] | None = None,
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
        # Validate explicit plans against the scheduled distance before any
        # mutable driver/event state is reset or any strategy RNG is consumed.
        driver_ids = tuple(driver.id for driver in drivers)
        normalized_pit_plans = validate_pit_plans(
            pit_plans,
            driver_ids=driver_ids,
            total_laps=track.total_laps,
            tire_inventory=tire_inventory,
        )
        validate_unique_ids(driver_ids, "drivers")
        validate_unique_ids(starting_grid, "starting_grid")
        starting_tires = validate_starting_tires(starting_tires, (d.id for d in drivers))
        ages = validate_starting_tire_ages(starting_tire_ages, starting_tires,
                                           (d.id for d in drivers))
        inventories = validate_tire_inventory(tire_inventory, starting_tires, ages,
                                              (d.id for d in drivers))
        # Reset mutable driver state as well as event state.  Monte Carlo
        # workers may intentionally reuse model instances between simulations.
        for driver in drivers:
            driver.reset_race_state()

        # Reset event manager
        self.event_manager.reset()
        self.weather_history = []
        self.suspensions.clear()

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
            inventory = selected_set = None
            if driver_id in inventories:
                inventory, selected_set = self._inventory_opening_set(
                    driver, car, track, current_weather, strategy, inventories[driver_id],
                    starting_tires.get(driver_id), ages.get(driver_id, 0),
                )
                tire_compound = selected_set.compound
            elif starting_tires and driver_id in starting_tires:
                tire_compound = starting_tires[driver_id]
            else:
                tire_compound = self._choose_starting_compound(
                    strategy,
                    track,
                    current_weather,
                    driver,
                    car,
                )

            driver.current_tire_laps = ages.get(driver_id, 0)
            pit_plans = self._plan_pit_lap_options(strategy, track)
            states.append(
                DriverRaceState(
                    driver=driver,
                    car=car,
                    position=pos,
                    tire_laps=ages.get(driver_id, 0),
                    prior_tire_laps=ages.get(driver_id, 0),
                    current_tire=TIRE_COMPOUNDS[tire_compound].model_copy(deep=True),
                    strategy_archetype=strategy,
                    planned_pit_laps=pit_plans[0],
                    pit_plan_options=pit_plans,
                    active_pit_plan_index=0,
                )
            )
            initialize_pit_plan_state(
                states[-1], normalized_pit_plans.get(driver_id)
                if driver_id in normalized_pit_plans else None,
            )
            if inventory is not None:
                self._initialize_inventory(states[-1], inventory, selected_set)

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
        pending_resume_time: float | None = None
        for lap in range(1, track.total_laps + 1):
            raise_if_cancelled()
            # A suspension is elapsed between completed crossings. Preserve a
            # scalar snapshot before exposing the common restart clock so a
            # retirement on the restart lap can roll back its uncompleted lap
            # without fabricating the suspension as running time.
            last_completed_states: list[DriverRaceState] | None = None
            resumed_at: float | None = None
            if pending_resume_time is not None:
                resumed_at = pending_resume_time
                last_completed_states = [replace(state) for state in states]
                for state in states:
                    if state.status == DriverStatus.RACING:
                        state.total_time = resumed_at
                pending_resume_time = None
            if not any(state.status == DriverStatus.RACING for state in states):
                break
            self._record_weather(lap, current_weather)
            leader = min((state for state in states if state.status == DriverStatus.RACING),
                         key=lambda state: state.position)
            planning_crossing_time = leader.total_time
            planning_next_lap_start = None
            if last_completed_states is not None:
                completed = next(
                    state for state in last_completed_states
                    if state.driver.id == leader.driver.id
                )
                planning_crossing_time = completed.total_time
                planning_next_lap_start = resumed_at
            planning_final_lap = forecast_final_lap(
                final_lap, lap - 1, planning_crossing_time,
                observed_running_pace.get(leader.driver.id), finish_clock.time_limit_seconds,
                self.event_manager.get_lap_time_modifier(),
                **({"next_lap_start_time": planning_next_lap_start}
                   if planning_next_lap_start is not None else {}),
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
            lap_safety_car = self.event_manager.safety_car_active
            lap_time_modifier = self.event_manager.get_lap_time_modifier()

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
            # A paid stop is completed before the lap starts.  Retirements
            # roll back only the uncompleted running lap, so retain the
            # post-service tyre age for a car that fails after rejoining.
            running_start_tire_laps = {
                state.driver.id: state.tire_laps
                for state in states if state.status == DriverStatus.RACING
            }
            pitting_ids = {state.driver.id for state in drivers_pitting}
            pit_lap_losses = {
                state.driver.id: state.total_time - lap_start_times[state.driver.id]
                for state in drivers_pitting
            }
            traffic_gaps = lap_start_gaps
            traffic_states = lap_start_states
            if pitting_ids:
                # Freeze post-service clocks and positions so every car sees
                # the same rejoin traffic. The full SC also uses this order
                # for its running queue; mode detection remains pre-stop.
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
                lap_times[state.driver.id] = lap_time

            # A full SC closes gaps through this lap's running. Resolve the
            # whole queue after every free pace is known, so input-list order
            # cannot make a follower use the previous lap's leader pace.
            if lap_safety_car:
                lap_times = self._safety_car_lap_times(
                    lap_times, traffic_states, lap_time_modifier,
                )
                queue_positions = {state.driver.id: state.position for state in traffic_states}
                for state in states:
                    if state.driver.id in lap_times:
                        state.position = queue_positions[state.driver.id]
            else:
                lap_times = {driver_id: time * lap_time_modifier
                             for driver_id, time in lap_times.items()}

            # This is an absolute elapsed-time sensitivity, not a tyre-physics
            # multiplier. Charge it after neutralization scaling.
            if self.tire_warmup:
                for state in states:
                    if state.driver.id in lap_times and state.fit_lap_pending:
                        lap_times[state.driver.id] += self._consume_tire_warmup(state)

            for state in states:
                if state.driver.id not in lap_times:
                    continue
                lap_time = lap_times[state.driver.id]
                state.total_time += lap_time
                # Pit loss already entered total_time before running this lap.
                # Include it in the recorded lap too, without charging it twice
                # or multiplying stationary/queue time by the SC lap modifier.
                state.last_lap_time = lap_time + pit_lap_losses.get(state.driver.id, 0.0)
                state.tire_laps += 1
                state.driver.current_tire_laps = state.tire_laps

            # Full-SC running already preserves the committed pit-exit order.
            # Merging it again could let a pitter pass its blocker at an equal
            # crossing time. Other laps resolve the batch by completed clocks.
            if not lap_safety_car:
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
                            self._damage_inventory_tire(state)

                        if state.driver.dnf:
                            state.status = DriverStatus.DNF
                            state.dnf_reason = state.driver.dnf_reason

            # Restore a retired car's last completed crossing. Neither the
            # sampled failure lap nor its service/incident time is race distance.
            rollback_states = (last_completed_states
                               if last_completed_states is not None else lap_start_states)
            for state, before in zip(states, rollback_states):
                if state.status == DriverStatus.DNF and before.status == DriverStatus.RACING:
                    state.total_time = before.total_time
                    state.last_lap_time = before.last_lap_time
                    if state.driver.id in lap_times:
                        state.tire_laps = running_start_tire_laps[state.driver.id]
                        state.driver.current_tire_laps = state.tire_laps
                    if state.tire_inventory is not None and state.driver.id in lap_times:
                        # Finite ledgers count completed tyre laps, just as the
                        # chronological engine does. Keep any paid fitting, but
                        # discard this failed crossing's provisional wear.
                        pool = state.tire_inventory
                        current = pool.sets[pool.current_set_id]
                        if current.age > state.tire_laps:
                            pool.sets[current.id] = replace(current, age=state.tire_laps)

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

            # Check if red flag was just deployed. Suspension setup happens
            # after all completed-lap accounting below, once the authoritative
            # leader crossing has been committed to the finish clock.
            red_flag_deployed_this_lap = any(e.event_type == EventType.RED_FLAG for e in lap_events)

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
                # Recharge according to the completed lap's starting state.
                # Race-control events are resolved after running, so a new
                # deployment applies to the next lap, not this one.
                neutralized=lap_started_neutralized,
            )

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

            # Preserve the final retirement lap's consequences, then stop.
            # Later scheduled laps cannot produce events with no running cars.
            # Update the green-lap streak first so a final red flag still
            # disqualifies green-only points even when every car retires.
            if not any(state.status == DriverStatus.RACING for state in states):
                break

            leader = min((state for state in states if state.status == DriverStatus.RACING),
                         key=lambda state: state.position)
            final_lap = finish_clock.observe_leader_crossing(lap, leader.total_time)
            if lap >= final_lap:
                break

            if red_flag_deployed_this_lap:
                # The leader's completed crossing starts collection. Every
                # active car's reconciled total_time is already a completed
                # crossing clock, so the latest one determines pit-lane arrival.
                active_states = [state for state in states
                                 if state.status == DriverStatus.RACING]
                collection_time = max(state.total_time for state in active_states)
                resume = collection_time + self.red_flag_pause_seconds
                finish_clock.begin_suspension(leader.total_time)
                finish_clock.end_suspension(resume)
                ordered_active_ids = tuple(
                    state.driver.id
                    for state in sorted(active_states, key=lambda state: state.position)
                )
                self.suspensions.append((leader.total_time, resume, ordered_active_ids))

                # End red control before choosing the restart set. The restart
                # lap consumes exactly one evolved weather snapshot; no
                # running, service, or RNG work is charged during collection.
                self.event_manager.end_red_flag()
                current_weather = current_weather.evolve(self.weather_rng)
                restart_final_lap = forecast_final_lap(
                    final_lap, lap, leader.total_time,
                    observed_running_pace.get(leader.driver.id),
                    finish_clock.time_limit_seconds,
                    next_lap_start_time=resume,
                )
                restart_track = (track if restart_final_lap == track.total_laps else
                                 track.model_copy(update={"total_laps": restart_final_lap}))
                self._fit_red_flag_tires(
                    states, current_weather, restart_track, lap,
                    **({"physical_total_laps": track.total_laps}
                       if restart_final_lap < track.total_laps else {}),
                )
                pending_resume_time = resume
            elif lap < final_lap:
                # The initial weather snapshot was used unchanged on lap one.
                # Evolve only when another lap will actually consume the result.
                current_weather = current_weather.evolve(self.weather_rng)

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
                    race_suspension_seconds=finish_clock.total_suspension_seconds,
                    pit_plan_history=finalize_pit_plan(
                        state,
                        "retired" if state.status == DriverStatus.DNF else "race_finished",
                    ),
                    **self._inventory_result_fields(state),
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
        # Compare precautionary intermediates whenever the numeric surface
        # permits them. A descriptive label must not change the candidate
        # set, or whether the opening choice consumes the actual race RNG.
        if self._check_tire_weather_mismatch(
            TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], weather,
        ) != "critical":
            if driver is not None and car is not None:
                costs = opening_policy_costs(
                    driver, car, track, weather, strategy,
                    self.strategy_tuning, self.strategy_profiles,
                    **({"tire_warmup": self.tire_warmup} if self.tire_warmup else {}),
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
                **({"tire_warmup": self.tire_warmup} if self.tire_warmup else {}),
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

    def _pit_plan_compulsory_reason(
        self, state: DriverRaceState, track: Track, weather: Weather, lap: int,
    ) -> str | None:
        """Return the existing safety reason that must outrank a plan."""
        if state.force_pit_next_lap:
            return "forced_repair"
        if self._check_tire_weather_mismatch(state.current_tire, weather) == "critical":
            return "critical_weather"
        if (state.tire_inventory is not None
                and state.tire_inventory.current_set_id in state.tire_inventory.unavailable_ids):
            return "forced_repair"
        if lap >= max(2, track.total_laps) and not self._stay_satisfies_tire_rule(state):
            return "compound_requirement"
        return None

    def _pit_plan_replacement(self, state, compound: TireCompound, weather: Weather):
        """Return a deterministic requested replacement, if one exists."""
        if weather.tire_mismatch(compound) == "critical":
            return None, "critical_requested_compound"
        inventory = state.tire_inventory
        if inventory is None:
            return compound, None
        candidates = [
            (index, item)
            for index, item in enumerate(inventory.replacements())
            if item.compound == compound
            and weather.tire_mismatch(item.compound) != "critical"
        ]
        if not candidates:
            return None, "requested_compound_unavailable"
        _, selected = min(candidates, key=lambda pair: (pair[1].age, pair[0]))
        return selected.compound, selected.id

    def _pit_plan_satisfies_rule(self, state: DriverRaceState, compound: TireCompound) -> bool:
        # The requested set replaces the currently fitted set before the
        # numbered lap starts.  Exclude an unrun red-flag refit here; adding it
        # would make a final-lap request appear to satisfy the two-slick rule
        # even though that set is immediately removed.
        prospective = self._actually_used_compounds(state) | {compound}
        return bool(prospective & {TireCompound.INTERMEDIATE, TireCompound.WET}) or len(
            prospective & {TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD}
        ) >= 2

    def _prepare_pit_plan_stop(
        self, state: DriverRaceState, track: Track, weather: Weather, lap: int,
    ) -> bool | None:
        """Prepare one due custom instruction.

        ``True`` requests a paid service, ``False`` records a definitive skip,
        and ``None`` leaves the existing automatic/compulsory policy in charge.
        """
        instruction = current_pit_plan_instruction(state, lap)
        if instruction is None or self.event_manager.red_flag_active:
            return None
        state.pit_plan_target = None
        state.pit_plan_target_set_id = None
        state.pit_plan_reason = None
        state.pit_plan_override_reason = None
        state.dry_pit_proposal = None
        state.weather_pit_proposal = None
        state.inventory_pit_proposal = None
        compound = TireCompound(instruction["compound"])
        compulsory = self._pit_plan_compulsory_reason(state, track, weather, lap)
        target, availability = self._pit_plan_replacement(state, compound, weather)
        if target is None:
            if compulsory is not None:
                state.pit_plan_override_reason = compulsory
                return None
            skip_pit_plan_instruction(state, availability)
            return False
        compound_requirement = (
            lap >= max(2, track.total_laps)
            and not self._pit_plan_satisfies_rule(state, compound)
        )
        if compound_requirement:
            state.pit_plan_override_reason = compulsory or "compound_requirement"
            return None
        state.pit_plan_target = compound
        state.pit_plan_target_set_id = availability if state.tire_inventory is not None else None
        state.pit_plan_reason = compulsory or "user_plan"
        if state.tire_inventory is not None:
            state.inventory_pit_proposal = (lap, availability)
        state.pit_decision_context = {
            "lap": int(lap),
            "decision_reason": state.pit_plan_reason,
            "forecast_saving_seconds": None,
        }
        return True

    @staticmethod
    def _commit_pit_plan_if_due(state: DriverRaceState, *, overridden=False):
        """Commit the due request after a caller has completed service."""
        history = state.pit_plan_history
        index = state.pit_plan_index
        if history is None or index >= len(history):
            return
        details = state.pit_stop_details[-1] if state.pit_stop_details else {}
        reason = state.pit_plan_override_reason or state.pit_plan_reason
        if reason is None:
            return
        actual_compound = details.get("to_compound")
        actual_set_id = details.get("to_set_id")
        status = "overridden" if overridden else "executed"
        commit_pit_plan_service(
            state,
            reason=reason,
            actual_compound=actual_compound,
            actual_set_id=actual_set_id,
            status=status,
        )

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
        expected_losses: dict[str, float] = {}
        pitting = []
        for state in ordered:
            arrival = arrivals[state.driver.id][0]
            team = state.car.team_id
            delay = max(0.0, expected_releases.get(team, arrival) - arrival)
            traffic_snapshot = self._standard_pit_traffic_snapshot(
                state,
                lap_start_states,
                states,
                track,
                delay,
                expected_losses,
            )
            forced_repair = state.force_pit_next_lap
            if forced_repair:
                # A forced stop bypasses policy evaluation.  Any proposal
                # left by an earlier veto must not label this execution.
                state.pit_decision_context = None
            custom_stop = self._prepare_pit_plan_stop(state, track, weather, lap)
            explicit_plan_suppresses = (
                state.pit_plan is not None
                and current_pit_plan_instruction(state, lap) is None
                and not self.event_manager.red_flag_active
                and self._pit_plan_compulsory_reason(state, track, weather, lap) is None
            )
            if custom_stop is True:
                should_pit = True
            elif custom_stop is False or explicit_plan_suppresses:
                should_pit = False
            else:
                should_pit = forced_repair or self._should_pit(
                    state, lap_start_states, track, lap,
                    self.event_manager.is_pit_window_open(), weather=weather,
                    additional_current_stop_cost=delay,
                    traffic_snapshot=traffic_snapshot,
                    **({"physical_total_laps": physical_total_laps}
                       if physical_total_laps is not None else {}),
                )
            if not should_pit and state.pit_plan_override_reason is not None:
                override_pit_plan_instruction(
                    state, state.pit_plan_override_reason,
                )
            if should_pit:
                if state.tire_inventory is not None and not self._prepare_inventory_pit(
                    state,
                    track,
                    weather,
                    lap,
                    physical_total_laps=physical_total_laps,
                    current_traffic_gaps=traffic_snapshot.current_traffic_gaps,
                    additional_current_stop_cost=delay,
                ):
                    continue
                pitting.append(state)
                expected_service = expected_stationary_time(state.car)
                expected_releases[team] = arrival + delay + expected_service
                expected_losses[state.driver.id] = (
                    track.pit_lane_delta * self._pit_lane_factor()
                    + expected_service
                    + delay
                )

        actual_releases: dict[str, float] = {}
        for state in pitting:
            state.total_time += self._execute_pit_stop(
                state, track, weather, current_lap=lap,
                pit_box_releases=actual_releases,
                arrival_time=arrivals[state.driver.id][0],
                **({"physical_total_laps": physical_total_laps}
                   if physical_total_laps is not None else {}),
            )
            if state.status != DriverStatus.RACING:
                continue
            state.pit_stops += 1
            state.pit_laps.append(lap)
            self._commit_pit_plan_if_due(
                state,
                overridden=state.pit_plan_override_reason is not None,
            )
            state.force_pit_next_lap = False
        return pitting

    def _standard_pit_traffic_snapshot(
        self,
        state: DriverRaceState,
        lap_start_states: list[DriverRaceState],
        states: list[DriverRaceState],
        track: Track,
        queue_delay: float,
        committed_losses: dict[str, float],
    ) -> StrategyTrafficSnapshot:
        """Project only earlier committed stops for this standard decision.

        Strategy decisions happen together at a lap boundary, while the
        constructor queue is committed in frozen arrival order.  The traffic
        projection therefore uses lap-start clocks and positions, then folds
        in expected losses for stops already selected in this batch.  It does
        not sample service, mutate live states, or make choices for rivals
        that have not reached their decision yet.
        """
        active_ids = {
            candidate.driver.id
            for candidate in states
            if candidate.status == DriverStatus.RACING
        }
        frozen = {
            candidate.driver.id: candidate
            for candidate in lap_start_states
            if candidate.status == DriverStatus.RACING
            and candidate.driver.id in active_ids
        }
        if state.driver.id not in frozen:
            # The caller only asks for snapshots for active candidates. Keep a
            # defensive empty snapshot for direct/instrumented callers that
            # violate that invariant.
            return StrategyTrafficSnapshot(None, None, 0.0, None)

        # The native gap and merge helpers read only driver.id, status,
        # position, and total_time. Use compact rows on this hot path instead
        # of cloning each DriverRaceState (which also carries tire history,
        # proposals, inventory, and other nested mutable state). Preserve the
        # historical full-state path for instrumented helpers and subclasses:
        # overrides may rely on any DriverRaceState field.
        native_helpers = (
            getattr(self._get_gap_to_car_ahead, "__func__", None)
            is RaceSimulator._NATIVE_PIT_TRAFFIC_GAP_AHEAD
            and getattr(self._get_gap_to_car_behind, "__func__", None)
            is RaceSimulator._NATIVE_PIT_TRAFFIC_GAP_BEHIND
            and getattr(self._handle_pit_batch_position_changes, "__func__", None)
            is RaceSimulator._NATIVE_PIT_BATCH_POSITION_CHANGES
        )

        # A failed finite-inventory preparation can retire a car between two
        # decisions. Compact only the copied rows so a surviving car keeps
        # seeing its nearest physical predecessor despite the old position
        # hole; the live and frozen race states remain untouched.
        if native_helpers:
            base_rows = [
                _PitTrafficProjectionRow(
                    candidate.driver,
                    candidate.position,
                    candidate.total_time,
                    candidate.status,
                )
                for candidate in frozen.values()
            ]
        else:
            base_rows = [replace(candidate) for candidate in frozen.values()]
        if sorted(candidate.position for candidate in base_rows) != list(
            range(1, len(base_rows) + 1)
        ):
            for position, candidate in enumerate(
                sorted(base_rows, key=lambda row: row.position), 1
            ):
                candidate.position = position

        observed_rows = (
            base_rows
            if native_helpers
            else [replace(candidate) for candidate in base_rows]
        )
        observed_by_id = {
            candidate.driver.id: candidate for candidate in observed_rows
        }
        observed = observed_by_id[state.driver.id]
        gap_ahead = self._get_gap_to_car_ahead(observed, observed_rows)
        gap_behind = self._get_gap_to_car_behind(observed, observed_rows)

        # Existing planning deliberately has no green-running traffic model
        # while control compresses the field.  Match that contract while
        # retaining the physical observations for reactive strategy rules.
        if not self.event_manager.is_active_aero_allowed():
            return StrategyTrafficSnapshot(gap_ahead, gap_behind, 0.0, None)

        def merged_gap(candidate_loss: float | None) -> float | None:
            # Recreate both alternatives from the same original physical
            # order. The stay branch may move a car ahead of a committed
            # pitter; that virtual position must not become the tie-breaker
            # for the candidate-inclusive pit branch.
            if native_helpers:
                projected = [
                    _PitTrafficProjectionRow(
                        row.driver, row.position, row.total_time, row.status,
                    )
                    for row in base_rows
                ]
            else:
                projected = [replace(row) for row in base_rows]
            pitting = []
            for row in projected:
                loss = committed_losses.get(row.driver.id)
                if loss is None and row.driver.id == state.driver.id:
                    loss = candidate_loss
                if loss is None:
                    continue
                row.total_time += loss
                pitting.append(row)
            self._handle_pit_batch_position_changes(pitting, projected)
            own = next(row for row in projected if row.driver.id == state.driver.id)
            return self._get_gap_to_car_ahead(own, projected)

        expected_stay_gap = merged_gap(None)
        expected_pit_gap = merged_gap(
            track.pit_lane_delta * self._pit_lane_factor()
            + expected_stationary_time(state.car)
            + queue_delay
        )
        traffic = self.lap_simulator.traffic_pace_contribution
        rejoin_cost = traffic(expected_pit_gap) - traffic(expected_stay_gap)
        return StrategyTrafficSnapshot(
            gap_ahead,
            gap_behind,
            rejoin_cost,
            (expected_stay_gap, expected_pit_gap),
        )

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
        weather_intervals: tuple[int, ...] | None = None,
        weather_clock: StrategyWeatherClock | None = None,
        current_overtake_mode_allowed: bool | None = None,
    ) -> bool:
        """Decide if driver should pit this lap."""
        state.dry_pit_proposal = None
        state.weather_pit_proposal = None
        state.pit_decision_context = None
        observed_gap = (self._get_gap_to_car_ahead(state, all_states)
                        if traffic_snapshot is None else traffic_snapshot.gap_ahead)
        mode_active = self._strategy_overtake_mode_active(
            state, track, lap, weather, observed_gap, current_overtake_mode_allowed,
        )
        if state.tire_inventory is not None:
            return self._should_pit_inventory(
                state, all_states, track, lap, weather if weather is not None else Weather(),
                additional_current_stop_cost, physical_total_laps,
                traffic_snapshot, weather_intervals, weather_clock,
                current_overtake_mode_active=mode_active,
            )
        clearly_dry = weather is None or (
            weather.track_wetness < 0.08 and weather.rain_intensity < 0.15
        )
        mixed_slick = (weather is not None and lap > 1 and not clearly_dry
                       and state.current_tire.compound in {
                           TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
                       })
        rain_transition = mixed_slick or (
            weather is not None and lap > 1 and self._has_used_wet_compound(state)
            and state.current_tire.compound in {TireCompound.INTERMEDIATE, TireCompound.WET}
            and not self._rain_stint_can_be_planned(
                state, track, weather, lap, weather_intervals,
            )
        )
        # A stop can advance an externally anchored weather clock across a
        # compound transition even when ordinary own-lap intervals do not.
        if (weather_clock is not None and lap > 1
                and state.current_tire.compound in {
                    TireCompound.INTERMEDIATE, TireCompound.WET,
                }):
            rain_transition = True
        # CRITICAL: Force pit if tires are completely wrong for conditions
        if weather is not None:
            tire_mismatch = self._check_tire_weather_mismatch(state.current_tire, weather)
            if tire_mismatch == "critical":
                self._capture_pit_decision_context(state, lap, "critical_weather")
                return True  # Must pit immediately
            elif tire_mismatch == "suboptimal" and not rain_transition:
                dry_rule_satisfied = self._stay_satisfies_tire_rule(state)
                if (
                    dry_rule_satisfied
                    and not self._weather_stop_can_pay(
                        state, track, weather, lap, additional_current_stop_cost,
                        weather_intervals=weather_intervals,
                        weather_clock=weather_clock,
                        traffic_possible=any(
                            other.status == DriverStatus.RACING
                            and other.driver.id != state.driver.id for other in all_states
                        ),
                        **({"current_overtake_mode_active": True} if mode_active else {}),
                        **({"physical_total_laps": physical_total_laps}
                           if physical_total_laps is not None else {}),
                    )
                ):
                    return False
                if self.rng.random() < 0.7:
                    self._capture_pit_decision_context(state, lap, "weather_reaction")
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

        dry_planning = clearly_dry and state.current_tire.compound in {
            TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
        }
        # Dry optimization chooses how many stops pay for themselves; style
        # must not exclude a faster legal schedule before it is evaluated.
        max_stops = (self._dry_stop_budget(state, track) if dry_planning
                     else self._ordinary_stop_budget(state, track))
        mixed_surfaces = (tuple(projected_surfaces(
            weather, track.total_laps - lap + 1, weather_intervals,
        )) if mixed_slick else ())
        if any(surface.track_wetness < .08 and surface.rain_intensity < .15
               for surface in mixed_surfaces):
            # Reserve later dry stops while the planner's damp allowance still
            # limits elective stops on the current wet surface.
            max_stops = max(max_stops, self._dry_stop_budget(state, track))
        if weather is not None and (
            weather.track_wetness > 0.3
            or state.current_tire.compound in {TireCompound.INTERMEDIATE, TireCompound.WET}
            or any(surface.fresh_rain_compound() is not None for surface in mixed_surfaces)
        ):
            # Retain the rain-stint allowance while the surface dries. Dropping
            # it at 0.3 would block a forecast intermediate-to-slick transition.
            max_stops = max(max_stops, 4)
        if rain_transition:
            max_stops = _timed_stop_budget_envelope(
                max_stops, self._dry_stop_budget(state, track),
                self._ordinary_stop_budget(state, track), weather_clock,
            )

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
            and dry_rule_required and not rain_transition
        ):
            self._capture_pit_decision_context(state, lap, "compound_requirement")
            return True

        # Elective stops need at least one lap on the starting set. Weather
        # and mandatory safeguards above still take priority.
        if lap <= 1:
            return False

        def current_traffic():
            if (self.event_manager.safety_car_active or self.event_manager.vsc_active
                    or self.event_manager.red_flag_active):
                return {}, 0.0
            if traffic_snapshot is None:
                gaps = self._pit_rejoin_traffic_gaps(
                    state, all_states, track, additional_current_stop_cost,
                )
            else:
                gaps = traffic_snapshot.current_traffic_gaps
                if gaps is None:
                    # Preserve explicit legacy snapshots that supply only a
                    # scalar cost. Native engines retain both observed gaps.
                    return {}, traffic_snapshot.rejoin_traffic_cost
            return {"current_traffic_gaps": gaps}, 0.0

        if dry_planning:
            tire_multiplier = (self.lap_simulator.weather_pace_multiplier(
                state.driver, state.car, weather,
            ) if weather is not None else 1.0)
            traffic_options, traffic_cost = current_traffic()
            traffic_cost *= tire_multiplier
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
                **traffic_options,
                current_set_used=state.tire_laps > state.prior_tire_laps,
                **({"tire_warmup": self.tire_warmup,
                    "current_fit_pending": state.fit_lap_pending}
                   if self.tire_warmup else {}),
            )
            mode_gain = self._strategy_mode_gain(
                state, track, weather, lap, mode_active,
                traffic_options.get("current_traffic_gaps", (None, None))[0],
                physical_total_laps,
            )
            if mode_gain:
                decision = replace(decision, wait_cost=decision.wait_cost - mode_gain)
            timing_bias = {
                TeamStrategyArchetype.AGGRESSIVE: 0.1,
                TeamStrategyArchetype.BALANCED: 0.0,
                TeamStrategyArchetype.CONSERVATIVE: -0.1,
            }[strategy]
            if decision.should_pit(timing_bias):
                state.dry_pit_proposal = (lap, decision.compound)
                self._capture_pit_decision_context(state, lap, "dry_forecast", decision)
                return True
            return False

        if weather is not None and (
            rain_transition or self._rain_stint_can_be_planned(
                state, track, weather, lap, weather_intervals,
            )
        ):
            traffic_options, traffic_cost = current_traffic()
            traffic_cost *= self.lap_simulator.weather_pace_multiplier(
                state.driver, state.car, weather,
            )
            planner = plan_rain_transition if rain_transition else plan_rain_stop
            decision = planner(
                state.driver, state.car, track, weather, state.current_tire,
                state.tire_laps, lap, max_stops - state.pit_stops,
                pit_lane_factor=self._pit_lane_factor(),
                additional_current_stop_cost=additional_current_stop_cost + traffic_cost,
                current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
                active_aero_enabled=self.event_manager.is_active_aero_allowed(),
                physical_total_laps=physical_total_laps,
                weather_intervals=weather_intervals,
                **traffic_options,
                **({"weather_clock": weather_clock} if weather_clock is not None else {}),
                **({"tire_warmup": self.tire_warmup,
                    "current_fit_pending": state.fit_lap_pending}
                   if self.tire_warmup else {}),
                **({
                    "used_compounds": self._actually_used_compounds(state),
                    "remaining_dry_stops": max(
                        0, self._dry_stop_budget(state, track) - state.pit_stops,
                    ),
                    "remaining_damp_stops": max(
                        0, self._ordinary_stop_budget(state, track) - state.pit_stops,
                    ),
                } if rain_transition else {}),
            )
            mode_gain = self._strategy_mode_gain(
                state, track, weather, lap, mode_active,
                traffic_options.get("current_traffic_gaps", (None, None))[0],
                physical_total_laps,
            )
            if mode_gain:
                decision = replace(decision, wait_cost=decision.wait_cost - mode_gain)
            if decision.should_pit():
                if rain_transition:
                    state.weather_pit_proposal = (lap, decision.compound)
                self._capture_pit_decision_context(state, lap, "rain_forecast", decision)
                return True
            return False

        # Slicks in damp conditions retain the reactive fallback windows.
        if lap <= 5 or lap >= track.total_laps - 5:
            return False

        def wet_stop_can_pay() -> bool:
            # A window proposes a stop; even an optimistic fresh-set plan
            # must recover its paid loss before we accept that proposal.
            if weather is None:
                return True  # Legacy callers supplied no surface to project.
            return self._weather_stop_can_pay(
                state, track, weather, lap, additional_current_stop_cost,
                weather_intervals=weather_intervals,
                weather_clock=weather_clock,
                traffic_possible=any(
                    other.status == DriverStatus.RACING and other.driver.id != state.driver.id
                    for other in all_states
                ),
                **({"current_overtake_mode_active": True} if mode_active else {}),
                **({"physical_total_laps": physical_total_laps}
                   if physical_total_laps is not None else {}),
            )

        # Pit window opportunity (under SC/VSC) - usually strong strategic value.
        if pit_window_open and state.tire_laps > 10 and state.pit_stops < max_stops:
            # If we have a free stop window (big gap behind), almost always take it.
            if gap_behind is not None and gap_behind > track.pit_lane_delta * 0.85:
                accepted = wet_stop_can_pay()
                if accepted:
                    self._capture_pit_decision_context(
                        state, lap, "neutralization_window",
                    )
                return accepted
            window_prob = np.clip(0.85 + strategy_bias, 0.55, 0.98)
            if self.rng.random() < window_prob:
                accepted = wet_stop_can_pay()
                if accepted:
                    self._capture_pit_decision_context(
                        state, lap, "neutralization_window",
                    )
                return accepted

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
                accepted = wet_stop_can_pay()
                if accepted:
                    self._capture_pit_decision_context(state, lap, "planned_window")
                return accepted

        return False

    @staticmethod
    def _rain_stint_can_be_planned(
        state: DriverRaceState, track: Track, weather: Weather, lap: int,
        weather_intervals: tuple[int, ...] | None = None,
    ) -> bool:
        compound = state.current_tire.compound
        if compound not in {TireCompound.INTERMEDIATE, TireCompound.WET}:
            return False
        for surface in projected_surfaces(weather, track.total_laps - lap + 1, weather_intervals):
            if surface.fresh_rain_compound() != compound:
                return False
        return True

    @staticmethod
    def _actually_used_compounds(state: DriverRaceState) -> set[TireCompound]:
        """Exclude only the current unrun fitted set, preserving earlier stints."""
        history = state.tire_compound_history
        if (state.tire_laps == state.prior_tire_laps and history
                and history[-1] == state.current_tire.compound.value):
            history = history[:-1]
        used = set()
        for compound in history:
            try:
                used.add(TireCompound(compound))
            except (TypeError, ValueError):
                continue
        if state.tire_laps > state.prior_tire_laps:
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
        if (state.tire_laps == state.prior_tire_laps and state.tire_compound_history
                and state.tire_compound_history[-1] == state.current_tire.compound.value):
            state.tire_compound_history[-1] = compound.value
        else:
            state.tire_compound_history.append(compound.value)
        state.current_tire = TIRE_COMPOUNDS[compound].model_copy(deep=True)
        state.tire_laps = 0
        state.prior_tire_laps = 0
        state.driver.current_tire_laps = 0
        state.fit_lap_pending = True

    def _consume_tire_warmup(self, state: DriverRaceState) -> float:
        """Consume one fitted set's cost on its first sampled running lap."""
        if not state.fit_lap_pending:
            return 0.0
        state.fit_lap_pending = False
        return tire_warmup_seconds(self.tire_warmup, state.current_tire.compound)

    def _choose_distinct_dry_compound(
        self,
        state: DriverRaceState,
        track: Track,
        current_lap: int,
        weather: Weather | None = None,
        *, physical_total_laps: int | None = None,
        weather_intervals: tuple[int, ...] | None = None,
        weather_clock: StrategyWeatherClock | None = None,
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
            weather_intervals=weather_intervals,
            weather_clock=weather_clock,
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
        weather_intervals: tuple[int, ...] | None = None,
        weather_clock: StrategyWeatherClock | None = None,
    ) -> TireCompound:
        """Rank fresh slicks using the same tyre pace as the actual race."""
        return self._rank_stint_compounds(
            state, track, current_lap,
            [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD],
            weather=weather, physical_total_laps=physical_total_laps,
            weather_intervals=weather_intervals,
            weather_clock=weather_clock,
        )

    def _projected_stint_weather(
        self, weather: Weather, weather_clock: StrategyWeatherClock | None,
        weather_intervals: tuple[int, ...] | None, target_stint: int,
        *, fit_delay: float = 0.0,
    ) -> tuple[Weather, tuple[int, ...] | None]:
        """Rebase a paid-stop weather path onto the replacement's outlap.

        ``weather`` is the observed entry snapshot.  A clock-aware stop first
        advances that snapshot through the expected physical stop, then
        supplies cumulative updates relative to the fresh set's outlap.  The
        ordinary cadence path is retained for callers that have no paid-stop
        clock.  Traffic costs are deliberately absent: they are strategy
        prices, not physical elapsed time.
        """
        if target_stint <= 0:
            return weather, ()
        if weather_clock is not None:
            if fit_delay and type(weather_clock) is not StrategyWeatherClock:
                raise ValueError("tire_warmup requires the native StrategyWeatherClock")
            if target_stint > len(weather_clock.lap_start_offsets):
                return weather, None
            try:
                first = weather_clock.updates(0, 1, True)
                counts = tuple(
                    (weather_clock.updates(offset, 1, True, fit_delay=fit_delay)
                     if fit_delay else weather_clock.updates(offset, 1, True)) - first
                    for offset in range(target_stint)
                )
            except (TypeError, ValueError, OverflowError):
                return weather, None
            surface = weather
            for _ in range(first):
                surface = surface.project_surface()
            return surface, counts
        if weather_intervals is None or len(weather_intervals) < target_stint:
            return weather, None
        origin = weather_intervals[0]
        surface = weather
        for _ in range(origin):
            surface = surface.project_surface()
        return surface, tuple(value - origin for value in weather_intervals[:target_stint])

    def _rank_stint_compounds(
        self, state: DriverRaceState, track: Track, current_lap: int,
        available: list[TireCompound],
        *, weather: Weather | None = None, physical_total_laps: int | None = None,
        weather_intervals: tuple[int, ...] | None = None,
        weather_clock: StrategyWeatherClock | None = None,
    ) -> TireCompound:
        target_stint = self._next_stint_laps(state, track, current_lap)
        costs = {}
        for compound in available:
            fit_cost = tire_warmup_seconds(self.tire_warmup, compound)
            stint_weather, stint_intervals = self._projected_stint_weather(
                weather if weather is not None else Weather(), weather_clock,
                weather_intervals, target_stint,
                fit_delay=fit_cost if self.tire_warmup else 0.0,
            )
            cost = self.lap_simulator.projected_stint_lap_cost(
                state.driver, state.car, track, TIRE_COMPOUNDS[compound], target_stint,
                current_lap, stint_weather,
                physical_total_laps=physical_total_laps,
                current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
                active_aero_enabled=self.event_manager.is_active_aero_allowed(),
                **({"weather_intervals": stint_intervals}
                   if stint_intervals is not None else {}),
            )
            costs[compound] = cost + fit_cost if fit_cost else cost
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
                **({"tire_warmup": self.tire_warmup,
                    "current_fit_pending": True}
                   if self.tire_warmup else {}),
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
        weather_intervals: tuple[int, ...] | None = None,
        weather_clock: StrategyWeatherClock | None = None,
    ) -> float:
        """Execute pit stop and return total time lost.

        Args:
            state: Driver state
            track: Current track
            weather: Current weather

        Returns:
            Time lost in seconds
        """
        forced_repair = bool(state.force_pit_next_lap)
        decision_context = state.pit_decision_context
        # Consume the proposal before any execution-side replanning.  A
        # failed preparation or later direct call must not reuse it.
        state.pit_decision_context = None
        decision_reason = "forced_repair" if forced_repair else None
        forecast_saving = None
        if not forced_repair and isinstance(decision_context, dict):
            if decision_context.get("lap") == int(current_lap):
                candidate_reason = decision_context.get("decision_reason")
                if candidate_reason in {
                    "critical_weather", "weather_reaction", "compound_requirement",
                    "dry_forecast", "rain_forecast", "inventory_forecast",
                    "neutralization_window", "planned_window", "forced_repair",
                    "user_plan",
                }:
                    decision_reason = candidate_reason
                    candidate_saving = decision_context.get("forecast_saving_seconds")
                    if (isinstance(candidate_saving, Real)
                            and not isinstance(candidate_saving, bool)):
                        try:
                            candidate_saving = float(candidate_saving)
                        except (OverflowError, TypeError, ValueError):
                            candidate_saving = None
                        if candidate_saving is not None and isfinite(candidate_saving):
                            forecast_saving = candidate_saving
        selected_set = None
        if state.tire_inventory is not None:
            if not self._prepare_inventory_pit(
                state, track, weather, current_lap, physical_total_laps=physical_total_laps,
                weather_intervals=weather_intervals, weather_clock=weather_clock,
            ):
                return 0.0
            selected_set = state.inventory_pit_proposal[1]
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
        weather_proposal = state.weather_pit_proposal
        custom_target = state.pit_plan_target
        state.dry_pit_proposal = None
        state.weather_pit_proposal = None
        if selected_set is not None:
            new_compound = state.tire_inventory.sets[selected_set].compound
        elif custom_target is not None:
            # Explicit instructions bypass automatic profitability and forecast
            # choices after the request has passed the safety checks.
            new_compound = custom_target
        elif weather_compound is not None:
            new_compound = weather_compound
        elif (
            weather_proposal is not None and weather_proposal[0] == current_lap
            and weather_proposal[1] in {TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD}
            and (self._has_used_wet_compound(state) or state.current_tire.compound in {
                TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
            })
        ):
            new_compound = weather_proposal[1]
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
                weather_intervals=weather_intervals, weather_clock=weather_clock,
            )
        else:
            new_compound = self._choose_compound_for_next_stint(
                state,
                track,
                current_lap,
                weather, physical_total_laps=physical_total_laps,
                weather_intervals=weather_intervals, weather_clock=weather_clock,
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
            "decision_reason": decision_reason,
            "forecast_saving_seconds": forecast_saving,
        })
        if selected_set is not None:
            state.pit_stop_details[-1].update(
                from_set_id=state.tire_inventory.current_set_id,
                to_set_id=selected_set,
                incoming_tire_age=state.tire_inventory.sets[selected_set].age,
            )
            self._fit_inventory_tire(state, selected_set, current_lap, "pit")
            state.inventory_pit_proposal = None
        else:
            self._fit_tire(state, new_compound)

        return total_loss

    def _pit_rejoin_traffic_cost(
        self, state: DriverRaceState, all_states: list[DriverRaceState],
        track: Track, queue_delay: float = 0.0,
    ) -> float:
        """Legacy unscaled dirty-air difference; native planning retains gaps."""
        stay_gap, rejoin_gap = self._pit_rejoin_traffic_gaps(state, all_states, track, queue_delay)
        return (
            self.lap_simulator.traffic_pace_contribution(rejoin_gap)
            - self.lap_simulator.traffic_pace_contribution(stay_gap)
        )

    def _pit_rejoin_traffic_gaps(
        self, state: DriverRaceState, all_states: list[DriverRaceState],
        track: Track, queue_delay: float = 0.0,
    ) -> tuple[float | None, float | None]:
        """Expected stay/rejoin gaps for the current green running lap.

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
        return stay_gap, rejoin_gap

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

    def _strategy_overtake_mode_active(
        self, state, track, lap, weather, gap, mode_allowed=None,
    ) -> bool:
        """Read current deployment eligibility without consuming the energy store."""
        allowed = (self.event_manager.is_overtake_mode_allowed(lap, weather)
                   if mode_allowed is None else mode_allowed)
        return bool(allowed and gap is not None
                    and gap <= track.overtake_mode_detection_gap
                    and float(np.clip(state.overtake_mode_energy, 0.0, 1.0)) + 1e-12
                    >= self.OVERTAKE_MODE_DEPLOYMENT_COST)

    def _strategy_mode_gain(
        self, state, track, weather, lap, active, gap, physical_total_laps=None,
    ) -> float:
        """Exact retained first-lap gain; future laps and paid stops receive none."""
        if not active:
            return 0.0
        driver = state.driver.model_copy(deep=True)
        driver.current_tire_laps = state.tire_laps
        physics = LapSimulator(np.random.default_rng(0))
        args = (driver, state.car, track, state.current_tire,
                weather if weather is not None else Weather(), lap,
                track.total_laps if physical_total_laps is None else physical_total_laps)
        options = dict(gap_to_car_ahead=gap, sample_variation=False,
                       active_aero_enabled=self.event_manager.is_active_aero_allowed())
        without = physics.calculate_lap_time(*args, **options, overtake_mode_active=False)
        deployed = physics.calculate_lap_time(*args, **options, overtake_mode_active=True)
        return (without - deployed) * self.event_manager.get_lap_time_modifier()

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

    # Stable references let the projection detect instrumentation applied at
    # either the instance or class level. Looking up RaceSimulator's current
    # helper attributes would miss class-level monkeypatches because both
    # sides of that comparison would resolve to the patched function.
    _NATIVE_PIT_TRAFFIC_GAP_AHEAD = _get_gap_to_car_ahead
    _NATIVE_PIT_TRAFFIC_GAP_BEHIND = _get_gap_to_car_behind
    _NATIVE_PIT_BATCH_POSITION_CHANGES = _handle_pit_batch_position_changes

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

    def _safety_car_lap_times(
        self,
        free_lap_times: dict[str, float],
        running_start_states: list[DriverRaceState],
        modifier: float,
    ) -> dict[str, float]:
        """Form a queue using positive running time and committed pit exits.

        The first car after the pit merge sets the common SC pace. Followers
        close excess gaps at no faster than their own free pace, using the
        preceding car's projected crossing so catch-up propagates down the
        queue. A slower predecessor can impose additional blocked time.
        Service and lane losses are already in the starting clocks; they are
        never multiplied by the SC modifier or removed from a completed lap.
        """
        queue = sorted(
            (state for state in running_start_states if state.driver.id in free_lap_times),
            key=lambda state: state.position,
        )
        if not queue:
            return {}
        queue_pace = free_lap_times[queue[0].driver.id] * modifier
        result = {}
        ahead_crossing = None
        for state in queue:
            free = free_lap_times[state.driver.id]
            nominal = max(free, queue_pace)
            gap = (None if ahead_crossing is None else
                   max(0.0, state.total_time + nominal - ahead_crossing))
            running = safety_car_running_time(free, nominal, gap)
            if ahead_crossing is not None:
                running = max(running, ahead_crossing - state.total_time)
            result[state.driver.id] = running
            ahead_crossing = state.total_time + running
        return result

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

        The standard red-flag regrouping approximation replaces the
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

        The live loop models collection and the shared restart clock itself.
        This direct helper performs only free refits/repairs and leaves past
        race clocks and physical order untouched.

        Args:
            states: Driver race states
            weather: Current weather conditions
            track: Race distance and tyre physics
            current_lap: Lap completed before suspension
            defer_tire_fit: Skip fitting when a caller will apply it separately
        """
        if not defer_tire_fit:
            self._fit_red_flag_tires(states, weather, track, current_lap)

        # Keep race-control ordering unchanged.  Clock advancement belongs to
        # the live race loop, which records collection and pause explicitly;
        # direct callers retain only the free-fit/repair behavior.
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

            if state.tire_inventory is not None:
                self._refit_inventory_free(
                    state, track, weather, current_lap, physical_total_laps=physical_total_laps,
                )
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
            state.pit_decision_context = None
            state.dry_pit_proposal = None
            state.weather_pit_proposal = None

    def _choose_red_flag_tire(
        self, state: DriverRaceState, weather: Weather, track: Track, current_lap: int,
        *, physical_total_laps: int | None = None,
        weather_intervals: tuple[int, ...] | None = None,
        weather_clock: StrategyWeatherClock | None = None,
    ) -> TireCompound:
        """Price a free set from the next lap, including future paid stops."""
        remaining_laps = track.total_laps - current_lap
        if remaining_laps <= 0:
            return state.current_tire.compound
        # Keep the inexpensive dry forecast when the whole no-stop weather
        # path stays clearly dry. A cadence or external clock can reveal a
        # future crossover even when the current snapshot still permits slicks.
        projected = projected_surfaces(weather, remaining_laps, weather_intervals)
        transition_forecast = any(
            surface.track_wetness >= .08 or surface.rain_intensity >= .15
            for surface in projected
        ) or (weather_clock is not None and weather.rain_intensity >= .08)

        used = self._used_slick_compounds(state)
        wet_exemption = self._has_used_wet_compound(state)

        if transition_forecast:
            actual_used = self._actually_used_compounds(state)
            dry_limit = self._dry_stop_budget(state, track)
            damp_limit = self._ordinary_stop_budget(state, track)

            def transition_cost(compound: TireCompound) -> float:
                # A free fit is usable whenever it is not currently critical;
                # its future suitability and every paid replacement remain in
                # the transition planner's bounded weather graph.
                if weather.tire_mismatch(compound) == "critical":
                    return float("inf")
                maximum = damp_limit
                if any(surface.track_wetness < .08
                       and surface.rain_intensity < .15 for surface in projected):
                    maximum = max(maximum, dry_limit)
                if (compound in {TireCompound.INTERMEDIATE, TireCompound.WET}
                        or any(surface.track_wetness > .3
                               or surface.fresh_rain_compound() is not None
                               for surface in projected)):
                    maximum = max(maximum, 4)
                maximum = _timed_stop_budget_envelope(
                    maximum, dry_limit, damp_limit, weather_clock,
                )
                remaining_stops = max(0, maximum - state.pit_stops)
                return plan_rain_transition(
                    state.driver, state.car, track, weather,
                    TIRE_COMPOUNDS[compound], 0, current_lap + 1,
                    remaining_stops,
                    current_lap_time_modifier=self.event_manager.get_lap_time_modifier(),
                    active_aero_enabled=self.event_manager.is_active_aero_allowed(),
                    physical_total_laps=physical_total_laps,
                    weather_intervals=weather_intervals,
                    remaining_dry_stops=max(0, dry_limit - state.pit_stops),
                    remaining_damp_stops=max(0, damp_limit - state.pit_stops),
                    used_compounds=actual_used | {compound},
                    weather_clock=weather_clock,
                    **({"tire_warmup": self.tire_warmup,
                        "current_fit_pending": True}
                       if self.tire_warmup else {}),
                ).wait_cost

            candidates = [compound for compound in TireCompound
                          if weather.tire_mismatch(compound) != "critical"]
            if candidates:
                return min(candidates, key=transition_cost)

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
                **({"tire_warmup": self.tire_warmup,
                    "current_fit_pending": True}
                   if self.tire_warmup else {}),
            ).wait_cost

        return min(
            (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD),
            key=finish_cost,
        )

    def _weather_stop_can_pay(
        self, state: DriverRaceState, track: Track, weather: Weather, current_lap: int,
        additional_current_stop_cost: float = 0.0, *, traffic_possible: bool = True,
        physical_total_laps: int | None = None,
        weather_intervals: tuple[int, ...] | None = None,
        weather_clock: StrategyWeatherClock | None = None,
        current_overtake_mode_active: bool = False,
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
            weather_intervals=weather_intervals,
            **({"weather_clock": weather_clock} if weather_clock is not None else {}),
            **({"tire_warmup": self.tire_warmup,
                "current_fit_pending": state.fit_lap_pending}
               if self.tire_warmup else {}),
        )
        gain = self._strategy_mode_gain(
            state, track, weather, current_lap, current_overtake_mode_active,
            0.0 if traffic_possible else None, physical_total_laps,
        )
        return costs.pit_now_cost < costs.stay_cost - gain

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
