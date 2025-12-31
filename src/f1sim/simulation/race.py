"""Race simulation engine."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.events import EventManager, RaceEvent
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.overtaking import OvertakingModel


class DriverStatus(str, Enum):
    """Driver race status."""

    RACING = "racing"
    FINISHED = "finished"
    DNF = "dnf"


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

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize race simulator.

        Args:
            rng: Random number generator
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.lap_simulator = LapSimulator(rng=self.rng)
        self.overtaking_model = OvertakingModel(rng=self.rng)
        self.event_manager = EventManager(rng=self.rng)

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
        # Reset event manager
        self.event_manager.reset()

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

            # Starting tire (default to medium for Q2+ drivers, soft for others)
            if starting_tires and driver_id in starting_tires:
                tire_compound = starting_tires[driver_id]
            else:
                tire_compound = TireCompound.MEDIUM if pos <= 10 else TireCompound.SOFT

            states.append(DriverRaceState(
                driver=driver,
                car=car,
                position=pos,
                current_tire=TIRE_COMPOUNDS[tire_compound].model_copy(deep=True),
            ))

        # Track fastest laps
        fastest_laps: dict[str, float] = {}
        current_weather = weather.model_copy(deep=True)
        all_events: list[RaceEvent] = []

        # Simulate each lap
        for lap in range(1, track.total_laps + 1):
            # Evolve weather
            current_weather = current_weather.evolve(self.rng)

            # Simulate lap for each driver
            lap_times: dict[str, float] = {}
            incidents_this_lap = 0
            drivers_pitting: list[DriverRaceState] = []

            # Phase 1: Determine who pits and calculate lap times
            for state in states:
                if state.status != DriverStatus.RACING:
                    continue

                # Check for pit stop decision
                should_pit = self._should_pit(
                    state, track, lap,
                    self.event_manager.is_pit_window_open()
                )

                if should_pit:
                    pit_time = self._execute_pit_stop(state, track, current_weather)
                    state.total_time += pit_time
                    state.pit_stops += 1
                    state.pit_laps.append(lap)
                    drivers_pitting.append(state)

                # Calculate lap time
                gap_ahead = self._get_gap_to_car_ahead(state, states)

                lap_time = self.lap_simulator.calculate_lap_time(
                    driver=state.driver,
                    car=state.car,
                    track=track,
                    tire=state.current_tire,
                    weather=current_weather,
                    lap_number=lap,
                    total_laps=track.total_laps,
                    gap_to_car_ahead=gap_ahead,
                    is_drs_enabled=lap > 2 and not self.event_manager.safety_car_active,
                )

                # Apply safety car modifier
                lap_time *= self.event_manager.get_lap_time_modifier()

                state.total_time += lap_time
                state.last_lap_time = lap_time
                state.tire_laps += 1
                state.driver.current_tire_laps = state.tire_laps

                lap_times[state.driver.id] = lap_time

                # Track fastest lap
                if state.driver.id not in fastest_laps:
                    fastest_laps[state.driver.id] = lap_time
                else:
                    fastest_laps[state.driver.id] = min(fastest_laps[state.driver.id], lap_time)

            # Phase 2: Handle position changes from pit stops (after all lap times calculated)
            # At high overtake difficulty tracks (Monaco, Singapore), pit stops have minimal
            # position impact because everyone pits in a narrow window and can't recover via overtaking
            if track.overtake_difficulty < 0.8:  # Only apply at easier-to-pass tracks
                for pitting_driver in drivers_pitting:
                    self._handle_pit_position_changes(pitting_driver, states)

            # Phase 3: Process overtakes (skip at extremely difficult tracks like Monaco)
            if track.overtake_difficulty < 0.9:
                incidents_this_lap += self._process_overtakes(
                    states, track, current_weather
                )

            # Process events
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
                        if state.driver.id == driver_id and state.driver.dnf:
                            state.status = DriverStatus.DNF
                            state.dnf_reason = state.driver.dnf_reason

            all_events.extend(lap_events)

            # Update positions
            self._update_positions(states)

        # Mark finished drivers
        for state in states:
            if state.status == DriverStatus.RACING:
                state.status = DriverStatus.FINISHED

        # Calculate overall fastest lap
        overall_fastest = min(fastest_laps.values()) if fastest_laps else float("inf")
        fastest_lap_driver = min(fastest_laps, key=fastest_laps.get) if fastest_laps else None

        # Build results - use track position (state.position), not total time
        # DNF drivers go after finishers
        sorted_states = sorted(states, key=lambda s: (s.status == DriverStatus.DNF, s.position))

        # Find leader time for gap calculation
        leader_time = sorted_states[0].total_time if sorted_states and sorted_states[0].status == DriverStatus.FINISHED else 0

        results = []
        for state in sorted_states:
            # Build strategy list
            strategy = [state.current_tire.compound.value]
            if state.pit_stops > 0:
                # Simplified strategy tracking
                strategy = ["medium", "hard"] if state.pit_stops == 1 else ["soft", "hard", "medium"]

            results.append(RaceResult(
                driver_id=state.driver.id,
                driver_name=state.driver.name,
                team=state.car.team_name,
                position=state.position,
                total_time=state.total_time,
                gap_to_leader=state.total_time - leader_time if state.status == DriverStatus.FINISHED else 0,
                pit_stops=state.pit_stops,
                fastest_lap=fastest_laps.get(state.driver.id, 0),
                status=state.status,
                dnf_reason=state.dnf_reason,
                strategy=strategy,
            ))

        return results

    def _should_pit(
        self,
        state: DriverRaceState,
        track: Track,
        lap: int,
        pit_window_open: bool,
    ) -> bool:
        """Decide if driver should pit this lap."""
        # Limit to realistic number of pit stops (1-2 for most races)
        max_stops = 2 if track.total_laps > 50 else 1
        if state.pit_stops >= max_stops:
            return False

        # Mandatory pit stop check (must stop at least once)
        if lap == track.total_laps - 1 and state.pit_stops == 0:
            return True

        # Don't pit on first 5 laps or last 5 laps
        if lap <= 5 or lap >= track.total_laps - 5:
            return False

        # Pit window opportunity (under SC/VSC) - good strategy
        if pit_window_open and state.tire_laps > 15 and state.pit_stops < max_stops:
            if self.rng.random() < 0.8:
                return True

        # Calculate optimal pit windows for 1-stop or 2-stop strategy
        if max_stops == 1:
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
            adjusted_prob = max(0.05, base_prob - position_reluctance)

            if self.rng.random() < adjusted_prob:
                return True

        return False

    def _execute_pit_stop(
        self,
        state: DriverRaceState,
        track: Track,
        weather: Weather,
    ) -> float:
        """Execute pit stop and return total time lost.

        Args:
            state: Driver state
            track: Current track
            weather: Current weather

        Returns:
            Time lost in seconds
        """
        # Pit lane time + stationary time
        pit_lane_time = track.pit_lane_delta
        stationary_time = self.lap_simulator.calculate_pit_stop_time(state.car)

        # Choose new tire compound
        if weather.requires_wet_tires():
            new_compound = TireCompound.WET
        elif weather.is_wet():
            new_compound = TireCompound.INTERMEDIATE
        else:
            # Strategic choice: if on softs, go harder; if on hards, maybe mediums
            current = state.current_tire.compound
            if current == TireCompound.SOFT:
                new_compound = TireCompound.HARD if self.rng.random() < 0.6 else TireCompound.MEDIUM
            elif current == TireCompound.MEDIUM:
                new_compound = TireCompound.HARD if self.rng.random() < 0.5 else TireCompound.SOFT
            else:
                new_compound = TireCompound.MEDIUM if self.rng.random() < 0.6 else TireCompound.SOFT

        state.current_tire = TIRE_COMPOUNDS[new_compound].model_copy(deep=True)
        state.tire_laps = 0

        return pit_lane_time + stationary_time

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
                return abs(state.total_time - other.total_time)

        return None

    def _process_overtakes(
        self,
        states: list[DriverRaceState],
        track: Track,
        weather: Weather,
    ) -> int:
        """Process overtaking opportunities. Returns number of incidents."""
        incidents = 0
        racing_states = [s for s in states if s.status == DriverStatus.RACING]

        # Sort by position to check if faster cars are stuck behind slower ones
        racing_states.sort(key=lambda s: s.position)

        for i in range(1, len(racing_states)):
            # Car behind (higher position number)
            attacker = racing_states[i]
            # Car ahead (lower position number)
            defender = racing_states[i - 1]

            # Gap: positive means attacker is behind in time (normal)
            # Negative means attacker has caught up and is faster
            gap = attacker.total_time - defender.total_time

            # If attacker hasn't caught up yet, no overtake attempt
            if gap > 1.5:
                continue

            # Gap for overtake purposes (how close they are)
            overtake_gap = max(0.0, gap)  # If negative, they're right on their tail

            # Check if should attempt
            if not self.overtaking_model.should_attempt_overtake(
                attacker.driver,
                defender.driver,
                overtake_gap,
                track.total_laps,  # Would need remaining laps
                attacker.position,
            ):
                continue

            # Attempt overtake
            success, incident = self.overtaking_model.attempt_overtake(
                attacker=attacker.driver,
                attacker_car=attacker.car,
                defender=defender.driver,
                defender_car=defender.car,
                track=track,
                gap=overtake_gap,
                has_drs=overtake_gap <= 1.0 and not weather.is_wet(),
                is_wet=weather.is_wet(),
            )

            if success:
                # Swap positions
                attacker.position, defender.position = defender.position, attacker.position

            if incident:
                incidents += 1
                # Small time loss for both drivers
                attacker.total_time += self.rng.uniform(1, 3)
                defender.total_time += self.rng.uniform(0.5, 2)

        return incidents

    def _handle_pit_position_changes(
        self,
        pitting_driver: DriverRaceState,
        all_states: list[DriverRaceState],
    ) -> None:
        """Handle position changes when a driver pits.

        Only cars that were close behind (within pit window gap) can pass.
        Typical pit stop costs ~22-25 seconds, so only cars within ~20-25s can realistically pass.
        """
        original_pos = pitting_driver.position

        # Get the pitting driver's time BEFORE pit stop was added
        # (pit_time was already added to total_time before this is called)
        # Estimate: typical pit loss is ~23 seconds
        estimated_pit_loss = 23.0

        # Get all racing cars sorted by their current position
        racing = [s for s in all_states if s.status == DriverStatus.RACING]
        racing.sort(key=lambda s: s.position)

        # Find cars that were behind the pitting driver
        cars_behind = [s for s in racing if s.position > original_pos]

        # Count how many cars pass - only if they were within the pit window gap
        cars_passing = 0
        for state in cars_behind:
            # Calculate how far behind this car was before the pit stop
            # If pitting driver now has more time, the difference includes pit loss
            time_diff = pitting_driver.total_time - state.total_time

            # Only pass if they were close enough that the pit stop put them ahead
            # AND they're now ahead in total time
            if time_diff > 0 and time_diff < estimated_pit_loss:
                cars_passing += 1
                state.position -= 1

        # Move the pitting driver back by the number of cars that passed
        pitting_driver.position = original_pos + cars_passing

    def _update_positions(self, states: list[DriverRaceState]) -> None:
        """Update positions, respecting track position (no auto-sort by time).

        Positions only change through:
        1. Successful overtakes (handled in _process_overtakes)
        2. Pit stops (handled in _handle_pit_position_changes)
        3. DNFs (moved to back)
        """
        # Separate racing and DNF drivers
        racing = [s for s in states if s.status == DriverStatus.RACING]
        dnf = [s for s in states if s.status == DriverStatus.DNF]

        # Keep racing drivers in their current positions (no re-sort by time)
        # Just ensure DNF drivers are moved to the back
        racing.sort(key=lambda s: s.position)

        # Re-assign positions to ensure no gaps from DNFs
        for pos, state in enumerate(racing, 1):
            state.position = pos

        # DNF drivers get positions after racing drivers
        for i, state in enumerate(dnf):
            state.position = len(racing) + i + 1
