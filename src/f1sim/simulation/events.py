"""Race events: safety car, failures, incidents."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from f1sim.models import Car, Driver, Track, Weather


class EventType(str, Enum):
    """Types of race events."""

    SAFETY_CAR = "safety_car"
    VIRTUAL_SAFETY_CAR = "virtual_safety_car"
    RED_FLAG = "red_flag"
    MECHANICAL_FAILURE = "mechanical_failure"
    COLLISION = "collision"
    SPIN = "spin"
    PUNCTURE = "puncture"


@dataclass
class RaceEvent:
    """Represents a race event."""

    event_type: EventType
    lap: int
    drivers_involved: list[str] = field(default_factory=list)
    duration_laps: int = 0
    description: str = ""


class EventManager:
    """Manages race events and their effects."""

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize the event manager.

        Args:
            rng: Random number generator
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.events: list[RaceEvent] = []
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.vsc_active = False
        self.vsc_laps_remaining = 0
        self.sc_just_ended = False  # Flag for restart lap
        self.sc_restart_lap = False  # True on the lap after SC ends
        # Red flag state
        self.red_flag_active = False
        self.red_flag_just_ended = False  # Flag for restart lap after red flag
        self.red_flag_restart_lap = False  # True on the lap after red flag ends
        # Manual trigger configuration
        self.forced_red_flag_laps: set[int] = set()  # Laps to force red flags
        self.forced_safety_car_laps: set[int] = set()  # Laps to force safety cars

    def reset(self) -> None:
        """Reset event state for new race."""
        self.events = []
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.vsc_active = False
        self.vsc_laps_remaining = 0
        self.sc_just_ended = False
        self.sc_restart_lap = False
        self.red_flag_active = False
        self.red_flag_just_ended = False
        self.red_flag_restart_lap = False
        # Note: forced laps are NOT reset - they persist across races

    def set_forced_red_flag(self, laps: list[int] | int) -> None:
        """Configure laps where red flags will be forced.

        Args:
            laps: Single lap number or list of lap numbers to force red flags
        """
        if isinstance(laps, int):
            self.forced_red_flag_laps.add(laps)
        else:
            self.forced_red_flag_laps.update(laps)

    def set_forced_safety_car(self, laps: list[int] | int) -> None:
        """Configure laps where safety cars will be forced.

        Args:
            laps: Single lap number or list of lap numbers to force safety cars
        """
        if isinstance(laps, int):
            self.forced_safety_car_laps.add(laps)
        else:
            self.forced_safety_car_laps.update(laps)

    def clear_forced_events(self) -> None:
        """Clear all forced event configurations."""
        self.forced_red_flag_laps.clear()
        self.forced_safety_car_laps.clear()

    def process_lap(
        self,
        lap: int,
        drivers: list[Driver],
        cars: dict[str, Car],
        track: Track,
        weather: Weather,
        incidents_this_lap: int = 0,
    ) -> list[RaceEvent]:
        """Process events for a lap.

        Args:
            lap: Current lap number
            drivers: Active drivers
            cars: Dictionary of cars by team_id
            track: Current track
            weather: Current weather
            incidents_this_lap: Number of overtake incidents

        Returns:
            List of events that occurred
        """
        lap_events: list[RaceEvent] = []

        # Track restart lap (lap after SC ended)
        self.sc_restart_lap = self.sc_just_ended
        self.sc_just_ended = False

        # Track red flag restart lap
        self.red_flag_restart_lap = self.red_flag_just_ended
        self.red_flag_just_ended = False

        # If red flag is active, race is suspended - no events processed
        # Red flag ending is handled externally by RaceSimulator
        if self.red_flag_active:
            return lap_events

        # Update active safety car/VSC
        if self.safety_car_active:
            self.safety_car_laps_remaining -= 1
            if self.safety_car_laps_remaining <= 0:
                self.safety_car_active = False
                self.sc_just_ended = True  # Next lap is restart

        if self.vsc_active:
            self.vsc_laps_remaining -= 1
            if self.vsc_laps_remaining <= 0:
                self.vsc_active = False

        # Check for mechanical failures
        for driver in drivers:
            if driver.dnf:
                continue

            car = cars.get(driver.team_id)
            if car is None:
                continue

            failure = self._check_mechanical_failure(driver, car, track, lap)
            if failure:
                lap_events.append(failure)

        # Check for random incidents (spins, etc.)
        if not self.safety_car_active:
            random_incident = self._check_random_incident(
                drivers, track, weather, lap
            )
            if random_incident:
                lap_events.append(random_incident)
                incidents_this_lap += 1

        # Check for forced red flag
        if lap in self.forced_red_flag_laps and not self.red_flag_active:
            red_flag_event = self.deploy_red_flag(lap, "Manual trigger")
            lap_events.append(red_flag_event)
            self.events.extend(lap_events)
            return lap_events

        # Check for forced safety car
        if lap in self.forced_safety_car_laps and not self.safety_car_active and not self.red_flag_active:
            self.safety_car_active = True
            self.safety_car_laps_remaining = self.rng.integers(3, 7)
            sc_event = RaceEvent(
                event_type=EventType.SAFETY_CAR,
                lap=lap,
                duration_laps=self.safety_car_laps_remaining,
                description="Safety car deployed (manual trigger)",
            )
            lap_events.append(sc_event)

        # Deploy safety car or red flag if needed (from incidents)
        if incidents_this_lap > 0 and not self.safety_car_active and not self.red_flag_active:
            sc_event = self._deploy_safety_measure(lap, incidents_this_lap, track, weather)
            if sc_event:
                lap_events.append(sc_event)

        self.events.extend(lap_events)
        return lap_events

    def _check_mechanical_failure(
        self,
        driver: Driver,
        car: Car,
        track: Track,
        lap: int,
    ) -> RaceEvent | None:
        """Check if a car has mechanical failure.

        Returns event if failure occurred.
        """
        # Base failure probability per lap
        base_prob = (1.0 - car.reliability) / 50  # Spread over ~50 laps

        # Higher failure rate at hot tracks
        temp_modifier = 1.0  # Would use actual track data

        if self.rng.random() < base_prob * temp_modifier:
            failure_types = [
                "engine failure",
                "gearbox failure",
                "hydraulics failure",
                "electrical failure",
                "brake failure",
            ]
            failure = self.rng.choice(failure_types)

            driver.dnf = True
            driver.dnf_reason = failure

            return RaceEvent(
                event_type=EventType.MECHANICAL_FAILURE,
                lap=lap,
                drivers_involved=[driver.id],
                description=f"{driver.name} retired with {failure}",
            )

        return None

    def _check_random_incident(
        self,
        drivers: list[Driver],
        track: Track,
        weather: Weather,
        lap: int,
    ) -> RaceEvent | None:
        """Check for random racing incidents."""
        active_drivers = [d for d in drivers if not d.dnf]
        if len(active_drivers) < 2:
            return None

        # Base incident probability
        base_prob = 0.005  # 0.5% per lap

        # Weather increases incident risk
        if weather.is_wet():
            base_prob *= 2.0
        if weather.requires_wet_tires():
            base_prob *= 3.0

        # Track difficulty affects incident rate
        base_prob *= (1.0 + track.overtake_difficulty * 0.5)

        if self.rng.random() < base_prob:
            # Select random driver for incident
            driver = self.rng.choice(active_drivers)

            # Determine incident severity
            severity_roll = self.rng.random()

            if severity_roll < 0.3:
                # Spin, continues
                return RaceEvent(
                    event_type=EventType.SPIN,
                    lap=lap,
                    drivers_involved=[driver.id],
                    description=f"{driver.name} spun but continues",
                )
            elif severity_roll < 0.6:
                # Puncture
                return RaceEvent(
                    event_type=EventType.PUNCTURE,
                    lap=lap,
                    drivers_involved=[driver.id],
                    description=f"{driver.name} suffered a puncture",
                )
            else:
                # Crash, DNF
                driver.dnf = True
                driver.dnf_reason = "crash"
                return RaceEvent(
                    event_type=EventType.COLLISION,
                    lap=lap,
                    drivers_involved=[driver.id],
                    description=f"{driver.name} crashed and retired",
                )

        return None

    def _deploy_safety_measure(
        self,
        lap: int,
        incidents: int,
        track: Track,
        weather: Weather | None = None,
    ) -> RaceEvent | None:
        """Deploy safety car, VSC, or red flag based on incidents and conditions."""
        # Check for red flag conditions first
        # Red flags are rare but occur for major incidents or dangerous weather
        red_flag_event = self._check_red_flag_conditions(lap, incidents, weather)
        if red_flag_event:
            return red_flag_event

        # Determine if safety car is needed
        sc_threshold = 0.4 + incidents * 0.3

        if self.rng.random() < sc_threshold:
            # Full safety car
            self.safety_car_active = True
            self.safety_car_laps_remaining = self.rng.integers(3, 7)

            return RaceEvent(
                event_type=EventType.SAFETY_CAR,
                lap=lap,
                duration_laps=self.safety_car_laps_remaining,
                description="Safety car deployed",
            )
        elif self.rng.random() < 0.5:
            # Virtual safety car
            self.vsc_active = True
            self.vsc_laps_remaining = self.rng.integers(2, 4)

            return RaceEvent(
                event_type=EventType.VIRTUAL_SAFETY_CAR,
                lap=lap,
                duration_laps=self.vsc_laps_remaining,
                description="Virtual safety car deployed",
            )

        return None

    def _check_red_flag_conditions(
        self,
        lap: int,
        incidents: int,
        weather: Weather | None = None,
    ) -> RaceEvent | None:
        """Check if conditions warrant a red flag.

        Red flags are deployed for:
        - Multiple serious incidents (3+ in a single lap)
        - Extremely dangerous weather conditions
        - Major track blockage (simulated by high incident severity)
        """
        red_flag_probability = 0.0

        # Multiple incidents significantly increase red flag chance
        if incidents >= 3:
            red_flag_probability += 0.4
        elif incidents >= 2:
            red_flag_probability += 0.15

        # Severe weather can trigger red flag
        if weather is not None:
            if weather.track_wetness > 0.9:
                # Standing water on track - very dangerous
                red_flag_probability += 0.3
            elif weather.track_wetness > 0.8 and weather.rain_intensity > 0.8:
                # Heavy rain with very wet track
                red_flag_probability += 0.15

        # Random major incident chance (rare)
        if incidents > 0 and self.rng.random() < 0.05:
            red_flag_probability += 0.3

        if red_flag_probability > 0 and self.rng.random() < red_flag_probability:
            return self.deploy_red_flag(lap, "Dangerous conditions")

        return None

    def deploy_red_flag(self, lap: int, reason: str = "Incident") -> RaceEvent:
        """Deploy a red flag, stopping the race.

        Args:
            lap: Current lap number
            reason: Description of why red flag was deployed

        Returns:
            RaceEvent for the red flag
        """
        self.red_flag_active = True
        # Clear any active SC/VSC
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.vsc_active = False
        self.vsc_laps_remaining = 0

        return RaceEvent(
            event_type=EventType.RED_FLAG,
            lap=lap,
            duration_laps=0,  # Duration determined by race director
            description=f"Red flag: {reason}",
        )

    def end_red_flag(self) -> None:
        """End the red flag period and prepare for restart."""
        self.red_flag_active = False
        self.red_flag_just_ended = True

    def is_red_flag_active(self) -> bool:
        """Check if red flag is currently active."""
        return self.red_flag_active

    def is_red_flag_restart_lap(self) -> bool:
        """Check if this is the restart lap after a red flag."""
        return self.red_flag_restart_lap

    def get_lap_time_modifier(self) -> float:
        """Get lap time modifier based on current safety status.

        Returns:
            Multiplier for lap times (>1 = slower)
        """
        if self.red_flag_active:
            return 0.0  # Race is stopped during red flag
        elif self.safety_car_active:
            return 1.4  # 40% slower under SC
        elif self.vsc_active:
            return 1.2  # 20% slower under VSC
        return 1.0

    def is_pit_window_open(self) -> bool:
        """Check if it's a good time to pit (under SC/VSC/red flag)."""
        return self.safety_car_active or self.vsc_active or self.red_flag_active

    def is_restart_lap(self) -> bool:
        """Check if this is a restart lap after SC or red flag."""
        return self.sc_restart_lap or self.red_flag_restart_lap

    def bunch_field(self, driver_states: list) -> None:
        """Bunch up the field behind safety car.

        Sets all gaps between cars to ~1 second, simulating
        the field catching the safety car and forming a queue.

        Args:
            driver_states: List of DriverRaceState objects
        """
        # Sort by position
        racing = [s for s in driver_states if s.status.value == "racing"]
        racing.sort(key=lambda s: s.position)

        if len(racing) < 2:
            return

        # Leader's time stays the same
        leader_time = racing[0].total_time

        # Each following car is set to ~0.8-1.2 seconds behind the car ahead
        for i, state in enumerate(racing[1:], 1):
            gap_to_ahead = self.rng.uniform(0.8, 1.2)
            state.total_time = racing[i - 1].total_time + gap_to_ahead
