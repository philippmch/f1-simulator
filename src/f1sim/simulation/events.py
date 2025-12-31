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

    def reset(self) -> None:
        """Reset event state for new race."""
        self.events = []
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.vsc_active = False
        self.vsc_laps_remaining = 0

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

        # Update active safety car/VSC
        if self.safety_car_active:
            self.safety_car_laps_remaining -= 1
            if self.safety_car_laps_remaining <= 0:
                self.safety_car_active = False

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

        # Deploy safety car if needed
        if incidents_this_lap > 0 and not self.safety_car_active:
            sc_event = self._deploy_safety_measure(lap, incidents_this_lap, track)
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
    ) -> RaceEvent | None:
        """Deploy safety car or VSC based on incidents."""
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

    def get_lap_time_modifier(self) -> float:
        """Get lap time modifier based on current safety status.

        Returns:
            Multiplier for lap times (>1 = slower)
        """
        if self.safety_car_active:
            return 1.4  # 40% slower under SC
        elif self.vsc_active:
            return 1.2  # 20% slower under VSC
        return 1.0

    def is_pit_window_open(self) -> bool:
        """Check if it's a good time to pit (under SC/VSC)."""
        return self.safety_car_active or self.vsc_active
