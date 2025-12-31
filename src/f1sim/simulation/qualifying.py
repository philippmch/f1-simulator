"""Qualifying session simulation (Q1, Q2, Q3)."""

from dataclasses import dataclass

import numpy as np

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator


@dataclass
class QualifyingResult:
    """Result for a single driver in qualifying."""

    driver_id: str
    driver_name: str
    position: int
    best_time: float
    q1_time: float | None
    q2_time: float | None
    q3_time: float | None
    eliminated_in: str | None  # "Q1", "Q2", or None


class QualifyingSimulator:
    """Simulates F1 qualifying sessions."""

    Q1_DRIVERS = 20  # All drivers
    Q2_DRIVERS = 15  # Top 15 from Q1
    Q3_DRIVERS = 10  # Top 10 from Q2

    Q1_ELIMINATED = 5  # Positions 16-20
    Q2_ELIMINATED = 5  # Positions 11-15

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize qualifying simulator.

        Args:
            rng: Random number generator
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.lap_simulator = LapSimulator(rng=self.rng)

    def simulate_qualifying(
        self,
        drivers: list[Driver],
        cars: dict[str, Car],
        track: Track,
        weather: Weather,
    ) -> list[QualifyingResult]:
        """Simulate full qualifying session (Q1, Q2, Q3).

        Args:
            drivers: All drivers participating
            cars: Dictionary of cars by team_id
            track: Circuit being raced
            weather: Weather conditions

        Returns:
            List of QualifyingResult sorted by position
        """
        results: dict[str, QualifyingResult] = {}

        # Initialize results
        for driver in drivers:
            results[driver.id] = QualifyingResult(
                driver_id=driver.id,
                driver_name=driver.name,
                position=0,
                best_time=float("inf"),
                q1_time=None,
                q2_time=None,
                q3_time=None,
                eliminated_in=None,
            )

        # Q1: All drivers, eliminate bottom 5
        q1_times = self._simulate_session(drivers, cars, track, weather, attempts=2)
        for driver_id, time in q1_times.items():
            results[driver_id].q1_time = time
            results[driver_id].best_time = min(results[driver_id].best_time, time)

        q1_sorted = sorted(q1_times.items(), key=lambda x: x[1])
        q1_qualifiers = [d[0] for d in q1_sorted[: self.Q2_DRIVERS]]
        for driver_id, _ in q1_sorted[self.Q2_DRIVERS :]:
            results[driver_id].eliminated_in = "Q1"

        # Q2: Top 15, eliminate bottom 5
        q2_drivers = [d for d in drivers if d.id in q1_qualifiers]
        q2_times = self._simulate_session(q2_drivers, cars, track, weather, attempts=2)
        for driver_id, time in q2_times.items():
            results[driver_id].q2_time = time
            results[driver_id].best_time = min(results[driver_id].best_time, time)

        q2_sorted = sorted(q2_times.items(), key=lambda x: x[1])
        q2_qualifiers = [d[0] for d in q2_sorted[: self.Q3_DRIVERS]]
        for driver_id, _ in q2_sorted[self.Q3_DRIVERS :]:
            results[driver_id].eliminated_in = "Q2"

        # Q3: Top 10, fight for pole
        q3_drivers = [d for d in drivers if d.id in q2_qualifiers]
        q3_times = self._simulate_session(q3_drivers, cars, track, weather, attempts=2)
        for driver_id, time in q3_times.items():
            results[driver_id].q3_time = time
            results[driver_id].best_time = min(results[driver_id].best_time, time)

        # Calculate final positions
        final_results = list(results.values())
        final_results.sort(key=lambda r: r.best_time)

        for pos, result in enumerate(final_results, 1):
            result.position = pos

        return final_results

    def _simulate_session(
        self,
        drivers: list[Driver],
        cars: dict[str, Car],
        track: Track,
        weather: Weather,
        attempts: int = 2,
    ) -> dict[str, float]:
        """Simulate a single qualifying session.

        Args:
            drivers: Drivers in this session
            cars: Cars dictionary
            track: Circuit
            weather: Weather conditions
            attempts: Number of flying laps per driver

        Returns:
            Dictionary of driver_id -> best lap time
        """
        best_times: dict[str, float] = {}

        # Use soft tires for qualifying
        soft_tire = TIRE_COMPOUNDS[TireCompound.SOFT]

        for driver in drivers:
            car = cars.get(driver.team_id)
            if car is None:
                continue

            driver_best = float("inf")

            for attempt in range(attempts):
                # Push level varies by attempt (more push on final attempt)
                push_level = 0.9 if attempt < attempts - 1 else 1.0

                lap_time = self.lap_simulator.calculate_qualifying_lap(
                    driver=driver,
                    car=car,
                    track=track,
                    tire=soft_tire,
                    weather=weather,
                    push_level=push_level,
                )

                driver_best = min(driver_best, lap_time)

            best_times[driver.id] = driver_best

        return best_times

    def get_starting_grid(
        self,
        results: list[QualifyingResult],
    ) -> list[str]:
        """Get starting grid order from qualifying results.

        Args:
            results: Qualifying results

        Returns:
            List of driver IDs in grid order
        """
        sorted_results = sorted(results, key=lambda r: r.position)
        return [r.driver_id for r in sorted_results]
