"""Qualifying session simulation (Q1, Q2, Q3)."""

from dataclasses import dataclass

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.validation import validate_unique_ids


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

    Q3_FIELD_SIZE = 10

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
        validate_unique_ids((driver.id for driver in drivers), "drivers")
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

        # Keep the input order as a deterministic tie-breaker.  This matters
        # for equal synthetic times and means a seeded run does not depend on
        # incidental dictionary ordering.
        driver_order = {driver.id: index for index, driver in enumerate(drivers)}

        def ordered_times(times: dict[str, float]) -> list[tuple[str, float]]:
            return sorted(times.items(), key=lambda item: (item[1], driver_order[item[0]]))

        # Q1: All drivers.  The current 22-car field eliminates six, while
        # the traditional 20-car field eliminates five.  Derive the number
        # advancing from the field size instead of hard-coding a 15-car Q2.
        field_size = len(drivers)
        q1_elimination_count = self._elimination_count(field_size)
        q2_target_count = max(field_size - q1_elimination_count, 0)
        q1_times = self._simulate_session(drivers, cars, track, weather, attempts=2)
        for driver_id, time in q1_times.items():
            results[driver_id].q1_time = time
            results[driver_id].best_time = min(results[driver_id].best_time, time)

        q1_sorted = ordered_times(q1_times)
        q2_count = min(q2_target_count, len(q1_sorted))
        q1_qualifiers = [driver_id for driver_id, _ in q1_sorted[:q2_count]]
        q1_eliminated = [driver_id for driver_id, _ in q1_sorted[q2_count:]]
        # A missing car should not silently disappear from classification.
        q1_eliminated.extend(driver.id for driver in drivers if driver.id not in q1_times)
        for driver_id in q1_eliminated:
            results[driver_id].eliminated_in = "Q1"

        # Q2: Top 15, eliminate bottom 5
        q2_drivers = [d for d in drivers if d.id in q1_qualifiers]
        q2_times = self._simulate_session(q2_drivers, cars, track, weather, attempts=2)
        for driver_id, time in q2_times.items():
            results[driver_id].q2_time = time
            results[driver_id].best_time = min(results[driver_id].best_time, time)

        q2_sorted = ordered_times(q2_times)
        # Keep ten Q3 places for a full field; for a reduced field, Q3 is the
        # largest sensible final session without exceeding the available
        # drivers.
        q3_count = min(self.Q3_FIELD_SIZE, len(q2_sorted), field_size)
        q2_qualifiers = [driver_id for driver_id, _ in q2_sorted[:q3_count]]
        q2_eliminated = [driver_id for driver_id, _ in q2_sorted[q3_count:]]
        q2_eliminated.extend(driver.id for driver in q2_drivers if driver.id not in q2_times)
        for driver_id in q2_eliminated:
            results[driver_id].eliminated_in = "Q2"

        # Q3: Top 10, fight for pole
        q3_drivers = [d for d in drivers if d.id in q2_qualifiers]
        q3_times = self._simulate_session(q3_drivers, cars, track, weather, attempts=2)
        for driver_id, time in q3_times.items():
            results[driver_id].q3_time = time
            results[driver_id].best_time = min(results[driver_id].best_time, time)

        # Qualifying classification is session based, rather than a sort by
        # each driver's best time across all sessions.  An eliminated Q2/Q1
        # driver can have a faster earlier-session time than a Q3 driver, but
        # the Q3 driver still starts ahead in the real sporting classification.
        q3_sorted = ordered_times(q3_times)
        q3_order = [driver_id for driver_id, _ in q3_sorted]
        q3_order.extend(driver.id for driver in q3_drivers if driver.id not in q3_times)
        classification_order = q3_order + q2_eliminated + q1_eliminated

        # Include any defensive leftovers once, preserving input order.
        classification_order.extend(
            driver.id for driver in drivers if driver.id not in classification_order
        )
        final_results = [results[driver_id] for driver_id in classification_order]

        for pos, result in enumerate(final_results, 1):
            result.position = pos

        return final_results

    @classmethod
    def _elimination_count(cls, field_size: int) -> int:
        """Return the Q1/Q2 elimination count for a starting field.

        FIA's current 22-car format removes six in each of Q1 and Q2. For
        reduced test fields, split the drivers outside the ten-car Q3 evenly
        between the first two sessions, assigning an odd extra elimination
        to Q1.
        """
        if field_size <= cls.Q3_FIELD_SIZE:
            return 0
        outside_q3 = field_size - cls.Q3_FIELD_SIZE
        return (outside_q3 + 1) // 2

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
