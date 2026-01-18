"""Monte Carlo simulation runner and statistics."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.simulation.events import EventType
from f1sim.simulation.qualifying import QualifyingResult, QualifyingSimulator
from f1sim.simulation.race import RaceResult, RaceSimulator


@dataclass
class DriverStatistics:
    """Aggregated statistics for a driver across simulations."""

    driver_id: str
    driver_name: str
    team: str
    wins: int = 0
    podiums: int = 0
    points_finishes: int = 0
    dnfs: int = 0
    total_points: float = 0.0
    avg_position: float = 0.0
    avg_qualifying: float = 0.0
    best_position: int = 20
    worst_position: int = 1
    positions: list[int] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        """Win percentage."""
        return self.wins / len(self.positions) * 100 if self.positions else 0

    @property
    def podium_rate(self) -> float:
        """Podium percentage."""
        return self.podiums / len(self.positions) * 100 if self.positions else 0

    @property
    def dnf_rate(self) -> float:
        """DNF percentage."""
        return self.dnfs / len(self.positions) * 100 if self.positions else 0


@dataclass
class RaceEventStatistics:
    """Aggregated event statistics across simulations."""

    safety_car_count: int = 0
    vsc_count: int = 0
    red_flag_count: int = 0
    total_incidents: int = 0
    races_with_safety_car: int = 0
    races_with_red_flag: int = 0

    @property
    def safety_car_rate(self) -> float:
        """Percentage of races with at least one safety car."""
        return 0.0  # Will be calculated after aggregation

    @property
    def red_flag_rate(self) -> float:
        """Percentage of races with at least one red flag."""
        return 0.0  # Will be calculated after aggregation


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation."""

    num_simulations: int
    track_name: str
    driver_stats: dict[str, DriverStatistics]
    race_results: list[list[RaceResult]]  # All individual race results
    qualifying_results: list[list[QualifyingResult]]  # All qualifying results
    event_stats: RaceEventStatistics = field(default_factory=RaceEventStatistics)

    def get_win_probabilities(self) -> dict[str, float]:
        """Get win probability for each driver."""
        return {
            driver_id: stats.win_rate
            for driver_id, stats in sorted(
                self.driver_stats.items(),
                key=lambda x: x[1].wins,
                reverse=True,
            )
        }

    def get_championship_projection(self) -> dict[str, float]:
        """Get projected points for each driver."""
        return {
            driver_id: stats.total_points / self.num_simulations
            for driver_id, stats in sorted(
                self.driver_stats.items(),
                key=lambda x: x[1].total_points,
                reverse=True,
            )
        }

    def get_position_distribution(self, driver_id: str) -> dict[int, float]:
        """Get position probability distribution for a driver."""
        if driver_id not in self.driver_stats:
            return {}

        positions = self.driver_stats[driver_id].positions
        counts: dict[int, int] = defaultdict(int)
        for pos in positions:
            counts[pos] += 1

        return {
            pos: count / len(positions) * 100
            for pos, count in sorted(counts.items())
        }


# F1 points system
POINTS_SYSTEM = {
    1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
    6: 8, 7: 6, 8: 4, 9: 2, 10: 1,
}


def _run_single_simulation(args: tuple) -> tuple[list[RaceResult], list[QualifyingResult], dict]:
    """Run a single race simulation (for multiprocessing).

    Args:
        args: Tuple of (drivers_data, cars_data, track_data, weather_data, seed, historical_grid)

    Returns:
        Tuple of (race_results, qualifying_results, event_counts)
    """
    drivers_data, cars_data, track_data, weather_data, seed, historical_grid = args

    # Reconstruct objects from serializable data
    drivers = [Driver.model_validate(d) for d in drivers_data]
    cars = {k: Car.model_validate(v) for k, v in cars_data.items()}
    track = Track.model_validate(track_data)
    weather = Weather.model_validate(weather_data)

    # Create RNG with seed
    rng = np.random.default_rng(seed)

    # Run qualifying (or use historical grid)
    quali_sim = QualifyingSimulator(rng=rng)
    if historical_grid:
        # Use historical grid - create dummy qualifying results
        starting_grid = historical_grid
        quali_results = [
            QualifyingResult(
                driver_id=driver_id,
                driver_name=driver_id,
                position=pos,
                best_time=0.0,
                q1_time=None,
                q2_time=None,
                q3_time=None,
                eliminated_in=None,
            )
            for pos, driver_id in enumerate(starting_grid, 1)
        ]
    else:
        quali_results = quali_sim.simulate_qualifying(drivers, cars, track, weather)
        starting_grid = quali_sim.get_starting_grid(quali_results)

    # Run race
    race_sim = RaceSimulator(rng=rng)
    race_results = race_sim.simulate_race(
        drivers=drivers,
        cars=cars,
        track=track,
        weather=weather,
        starting_grid=starting_grid,
    )

    # Collect event statistics
    events = race_sim.event_manager.events
    event_counts = {
        "safety_car": sum(1 for e in events if e.event_type == EventType.SAFETY_CAR),
        "vsc": sum(1 for e in events if e.event_type == EventType.VIRTUAL_SAFETY_CAR),
        "red_flag": sum(1 for e in events if e.event_type == EventType.RED_FLAG),
        "incidents": len([e for e in events if e.event_type in (
            EventType.COLLISION, EventType.SPIN, EventType.PUNCTURE, EventType.MECHANICAL_FAILURE
        )]),
    }

    return race_results, quali_results, event_counts


class MonteCarloRunner:
    """Runs Monte Carlo simulations for F1 races."""

    def __init__(
        self,
        drivers: list[Driver],
        cars: dict[str, Car],
        track: Track,
        weather: Weather,
        seed: int | None = None,
        historical_grid: list[str] | None = None,
    ):
        """Initialize Monte Carlo runner.

        Args:
            drivers: List of drivers
            cars: Dictionary of cars by team_id
            track: Circuit to simulate
            weather: Initial weather conditions
            seed: Random seed for reproducibility
            historical_grid: If provided, use this grid instead of simulating qualifying
        """
        self.drivers = drivers
        self.cars = cars
        self.track = track
        self.weather = weather
        self.base_seed = seed if seed is not None else np.random.default_rng().integers(0, 2**31)
        self.historical_grid = historical_grid

    def run(
        self,
        num_simulations: int = 1000,
        parallel: bool = True,
        max_workers: int | None = None,
    ) -> SimulationResults:
        """Run Monte Carlo simulations.

        Args:
            num_simulations: Number of simulations to run
            parallel: Whether to use parallel processing
            max_workers: Maximum parallel workers (None = CPU count)

        Returns:
            SimulationResults with aggregated statistics
        """
        # Prepare serializable data for multiprocessing
        drivers_data = [d.model_dump() for d in self.drivers]
        cars_data = {k: v.model_dump() for k, v in self.cars.items()}
        track_data = self.track.model_dump()
        weather_data = self.weather.model_dump()

        # Generate unique seeds for each simulation
        seeds = [self.base_seed + i for i in range(num_simulations)]

        args_list = [
            (drivers_data, cars_data, track_data, weather_data, seed, self.historical_grid)
            for seed in seeds
        ]

        all_race_results: list[list[RaceResult]] = []
        all_quali_results: list[list[QualifyingResult]] = []
        all_event_counts: list[dict] = []

        if parallel and num_simulations > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(_run_single_simulation, args_list))
                for race_res, quali_res, event_counts in results:
                    all_race_results.append(race_res)
                    all_quali_results.append(quali_res)
                    all_event_counts.append(event_counts)
        else:
            for args in args_list:
                race_res, quali_res, event_counts = _run_single_simulation(args)
                all_race_results.append(race_res)
                all_quali_results.append(quali_res)
                all_event_counts.append(event_counts)

        # Aggregate statistics
        driver_stats = self._aggregate_statistics(all_race_results, all_quali_results)
        event_stats = self._aggregate_event_statistics(all_event_counts)

        return SimulationResults(
            num_simulations=num_simulations,
            track_name=self.track.name,
            driver_stats=driver_stats,
            race_results=all_race_results,
            qualifying_results=all_quali_results,
            event_stats=event_stats,
        )

    def _aggregate_statistics(
        self,
        race_results: list[list[RaceResult]],
        quali_results: list[list[QualifyingResult]],
    ) -> dict[str, DriverStatistics]:
        """Aggregate statistics from all simulations."""
        stats: dict[str, DriverStatistics] = {}

        # Initialize stats for all drivers
        for driver in self.drivers:
            car = self.cars.get(driver.team_id)
            stats[driver.id] = DriverStatistics(
                driver_id=driver.id,
                driver_name=driver.name,
                team=car.team_name if car else "Unknown",
            )

        # Process race results
        for sim_results in race_results:
            for result in sim_results:
                if result.driver_id not in stats:
                    continue

                driver_stat = stats[result.driver_id]
                driver_stat.positions.append(result.position)

                if result.position == 1:
                    driver_stat.wins += 1
                if result.position <= 3:
                    driver_stat.podiums += 1
                if result.position <= 10:
                    driver_stat.points_finishes += 1
                    driver_stat.total_points += POINTS_SYSTEM.get(result.position, 0)

                if result.status.value == "dnf":
                    driver_stat.dnfs += 1

                driver_stat.best_position = min(driver_stat.best_position, result.position)
                driver_stat.worst_position = max(driver_stat.worst_position, result.position)

        # Process qualifying results
        quali_positions: dict[str, list[int]] = defaultdict(list)
        for sim_quali in quali_results:
            for result in sim_quali:
                quali_positions[result.driver_id].append(result.position)

        # Calculate averages
        for driver_id, driver_stat in stats.items():
            if driver_stat.positions:
                driver_stat.avg_position = np.mean(driver_stat.positions)
            if driver_id in quali_positions and quali_positions[driver_id]:
                driver_stat.avg_qualifying = np.mean(quali_positions[driver_id])

        return stats

    def _aggregate_event_statistics(
        self,
        event_counts: list[dict],
    ) -> RaceEventStatistics:
        """Aggregate event statistics from all simulations."""
        stats = RaceEventStatistics()

        for counts in event_counts:
            stats.safety_car_count += counts.get("safety_car", 0)
            stats.vsc_count += counts.get("vsc", 0)
            stats.red_flag_count += counts.get("red_flag", 0)
            stats.total_incidents += counts.get("incidents", 0)

            if counts.get("safety_car", 0) > 0:
                stats.races_with_safety_car += 1
            if counts.get("red_flag", 0) > 0:
                stats.races_with_red_flag += 1

        return stats

    def run_quick(self, num_simulations: int = 100) -> SimulationResults:
        """Run a quick simulation without parallelization.

        Useful for testing or when running in environments
        where multiprocessing is problematic.
        """
        return self.run(num_simulations=num_simulations, parallel=False)
