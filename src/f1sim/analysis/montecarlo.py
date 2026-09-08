"""Monte Carlo simulation runner and statistics."""

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from math import isclose, isfinite, sqrt
from numbers import Integral, Real
from statistics import NormalDist

import numpy as np

from f1sim.analysis.provenance import simulation_runtime
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType
from f1sim.simulation.execution import validate_race_engine, validate_starting_tires
from f1sim.simulation.qualifying import QualifyingResult, QualifyingSimulator
from f1sim.simulation.race import (
    DriverStatus,
    RaceResult,
    RaceSimulator,
    result_is_classified,
)
from f1sim.simulation.race_points import POINTS_SYSTEM as POINTS_SYSTEM
from f1sim.simulation.race_points import points_for_result
from f1sim.simulation.validation import validate_unique_ids


def wilson_interval(successes: int, trials: int) -> dict[str, float]:
    """Return a 95% Wilson binomial score interval in percentage units.

    These bounds describe sampling error across independent Monte Carlo trials,
    conditional on the model and inputs. They do not measure model calibration
    or uncertainty about a real race. With no observations, return [0, 100].
    """
    if (
        isinstance(successes, bool)
        or isinstance(trials, bool)
        or not isinstance(successes, Integral)
        or not isinstance(trials, Integral)
    ):
        raise ValueError("successes and trials must be integers")
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("counts must satisfy 0 <= successes <= trials")
    if trials == 0:
        return {"lower": 0.0, "upper": 100.0}

    z_squared = NormalDist().inv_cdf(0.975) ** 2
    proportion = successes / trials
    denominator = 1 + z_squared / trials
    center = (proportion + z_squared / (2 * trials)) / denominator
    margin = sqrt(
        z_squared * (proportion * (1 - proportion) + z_squared / (4 * trials)) / trials
    ) / denominator
    return {
        "lower": 0.0 if successes == 0 else max(0.0, (center - margin) * 100),
        "upper": 100.0 if successes == trials else min(100.0, (center + margin) * 100),
    }


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
    mechanical_failure_breakdown: dict[str, int] = field(default_factory=dict)
    num_simulations: int = 0

    @property
    def safety_car_rate(self) -> float:
        """Percentage of races with at least one safety car."""
        return (
            self.races_with_safety_car / self.num_simulations * 100
            if self.num_simulations else 0.0
        )

    @property
    def red_flag_rate(self) -> float:
        """Percentage of races with at least one red flag."""
        return (
            self.races_with_red_flag / self.num_simulations * 100
            if self.num_simulations else 0.0
        )


@dataclass
class SimulationResults:
    """Results from Monte Carlo simulation."""

    num_simulations: int
    track_name: str
    driver_stats: dict[str, DriverStatistics]
    race_results: list[list[RaceResult]]  # All individual race results
    qualifying_results: list[list[QualifyingResult]]  # All qualifying results
    event_stats: RaceEventStatistics = field(default_factory=RaceEventStatistics)
    seed: int | None = None
    parallel: bool = True
    max_workers: int | None = None
    race_engine: str = "standard"
    input_snapshot: dict | None = None
    weather_histories: list[list[dict]] = field(default_factory=list)

    def get_race_distance_statistics(self) -> dict[str, int | float | None]:
        """Summarize recorded distances, with rates as fractions in [0, 1].

        Only a finished position-one result identifies the winner: a retired
        car can have completed more laps in a time-limited race. Missing legacy
        distances stay unknown and do not enter distance-based denominators.
        """
        def positive_laps(result: RaceResult) -> int | None:
            laps = getattr(result, "laps_completed", None)
            if isinstance(laps, Integral) and not isinstance(laps, bool) and laps > 0:
                return int(laps)
            return None

        winners = 0
        winner_distances: list[int] = []
        time_limited = 0
        finishers = 0
        comparable = 0
        lapped = 0
        for race in self.race_results:
            finished = [result for result in race if result.status == DriverStatus.FINISHED]
            winner = next((result for result in finished if result.position == 1), None)
            winner_laps = positive_laps(winner) if winner is not None else None
            winners += int(winner is not None)
            if winner_laps is not None:
                winner_distances.append(winner_laps)
            time_limited += int(any(getattr(result, "race_time_limited", False) for result in race))
            finishers += len(finished)
            for result in finished:
                laps = positive_laps(result)
                if winner_laps is not None and laps is not None:
                    comparable += 1
                    lapped += int(laps < winner_laps)

        recorded = len(self.race_results)
        return {
            "recorded_races": recorded,
            "races_with_winner": winners,
            "races_without_winner": recorded - winners,
            "races_with_known_winner_distance": len(winner_distances),
            "mean_winner_laps": (
                sum(winner_distances) / len(winner_distances) if winner_distances else None
            ),
            "time_limited_races": time_limited,
            "time_limited_race_rate": time_limited / recorded if recorded else None,
            "finishing_cars": finishers,
            "finishers_with_comparable_distance": comparable,
            "lapped_finishers": lapped,
            "lapped_finisher_rate": lapped / comparable if comparable else None,
        }

    def get_strategy_statistics(self) -> dict[str, dict]:
        """Count recorded tyre sequences per observed driver race row.

        Shares are fractions of rows with a valid recorded sequence, including
        DNFs whose sequences can be truncated. Fittings can include free red
        flag changes, so sequence length is not a paid pit-stop count. Missing
        or malformed legacy sequences stay unknown. These frequencies describe
        observed strategies, not their causal effect on race outcomes.
        """
        observations: dict[str, int] = defaultdict(int)
        variants: dict[str, dict[tuple[str, ...], dict[str, int]]] = defaultdict(dict)
        for race in self.race_results:
            for result in race:
                driver_id = result.driver_id
                observations[driver_id] += 1
                sequence = getattr(result, "strategy", None)
                if not isinstance(sequence, (list, tuple)) or not sequence:
                    continue
                if not all(isinstance(compound, str) and compound for compound in sequence):
                    continue
                key = tuple(
                    compound.value if isinstance(compound, TireCompound) else str(compound)
                    for compound in sequence
                )
                counts = variants[driver_id].setdefault(
                    key, {"races": 0, "finished_races": 0, "dnf_races": 0}
                )
                counts["races"] += 1
                counts["finished_races"] += int(result.status == DriverStatus.FINISHED)
                counts["dnf_races"] += int(result.status == DriverStatus.DNF)

        summaries = {}
        for driver_id, races in sorted(observations.items()):
            recorded = sum(counts["races"] for counts in variants[driver_id].values())
            summaries[driver_id] = {
                "races": races,
                "races_with_recorded_strategy": recorded,
                "missing_strategy_races": races - recorded,
                "strategies": [
                    {"compounds": list(sequence), **counts, "share": counts["races"] / recorded}
                    for sequence, counts in sorted(
                        variants[driver_id].items(), key=lambda item: (-item[1]["races"], item[0])
                    )
                ],
            }
        return summaries

    def get_pit_stop_statistics(self) -> dict[str, dict]:
        """Paid stops per observed race row, including retired entrants.

        Missing observations are not zero-stop races. Free tyre changes do
        not enter RaceResult.pit_stops and therefore do not affect this count.
        """
        distributions: dict[str, dict[int, int]] = {}
        for race in self.race_results:
            for result in race:
                counts = distributions.setdefault(result.driver_id, {})
                counts[result.pit_stops] = counts.get(result.pit_stops, 0) + 1
        return {
            driver_id: {
                "races": sum(counts.values()),
                "average_stops": sum(stops * count for stops, count in counts.items())
                / sum(counts.values()),
                "stop_count_distribution": dict(sorted(counts.items())),
            }
            for driver_id, counts in distributions.items()
        }

    def get_pit_loss_statistics(self) -> dict[str, dict]:
        """Modeled paid-stop losses per completely recorded driver race.

        Include retirements and observed zero-stop races. Missing, partial or
        inconsistent detail rows contribute only to the missing count, never
        zero seconds. Means and queue race rates use races_with_recorded_details,
        not requested simulations or recorded stop counts. Free fits are absent
        from paid-stop details and contribute no service or queue loss.
        """
        fields = ("total_loss", "lane_loss", "service_time", "queue_time")

        def valid_component(value):
            if isinstance(value, bool) or not isinstance(value, Real):
                return False
            try:
                return isfinite(value) and value >= 0
            except OverflowError:
                return False  # An integer too large for the loss representation.

        summaries = {}
        for race in self.race_results:
            for result in race:
                summary = summaries.setdefault(result.driver_id, {
                    "races": 0, "races_with_recorded_details": 0,
                    "missing_details_races": 0, "recorded_stops": 0,
                    "queued_stops": 0, "races_with_queue": 0,
                    "queue_race_rate": None,
                    **{f"mean_{name}_per_race": None for name in fields},
                })
                summary["races"] += 1
                stops = getattr(result, "pit_stops", None)
                details = getattr(result, "pit_stop_details", None)
                valid = (isinstance(stops, Integral) and not isinstance(stops, bool)
                         and stops >= 0 and isinstance(details, (list, tuple))
                         and len(details) == stops)
                values = []
                if valid:
                    for stop in details:
                        if not isinstance(stop, dict) or any(
                            not valid_component(stop.get(name))
                            for name in fields
                        ):
                            valid = False
                            break
                        row = {name: float(stop[name]) for name in fields}
                        if not isclose(row["total_loss"], row["lane_loss"]
                                       + row["service_time"] + row["queue_time"],
                                       rel_tol=1e-9, abs_tol=1e-9):
                            valid = False
                            break
                        values.append(row)
                totals = {name: sum(row[name] for row in values) for name in fields}
                if not all(isfinite(value) for value in totals.values()):
                    valid = False
                if not valid:
                    summary["missing_details_races"] += 1
                    continue
                summary["races_with_recorded_details"] += 1
                observed = summary["races_with_recorded_details"]
                summary["recorded_stops"] += int(stops)
                queued = sum(row["queue_time"] > 0 for row in values)
                summary["queued_stops"] += queued
                summary["races_with_queue"] += int(queued > 0)
                summary["queue_race_rate"] = summary["races_with_queue"] / observed
                for name in fields:
                    key = f"mean_{name}_per_race"
                    previous = summary[key] or 0.0
                    summary[key] = previous * ((observed - 1) / observed) + totals[name] / observed
        return summaries

    def get_probability_intervals(self) -> dict[str, dict]:
        """95% sampling intervals for win, podium and DNF rates, in percent.

        Use each driver's observed race count, including DNFs, matching the
        denominators of the existing point estimates. Counts come from the
        classification-aware aggregation; retirement is counted independently.
        These are individual intervals, not simultaneous bounds across drivers.
        """
        return {
            driver_id: {
                "confidence": 0.95,
                "method": "wilson",
                "scope": "monte_carlo_sampling",
                "trials": len(stats.positions),
                "win": wilson_interval(stats.wins, len(stats.positions)),
                "podium": wilson_interval(stats.podiums, len(stats.positions)),
                "dnf": wilson_interval(stats.dnfs, len(stats.positions)),
            }
            for driver_id, stats in self.driver_stats.items()
        }

    def get_win_probabilities(self) -> dict[str, float]:
        """Get win probabilities ranked by observed rate, including partial samples."""
        return {
            driver_id: stats.win_rate
            for driver_id, stats in sorted(
                self.driver_stats.items(),
                key=lambda x: x[1].win_rate,
                reverse=True,
            )
        }

    def get_championship_projection(self) -> dict[str, float]:
        """Mean points per observed driver race; omit unobserved drivers."""
        projections = {
            driver_id: stats.total_points / len(stats.positions)
            for driver_id, stats in self.driver_stats.items() if stats.positions
        }
        return dict(sorted(projections.items(), key=lambda item: item[1], reverse=True))

    def get_position_distribution(self, driver_id: str) -> dict[int, float]:
        """Get position probability distribution for a driver."""
        if driver_id not in self.driver_stats:
            return {}

        positions = self.driver_stats[driver_id].positions
        counts: dict[int, int] = defaultdict(int)
        for pos in positions:
            counts[pos] += 1

        return {pos: count / len(positions) * 100 for pos, count in sorted(counts.items())}

    def get_top_n_finish_probabilities(self, n: int = 10) -> dict[str, float]:
        """Get probability of a classified top-N result for each driver."""
        if n <= 0:
            msg = "n must be greater than 0"
            raise ValueError(msg)

        # Raw ordinal positions include unclassified retirements. Full race
        # results let us distinguish them from eligible late retirements.
        finish_counts: dict[str, int] = defaultdict(int)
        for race in self.race_results:
            for result in race:
                if result_is_classified(result) and result.position <= n:
                    finish_counts[result.driver_id] += 1

        probs: dict[str, float] = {}
        for driver_id, stats in self.driver_stats.items():
            if not stats.positions:
                probs[driver_id] = 0.0
                continue
            top_n_count = (
                finish_counts[driver_id]
                if self.race_results
                else sum(1 for pos in stats.positions if pos <= n)
            )
            probs[driver_id] = top_n_count / len(stats.positions) * 100

        return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    def get_points_finish_probabilities(self) -> dict[str, float]:
        """Get the probability of earning points, including reduced awards."""
        probabilities = {
            driver_id: stats.points_finishes / len(stats.positions) * 100
            if stats.positions else 0.0
            for driver_id, stats in self.driver_stats.items()
        }
        return dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

    def get_position_percentiles(
        self,
        driver_id: str,
        percentiles: tuple[int, ...] = (10, 50, 90),
    ) -> dict[int, float]:
        """Get finishing-position percentiles for one driver."""
        if driver_id not in self.driver_stats:
            return {}

        positions = self.driver_stats[driver_id].positions
        if not positions:
            return {}

        values = np.array(positions)
        return {p: float(np.percentile(values, p)) for p in percentiles}

    def get_team_championship_projection(self) -> dict[str, float]:
        """Sum listed drivers' observed means, omitting incompletely observed teams.

        Each driver's mean uses their own observation count. This is a lineup
        projection, not an average over a reconstructed union of team races.
        """
        team_points: dict[str, float] = defaultdict(float)
        incomplete: set[str] = set()

        for stats in self.driver_stats.values():
            if not stats.positions:
                incomplete.add(stats.team)
            else:
                team_points[stats.team] += stats.total_points / len(stats.positions)

        return dict(sorted(
            ((team, points) for team, points in team_points.items() if team not in incomplete),
            key=lambda item: item[1], reverse=True,
        ))

    def get_event_rate_trials(self) -> int:
        """Event ledger denominator, retaining nominal counts for legacy aggregates."""
        return max(self.event_stats.num_simulations or self.num_simulations, 0)

    def get_event_rates(self) -> dict[str, float]:
        """Get normalized event-rate metrics per race."""
        sims = max(self.get_event_rate_trials(), 1)
        return {
            "safety_car_race_rate": self.event_stats.races_with_safety_car / sims,
            "red_flag_race_rate": self.event_stats.races_with_red_flag / sims,
            "avg_safety_cars": self.event_stats.safety_car_count / sims,
            "avg_vsc": self.event_stats.vsc_count / sims,
            "avg_incidents": self.event_stats.total_incidents / sims,
        }

    def get_safety_car_calibration_delta(self, expected_race_rate: float) -> float:
        """Absolute delta between observed and expected race-level SC rate."""
        observed = self.get_event_rates()["safety_car_race_rate"]
        return abs(observed - expected_race_rate)

    def get_mechanical_failure_component_rates(self) -> dict[str, float]:
        """Get normalized component breakdown rates for mechanical failures."""
        breakdown = self.event_stats.mechanical_failure_breakdown
        total = sum(breakdown.values())
        if total <= 0:
            return {}

        return {
            component: count / total
            for component, count in sorted(breakdown.items(), key=lambda x: x[1], reverse=True)
        }

    def get_mechanical_calibration_delta(
        self,
        expected_component_rates: dict[str, float],
    ) -> float | None:
        """Mean component-share error, or None when no failures were observed."""
        observed = self.get_mechanical_failure_component_rates()
        if not expected_component_rates:
            return 0.0
        if not observed:
            return None

        keys = set(expected_component_rates) | set(observed)
        if not keys:
            return 0.0

        deltas = [
            abs(observed.get(key, 0.0) - expected_component_rates.get(key, 0.0))
            for key in keys
        ]
        return float(np.mean(deltas))

    def get_mechanical_tuning_suggestions(
        self,
        expected_component_rates: dict[str, float],
    ) -> dict[str, str]:
        """Suggest direction of reliability tuning per component."""
        observed = self.get_mechanical_failure_component_rates()
        if not observed:
            return {}
        suggestions: dict[str, str] = {}

        for component, expected in expected_component_rates.items():
            obs = observed.get(component, 0.0)
            delta = obs - expected
            if delta > 0.03:
                suggestions[component] = "increase reliability"
            elif delta < -0.03:
                suggestions[component] = "decrease reliability"
            else:
                suggestions[component] = "keep as-is"

        return suggestions

    def get_reliability_adjustment_recommendations(
        self,
        expected_component_rates: dict[str, float],
        gain: float = 0.12,
        clamp: float = 0.05,
    ) -> dict[str, float]:
        """Return suggested reliability adjustments per component.

        Negative => lower reliability, positive => increase reliability.
        """
        observed = self.get_mechanical_failure_component_rates()
        if not observed:
            return {}
        adjustments: dict[str, float] = {}

        for component, expected in expected_component_rates.items():
            obs = observed.get(component, 0.0)
            delta = obs - expected
            # If observed failures are too high (delta > 0), increase reliability.
            adjustment = float(np.clip(delta * gain, -clamp, clamp))
            adjustments[component] = adjustment

        return adjustments


def _run_single_simulation(args: tuple) -> tuple[list[RaceResult], list[QualifyingResult], dict]:
    """Run a single race simulation (for multiprocessing).

    Args:
        args: Tuple of (drivers_data, cars_data, track_data, weather_data, seed,
            race_engine, starting_tires). Legacy five/six-item calls retain defaults.

    Returns:
        Tuple of (race_results, qualifying_results, event_counts)
    """
    starting_tires = None
    if len(args) == 5:
        drivers_data, cars_data, track_data, weather_data, seed = args
        race_engine = "standard"
    elif len(args) == 6:
        drivers_data, cars_data, track_data, weather_data, seed, race_engine = args
    else:
        (drivers_data, cars_data, track_data, weather_data, seed,
         race_engine, starting_tires) = args
    race_engine = validate_race_engine(race_engine)

    # Reconstruct objects from serializable data
    drivers = [Driver.model_validate(d) for d in drivers_data]
    opening_compounds = {
        key: TireCompound(value) for key, value in
        validate_starting_tires(starting_tires, (driver.id for driver in drivers)).items()
    }
    cars = {k: Car.model_validate(v) for k, v in cars_data.items()}
    track = Track.model_validate(track_data)
    weather = Weather.model_validate(weather_data)

    # Create RNG with seed
    rng = np.random.default_rng(seed)

    # Every run receives a newly simulated qualifying session.
    quali_sim = QualifyingSimulator(rng=rng)
    quali_results = quali_sim.simulate_qualifying(drivers, cars, track, weather)
    starting_grid = quali_sim.get_starting_grid(quali_results)

    # Run race
    race_sim = RaceSimulator(rng=rng)
    simulate = (
        ChronologicalRace(race_sim).run
        if race_engine == "chronological" else race_sim.simulate_race
    )
    race_results = simulate(
        drivers=drivers,
        cars=cars,
        track=track,
        weather=weather,
        starting_grid=starting_grid,
        **({"starting_tires": opening_compounds} if opening_compounds else {}),
    )

    # Collect event statistics
    events = race_sim.event_manager.events
    mech_failures = [
        e for e in events if e.event_type == EventType.MECHANICAL_FAILURE
    ]
    mech_breakdown: dict[str, int] = defaultdict(int)
    for event in mech_failures:
        desc = event.description.lower()
        if "engine" in desc:
            mech_breakdown["engine"] += 1
        elif "gearbox" in desc:
            mech_breakdown["gearbox"] += 1
        elif "brake" in desc:
            mech_breakdown["brakes"] += 1
        elif "electrical" in desc:
            mech_breakdown["electrical"] += 1
        elif "cooling" in desc:
            mech_breakdown["cooling"] += 1
        else:
            mech_breakdown["other"] += 1

    event_counts = {
        "safety_car": sum(1 for e in events if e.event_type == EventType.SAFETY_CAR),
        "vsc": sum(1 for e in events if e.event_type == EventType.VIRTUAL_SAFETY_CAR),
        "red_flag": sum(1 for e in events if e.event_type == EventType.RED_FLAG),
        "incidents": len([e for e in events if e.event_type in (
            EventType.COLLISION, EventType.SPIN, EventType.PUNCTURE, EventType.MECHANICAL_FAILURE
        )]),
        "mechanical_failure_breakdown": dict(mech_breakdown),
        "weather_history": race_sim.weather_history,
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
        race_engine: str = "standard",
        starting_tires: dict[str, str | TireCompound] | None = None,
    ):
        """Initialize Monte Carlo runner.

        Args:
            drivers: List of drivers
            cars: Dictionary of cars by team_id
            track: Circuit to simulate
            weather: Initial weather conditions
            seed: Random seed for reproducibility
            race_engine: Standard lap loop or experimental chronological execution
            starting_tires: Explicit opening compounds by driver ID; omitted drivers use policy
        """
        self.race_engine = validate_race_engine(race_engine)
        validate_unique_ids((driver.id for driver in drivers), "drivers")
        self.starting_tires = validate_starting_tires(starting_tires, (d.id for d in drivers))
        self.drivers = drivers
        self.cars = cars
        self.track = track
        self.weather = weather
        self.base_seed = seed if seed is not None else np.random.default_rng().integers(0, 2**31)

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
        if (
            isinstance(num_simulations, bool)
            or not isinstance(num_simulations, Integral)
            or num_simulations <= 0
        ):
            msg = "num_simulations must be greater than 0 (integer required)"
            raise ValueError(msg)
        num_simulations = int(num_simulations)

        if max_workers is not None and (
            isinstance(max_workers, bool)
            or not isinstance(max_workers, Integral)
            or max_workers <= 0
        ):
            msg = "max_workers must be greater than 0 (integer required)"
            raise ValueError(msg)
        if max_workers is not None:
            max_workers = int(max_workers)

        validate_unique_ids((driver.id for driver in self.drivers), "drivers")
        starting_tires = validate_starting_tires(
            self.starting_tires, (driver.id for driver in self.drivers),
        )

        # Prepare serializable data for multiprocessing
        drivers_data = [d.model_dump() for d in self.drivers]
        cars_data = {k: v.model_dump() for k, v in self.cars.items()}
        track_data = self.track.model_dump()
        weather_data = self.weather.model_dump()
        input_snapshot = {
            "schema_version": 1,
            "drivers": deepcopy(drivers_data),
            "cars": deepcopy(cars_data),
            "track": deepcopy(track_data),
            "weather": deepcopy(weather_data),
            "starting_tires": starting_tires.copy(),
            "runtime": simulation_runtime(),
        }

        # Generate unique seeds for each simulation
        seeds = [self.base_seed + i for i in range(num_simulations)]

        args_list = [
            (drivers_data, cars_data, track_data, weather_data, seed,
             self.race_engine, starting_tires)
            for seed in seeds
        ]

        all_race_results: list[list[RaceResult]] = []
        all_quali_results: list[list[QualifyingResult]] = []
        all_event_counts: list[dict] = []

        if parallel and num_simulations > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for race_res, quali_res, event_counts in executor.map(
                    _run_single_simulation,
                    args_list,
                ):
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
            seed=int(self.base_seed),
            parallel=parallel,
            max_workers=max_workers,
            race_engine=self.race_engine,
            input_snapshot=input_snapshot,
            weather_histories=[counts.get("weather_history", []) for counts in all_event_counts],
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
                is_dnf = getattr(result.status, "value", result.status) == DriverStatus.DNF.value

                # Classification and operational retirement are independent:
                # a sufficiently late retirement can still score points.
                if result_is_classified(result):
                    if result.position == 1:
                        driver_stat.wins += 1
                    if result.position <= 3:
                        driver_stat.podiums += 1

                points = points_for_result(result)
                if points > 0:
                    driver_stat.points_finishes += 1
                driver_stat.total_points += points

                if is_dnf:
                    driver_stat.dnfs += 1

                # The fixed 20-car defaults (20th best/1st worst) are useful
                # only as empty-state sentinels.  Replace both on the first
                # observation so a first P21/P22 result is not clipped to a
                # fictional 20-car field.
                if len(driver_stat.positions) == 1:
                    driver_stat.best_position = result.position
                    driver_stat.worst_position = result.position
                else:
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
        stats = RaceEventStatistics(num_simulations=len(event_counts))

        breakdown: dict[str, int] = defaultdict(int)

        for counts in event_counts:
            stats.safety_car_count += counts.get("safety_car", 0)
            stats.vsc_count += counts.get("vsc", 0)
            stats.red_flag_count += counts.get("red_flag", 0)
            stats.total_incidents += counts.get("incidents", 0)

            for key, value in counts.get("mechanical_failure_breakdown", {}).items():
                breakdown[key] += int(value)

            if counts.get("safety_car", 0) > 0:
                stats.races_with_safety_car += 1
            if counts.get("red_flag", 0) > 0:
                stats.races_with_red_flag += 1

        stats.mechanical_failure_breakdown = dict(breakdown)

        return stats

    def run_quick(self, num_simulations: int = 100) -> SimulationResults:
        """Run a quick simulation without parallelization.

        Useful for testing or when running in environments
        where multiprocessing is problematic.
        """
        return self.run(num_simulations=num_simulations, parallel=False)
