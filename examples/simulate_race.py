#!/usr/bin/env python3
"""Simulate a current-season F1 race using freshly fetched data.

This script demonstrates the full workflow:
1. Fetch the live current-season calendar, roster, and results
2. Create driver/car/track models from current-season data
3. Run Monte Carlo simulations
4. Display and export results

Usage:
    python examples/simulate_race.py [--race RACE] [--simulations N]

Examples:
    python examples/simulate_race.py --race "Monaco" --simulations 100
    python examples/simulate_race.py --race 1 --simulations 1000
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.analysis import (
    MonteCarloRunner,
    parse_scenario_labels,
    scenario_weather_from_label,
)
from f1sim.data import CurrentSeasonDataLoader
from f1sim.models import Weather, WeatherCondition
from f1sim.output import ConsoleOutput, Exporter
from f1sim.simulation.execution import RACE_ENGINES, validate_starting_tires

MAX_SIMULATIONS = 1000
MAX_WORKERS = 16
MAX_TOP_N = 22
MAX_SEED = 2**32 - 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _bounded_positive_int(value: str, maximum: int) -> int:
    parsed = _positive_int(value)
    if parsed > maximum:
        raise argparse.ArgumentTypeError(f"must be at most {maximum}")
    return parsed


def _simulation_count(value: str) -> int:
    return _bounded_positive_int(value, MAX_SIMULATIONS)


def _worker_count(value: str) -> int:
    return _bounded_positive_int(value, MAX_WORKERS)


def _top_n_count(value: str) -> int:
    return _bounded_positive_int(value, MAX_TOP_N)


def _seed_value(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed > MAX_SEED:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_SEED}")
    return parsed


def _starting_tires(value: str) -> dict[str, str]:
    overrides = {}
    for assignment in value.split(","):
        driver, separator, compound = assignment.strip().partition("=")
        driver, compound = driver.strip(), compound.strip().lower()
        if not separator or not driver or not compound:
            raise argparse.ArgumentTypeError("use DRIVER=compound, e.g. VER=hard,NOR=soft")
        if driver in overrides:
            raise argparse.ArgumentTypeError(f"duplicate starting tyre override for {driver}")
        overrides[driver] = compound
    try:
        return validate_starting_tires(overrides)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate F1 race with Monte Carlo")
    parser.add_argument(
        "--race",
        default="1",
        help="Current-season race name or round number (default: round 1)",
    )
    parser.add_argument(
        "--simulations",
        "-n",
        type=_simulation_count,
        default=100,
        help=f"Number of simulations (1-{MAX_SIMULATIONS}; default: 100)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=True,
        help="Use parallel processing (default: True)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_false",
        dest="parallel",
        help="Disable parallel processing",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export results to CSV/JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory for exports (default: output)",
    )
    parser.add_argument(
        "--driver",
        help="Show detailed analysis for specific driver (e.g., VER, HAM)",
    )
    parser.add_argument(
        "--top-n",
        type=_top_n_count,
        default=10,
        help=f"Top-N finish probability table size (1-{MAX_TOP_N}; default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=_seed_value,
        default=42,
        help=f"Random seed for reproducible runs (0-{MAX_SEED}; default: 42)",
    )
    parser.add_argument(
        "--max-workers",
        type=_worker_count,
        help=f"Maximum worker processes for parallel mode (1-{MAX_WORKERS})",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help=(
            "Optional comma-separated weather scenarios to compare "
            "(dry,cloudy,light_rain,heavy_rain)"
        ),
    )
    parser.add_argument(
        "--race-engine", choices=RACE_ENGINES, default="standard",
        help="Race model (default: standard; chronological is experimental)",
    )
    parser.add_argument(
        "--starting-tyres", "--starting-tires", dest="starting_tires", type=_starting_tires,
        help="Optional DRIVER=compound pairs, e.g. VER=hard,NOR=soft; others stay automatic",
    )
    args = parser.parse_args()
    print(f"Race model: {args.race_engine}")
    try:
        scenario_labels = parse_scenario_labels(args.scenarios) if args.scenarios else ["dry"]
    except ValueError as exc:
        parser.error(str(exc))

    print("F1 Monte Carlo Race Simulation")
    print(f"{'=' * 40}")
    current_season = datetime.now(timezone.utc).year
    print(f"Season: {current_season} (live only)")
    print(f"Race: {args.race}")
    print(f"Simulations: {args.simulations}")
    print(f"Parallel: {args.parallel}")
    print(f"Seed: {args.seed}")
    print(f"Top-N table: {args.top_n}")
    if args.max_workers is not None:
        print(f"Max workers: {args.max_workers}")
    print()

    # Initialize the live, current-season-only data loader.
    print("Fetching the current calendar, official roster, and season form...")
    loader = CurrentSeasonDataLoader(current_year=current_season)

    # Get driver stats using weighted form + track performance
    print("Loading current driver, team, and form statistics...")
    try:
        driver_stats = loader.get_weighted_driver_stats(
            year=current_season,
            target_race=args.race,
            form_races=3,
            track_weight=0.5,
            form_weight=0.3,
            quali_weight=0.2,
        )
        print(f"Loaded data for {len(driver_stats)} drivers")
    except Exception as e:
        print(f"Error loading driver data: {e}")
        return 1

    # Result rows loaded above authoritatively mark completed rounds before
    # the track model optionally calibrates its fastest-lap reference.
    try:
        track_stats = loader.get_track_stats(current_season, args.race)
        print(f"Track: {track_stats.track_name} ({track_stats.country})")
        print(f"Laps: {track_stats.total_laps}")
        print(f"Avg lap time: {track_stats.avg_lap_time:.3f}s")
    except Exception as e:
        print(f"Error loading track data: {e}")
        print("Make sure this machine can access Jolpica and Formula1.com.")
        return 1

    # Create models from fresh current-season data.
    print("\nCreating simulation models...")
    drivers = loader.create_drivers_from_stats(driver_stats)
    try:
        starting_tires = validate_starting_tires(args.starting_tires, (d.id for d in drivers))
    except ValueError as exc:
        parser.error(str(exc))
    if starting_tires:
        print("Starting tyres: " + ", ".join(f"{key}={value}"
                                             for key, value in starting_tires.items()))
    cars = loader.create_cars_from_stats(driver_stats)
    track = loader.create_track_from_stats(track_stats)

    # Set up weather (default to dry)
    weather = Weather(
        condition=WeatherCondition.DRY,
        track_temperature=35.0,
        air_temperature=25.0,
        change_probability=track.weather_variability,
    )

    print(f"\nDrivers: {len(drivers)}")
    print(f"Teams: {len(cars)}")

    # Run Monte Carlo simulation
    print(f"\nRunning {args.simulations} simulations...")
    print("(This may take a while for large numbers of simulations)")

    scenario_results = {}

    for idx, label in enumerate(scenario_labels):
        scenario = scenario_weather_from_label(weather, label)
        scenario_seed = args.seed + idx * 1000

        print(f"\n--- Scenario: {scenario.name} (seed={scenario_seed}) ---")
        runner = MonteCarloRunner(
            drivers=drivers,
            cars=cars,
            track=track,
            weather=scenario.weather,
            seed=scenario_seed,
            race_engine=args.race_engine,
            **({"starting_tires": starting_tires} if starting_tires else {}),
        )

        scenario_result = runner.run(
            num_simulations=args.simulations,
            parallel=args.parallel,
            max_workers=args.max_workers,
        )
        scenario_results[scenario.name] = scenario_result

    # Display results
    if len(scenario_results) == 1:
        results = next(iter(scenario_results.values()))
        ConsoleOutput.print_monte_carlo_summary(results, top_n=args.top_n)
        ConsoleOutput.print_event_calibration(
            results,
            expected_sc_race_rate=track.safety_car_probability,
        )

        # Show detailed driver analysis if requested
        if args.driver:
            ConsoleOutput.print_driver_deep_dive(results, args.driver.upper())
    else:
        ConsoleOutput.print_scenario_comparison(scenario_results, top_n=args.top_n)
        print("\nSCENARIO EVENT CALIBRATION")
        print("-" * 50)
        for scenario_name, scenario_result in scenario_results.items():
            delta = scenario_result.get_safety_car_calibration_delta(track.safety_car_probability)
            observed = scenario_result.get_event_rates()["safety_car_race_rate"]
            print(
                f"{scenario_name:<14} expected={track.safety_car_probability * 100:5.1f}% "
                f"observed={observed * 100:5.1f}% delta={delta * 100:5.1f}%"
            )

    # Export results if requested
    if args.export:
        print(f"\nExporting results to {args.output_dir}/...")
        exporter = Exporter(output_dir=args.output_dir)

        print("Exported files:")
        for scenario_name, scenario_result in scenario_results.items():
            files = exporter.export_all(
                scenario_result,
                prefix=f"{current_season}_{track.id}_{scenario_name}",
            )
            for fmt, path in files.items():
                print(f"  {scenario_name}:{fmt}: {path}")

        if len(scenario_results) > 1:
            comparison_prefix = f"{current_season}_{track.id}_scenario_comparison_{uuid4().hex}"
            comparison = exporter.export_scenario_comparison_json(
                scenario_results,
                filename=f"{comparison_prefix}.json",
            )
            print(f"  comparison_json: {comparison}")
            report = exporter.export_scenario_comparison_html(
                scenario_results, filename=f"{comparison_prefix}.html",
            )
            print(f"  comparison_html: {report}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
