#!/usr/bin/env python3
"""Example: Simulate an F1 race using historical data.

This script demonstrates the full workflow:
1. Load historical data from FastF1
2. Create driver/car/track models from real data
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
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.analysis import (
    MonteCarloRunner,
    parse_scenario_labels,
    scenario_weather_from_label,
)
from f1sim.data import HistoricalDataLoader
from f1sim.models import Weather, WeatherCondition
from f1sim.output import ConsoleOutput, Exporter


def main():
    parser = argparse.ArgumentParser(description="Simulate F1 race with Monte Carlo")
    parser.add_argument(
        "--race",
        default="Bahrain",
        help="Race name or round number (default: Bahrain)",
    )
    parser.add_argument(
        "--simulations",
        "-n",
        type=int,
        default=100,
        help="Number of simulations (default: 100)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Season year (default: 2025)",
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
        type=int,
        default=10,
        help="Top-N finish probability table size (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible simulation runs (default: 42)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum worker processes for parallel mode (default: CPU count)",
    )
    parser.add_argument(
        "--scenarios",
        default="",
        help=(
            "Optional comma-separated weather scenarios to compare "
            "(dry,cloudy,light_rain,heavy_rain)"
        ),
    )
    args = parser.parse_args()

    print("F1 Monte Carlo Race Simulation")
    print(f"{'=' * 40}")
    print(f"Year: {args.year}")
    print(f"Race: {args.race}")
    print(f"Simulations: {args.simulations}")
    print(f"Parallel: {args.parallel}")
    print(f"Seed: {args.seed}")
    print(f"Top-N table: {args.top_n}")
    if args.max_workers:
        print(f"Max workers: {args.max_workers}")
    print()

    # Initialize data loader
    print("Loading historical data from FastF1...")
    loader = HistoricalDataLoader(cache_dir="data/cache")

    # Get track stats for the specified race
    try:
        track_stats = loader.get_track_stats(args.year, args.race)
        print(f"Track: {track_stats.track_name} ({track_stats.country})")
        print(f"Laps: {track_stats.total_laps}")
        print(f"Avg lap time: {track_stats.avg_lap_time:.3f}s")
    except Exception as e:
        print(f"Error loading track data: {e}")
        print("Make sure FastF1 can access the race data.")
        return 1

    # Get driver stats using weighted form + track performance
    print("\nLoading driver statistics (form + track-specific)...")
    try:
        # Weighted combination: 50% track race, 30% recent form, 20% qualifying
        driver_stats = loader.get_weighted_driver_stats(
            year=args.year,
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

    # Create models from historical data
    print("\nCreating simulation models...")
    drivers = loader.create_drivers_from_stats(driver_stats)
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

    # Load historical qualifying grid
    print("\nLoading historical qualifying grid...")
    historical_grid = loader.get_historical_grid(args.year, args.race)
    print(f"Grid: {' '.join(historical_grid[:5])}...")

    # Run Monte Carlo simulation
    print(f"\nRunning {args.simulations} simulations...")
    print("(This may take a while for large numbers of simulations)")

    scenario_results = {}

    scenario_labels = parse_scenario_labels(args.scenarios) if args.scenarios else ["dry"]

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
            historical_grid=historical_grid,
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

        # Show detailed driver analysis if requested
        if args.driver:
            ConsoleOutput.print_driver_deep_dive(results, args.driver.upper())
    else:
        ConsoleOutput.print_scenario_comparison(scenario_results, top_n=args.top_n)

    # Export results if requested
    if args.export:
        print(f"\nExporting results to {args.output_dir}/...")
        exporter = Exporter(output_dir=args.output_dir)

        print("Exported files:")
        for scenario_name, scenario_result in scenario_results.items():
            files = exporter.export_all(
                scenario_result,
                prefix=f"{args.year}_{track.id}_{scenario_name}",
            )
            for fmt, path in files.items():
                print(f"  {scenario_name}:{fmt}: {path}")

        if len(scenario_results) > 1:
            comparison = exporter.export_scenario_comparison_json(
                scenario_results,
                filename=f"{args.year}_{track.id}_scenario_comparison.json",
            )
            print(f"  comparison_json: {comparison}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
