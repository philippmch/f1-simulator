#!/usr/bin/env python3
"""Example: Simulate an F1 race using historical 2024 data.

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

from f1sim.analysis import MonteCarloRunner
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
        default=2024,
        help="Season year (default: 2024)",
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
    args = parser.parse_args()

    print(f"F1 Monte Carlo Race Simulation")
    print(f"{'=' * 40}")
    print(f"Year: {args.year}")
    print(f"Race: {args.race}")
    print(f"Simulations: {args.simulations}")
    print(f"Parallel: {args.parallel}")
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

    runner = MonteCarloRunner(
        drivers=drivers,
        cars=cars,
        track=track,
        weather=weather,
        seed=42,  # For reproducibility
        historical_grid=historical_grid,
    )

    results = runner.run(
        num_simulations=args.simulations,
        parallel=args.parallel,
    )

    # Display results
    ConsoleOutput.print_monte_carlo_summary(results)

    # Show detailed driver analysis if requested
    if args.driver:
        ConsoleOutput.print_driver_deep_dive(results, args.driver.upper())

    # Export results if requested
    if args.export:
        print(f"\nExporting results to {args.output_dir}/...")
        exporter = Exporter(output_dir=args.output_dir)
        files = exporter.export_all(results, prefix=f"{args.year}_{track.id}")

        print("Exported files:")
        for fmt, path in files.items():
            print(f"  {fmt}: {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
