#!/usr/bin/env python3
"""Quick simulation example using synthetic data.

This example doesn't require FastF1 data downloads,
making it useful for testing the simulation engine.

Usage:
    python examples/quick_simulation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.analysis import MonteCarloRunner
from f1sim.models import Car, DRSZone, Driver, Sector, Track, Weather, WeatherCondition
from f1sim.output import ConsoleOutput, Exporter
from f1sim.simulation import QualifyingSimulator, RaceSimulator


def create_2024_grid() -> tuple[list[Driver], dict[str, Car]]:
    """Create a simplified 2024 F1 grid with realistic-ish performance."""

    # Define teams with car performance
    teams_data = [
        ("red_bull", "Red Bull Racing", 1.00, 0.97),
        ("ferrari", "Scuderia Ferrari", 0.96, 0.95),
        ("mclaren", "McLaren F1 Team", 0.95, 0.94),
        ("mercedes", "Mercedes-AMG F1", 0.94, 0.96),
        ("aston_martin", "Aston Martin", 0.88, 0.94),
        ("alpine", "Alpine F1 Team", 0.84, 0.93),
        ("williams", "Williams Racing", 0.82, 0.92),
        ("haas", "MoneyGram Haas", 0.80, 0.91),
        ("alfa_romeo", "Stake F1 Team", 0.79, 0.92),
        ("alphatauri", "Visa RB", 0.81, 0.93),
    ]

    # Define drivers with skills
    drivers_data = [
        ("VER", "Max Verstappen", "red_bull", 0.98, 0.96, 1.05),
        ("PER", "Sergio Perez", "red_bull", 0.88, 0.85, 0.95),
        ("LEC", "Charles Leclerc", "ferrari", 0.94, 0.88, 1.00),
        ("SAI", "Carlos Sainz", "ferrari", 0.91, 0.92, 0.98),
        ("NOR", "Lando Norris", "mclaren", 0.93, 0.90, 1.02),
        ("PIA", "Oscar Piastri", "mclaren", 0.89, 0.91, 0.97),
        ("HAM", "Lewis Hamilton", "mercedes", 0.95, 0.94, 1.08),
        ("RUS", "George Russell", "mercedes", 0.91, 0.89, 1.00),
        ("ALO", "Fernando Alonso", "aston_martin", 0.92, 0.95, 1.05),
        ("STR", "Lance Stroll", "aston_martin", 0.82, 0.80, 0.92),
        ("GAS", "Pierre Gasly", "alpine", 0.86, 0.88, 0.98),
        ("OCO", "Esteban Ocon", "alpine", 0.85, 0.86, 0.96),
        ("ALB", "Alexander Albon", "williams", 0.87, 0.88, 0.97),
        ("SAR", "Logan Sargeant", "williams", 0.75, 0.78, 0.90),
        ("MAG", "Kevin Magnussen", "haas", 0.83, 0.82, 0.95),
        ("HUL", "Nico Hulkenberg", "haas", 0.84, 0.86, 0.94),
        ("BOT", "Valtteri Bottas", "alfa_romeo", 0.85, 0.90, 0.96),
        ("ZHO", "Zhou Guanyu", "alfa_romeo", 0.79, 0.82, 0.92),
        ("TSU", "Yuki Tsunoda", "alphatauri", 0.84, 0.80, 0.95),
        ("RIC", "Daniel Ricciardo", "alphatauri", 0.86, 0.84, 0.98),
    ]

    cars = {}
    for team_id, team_name, pace, reliability in teams_data:
        cars[team_id] = Car(
            team_id=team_id,
            team_name=team_name,
            base_pace=pace,
            reliability=reliability,
            downforce_level=0.85,
            straight_line_speed=0.85,
            pit_stop_avg=2.5,
            pit_stop_std=0.3,
        )

    drivers = []
    for driver_id, name, team, skill, consistency, wet_skill in drivers_data:
        drivers.append(Driver(
            id=driver_id,
            name=name,
            team_id=team,
            skill_rating=skill,
            consistency=consistency,
            wet_skill_modifier=wet_skill,
            overtaking_skill=skill * 0.95,
            tire_management=consistency,
        ))

    return drivers, cars


def create_monza_track() -> Track:
    """Create Monza circuit configuration."""
    return Track(
        id="monza",
        name="Italian Grand Prix - Monza",
        country="Italy",
        total_laps=53,
        base_lap_time=82.0,  # ~1:22
        pit_lane_delta=22.0,
        sectors=[
            Sector(number=1, base_time=27.0, is_high_speed=True, overtake_opportunity=0.7),
            Sector(number=2, base_time=28.0, is_high_speed=True, overtake_opportunity=0.5),
            Sector(number=3, base_time=27.0, is_high_speed=False, overtake_opportunity=0.3),
        ],
        drs_zones=[
            DRSZone(zone_id=1, sector=1, time_gain=0.4),
            DRSZone(zone_id=2, sector=2, time_gain=0.35),
        ],
        overtake_difficulty=0.3,  # Easy to overtake
        tire_stress=0.4,
        safety_car_probability=0.25,
        weather_variability=0.15,
    )


def main():
    print("F1 Monte Carlo Simulation - Quick Example")
    print("=" * 50)

    # Create grid and track
    drivers, cars = create_2024_grid()
    track = create_monza_track()
    weather = Weather(condition=WeatherCondition.DRY)

    print(f"Track: {track.name}")
    print(f"Drivers: {len(drivers)}")
    print(f"Laps: {track.total_laps}")
    print()

    # Run a single race first to show detailed output
    print("Running single race simulation...")
    print("-" * 50)

    import numpy as np
    rng = np.random.default_rng(42)

    # Qualifying
    quali_sim = QualifyingSimulator(rng=rng)
    quali_results = quali_sim.simulate_qualifying(drivers, cars, track, weather)
    ConsoleOutput.print_qualifying_results(quali_results)

    # Race
    starting_grid = quali_sim.get_starting_grid(quali_results)
    race_sim = RaceSimulator(rng=rng)
    race_results = race_sim.simulate_race(
        drivers=drivers,
        cars=cars,
        track=track,
        weather=weather,
        starting_grid=starting_grid,
    )
    ConsoleOutput.print_race_results(race_results)

    # Now run Monte Carlo simulation
    print("\n" + "=" * 50)
    print("Running Monte Carlo simulation (100 races)...")
    print("=" * 50)

    runner = MonteCarloRunner(
        drivers=drivers,
        cars=cars,
        track=track,
        weather=weather,
        seed=123,
    )

    # Use quick run (non-parallel) for simplicity
    results = runner.run_quick(num_simulations=100)

    # Display summary
    ConsoleOutput.print_monte_carlo_summary(results)

    # Show Verstappen deep dive
    ConsoleOutput.print_driver_deep_dive(results, "VER")

    # Export results
    print("\nExporting results...")
    exporter = Exporter(output_dir="output")
    files = exporter.export_all(results, prefix="monza_quick")
    for fmt, path in files.items():
        print(f"  {fmt}: {path}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
