"""Reproducible synthetic checks and optional current-season observed event rates.

This is an offline model diagnostic unless --observed is supplied. Provider rows
stay in memory. Observed rainfall is binary and cannot fit intensity or drainage.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.simulation.execution import RACE_ENGINES, validate_race_engine
from f1sim.simulation.race import DriverStatus


def observed_summary() -> dict:
    """Summarize completed current-season races; fail if a source is incomplete."""
    now = datetime.now(timezone.utc)
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    deadline = time.monotonic() + 120.0

    def fetch(path: str) -> list[dict]:
        time.sleep(0.3)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Observed-data fetch budget exceeded")
        with urlopen("https://api.openf1.org/v1/" + path, timeout=min(15, remaining)) as response:
            raw = response.read(4_000_001)
        if len(raw) > 4_000_000:
            raise RuntimeError("Observed response exceeds size budget")
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise RuntimeError("Expected OpenF1 records")
        return rows

    races = []
    for session in fetch(f"sessions?year={now.year}&session_name=Race"):
        if session["year"] != now.year or session["session_name"] != "Race":
            raise RuntimeError("Observed source returned an unexpected season/session")
        if session.get("is_cancelled") or datetime.fromisoformat(session["date_end"]) >= cutoff:
            continue
        key = session["session_key"]
        start = datetime.fromisoformat(session["date_start"])
        control = fetch(f"race_control?session_key={key}")
        finishes = [datetime.fromisoformat(row["date"]) for row in control
                    if row.get("flag") == "CHEQUERED"]
        if not finishes:
            raise RuntimeError(f"No completed-race evidence for session {key}")
        end = max(finishes)
        control = [row for row in control if start <= datetime.fromisoformat(row["date"]) <= end]
        weather = [row for row in fetch(f"weather?session_key={key}")
                   if start <= datetime.fromisoformat(row["date"]) <= end]
        if not weather:
            raise RuntimeError(f"No race weather observations for session {key}")
        races.append({
            "session": key,
            "venue": session["location"],
            "rain_observed": any(row["rainfall"] for row in weather),
            # Some current records carry RED FLAG only in message, with flag=null.
            "red_flag": any(row.get("flag") == "RED" or
                            row.get("message", "").startswith("RED FLAG") for row in control),
        })
    return {"year": now.year, "completed_before": cutoff.isoformat(), "races": races}


def model_summary(simulations: int, seed: int = 42, race_engine: str = "standard") -> list[dict]:
    """Hold car/track parameters fixed while varying initial weather."""
    validate_race_engine(race_engine)
    drivers = [Driver(id=f"D{i:02}", name=f"Driver {i}", team_id=f"T{i // 2}")
               for i in range(22)]
    cars = {driver.team_id: Car(team_id=driver.team_id, team_name=driver.team_id,
                               reliability=1.0) for driver in drivers}
    track = Track(id="diagnostic", name="Diagnostic", country="Synthetic",
                  total_laps=50, base_lap_time=90.0, safety_car_probability=0.3)
    conditions = {
        "fixed_dry": Weather(change_probability=0),
        "fixed_light_rain": Weather(condition=WeatherCondition.LIGHT_RAIN,
                                    rain_intensity=0.35, track_wetness=0.45,
                                    change_probability=0),
        "fixed_heavy_rain": Weather(condition=WeatherCondition.HEAVY_RAIN,
                                    rain_intensity=0.85, track_wetness=0.85,
                                    change_probability=0),
        "evolving_dry_start": Weather(change_probability=0.1),
    }
    summaries = []
    for name, weather in conditions.items():
        result = MonteCarloRunner(
            drivers, cars, track, weather, seed=seed, race_engine=race_engine,
        ).run(
            num_simulations=simulations, parallel=False,
        )
        finishers = [row for race in result.race_results for row in race
                     if row.status == DriverStatus.FINISHED]
        winners = [row for row in finishers if row.position == 1]
        lapped = sum(
            row.laps_completed < winner.laps_completed
            for race in result.race_results
            for winner in race if winner.position == 1 and winner.status == DriverStatus.FINISHED
            for row in race if row.status == DriverStatus.FINISHED
        )
        entrants = [row for race in result.race_results for row in race]
        summaries.append({
            "scenario": name, "simulations": simulations, "seed": seed,
            "race_engine": result.race_engine,
            "red_flag_race_rate": result.event_stats.races_with_red_flag / simulations,
            "red_flags_per_race": result.event_stats.red_flag_count / simulations,
            "finishing_cars": len(finishers),
            "lapped_finishers": lapped,
            "lapped_finisher_rate": lapped / len(finishers) if finishers else None,
            "races_with_winner": len(winners),
            "mean_winner_seconds": (sum(row.total_time for row in winners) / len(winners)
                                    if winners else None),
            "mean_pit_stops_per_entrant": (sum(row.pit_stops for row in entrants) / len(entrants)
                                           if entrants else None),
            "time_limited_races": sum(row.race_time_limited for row in winners),
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", action="store_true", help="Fetch current-season observations")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--race-engine", choices=(*RACE_ENGINES, "both"), default="standard",
                        help="Model to diagnose; both compares identical inputs and seeds")
    args = parser.parse_args()
    if not 1 <= args.simulations <= 1000:
        parser.error("simulations must be between 1 and 1000")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be between 0 and 4294967295")
    engines = RACE_ENGINES if args.race_engine == "both" else (args.race_engine,)
    observed = observed_summary() if args.observed else None
    with contextlib.redirect_stdout(sys.stderr):
        summaries = [row for engine in engines
                     for row in model_summary(args.simulations, args.seed, engine)]
    output = {"observed": observed, "model": summaries} if args.observed else summaries
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
