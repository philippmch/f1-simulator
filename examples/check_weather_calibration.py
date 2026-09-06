"""Reproducible synthetic checks and optional current-season observed event rates.

This is an offline model diagnostic unless --observed is supplied. Provider rows
stay in memory. Observed rainfall is binary and cannot fit intensity or drainage.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from urllib.request import urlopen

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition


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


def model_summary(simulations: int, seed: int = 42) -> list[dict]:
    """Hold car/track parameters fixed while varying initial weather."""
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
        result = MonteCarloRunner(drivers, cars, track, weather, seed=seed).run(
            num_simulations=simulations, parallel=False,
        )
        summaries.append({
            "scenario": name, "simulations": simulations, "seed": seed,
            "red_flag_race_rate": result.event_stats.races_with_red_flag / simulations,
            "red_flags_per_race": result.event_stats.red_flag_count / simulations,
        })
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", action="store_true", help="Fetch current-season observations")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.simulations <= 1000:
        parser.error("simulations must be between 1 and 1000")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be between 0 and 4294967295")
    if args.observed:
        print(json.dumps(observed_summary(), indent=2), flush=True)
    print(json.dumps(model_summary(args.simulations, args.seed), indent=2))


if __name__ == "__main__":
    main()
