"""Reproducible synthetic checks and optional current-season observed event rates.

This is an offline model diagnostic unless an --observed option is supplied.
Provider rows stay in memory. Observed rainfall is binary and cannot fit intensity
or drainage; observed strategy evidence does not establish optimal tyre choices.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from math import isfinite
from urllib.error import HTTPError
from urllib.request import urlopen

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.observed_strategy import observed_strategy_summary
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.simulation.execution import RACE_ENGINES, validate_race_engine
from f1sim.simulation.race import DriverStatus


def _records(rows, session_key=None) -> list[dict]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("Expected OpenF1 records")
    if session_key is not None and any(
        type(row.get("session_key")) is not int or row["session_key"] != session_key
        for row in rows
    ):
        raise RuntimeError("Observed source returned an unexpected session")
    return rows


def _timestamp(value) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.utcoffset() is None:
            raise ValueError("Missing timezone")
        return parsed
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Expected an OpenF1 timestamp with a timezone") from exc


def rain_stint_summary(rows: list[dict], session_key: int) -> dict:
    """Describe reported rain stints without inferring wear or why they ended."""
    _records(rows, session_key)
    if not rows:
        raise RuntimeError(f"No stint evidence for session {session_key}")
    stints, excluded, seen = [], {}, {}

    def exclude(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    def integer(value, minimum):
        return isinstance(value, int) and not isinstance(value, bool) and value >= minimum

    for row in rows:
        if row.get("session_key") != session_key:
            raise RuntimeError("Stint source returned an unexpected session")
        identity = (row.get("driver_number"), row.get("stint_number"))
        if not all(integer(value, 1) for value in identity):
            exclude("missing_identity")
            continue
        if identity in seen:
            if seen[identity] != row:
                raise RuntimeError("Conflicting stint records")
            exclude("duplicate_record")
            continue
        seen[identity] = row
        compound = row.get("compound")
        if compound in {"SOFT", "MEDIUM", "HARD"}:
            continue
        if compound not in {"INTERMEDIATE", "WET"}:
            exclude("unknown_compound")
            continue
        start, end = row.get("lap_start"), row.get("lap_end")
        if not integer(start, 1) or not integer(end, start):
            exclude("incomplete_lap_range")
            continue
        age = row.get("tyre_age_at_start")
        age = age if integer(age, 0) else None
        laps = end - start + 1
        stints.append({
            "driver_number": identity[0], "stint_number": identity[1], "compound": compound,
            "lap_start": start, "lap_end": end, "reported_laps": laps,
            "tyre_age_at_start": age,
            "tyre_age_at_end": age + laps if age is not None else None,
            "end_reason": "unknown",
        })
    stints.sort(key=lambda row: (row["driver_number"], row["stint_number"]))
    return {"source_records": len(rows), "excluded_records": excluded, "stints": stints}


def observed_summary(*, include_stints: bool = False, include_strategy: bool = False) -> dict:
    """Summarize current-season evidence; fail on invalid or unavailable required feeds."""
    now = datetime.now(timezone.utc)
    cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
    budget = 360.0 if include_strategy else 240.0 if include_stints else 180.0
    deadline = time.monotonic() + budget

    def fetch(path: str, session_key: int | None = None) -> list[dict]:
        for attempt in range(3):
            if deadline - time.monotonic() <= 2.1:
                raise RuntimeError("Observed-data fetch budget exceeded")
            time.sleep(2.1)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Observed-data fetch budget exceeded")
            try:
                with urlopen("https://api.openf1.org/v1/" + path,
                             timeout=min(15, remaining)) as response:
                    raw = response.read(4_000_001)
                break
            except HTTPError as exc:
                if exc.code != 429 or attempt == 2:
                    raise RuntimeError(
                        f"Observed request failed for {path}: HTTP {exc.code}"
                    ) from exc
                retry = (exc.headers or {}).get("Retry-After", "")
                try:
                    delay = float(retry)
                except ValueError:
                    try:
                        delay = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)
                                 ).total_seconds()
                    except (TypeError, ValueError, OverflowError):
                        delay = 5.0 * (attempt + 1)
                if not isfinite(delay) or delay < 0:
                    delay = 5.0 * (attempt + 1)
                if delay > 60 or delay + 2.1 >= deadline - time.monotonic():
                    raise RuntimeError(
                        "Provider retry delay exceeds observed fetch budget"
                    ) from exc
                time.sleep(delay)
        if len(raw) > 4_000_000:
            raise RuntimeError("Observed response exceeds size budget")
        return _records(json.loads(raw), session_key)

    races = []
    seen_sessions = {}
    for session in fetch(f"sessions?year={now.year}&session_name=Race"):
        if type(session.get("year")) is not int or session["year"] != now.year or (
            session.get("session_name") != "Race"
        ):
            raise RuntimeError("Observed source returned an unexpected season/session")
        key = session.get("session_key")
        if type(key) is not int or key < 1:
            raise RuntimeError("Expected a positive OpenF1 session key")
        if key in seen_sessions:
            if seen_sessions[key] != session:
                raise RuntimeError("Conflicting session records")
            continue
        seen_sessions[key] = session
        if session.get("is_cancelled"):
            continue
        scheduled_end = _timestamp(session.get("date_end"))
        start = _timestamp(session.get("date_start"))
        if scheduled_end < start:
            raise RuntimeError("Observed session ends before it starts")
        if scheduled_end >= cutoff:
            continue
        control = fetch(f"race_control?session_key={key}", key)
        dated_control = [(_timestamp(row.get("date")), row) for row in control]
        finishes = [date for date, row in dated_control
                    if row.get("flag") == "CHEQUERED" and start <= date < cutoff]
        if not finishes:
            raise RuntimeError(f"No completed-race evidence for session {key}")
        end = max(finishes)
        control = [row for date, row in dated_control if start <= date <= end]
        weather = [row for row in fetch(f"weather?session_key={key}", key)
                   if start <= _timestamp(row.get("date")) <= end]
        if not weather:
            raise RuntimeError(f"No race weather observations for session {key}")
        if any(row.get("rainfall") not in (0, 1) for row in weather):
            raise RuntimeError("Expected binary OpenF1 rainfall observations")
        if any(row.get("message") is not None and not isinstance(row["message"], str)
               for row in control):
            raise RuntimeError("Expected OpenF1 race-control messages")
        race = {
            "session": key,
            "venue": session["location"],
            "rain_observed": any(row["rainfall"] for row in weather),
            # Some current records carry RED FLAG only in message, with flag=null.
            "red_flag": any(row.get("flag") == "RED" or
                            (row.get("message") or "").startswith("RED FLAG") for row in control),
        }
        if include_stints or include_strategy:
            stint_rows = fetch(f"stints?session_key={key}", key)
        if include_stints:
            race["rain_stints"] = rain_stint_summary(stint_rows, key)
        if include_strategy:
            race["strategy"] = observed_strategy_summary(
                fetch(f"session_result?session_key={key}", key), stint_rows,
                fetch(f"pit?session_key={key}", key), key,
            )
        races.append(race)
    races.sort(key=lambda race: race["session"])
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
    parser.add_argument("--observed-stints", action="store_true",
                        help="Include reported rain stints; implies --observed")
    parser.add_argument("--observed-strategy", action="store_true",
                        help="Check stint coverage and pit-lane evidence; implies --observed")
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
    observed = (observed_summary(include_stints=args.observed_stints,
                                 include_strategy=args.observed_strategy)
                if args.observed or args.observed_stints or args.observed_strategy else None)
    with contextlib.redirect_stdout(sys.stderr):
        summaries = [row for engine in engines
                     for row in model_summary(args.simulations, args.seed, engine)]
    output = {"observed": observed, "model": summaries} if observed is not None else summaries
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
