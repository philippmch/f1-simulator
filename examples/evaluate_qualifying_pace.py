"""Evaluate current-season qualifying pace without target or later-round performance inputs.

Fetches today's revised provider data. The target qualifying entrants are known;
weather is a fixed model scenario. Prints one JSON report and retains no feed cache.
"""

import argparse
import json
from datetime import datetime, timezone

from f1sim.analysis.pace_evaluation import evaluate_qualifying_pace
from f1sim.analysis.scenarios import scenario_weather_from_label
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.models import Weather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race", help="Completed race name or round; default: all completed races")
    parser.add_argument("--form-races", type=int, default=3)
    parser.add_argument("--scenario", choices=("dry", "light_rain", "heavy_rain"), default="dry")
    parser.add_argument("--fetch-budget", type=float, default=120,
                        help="Total live fetch budget in seconds (1–300)")
    args = parser.parse_args()
    if not 0 <= args.form_races <= 24:
        parser.error("form-races must be between 0 and 24")
    if not 1 <= args.fetch_budget <= 300:
        parser.error("fetch-budget must be between 1 and 300 seconds")
    target = int(args.race) if args.race and args.race.isdecimal() else args.race
    try:
        result = evaluate_qualifying_pace(
            CurrentSeasonDataLoader(fetch_budget=args.fetch_budget),
            datetime.now(timezone.utc).year, target_race=target, form_races=args.form_races,
            weather=scenario_weather_from_label(Weather(), args.scenario).weather,
        )
    except (CurrentSeasonDataError, ValueError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
