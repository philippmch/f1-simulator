"""Validate actual qualifying calibration against Q1/Q2/Q3 and earlier-Q1 pace."""

import argparse
import json
from datetime import datetime, timezone

from f1sim.analysis.qualifying_validation import evaluate_qualifying_calibration
from f1sim.analysis.scenarios import scenario_weather_from_label
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.models import Weather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race", help="Name or round; default: all completed current-season races")
    parser.add_argument("--qualifying-only", action="store_true")
    parser.add_argument("--form-races", type=int, default=3)
    parser.add_argument("--scenario", choices=("dry", "light_rain", "heavy_rain"), default="dry")
    parser.add_argument("--fetch-budget", type=float, default=120)
    args = parser.parse_args()
    if not 0 <= args.form_races <= 24:
        parser.error("form-races must be between 0 and 24")
    if not 1 <= args.fetch_budget <= 300:
        parser.error("fetch-budget must be between 1 and 300 seconds")
    target = int(args.race) if args.race and args.race.isdecimal() else args.race
    try:
        report = evaluate_qualifying_calibration(
            CurrentSeasonDataLoader(fetch_budget=args.fetch_budget),
            datetime.now(timezone.utc).year,
            target_race=target, form_races=args.form_races, qualifying_only=args.qualifying_only,
            weather=scenario_weather_from_label(Weather(), args.scenario).weather,
        )
    except (CurrentSeasonDataError, ValueError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
