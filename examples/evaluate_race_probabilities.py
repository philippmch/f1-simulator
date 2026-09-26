"""Score simulated race-winner probabilities against completed current-season races."""

import argparse
import json
import sys
from datetime import datetime, timezone

from f1sim.analysis.race_probability_evaluation import evaluate_race_probabilities
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader


def _race_value(value: str) -> str | int:
    return int(value) if value.isascii() and value.isdecimal() else value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--race", help="Completed race name or round number")
    selection.add_argument(
        "--all", dest="all_targets", action="store_true",
        help="Evaluate every completed current-season race",
    )
    parser.add_argument("--trials", type=int, default=100,
                        help="Monte Carlo trials per selected race (1–10000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed for stable per-race random streams")
    parser.add_argument("--form-races", type=int, default=3,
                        help="Earlier eligible form rounds to include (0–24)")
    parser.add_argument("--scenario", choices=("dry", "light_rain", "heavy_rain"),
                        default="dry")
    parser.add_argument("--engine", choices=("standard", "chronological"),
                        default="standard", help="Race execution engine")
    parser.add_argument("--fetch-budget", type=float, default=120,
                        help="Total live fetch budget in seconds (1–300)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.trials <= 10_000:
        parser.error("trials must be between 1 and 10000")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be between 0 and 4294967295")
    if not 0 <= args.form_races <= 24:
        parser.error("form-races must be between 0 and 24")
    if not 1 <= args.fetch_budget <= 300:
        parser.error("fetch-budget must be between 1 and 300 seconds")

    year = datetime.now(timezone.utc).year
    try:
        result = evaluate_race_probabilities(
            CurrentSeasonDataLoader(fetch_budget=args.fetch_budget),
            year,
            target_race=None if args.all_targets else _race_value(args.race),
            all_targets=args.all_targets,
            trials=args.trials,
            seed=args.seed,
            form_races=args.form_races,
            scenario=args.scenario,
            race_engine=args.engine,
        )
    except (CurrentSeasonDataError, ValueError) as exc:
        parser.exit(1, f"Evaluation failed: {exc}\n")
    sys.stdout.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
