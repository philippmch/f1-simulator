"""Score simulated race-winner probabilities against completed current-season races."""

import argparse
import json
import signal
import sys
import time
from datetime import datetime, timezone
from threading import Event

from f1sim.analysis.race_probability_evaluation import evaluate_race_probabilities
from f1sim.cancellation import SimulationCancelled
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.simulation.execution import DEFAULT_RACE_ENGINE, RACE_ENGINES


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
    parser.add_argument("--engine", choices=RACE_ENGINES,
                        default=DEFAULT_RACE_ENGINE,
                        help=f"Race execution engine (default: {DEFAULT_RACE_ENGINE})")
    parser.add_argument("--fetch-budget", type=float, default=120,
                        help="Total live fetch budget in seconds (1–300)")
    parser.add_argument("--parallel", action="store_true",
                        help="Distribute each event's trials across worker processes")
    parser.add_argument("--workers", type=int,
                        help="Maximum worker processes with --parallel (1–61)")
    parser.add_argument("--progress", action="store_true",
                        help="Write collection and trial progress to stderr; JSON stays on stdout")
    return parser


def _progress_writer():
    last_key = None
    last_time = 0.0

    def report(update):
        nonlocal last_key, last_time
        phase = update["phase"]
        key = (phase, update.get("event_index"))
        now = time.monotonic()
        final_trial = (phase == "simulating" and update["event_trials_completed"]
                       == update["event_trials_total"])
        if key == last_key and now - last_time < 2 and not final_trial:
            return
        last_key, last_time = key, now
        if phase == "loading":
            message = "Loading current-season calendar and results."
        elif phase == "collecting":
            message = (f"Collecting event {update['event_index']}/{update['events_total']} "
                       f"(round {update['round']}).")
        elif phase == "collected":
            message = (f"Inputs collected: {update['events_total']} events, "
                       f"{update['trials_total']} simulation trials planned.")
        elif phase == "simulating":
            message = (f"Round {update['round']}: {update['event_trials_completed']}/"
                       f"{update['event_trials_total']} trials; total "
                       f"{update['trials_completed']}/{update['trials_total']}.")
        elif phase == "event_complete":
            message = (f"Event {update['events_completed']}/{update['events_total']} "
                       f"{update['status']}; {update['trials_completed']} trials completed.")
        else:
            message = (f"Evaluation complete: {update['events_completed']} events, "
                       f"{update['trials_completed']} trials.")
        print(message, file=sys.stderr, flush=True)

    return report


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
    if args.workers is not None:
        if not 1 <= args.workers <= 61:
            parser.error("workers must be between 1 and 61")
        if not args.parallel:
            parser.error("workers requires --parallel")

    year = datetime.now(timezone.utc).year
    cancelled = Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_cancel(_signal, _frame):
        if cancelled.is_set():
            raise KeyboardInterrupt
        cancelled.set()

    signal.signal(signal.SIGINT, request_cancel)
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
            parallel=args.parallel,
            max_workers=args.workers,
            progress_callback=_progress_writer() if args.progress else None,
            cancel_requested=cancelled.is_set,
        )
        report_json = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if cancelled.is_set():
            raise SimulationCancelled("cancelled before report output")
        sys.stdout.write(report_json)
    except (SimulationCancelled, KeyboardInterrupt):
        parser.exit(130, "Evaluation cancelled; no complete report was produced.\n")
    except Exception as exc:
        if cancelled.is_set():
            parser.exit(130, "Evaluation cancelled; no complete report was produced.\n")
        if isinstance(exc, (CurrentSeasonDataError, ValueError)):
            parser.exit(1, f"Evaluation failed: {exc}\n")
        raise
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main()
