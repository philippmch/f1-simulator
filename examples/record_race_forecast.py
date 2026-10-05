"""Save a sealed forecast before qualifying; preserve it for later scoring."""

import argparse
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Event
from time import monotonic

from f1sim.analysis.recorded_forecast import record_race_forecast, save_recorded_forecast
from f1sim.analysis.scenarios import scenario_weather_from_label
from f1sim.cancellation import SimulationCancelled, raise_if_cancelled
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.models import Weather


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race", required=True, help="Future current-season round or name")
    parser.add_argument("--output", required=True, help="New JSON file; refuses to overwrite")
    parser.add_argument("--scenario", choices=("dry", "light_rain", "heavy_rain"), required=True)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--fetch-budget", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.fetch_budget <= 300:
        parser.error("fetch-budget must be from 1 to 300 seconds")
    if not 1 <= args.trials <= 10_000:
        parser.error("trials must be from 1 to 10000")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be from 0 to 4294967295")
    if args.workers is not None and (not args.parallel or not 1 <= args.workers <= 61):
        parser.error("workers requires --parallel and must be from 1 to 61")
    if Path(args.output).exists():
        parser.error("output already exists; choose a new forecast file")
    last_update = [float("-inf")]

    def progress(done, total):
        instant = monotonic()
        if done == total or instant - last_update[0] >= 2:
            print(f"Trials {done}/{total}", file=sys.stderr)
            last_update[0] = instant

    target = int(args.race) if args.race.isdecimal() else args.race
    cancelled = Event()
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_cancel(_signal, _frame):
        if cancelled.is_set():
            raise KeyboardInterrupt
        cancelled.set()

    signal.signal(signal.SIGINT, request_cancel)
    try:
        record = record_race_forecast(
            CurrentSeasonDataLoader(fetch_budget=args.fetch_budget),
            datetime.now(timezone.utc).year, target, trials=args.trials, seed=args.seed,
            weather=scenario_weather_from_label(Weather(), args.scenario).weather,
            parallel=args.parallel, max_workers=args.workers,
            progress_callback=progress,
            cancel_requested=cancelled.is_set,
        )
        raise_if_cancelled(cancelled.is_set)
        path = save_recorded_forecast(args.output, record)
    except (SimulationCancelled, KeyboardInterrupt):
        parser.exit(130, "Forecast cancelled; no forecast was published.\n")
    except (CurrentSeasonDataError, ValueError, OSError) as error:
        parser.exit(1, f"Forecast failed: {error}\n")
    finally:
        signal.signal(signal.SIGINT, previous_handler)
    print(json.dumps({"path": str(path.resolve()), "content_sha256": record["content_sha256"],
                      "recorded_at": record["recorded_at"],
                      "qualifying_starts_at": record["qualifying_starts_at"],
                      "timing_evidence": record["timing_evidence"]}, indent=2))


if __name__ == "__main__":
    main()
