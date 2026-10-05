"""Score a sealed pre-event forecast using current-season results; run no trials."""

import argparse
import json

from f1sim.analysis.recorded_forecast import load_recorded_forecast, score_recorded_forecast
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("forecast", help="Previously recorded JSON forecast")
    parser.add_argument("--fetch-budget", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.fetch_budget <= 300:
        parser.error("fetch-budget must be from 1 to 300 seconds")
    try:
        record = load_recorded_forecast(args.forecast)
        loader = CurrentSeasonDataLoader(fetch_budget=args.fetch_budget)
        results, qualifying = loader._season_data(record["year"])
        report = score_recorded_forecast(record, loader, results, qualifying)
    except (CurrentSeasonDataError, ValueError, OSError) as error:
        parser.exit(1, f"Scoring failed: {error}\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
