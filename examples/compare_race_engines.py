#!/usr/bin/env python3
"""Compare race execution models against one saved input snapshot offline."""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.analysis.strategy_comparison import compare_saved_race_engines
from f1sim.output import Exporter


def _simulations(value: str) -> int:
    count = int(value)
    if not 1 <= count <= 1000:
        raise argparse.ArgumentTypeError("simulations must be between 1 and 1000")
    return count


def _workers(value: str) -> int:
    count = int(value)
    if not 1 <= count <= 16:
        raise argparse.ArgumentTypeError("max-workers must be between 1 and 16")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Standard and Lap-aware engines using saved inputs offline.",
        epilog="Lap-aware (chronological) remains experimental. Equal seeds do not freeze "
        "later race events. Differences describe model sensitivity, not real-race accuracy.",
    )
    parser.add_argument("path", type=Path, help="Saved statistics or dashboard JSON with inputs")
    parser.add_argument("--scenario", help="Exact source scenario when the file contains several")
    parser.add_argument("--engines", default="standard,chronological",
                        help="Distinct engines separated by commas, in comparison order")
    parser.add_argument("--simulations", type=_simulations, default=100,
                        help="Trials per engine (1-1000, default: 100)")
    parser.add_argument("--parallel", action="store_true", help="Use process workers")
    parser.add_argument("--max-workers", type=_workers)
    parser.add_argument("--export", action="store_true",
                        help="Write unique replayable bundles, comparison JSON and HTML")
    parser.add_argument("--output-dir", type=Path, default=Path("output/engine-comparisons"))
    args = parser.parse_args()
    engines = tuple(value.strip().lower() for value in args.engines.split(","))
    try:
        results = compare_saved_race_engines(
            args.path, engines, scenario=args.scenario, num_simulations=args.simulations,
            parallel=args.parallel, max_workers=args.max_workers,
        )
        first = next(iter(results.values()))
        print(f"{first.track_name} | {args.simulations} trials per engine | "
              f"seeds {first.seed}–{first.seed + args.simulations - 1}")
        print("Same saved roster, cars, track, weather and starting tyres; "
              "only the engine changes.")
        print("Lap-aware (chronological) is experimental. "
              "Equal seeds do not freeze later race events.")
        print("Intervals measure sampling uncertainty; differences show model sensitivity.")
        for engine, result in results.items():
            distance = result.get_race_distance_statistics()
            mean = distance["mean_winner_laps"]
            mean_text = "Not recorded" if mean is None else f"{mean:.2f} laps"
            print(f"\n{engine}: mean winning distance {mean_text} "
                  f"({distance['races_with_known_winner_distance']} known winners)")
            print(f"Time-limited: {distance['time_limited_races']} / "
                  f"{distance['recorded_races']} races; lapped finishers: "
                  f"{distance['lapped_finishers']} / "
                  f"{distance['finishers_with_comparable_distance']} comparable finishers")
            print("Driver        Trials     Win % [95% range]       Podium %   DNF %   Points/race")
            intervals = result.get_probability_intervals()
            for driver, stats in result.driver_stats.items():
                trials = len(stats.positions)
                if not trials:
                    print(f"{driver:<12}       0  Not recorded")
                    continue
                interval = intervals[driver]["win"]
                print(f"{driver:<12} {trials:>6}  {stats.win_rate:>6.1f} "
                      f"[{interval['lower']:>5.1f}, {interval['upper']:>5.1f}]"
                      f"       {stats.podium_rate:>6.1f} {stats.dnf_rate:>7.1f} "
                      f"{stats.total_points / trials:>12.2f}")
        if args.export:
            exporter = Exporter(args.output_dir)
            for engine, result in results.items():
                paths = exporter.export_all(result, prefix=f"engine_{engine}")
                print(f"{engine}: {paths['statistics_json']}")
            prefix = f"race_engines_{uuid4().hex}"
            comparison = exporter.export_scenario_comparison_json(
                results, filename=f"{prefix}.json",
            )
            report = exporter.export_scenario_comparison_html(results, filename=f"{prefix}.html")
            print(f"Comparison: {comparison}")
            print(f"Comparison report: {report}")
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
