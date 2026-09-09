#!/usr/bin/env python3
"""Compare opening choices using one saved input snapshot, entirely offline."""

import argparse
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.analysis.strategy_comparison import compare_saved_starting_tires
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
        description="Compare starting tyres using saved inputs and matching seed ranges.",
        epilog="Uses installed simulator code. Equal seeds do not freeze later race events; "
        "95% intervals describe Monte Carlo sampling, not real-race accuracy.",
    )
    parser.add_argument("path", type=Path, help="Saved statistics or dashboard JSON with inputs")
    parser.add_argument("--driver", required=True, help="Exact driver code from the saved roster")
    parser.add_argument("--scenario", help="Exact source scenario when the file contains several")
    parser.add_argument("--compounds", default="automatic,soft,medium,hard",
                        help="Choices separated by commas; automatic restores the usual policy")
    parser.add_argument("--simulations", type=_simulations, default=100,
                        help="Trials per choice (1-1000, default: 100)")
    parser.add_argument("--parallel", action="store_true", help="Use process workers")
    parser.add_argument("--max-workers", type=_workers)
    parser.add_argument("--independent-weather", action="store_true",
                        help="Use independent weather draws, including for older saved inputs")
    parser.add_argument("--export", action="store_true",
                        help="Write unique bundles and comparison JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("output/strategy-comparisons"))
    args = parser.parse_args()
    compounds = tuple(value.strip().lower() for value in args.compounds.split(","))
    try:
        results = compare_saved_starting_tires(
            args.path, args.driver, compounds, scenario=args.scenario,
            num_simulations=args.simulations, parallel=args.parallel, max_workers=args.max_workers,
            rng_policy="isolated_weather_v1" if args.independent_weather else None,
        )
        first = next(iter(results.values()))
        print(f"{first.track_name} | driver: {args.driver} | model: {first.race_engine}")
        print(f"{args.simulations} trials per choice | seeds {first.seed}–"
              f"{first.seed + args.simulations - 1}")
        print("Same saved inputs; only this driver's opening choice changes.")
        print("Weather draws: " + (
            "independent of race decisions (shared sequence by weather interval)."
            if first.input_snapshot["rng_policy"] == "isolated_weather_v1"
            else "shared with race events (legacy); strategy can change later weather."
        ))
        print("Equal seeds do not freeze later race events. "
              "Intervals measure sampling uncertainty.")
        print("Choice       Trials     Win % [95% range]       Podium %   DNF %   Points/race")
        for label, result in results.items():
            stats = result.driver_stats[args.driver]
            interval = result.get_probability_intervals()[args.driver]["win"]
            trials = len(stats.positions)
            points = stats.total_points / trials
            print(f"{label:<12} {trials:>6}  {stats.win_rate:>6.1f} "
                  f"[{interval['lower']:>5.1f}, {interval['upper']:>5.1f}]"
                  f"       {stats.podium_rate:>6.1f} {stats.dnf_rate:>7.1f} {points:>12.2f}")
        if args.export:
            exporter = Exporter(args.output_dir)
            for label, result in results.items():
                paths = exporter.export_all(result, prefix=f"starting_{label}")
                print(f"{label}: {paths['statistics_json']}")
            prefix = f"starting_tyres_{uuid4().hex}"
            comparison = exporter.export_scenario_comparison_json(
                results, filename=f"{prefix}.json",
            )
            report = exporter.export_scenario_comparison_html(
                results, filename=f"{prefix}.html", focus_driver=args.driver,
            )
            print(f"Comparison: {comparison}")
            print(f"Comparison report: {report}")
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
