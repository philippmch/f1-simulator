#!/usr/bin/env python3
"""Compare named custom pit plans using one saved input snapshot offline."""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.analysis.strategy_comparison import compare_saved_pit_plans
from f1sim.output import ConsoleOutput, Exporter
from f1sim.simulation.randomness import RNG_POLICIES


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


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_plans(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    if not isinstance(value, dict):
        raise ValueError("plans JSON must be an object mapping labels to null or instruction lists")
    return value


def _plan_text(value) -> str:
    if value is None:
        return "automatic policy"
    if value == []:
        return "no elective stops (compulsory repairs/corrections remain)"
    return ", ".join(
        f"{record.get('lap', 'not recorded')} own lap: {record.get('compound', 'not recorded')}"
        for record in value
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare named custom pit plans using saved inputs and matching seed ranges.",
        epilog="Requested numbers are each driver's own lap. Equal seeds do not freeze later "
               "race events; intervals describe Monte Carlo sampling uncertainty.",
    )
    parser.add_argument("path", type=Path, help="Saved statistics or dashboard JSON with inputs")
    parser.add_argument(
        "--driver", required=True, help="Exact target driver ID from the saved roster",
    )
    parser.add_argument("--plans", required=True, type=Path,
                        help="JSON object mapping variant labels to null or instruction lists")
    parser.add_argument("--scenario", help="Exact source scenario when the file contains several")
    parser.add_argument("--reference", help="Variant label used for paired output (default: first)")
    parser.add_argument("--simulations", type=_simulations, default=100,
                        help="Trials per plan (1-1000, default: 100)")
    parser.add_argument("--parallel", action="store_true", help="Use process workers")
    parser.add_argument("--max-workers", type=_workers)
    policy_group = parser.add_mutually_exclusive_group()
    policy_group.add_argument("--rng-policy", choices=RNG_POLICIES,
                              help="Select the random-stream policy for all variants")
    policy_group.add_argument("--independent-weather", action="store_true",
                              help="Use independent weather draws for every variant")
    parser.add_argument(
        "--export", action="store_true",
        help="Write replayable bundles and comparison files",
    )
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/pit-plan-comparisons"))
    args = parser.parse_args()

    try:
        plans = _load_plans(args.plans)
        if args.reference is not None and args.reference not in plans:
            parser.error("reference must name one of the supplied plan labels")
        reference = args.reference if args.reference is not None else next(iter(plans), None)
        rng_policy = args.rng_policy or (
            "isolated_weather_v1" if args.independent_weather else None
        )
        results = compare_saved_pit_plans(
            args.path, args.driver, plans, scenario=args.scenario,
            num_simulations=args.simulations, parallel=args.parallel,
            max_workers=args.max_workers, rng_policy=rng_policy,
        )
        first = next(iter(results.values()))
        print(f"{first.track_name} | driver: {args.driver} | model: {first.race_engine}")
        print(f"{args.simulations} trials per plan | seeds {first.seed}–"
              f"{first.seed + args.simulations - 1}")
        print("Requested lap numbers are the target driver's own lap at pit entry.")
        for label, value in plans.items():
            print(f"  {label}: {_plan_text(value)}")
        print("Same saved models, openings, finite pools and seed range are used for every plan.")
        print("None restores automatic strategy; [] disables elective stops while compulsory "
              "repairs and weather corrections remain active.")
        ConsoleOutput.print_paired_comparison(results, reference, driver_id=args.driver)
        if args.export:
            exporter = Exporter(args.output_dir)
            for index, (label, result) in enumerate(results.items()):
                paths = exporter.export_all(result, prefix=f"pit_plan_{index:02d}")
                print(f"{label}: {paths['statistics_json']}")
            prefix = f"pit_plan_comparison_{uuid4().hex}"
            comparison = exporter.export_scenario_comparison_json(
                results, filename=f"{prefix}.json", reference_scenario=reference,
            )
            report = exporter.export_scenario_comparison_html(
                results, filename=f"{prefix}.html", focus_driver=args.driver,
                reference_scenario=reference,
            )
            print(f"Comparison: {comparison}")
            print(f"Comparison report: {report}")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
