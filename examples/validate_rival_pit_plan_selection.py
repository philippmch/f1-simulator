#!/usr/bin/env python3
"""Select a target pit plan across fixed rival scenarios, then validate it."""

import argparse
import json
import math
import sys
from numbers import Real
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare_pit_plans import _load_plans, _workers

from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.provenance import format_saved_runtime_status, saved_runtime_status
from f1sim.analysis.rival_strategy_selection import evaluate_saved_rival_pit_plan_selection
from f1sim.output import Exporter
from f1sim.simulation.randomness import RNG_POLICIES


def _phase_simulations(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "simulations must be an integer between 1 and 1000",
        ) from exc
    if not 1 <= count <= 1000:
        raise argparse.ArgumentTypeError("simulations must be between 1 and 1000")
    return count


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_number(value: str):
    raise ValueError(f"invalid JSON number: {value}")


def _load_rival_scenarios(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonstandard_number,
        )
    if not isinstance(value, dict) or not 1 <= len(value) <= 10:
        raise ValueError("rival scenarios JSON must name between 1 and 10 scenarios")
    for label, definition in value.items():
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise ValueError(
                "rival scenario names must be nonempty strings of at most 80 characters",
            )
        if not isinstance(definition, dict) or set(definition) != {"weight", "pit_plans"}:
            raise ValueError(
                f"rival scenario {label!r} must contain exactly weight and pit_plans",
            )
        weight = definition["weight"]
        try:
            finite_weight = math.isfinite(float(weight))
        except (OverflowError, TypeError, ValueError):
            finite_weight = isinstance(weight, int) and not isinstance(weight, bool) and weight > 0
        if (not isinstance(weight, Real) or isinstance(weight, bool)
                or not finite_weight or weight <= 0):
            raise ValueError(f"rival scenario {label!r} weight must be positive and finite")
        overrides = definition["pit_plans"]
        if not isinstance(overrides, dict):
            raise ValueError(f"rival scenario {label!r} pit_plans must be an object")
        if any(not isinstance(driver_id, str) or not driver_id for driver_id in overrides):
            raise ValueError(f"rival scenario {label!r} has an invalid driver ID")
        for driver_id, plan in overrides.items():
            if plan is not None and not isinstance(plan, list):
                raise ValueError(
                    f"rival scenario {label!r} plan for {driver_id!r} must be a list or null",
                )
    return value


def _print_validation_scenarios(
    validation_results: dict,
    selection: dict,
    normalized_weights: dict[str, float],
) -> None:
    target_key = (
        "driver_statistics" if selection["target_mode"] == "driver"
        else "constructor_statistics"
    )
    reference = selection["reference_label"]
    selected = selection["selected_label"]
    target_id = selection["target_id"]
    print("Per-scenario held-out comparisons:")
    for label, variants in validation_results.items():
        normalized = normalized_weights[label]
        print(f"  {label} (normalized weight {normalized:.3f})")
        if selected == reference:
            print("    identity vs reference: 0 points; no separate standard error estimated")
            continue
        paired = paired_comparison_statistics(variants, reference)
        summary = paired["variants"][selected]
        statistic = summary.get(target_key, {}).get(target_id)
        if summary.get("status") != "paired" or statistic is None:
            reason = summary.get("reason") or "target comparison unavailable"
            raise ValueError(
                f"held-out comparison for rival scenario {label!r} is unavailable: {reason}",
            )
        error = statistic.get("points_difference_standard_error")
        error_text = (
            "SE not estimated with one paired trial" if error is None else f"SE {error:.3f}"
        )
        print(
            f"    selected-minus-reference mean {statistic['mean_points_difference']:.3f} points; "
            f"{error_text}; {statistic['paired_races']} paired races",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select one pit plan across predeclared, weighted rival-plan scenarios, "
            "then validate the frozen choice on disjoint seeds."
        ),
        epilog=(
            "Rival-scenarios JSON maps names to {weight, pit_plans}; each pit_plans object "
            "maps rival driver IDs to null (automatic policy), [] (no elective stops), or "
            "normal pit-plan instruction lists. Scenario weights are user-supplied analysis "
            "assumptions, not probabilities learned from race data."
        ),
    )
    parser.add_argument("path", type=Path, help="Saved statistics or comparison JSON with inputs")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--driver", help="Exact target driver ID from saved inputs")
    target_group.add_argument("--constructor", help="Exact saved constructor/team ID")
    parser.add_argument(
        "--plans", required=True, type=Path, help="JSON file of target candidate plans",
    )
    parser.add_argument("--reference", required=True, help="Fixed comparison plan label")
    parser.add_argument(
        "--rival-scenarios", required=True, type=Path,
        help="JSON file mapping scenario names to positive weights and rival plan overrides",
    )
    parser.add_argument("--scenario", help="Exact source scenario when the file contains several")
    parser.add_argument("--training-simulations", type=_phase_simulations, default=100,
                        help="Trials per candidate in each rival scenario (1-1000, default: 100)")
    parser.add_argument("--validation-simulations", type=_phase_simulations, default=100,
                        help="Fresh trials per rival scenario (1-1000, default: 100)")
    parser.add_argument("--parallel", action="store_true", help="Use process workers")
    parser.add_argument("--max-workers", type=_workers)
    parser.add_argument("--rng-policy", choices=RNG_POLICIES,
                        help="Select the random-stream policy for every phase and scenario")
    parser.add_argument("--export", action="store_true",
                        help="Write scenario-specific replayable comparisons and a manifest")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/rival-pit-plan-selections"))
    args = parser.parse_args()

    try:
        plans = _load_plans(args.plans)
        rival_scenarios = _load_rival_scenarios(args.rival_scenarios)
        runtime_status = saved_runtime_status(args.path, args.scenario)
        outcome = evaluate_saved_rival_pit_plan_selection(
            args.path,
            plans,
            args.reference,
            rival_scenarios,
            driver_id=args.driver,
            constructor_id=args.constructor,
            scenario=args.scenario,
            training_simulations=args.training_simulations,
            validation_simulations=args.validation_simulations,
            parallel=args.parallel,
            max_workers=args.max_workers,
            rng_policy=args.rng_policy,
        )
        selection = outcome["selection"]
        training_results = outcome["training_results"]
        validation_results = outcome["validation_results"]
        first_scenario = next(iter(training_results.values()))
        first = next(iter(first_scenario.values()))
        print(format_saved_runtime_status(runtime_status))
        target_text = f"{selection['target_mode']}: {selection['target_id']}"
        if selection["target_mode"] == "constructor":
            target_text += f" | members: {', '.join(selection['target_member_ids'])}"
        print(f"{first.track_name} | {target_text} | model: {first.race_engine}")
        print(f"Fixed reference: {args.reference}")
        ranges = selection["seed_ranges"]
        training_range = ranges["training"]
        validation_range = ranges["validation"]
        print(
            f"Training seeds {training_range['first_seed']}–{training_range['last_seed']} "
            f"({training_range['trials']} per candidate and rival scenario); validation seeds "
            f"{validation_range['first_seed']}–{validation_range['last_seed']} "
            f"({validation_range['trials']} per rival scenario).",
        )
        scenario_metadata = {
            row["name"]: row for row in selection["rival_scenarios"]
        }
        normalized_weights = {
            label: scenario_metadata[label]["normalized_weight"]
            for label in rival_scenarios
        }
        print("Predeclared rival scenarios (weights are supplied assumptions):")
        for label, definition in rival_scenarios.items():
            weight_text = str(definition["weight"])
            print(
                f"  {label}: supplied {weight_text}, "
                f"normalized {normalized_weights[label]:.3f}",
            )
        print("Weighted training mean points:")
        for row in selection["training_score_table"]:
            print(f"  {row['label']}: {row['mean_points']:.3f}")
        if selection["tiebreak_applied"] == "reference_preferred_on_exact_tie":
            tie_text = "an exact training tie preferred the reference"
        elif selection["tiebreak_applied"] == "first_plan_order_on_exact_tie":
            tie_text = "an exact training tie used the first candidate in the plans file"
        else:
            tie_text = "it had the highest weighted training mean"
        print(f"Selected and frozen: {selection['selected_label']} because {tie_text}.")
        metrics = selection["validation_target_metrics"]
        if selection["validation_status"] == "no_change":
            print(
                "Weighted held-out validation: no change; the selected plan is the reference. "
                "Identity difference is 0 points; no separate standard error is estimated.",
            )
        else:
            standard_error = metrics["points_difference_standard_error"]
            uncertainty = (
                "standard error not estimated with one validation trial"
                if standard_error is None
                else f"sample standard error {standard_error:.3f}"
            )
            print(
                f"Weighted held-out selected-minus-reference mean: "
                f"{metrics['mean_points_difference']:.3f} points; {uncertainty}. "
                "The selected plan stays frozen regardless of this result.",
            )
        _print_validation_scenarios(validation_results, selection, normalized_weights)
        print(
            "Scenario points are combined within each seed before the standard error is computed, "
            "so cross-scenario covariance is retained. Validation differences describe this "
            "simulator experiment, not causal proof of a real-world advantage. Repeating the "
            "same request reuses validation seeds.",
        )

        if args.export:
            exporter = Exporter(args.output_dir)
            run_id = uuid4().hex
            exported_scenarios = {}
            for index, label in enumerate(rival_scenarios):
                prefix = f"rival_selection_{run_id}_scenario_{index:02d}"
                train_json = exporter.export_scenario_comparison_json(
                    training_results[label], filename=f"{prefix}_training.json",
                    reference_scenario=args.reference,
                )
                train_html = exporter.export_scenario_comparison_html(
                    training_results[label], filename=f"{prefix}_training.html",
                    focus_driver=args.driver, reference_scenario=args.reference,
                )
                valid_json = exporter.export_scenario_comparison_json(
                    validation_results[label], filename=f"{prefix}_validation.json",
                    reference_scenario=args.reference,
                )
                valid_html = exporter.export_scenario_comparison_html(
                    validation_results[label], filename=f"{prefix}_validation.html",
                    focus_driver=args.driver, reference_scenario=args.reference,
                )
                exported_scenarios[label] = {
                    "supplied_weight": scenario_metadata[label]["weight"],
                    "normalized_weight": normalized_weights[label],
                    "training_comparison_json": train_json.name,
                    "training_comparison_html": train_html.name,
                    "validation_comparison_json": valid_json.name,
                    "validation_comparison_html": valid_html.name,
                }
                print(
                    f"{label}: training comparison {train_json}; "
                    f"validation comparison {valid_json}",
                )
            manifest_path = args.output_dir / f"rival_selection_manifest_{run_id}.json"
            report_name = f"rival_selection_{run_id}_summary.html"
            manifest = {
                "selection": selection,
                "rival_scenarios": exported_scenarios,
                "target_plans": {
                    label: plans[label]
                    for label in dict.fromkeys((selection["reference_label"],
                                                selection["selected_label"]))
                },
                "report_context": {
                    "track_name": first.track_name,
                    "race_engine": first.race_engine,
                },
                "selection_report_html": report_name,
            }
            report_path = exporter.export_rival_strategy_selection_html(
                manifest, filename=report_name, manifest_filename=manifest_path.name,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
            print(f"Consolidated selection report: {report_path}")
            print(f"Selection manifest: {manifest_path}")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
