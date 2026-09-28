#!/usr/bin/env python3
"""Select a saved pit plan on training seeds and validate it on later seeds."""

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from compare_pit_plans import _load_plans, _workers

from f1sim.analysis.provenance import format_saved_runtime_status, saved_runtime_status
from f1sim.analysis.strategy_selection import evaluate_saved_pit_plan_selection
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


def _print_points_outcome_profile(profile: dict | None, *, identity: bool = False) -> None:
    if identity:
        print(
            "Held-out paired points outcome profile: not independently estimated; "
            "the selected plan is the reference.",
        )
        return
    if not isinstance(profile, dict):
        print("Held-out paired points outcome profile: not recorded.")
        return
    gain = profile["mean_points_gain_when_ahead"]
    loss = profile["mean_points_loss_when_behind"]
    gain_text = "none (no more-points seeds)" if gain is None else f"{gain:.3f} points"
    loss_text = "none (no fewer-points seeds)" if loss is None else f"{loss:.3f} points"
    print(
        "Held-out paired points outcomes (more/equal/fewer): "
        f"{profile['more_points_races']}/{profile['equal_points_races']}/"
        f"{profile['fewer_points_races']} across {profile['paired_races']} seeds; "
        f"mean gain when ahead {gain_text}; mean loss when behind {loss_text}.",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Choose a named pit plan on complete training outcomes and assess the fixed choice "
            "on disjoint later seeds."
        ),
        epilog=(
            "Plans JSON uses the same format as compare_pit_plans.py. Driver alternatives are "
            "null or an instruction list; constructor alternatives are null or a complete "
            "mapping of saved member IDs to null or instruction lists. An empty list disables "
            "elective stops."
        ),
    )
    parser.add_argument("path", type=Path, help="Saved statistics or comparison JSON with inputs")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--driver", help="Exact target driver ID from saved inputs")
    target_group.add_argument("--constructor", help="Exact saved constructor/team ID")
    parser.add_argument("--plans", required=True, type=Path, help="JSON file of candidate plans")
    parser.add_argument("--reference", required=True, help="Fixed comparison plan label")
    parser.add_argument("--scenario", help="Exact source scenario when the file contains several")
    parser.add_argument("--training-simulations", type=_phase_simulations, default=100,
                        help="Trials per candidate for selection (1-1000, default: 100)")
    parser.add_argument("--validation-simulations", type=_phase_simulations, default=100,
                        help="Fresh trials for validation (1-1000, default: 100)")
    parser.add_argument("--parallel", action="store_true", help="Use process workers")
    parser.add_argument("--max-workers", type=_workers)
    parser.add_argument("--rng-policy", choices=RNG_POLICIES,
                        help="Select the random-stream policy for every phase")
    parser.add_argument("--export", action="store_true",
                        help="Write replayable phase comparisons and a selection manifest")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/pit-plan-selections"))
    args = parser.parse_args()

    try:
        plans = _load_plans(args.plans)
        runtime_status = saved_runtime_status(args.path, args.scenario)
        outcome = evaluate_saved_pit_plan_selection(
            args.path,
            plans,
            args.reference,
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
        first = next(iter(training_results.values()))
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
            f"({training_range['trials']} trials per candidate); validation seeds "
            f"{validation_range['first_seed']}–{validation_range['last_seed']} "
            f"({validation_range['trials']} trials).",
        )
        print("Training mean points:")
        for row in selection["training_score_table"]:
            print(f"  {row['label']}: {row['mean_points']:.3f}")
        if selection["tiebreak_applied"] == "reference_preferred_on_exact_tie":
            tie_text = "an exact training tie preferred the reference"
        elif selection["tiebreak_applied"] == "first_plan_order_on_exact_tie":
            tie_text = "an exact training tie used the first candidate in the plans file"
        else:
            tie_text = "it had the highest training mean"
        print(f"Selected: {selection['selected_label']} because {tie_text}.")
        metrics = selection["validation_target_metrics"]
        if selection["validation_status"] == "no_change":
            print(
                "Held-out validation: no change; the selected plan is the reference. "
                "Identity difference is 0 points; no separate standard error is estimated.",
            )
            _print_points_outcome_profile(metrics.get("points_outcome_profile"), identity=True)
        else:
            standard_error = metrics["points_difference_standard_error"]
            uncertainty = (
                "standard error not estimated with one validation trial"
                if standard_error is None
                else f"sample standard error {standard_error:.3f}"
            )
            print(
                f"Held-out selected-minus-reference mean: "
                f"{metrics['mean_points_difference']:.3f} points; {uncertainty}. "
                f"The selected plan stays {selection['selected_label']} regardless of this result."
            )
            _print_points_outcome_profile(metrics.get("points_outcome_profile"))
        print(
            "These results are conditional on the saved simulator inputs, not real-race "
            "calibration. Validation seeds are disjoint within this run; repeating the same "
            "request reuses them.",
        )

        if args.export:
            exporter = Exporter(args.output_dir)
            run_id = uuid4().hex
            training_json = exporter.export_scenario_comparison_json(
                training_results,
                filename=f"pit_plan_selection_training_{run_id}.json",
                reference_scenario=args.reference,
            )
            training_html = exporter.export_scenario_comparison_html(
                training_results,
                filename=f"pit_plan_selection_training_{run_id}.html",
                focus_driver=args.driver,
                reference_scenario=args.reference,
            )
            validation_json = exporter.export_scenario_comparison_json(
                validation_results,
                filename=f"pit_plan_selection_validation_{run_id}.json",
                reference_scenario=args.reference,
            )
            validation_html = exporter.export_scenario_comparison_html(
                validation_results,
                filename=f"pit_plan_selection_validation_{run_id}.html",
                focus_driver=args.driver,
                reference_scenario=args.reference,
            )
            manifest_path = args.output_dir / f"selection_manifest_{run_id}.json"
            manifest = {
                "selection": selection,
                "training_comparison_json": training_json.name,
                "training_comparison_html": training_html.name,
                "validation_comparison_json": validation_json.name,
                "validation_comparison_html": validation_html.name,
            }
            manifest_path.write_text(
                json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8",
            )
            print(f"Training comparison: {training_json}")
            print(f"Validation comparison: {validation_json}")
            print(f"Selection manifest: {manifest_path}")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
