"""Select saved pit plans on training seeds and assess them on fresh seeds."""

from collections.abc import Mapping
from math import isfinite
from numbers import Integral
from statistics import mean
from typing import Any

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.analysis.paired_comparison import (
    _observation,
    _ordered_trials,
    _qualifying_valid,
    _snapshot,
    paired_comparison_statistics,
)
from f1sim.analysis.strategy_comparison import (
    _prepare_saved_pit_plan_variants,
    _runner_variant,
)
from f1sim.simulation.randomness import validate_rng_policy

_MAX_SEED = 2**32 - 1


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _validate_labels(plans: Mapping[str, Any], reference_label: str) -> list[str]:
    if not isinstance(plans, Mapping) or not 2 <= len(plans) <= 10:
        raise ValueError("plans must be a mapping containing between 2 and 10 variants")
    labels = list(plans)
    for label in labels:
        if not isinstance(label, str) or not label.strip() or len(label) > 80:
            raise ValueError("plan labels must be nonempty strings of at most 80 characters")
    if not isinstance(reference_label, str) or reference_label not in plans:
        raise ValueError("reference_label must name one of the supplied plan labels")
    return labels


def _validate_seed_ranges(source_seed: int, saved_count: int, train_count: int, valid_count: int):
    source_end = source_seed + saved_count - 1
    train_start = source_end + 1
    train_end = train_start + train_count - 1
    valid_start = train_end + 1
    valid_end = valid_start + valid_count - 1
    if valid_end > _MAX_SEED:
        raise ValueError(f"source, training, and validation seeds must not exceed {_MAX_SEED}")
    return {
        "source": {"first_seed": source_seed, "last_seed": source_end, "trials": saved_count},
        "training": {"first_seed": train_start, "last_seed": train_end, "trials": train_count},
        "validation": {"first_seed": valid_start, "last_seed": valid_end, "trials": valid_count},
    }


def _run_variants(
    variants: Mapping[str, MonteCarloRunner],
    count: int,
    *,
    parallel: bool,
    max_workers: int | None,
) -> dict[str, SimulationResults]:
    return {
        label: runner.run(
            count,
            parallel=parallel,
            max_workers=None if max_workers is None else int(max_workers),
        )
        for label, runner in variants.items()
    }


def _phase_points(
    results: Mapping[str, SimulationResults],
    labels: list[str],
    reference_label: str,
    *,
    expected_runner: MonteCarloRunner,
    expected_seed: int,
    expected_count: int,
    target_members: list[str],
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    """Require one complete, matching qualifying and target-point cohort."""
    if reference_label not in results or list(results) != labels:
        raise ValueError("simulation results do not contain the requested plan variants in order")

    normalized_inputs = {}
    for label, result in results.items():
        if (
            not _ordered_trials(result)
            or result.seed != expected_seed
            or result.num_simulations != expected_count
            or len(result.race_results) != expected_count
            or len(result.qualifying_results) != expected_count
        ):
            raise ValueError(f"{label!r} has incomplete or out-of-range trial records")
        snapshot = _snapshot(result)
        if snapshot is None:
            raise ValueError(f"{label!r} has invalid or incomplete saved input evidence")
        if result.race_engine != expected_runner.race_engine:
            raise ValueError(f"{label!r} used a different race engine")
        raw_inputs = result.input_snapshot
        if (
            raw_inputs.get("starting_tires", {}) != expected_runner.starting_tires
            or raw_inputs.get("starting_tire_ages", {}) != expected_runner.starting_tire_ages
        ):
            raise ValueError(f"{label!r} used different opening tyre inputs")
        normalized_inputs[label] = snapshot
    baseline = normalized_inputs[reference_label]
    if any(snapshot != baseline for snapshot in normalized_inputs.values()):
        raise ValueError("plan variants have incompatible saved inputs or runtime provenance")

    roster_ids = baseline[1]
    runnable_ids = baseline[2]
    if any(member not in runnable_ids for member in target_members):
        raise ValueError("the selected target has no runnable saved member")
    if not target_members or len(set(target_members)) != len(target_members):
        raise ValueError("the selected target has invalid saved membership")
    scheduled_laps = baseline[0]["track"]["total_laps"]
    points_by_label: dict[str, list[int | float]] = {label: [] for label in labels}
    for index in range(expected_count):
        qualifying = {}
        for label, result in results.items():
            rows = result.qualifying_results[index]
            if not _qualifying_valid(rows, roster_ids, runnable_ids):
                raise ValueError(
                    f"{label!r} has missing, duplicate, or malformed qualifying evidence "
                    f"at trial {index + 1}",
                )
            qualifying[label] = rows
        reference_rows = qualifying[reference_label]
        if any(rows != reference_rows for rows in qualifying.values()):
            raise ValueError(f"qualifying results do not match across plans at trial {index + 1}")

        for label, result in results.items():
            target_points = []
            for member in target_members:
                observation = _observation(result.race_results[index], member, scheduled_laps)
                if observation is None:
                    raise ValueError(
                        f"{label!r} has a missing, duplicate, or malformed points outcome "
                        f"for {member!r} at trial {index + 1}",
                    )
                target_points.append(observation[0])
            # Sum team points within the seed before computing means or paired variance.
            points_by_label[label].append(sum(target_points))
    coverage = {
        "required_trials": expected_count,
        "complete_trials": expected_count,
        "qualifying_match_trials": expected_count,
        "target_observations_per_variant": expected_count * len(target_members),
    }
    return points_by_label, coverage


def _json_number(value: int | float) -> int | float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(value):
        raise ValueError("simulation points must be finite JSON numbers")
    return value


def evaluate_saved_pit_plan_selection(
    path,
    plans: Mapping[str, Any],
    reference_label: str,
    *,
    driver_id: str | None = None,
    constructor_id: str | None = None,
    scenario: str | None = None,
    training_simulations: int = 100,
    validation_simulations: int = 100,
    parallel: bool = False,
    max_workers: int | None = None,
    rng_policy: str | None = None,
) -> dict[str, Any]:
    """Choose a pit plan on complete training outcomes and validate it on later seeds.

    The source export's trials are skipped. All candidate plans use one shared
    training seed range; only the fixed reference and selected plan use the next,
    disjoint validation range. Every requested target outcome must be valid.
    """
    training_count = _positive_int(training_simulations, "training_simulations")
    validation_count = _positive_int(validation_simulations, "validation_simulations")
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    if rng_policy is not None:
        rng_policy = validate_rng_policy(rng_policy)
    labels = _validate_labels(plans, reference_label)
    if (driver_id is None) == (constructor_id is None):
        raise ValueError("exactly one driver_id or constructor_id is required")

    runner, saved_count, source_variants = _prepare_saved_pit_plan_variants(
        path, plans, driver_id=driver_id, constructor_id=constructor_id,
        scenario=scenario, rng_policy=rng_policy,
    )
    ranges = _validate_seed_ranges(
        int(runner.base_seed), saved_count, training_count, validation_count,
    )
    train_start = ranges["training"]["first_seed"]
    valid_start = ranges["validation"]["first_seed"]
    # Build independent phase runners for every candidate before spending trial work.
    training_runners = {
        label: _runner_variant(source_variants[label], seed=train_start)
        for label in labels
    }
    validation_candidates = {
        label: _runner_variant(source_variants[label], seed=valid_start)
        for label in labels
    }
    target_members = (
        [driver_id] if driver_id is not None else [
            driver.id for driver in runner.drivers if driver.team_id == constructor_id
        ]
    )

    training_results = _run_variants(
        training_runners, training_count, parallel=parallel, max_workers=max_workers,
    )
    training_points, training_coverage = _phase_points(
        training_results, labels, reference_label,
        expected_runner=training_runners[reference_label],
        expected_seed=train_start, expected_count=training_count,
        target_members=target_members,
    )
    training_scores = {
        label: _json_number(mean(values)) for label, values in training_points.items()
    }
    best_score = max(training_scores.values())
    tied_labels = [label for label in labels if training_scores[label] == best_score]
    if len(tied_labels) == 1:
        selected_label = tied_labels[0]
        tiebreak = "unique_highest_training_mean"
    elif reference_label in tied_labels:
        selected_label = reference_label
        tiebreak = "reference_preferred_on_exact_tie"
    else:
        selected_label = tied_labels[0]
        tiebreak = "first_plan_order_on_exact_tie"

    validation_labels = (
        [reference_label] if selected_label == reference_label
        else [reference_label, selected_label]
    )
    validation_runners = {
        label: validation_candidates[label] for label in validation_labels
    }
    validation_results = _run_variants(
        validation_runners, validation_count,
        parallel=parallel, max_workers=max_workers,
    )
    validation_points, validation_coverage = _phase_points(
        validation_results, validation_labels, reference_label,
        expected_runner=validation_runners[reference_label],
        expected_seed=valid_start, expected_count=validation_count,
        target_members=target_members,
    )

    no_change = selected_label == reference_label
    if no_change:
        target_metrics = {
            "reference_mean_points": _json_number(mean(validation_points[reference_label])),
            "selected_mean_points": _json_number(mean(validation_points[reference_label])),
            "mean_points_difference": 0,
            "points_difference_standard_error": None,
            "paired_races": validation_count,
            "comparison": "identity; no separate alternative was estimated",
        }
        validation_status = "no_change"
    else:
        paired = paired_comparison_statistics(validation_results, reference_label)
        summary = paired["variants"][selected_label]
        if summary.get("status") != "paired":
            raise ValueError(
                "held-out paired comparison is unavailable: "
                f"{summary.get('reason') or 'incompatible evidence'}",
            )
        statistic_key = "driver_statistics" if driver_id is not None else "constructor_statistics"
        statistic_id = driver_id if driver_id is not None else constructor_id
        statistic = summary[statistic_key].get(statistic_id)
        if statistic is None or statistic.get("paired_races") != validation_count:
            raise ValueError("held-out comparison does not cover every validation trial")
        target_metrics = {
            "reference_mean_points": _json_number(statistic["reference_mean_points"]),
            "selected_mean_points": _json_number(statistic["variant_mean_points"]),
            "mean_points_difference": _json_number(statistic["mean_points_difference"]),
            "points_difference_standard_error": (
                None if statistic["points_difference_standard_error"] is None
                else _json_number(statistic["points_difference_standard_error"])
            ),
            "paired_races": statistic["paired_races"],
            "comparison": "selected plan minus fixed reference, paired by seed",
        }
        validation_status = "evaluated"

    score_table = [
        {
            "label": label,
            "total_points": _json_number(sum(training_points[label])),
            "mean_points": training_scores[label],
            "trials": training_count,
        }
        for label in labels
    ]
    selection = {
        "schema_version": 1,
        "method": "complete_cohort_training_then_disjoint_seed_validation",
        "target_mode": "driver" if driver_id is not None else "constructor",
        "target_id": driver_id if driver_id is not None else constructor_id,
        "target_member_ids": target_members,
        "candidate_order": labels,
        "reference_label": reference_label,
        "selected_label": selected_label,
        "selection_status": "no_change" if no_change else "selected",
        "selection_rule": "highest training mean points; exact ties prefer the reference, "
        "then candidate mapping order",
        "tiebreak_applied": tiebreak,
        "training_score_table": score_table,
        "training_coverage": training_coverage,
        "validation_coverage": validation_coverage,
        "seed_ranges": ranges,
        "validation_status": validation_status,
        "validation_target_metrics": target_metrics,
        "methodology_limits": [
            "This evaluates outcomes under the saved simulator inputs and does not establish "
            "real-world calibration, causality, or a globally optimal strategy.",
            "Matching qualifying and seed records do not freeze later race events across plans.",
            "Repeating the same request reuses the same validation seed range.",
        ],
    }
    return {
        "selection": selection,
        "training_results": training_results,
        "validation_results": validation_results,
    }
