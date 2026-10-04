"""Select saved pit plans on training seeds and assess them on fresh seeds."""

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from math import isfinite, sqrt
from numbers import Integral
from statistics import mean, stdev
from typing import Any, Callable

from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.analysis.paired_comparison import (
    _observation,
    _ordered_trials,
    _qualifying_valid,
    _snapshot,
    paired_comparison_statistics,
)
from f1sim.analysis.strategy_comparison import (
    _build_pit_plan_variant_runners,
    _prepare_saved_pit_plan_variants,
    _runner_variant,
    _validate_pit_plan_variant_requests,
)
from f1sim.simulation.race import result_is_classified
from f1sim.simulation.randomness import validate_rng_policy

_MAX_SEED = 2**32 - 1
SELECTION_OBJECTIVES = ("points", "win", "podium")


def _validate_objective(objective: object) -> str:
    if not isinstance(objective, str) or objective not in SELECTION_OBJECTIVES:
        raise ValueError("objective must be one of points, win, podium")
    return objective


def _objective_description(objective: str, constructor: bool) -> str:
    if objective == "points":
        return "Expected constructor points" if constructor else "Expected driver points"
    outcome = "classified race win" if objective == "win" else "classified podium"
    target = "at least one constructor driver" if constructor else "the target driver"
    return f"Probability of a {outcome} for {target}"


def _phase_scores(
    results: Mapping[str, SimulationResults], points: dict[str, list[int | float]],
    target_members: list[str], objective: str,
) -> dict[str, list[int | float]]:
    """Score an already validated complete cohort, with one event per team race."""
    if objective == "points":
        return points
    position_limit = 1 if objective == "win" else 3
    members = set(target_members)
    return {
        label: [
            int(any(row.driver_id in members and result_is_classified(row)
                    and row.position <= position_limit for row in race))
            for race in result.race_results
        ]
        for label, result in results.items()
    }


def _score_number(value: int | float | Fraction) -> int | float:
    return _json_number(float(value) if isinstance(value, Fraction) else value)


def _score_metrics(
    reference_values: list[int | float | Fraction],
    selected_values: list[int | float | Fraction], *, identity: bool = False,
) -> dict[str, int | float | None]:
    """Use paired seed differences, including scenario covariance when weighted."""
    if len(reference_values) != len(selected_values) or not reference_values:
        raise ValueError("paired objective scores must have matching nonempty seed cohorts")
    differences = [selected - reference for reference, selected in zip(
        reference_values, selected_values,
    )]
    error = None
    if not identity and len(differences) > 1:
        error = stdev(differences) / sqrt(len(differences))
    return {
        "reference_mean_score": _score_number(mean(reference_values)),
        "selected_mean_score": _score_number(mean(selected_values)),
        "mean_score_difference": _score_number(mean(differences)),
        "score_difference_standard_error": None if error is None else _score_number(error),
    }


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
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, SimulationResults]:
    results = {}
    for label, runner in variants.items():
        _check_cancelled(cancel_requested)
        run_kwargs: dict[str, Any] = {
            "parallel": parallel,
            "max_workers": None if max_workers is None else int(max_workers),
        }
        if cancel_requested is not None:
            run_kwargs["cancel_requested"] = cancel_requested
        result = runner.run(count, **run_kwargs)
        _check_cancelled(cancel_requested)
        results[label] = result
    return results


def _check_cancelled(cancel_requested: Callable[[], bool] | None) -> None:
    if cancel_requested is not None and cancel_requested():
        raise SimulationCancelled("Pit-plan selection was cancelled.")


def validate_pit_plan_selection_request(
    plans: Mapping[str, Any],
    reference_label: str,
    *,
    driver_id: str | None,
    constructor_id: str | None,
    training_simulations: int,
    validation_simulations: int,
    objective: str = "points",
    max_count: int | None = None,
) -> list[str]:
    """Validate request fields that do not depend on a loaded runner."""
    _validate_objective(objective)
    labels = _validate_labels(plans, reference_label)
    if (driver_id is None) == (constructor_id is None):
        raise ValueError("exactly one driver_id or constructor_id is required")
    target_id = driver_id if driver_id is not None else constructor_id
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("driver_id or constructor_id must be a nonempty string")
    training_count = _positive_int(training_simulations, "training_simulations")
    validation_count = _positive_int(validation_simulations, "validation_simulations")
    if max_count is not None:
        if training_count > max_count:
            raise ValueError(f"training_simulations must be at most {max_count}")
        if validation_count > max_count:
            raise ValueError(f"validation_simulations must be at most {max_count}")
    constructor_mode = constructor_id is not None
    _validate_pit_plan_variant_requests(plans, constructor_mode=constructor_mode)
    from f1sim.simulation.pit_plans import validate_pit_plans

    for label, requested in plans.items():
        if requested is None:
            continue
        if constructor_mode:
            if any(
                not isinstance(member_id, str) or not member_id.strip()
                for member_id in requested
            ):
                raise ValueError(
                    f"plan variant {label!r} must use nonempty constructor member IDs",
                )
            member_plans = requested.items()
        else:
            member_plans = ((driver_id, requested),)
        for member_id, instructions in member_plans:
            if instructions is not None:
                validate_pit_plans({member_id: instructions})
    return labels


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


def _points_outcome_profile(
    reference_values: list[int | float], selected_values: list[int | float],
) -> dict[str, int | float | None]:
    """Describe the distribution of paired target-point changes by seed."""
    if len(reference_values) != len(selected_values):
        raise ValueError("paired point outcomes must have matching seed cohorts")
    differences = [selected - reference for reference, selected in zip(
        reference_values, selected_values,
    )]
    return _points_outcome_profile_from_differences(differences)


def _points_outcome_profile_from_differences(
    differences: list,
) -> dict[str, int | float | None]:
    """Summarize an already paired difference vector without changing its signs."""
    gains = [difference for difference in differences if difference > 0]
    losses = [-difference for difference in differences if difference < 0]
    return {
        "paired_races": len(differences),
        "more_points_races": len(gains),
        "equal_points_races": sum(difference == 0 for difference in differences),
        "fewer_points_races": len(losses),
        "mean_points_gain_when_ahead": (
            _json_number(float(mean(gains))) if gains else None
        ),
        "mean_points_loss_when_behind": (
            _json_number(float(mean(losses))) if losses else None
        ),
    }


def evaluate_saved_pit_plan_selection(
    path,
    plans: Mapping[str, Any],
    reference_label: str,
    *,
    driver_id: str | None = None,
    constructor_id: str | None = None,
    objective: str = "points",
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
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    if rng_policy is not None:
        rng_policy = validate_rng_policy(rng_policy)
    validate_pit_plan_selection_request(
        plans, reference_label, driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_simulations,
        validation_simulations=validation_simulations,
        objective=objective,
    )
    runner, saved_count, source_variants = _prepare_saved_pit_plan_variants(
        path, plans, driver_id=driver_id, constructor_id=constructor_id,
        scenario=scenario, rng_policy=rng_policy,
    )
    prepared = _prepare_selection_from_variants(
        runner, saved_count, plans, reference_label, source_variants,
        driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_simulations,
        validation_simulations=validation_simulations,
        objective=objective,
    )
    return evaluate_prepared_pit_plan_selection(
        prepared, parallel=parallel, max_workers=max_workers,
    )


def prepare_pit_plan_selection(
    runner: MonteCarloRunner,
    source_simulations: int,
    plans: Mapping[str, Any],
    reference_label: str,
    *,
    driver_id: str | None = None,
    constructor_id: str | None = None,
    objective: str = "points",
    training_simulations: int = 100,
    validation_simulations: int = 100,
) -> dict[str, Any]:
    """Validate and build all in-memory candidate and phase runners before trials."""
    labels = validate_pit_plan_selection_request(
        plans, reference_label, driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_simulations,
        validation_simulations=validation_simulations,
        objective=objective,
    )
    source_count = _positive_int(source_simulations, "source_simulations")
    source_variants = _build_pit_plan_variant_runners(
        runner, plans, driver_id=driver_id, constructor_id=constructor_id,
    )
    return _prepare_selection_from_variants(
        runner, source_count, plans, reference_label, source_variants,
        driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_simulations,
        validation_simulations=validation_simulations,
        objective=objective,
        labels=labels,
    )


def evaluate_pit_plan_selection(
    runner: MonteCarloRunner,
    source_simulations: int,
    plans: Mapping[str, Any],
    reference_label: str,
    *,
    driver_id: str | None = None,
    constructor_id: str | None = None,
    objective: str = "points",
    training_simulations: int = 100,
    validation_simulations: int = 100,
    parallel: bool = False,
    max_workers: int | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Select and validate candidates using a trusted in-memory runner."""
    prepared = prepare_pit_plan_selection(
        runner, source_simulations, plans, reference_label,
        driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_simulations,
        validation_simulations=validation_simulations,
        objective=objective,
    )
    return evaluate_prepared_pit_plan_selection(
        prepared, parallel=parallel, max_workers=max_workers,
        cancel_requested=cancel_requested,
    )


def _prepare_selection_from_variants(
    runner: MonteCarloRunner,
    source_count: int,
    plans: Mapping[str, Any],
    reference_label: str,
    source_variants: Mapping[str, MonteCarloRunner],
    *,
    driver_id: str | None,
    constructor_id: str | None,
    training_simulations: int,
    validation_simulations: int,
    objective: str = "points",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    training_count = _positive_int(training_simulations, "training_simulations")
    validation_count = _positive_int(validation_simulations, "validation_simulations")
    labels = labels or _validate_labels(plans, reference_label)
    ranges = _validate_seed_ranges(
        int(runner.base_seed), source_count, training_count, validation_count,
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
    frozen_plans = {
        label: deepcopy(getattr(source_variants[label], "pit_plans", None) or {})
        for label in labels
    }
    return {
        "labels": labels,
        "reference_label": reference_label,
        "driver_id": driver_id,
        "constructor_id": constructor_id,
        "objective": _validate_objective(objective),
        "training_count": training_count,
        "validation_count": validation_count,
        "train_start": train_start,
        "valid_start": valid_start,
        "ranges": ranges,
        "training_runners": training_runners,
        "validation_candidates": validation_candidates,
        "target_members": target_members,
        "plans": frozen_plans,
    }


def evaluate_prepared_pit_plan_selection(
    prepared: Mapping[str, Any],
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run prepared training/validation cohorts and return the frozen selection."""
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    _check_cancelled(cancel_requested)
    labels = prepared["labels"]
    reference_label = prepared["reference_label"]
    driver_id = prepared["driver_id"]
    constructor_id = prepared["constructor_id"]
    objective = _validate_objective(prepared.get("objective", "points"))
    training_count = prepared["training_count"]
    validation_count = prepared["validation_count"]
    train_start = prepared["train_start"]
    valid_start = prepared["valid_start"]
    ranges = prepared["ranges"]
    training_runners = prepared["training_runners"]
    validation_candidates = prepared["validation_candidates"]
    target_members = prepared["target_members"]

    training_results = _run_variants(
        training_runners, training_count, parallel=parallel, max_workers=max_workers,
        cancel_requested=cancel_requested,
    )
    _check_cancelled(cancel_requested)
    training_points, training_coverage = _phase_points(
        training_results, labels, reference_label,
        expected_runner=training_runners[reference_label],
        expected_seed=train_start, expected_count=training_count,
        target_members=target_members,
    )
    _check_cancelled(cancel_requested)
    training_values = _phase_scores(training_results, training_points, target_members, objective)
    training_scores = {
        label: mean([Fraction(value) for value in values])
        for label, values in training_values.items()
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
        cancel_requested=cancel_requested,
    )
    _check_cancelled(cancel_requested)
    validation_points, validation_coverage = _phase_points(
        validation_results, validation_labels, reference_label,
        expected_runner=validation_runners[reference_label],
        expected_seed=valid_start, expected_count=validation_count,
        target_members=target_members,
    )
    _check_cancelled(cancel_requested)

    no_change = selected_label == reference_label
    if no_change:
        target_metrics = {
            "reference_mean_points": _json_number(mean(validation_points[reference_label])),
            "selected_mean_points": _json_number(mean(validation_points[reference_label])),
            "mean_points_difference": 0,
            "points_difference_standard_error": None,
            "paired_races": validation_count,
            "points_outcome_profile": None,
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
            "points_outcome_profile": _points_outcome_profile(
                validation_points[reference_label], validation_points[selected_label],
            ),
            "comparison": "selected plan minus fixed reference, paired by seed",
        }
        validation_status = "evaluated"

    validation_values = _phase_scores(
        validation_results, validation_points, target_members, objective,
    )
    target_metrics.update(_score_metrics(
        validation_values[reference_label], validation_values[selected_label], identity=no_change,
    ))

    score_table = [
        {
            "label": label,
            "total_points": _json_number(sum(training_points[label])),
            "mean_points": _json_number(mean(training_points[label])),
            "total_score": _score_number(sum(training_values[label])),
            "mean_score": _score_number(training_scores[label]),
            "mean_score_behind_selected": _score_number(best_score - training_scores[label]),
            "tied_for_best": training_scores[label] == best_score,
            "trials": training_count,
        }
        for label in labels
    ]
    selection = {
        "schema_version": 1,
        "method": "complete_cohort_training_then_disjoint_seed_validation",
        "objective": objective,
        "objective_description": _objective_description(objective, constructor_id is not None),
        "score_unit": "points" if objective == "points" else "probability",
        "target_mode": "driver" if driver_id is not None else "constructor",
        "target_id": driver_id if driver_id is not None else constructor_id,
        "target_member_ids": target_members,
        "candidate_order": labels,
        "reference_label": reference_label,
        "selected_label": selected_label,
        "selection_status": "no_change" if no_change else "selected",
        "selection_rule": f"highest training mean {objective} score; exact ties prefer the "
        "reference, then candidate mapping order",
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
