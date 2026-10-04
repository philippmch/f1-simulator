"""Select saved pit plans across explicit rival-plan scenarios offline."""

from collections.abc import Mapping
from copy import deepcopy
from fractions import Fraction
from math import fsum, isfinite, sqrt
from numbers import Integral, Real
from statistics import mean, stdev
from typing import Any, Callable

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.analysis.paired_comparison import _snapshot
from f1sim.analysis.replay import _load_saved_runner
from f1sim.analysis.strategy_comparison import (
    _build_pit_plan_variant_runners,
    _runner_variant,
    _validate_pit_plans,
)
from f1sim.analysis.strategy_selection import (
    _check_cancelled,
    _json_number,
    _objective_description,
    _phase_points,
    _phase_scores,
    _points_outcome_profile,
    _points_outcome_profile_from_differences,
    _positive_int,
    _run_variants,
    _score_metrics,
    _validate_objective,
    _validate_seed_ranges,
    validate_pit_plan_selection_request,
)
from f1sim.models import Weather
from f1sim.simulation.control_schedule import validate_control_schedule
from f1sim.simulation.qualifying_weather import (
    effective_qualifying_weather,
    validate_qualifying_weather,
)
from f1sim.simulation.randomness import validate_rng_policy
from f1sim.simulation.weather_schedule import validate_weather_schedule

RIVAL_SELECTION_METHODS = ("weighted_mean", "minimax_regret")


def validate_rival_selection_method(value: object) -> str:
    """Freeze the decision criterion before loading inputs or running trials."""
    if not isinstance(value, str) or value not in RIVAL_SELECTION_METHODS:
        raise ValueError("selection_method must be one of weighted_mean, minimax_regret")
    return value


def _weight_number(value: object, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite real number")
    try:
        numeric = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite real number") from exc
    if not isfinite(numeric) or numeric <= 0:
        raise ValueError(f"{name} must be a positive finite real number")
    if isinstance(value, Integral):
        return int(value)
    return numeric


def _validate_rival_scenarios(rival_scenarios: Mapping) -> tuple[list[str], dict, dict]:
    if not isinstance(rival_scenarios, Mapping) or not 1 <= len(rival_scenarios) <= 10:
        raise ValueError("rival_scenarios must be a mapping of 1 to 10 named scenarios")

    names = list(rival_scenarios)
    supplied_weights = {}
    rival_plans = {}
    weather_overrides = {}
    weather_schedules = {}
    control_schedules = {}
    for name, scenario in rival_scenarios.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 80:
            raise ValueError(
                "rival scenario names must be nonempty strings of at most 80 characters",
            )
        required = {"weight", "pit_plans"}
        if (not isinstance(scenario, Mapping) or not required <= set(scenario)
                or set(scenario) - required - {"weather", "weather_schedule", "control_schedule"}):
            raise ValueError(
                f"rival scenario {name!r} must contain weight and pit_plans, "
                "with optional weather, weather_schedule and control_schedule",
            )
        weather = scenario.get("weather")
        try:
            if weather is not None:
                if not isinstance(weather, dict):
                    raise ValueError("weather must be a field object or null")
                validated = validate_qualifying_weather({"Q1": weather})["Q1"]
                # Omitted fields inherit the source rather than Weather defaults.
                weather = {key: validated[key] for key in weather}
            schedule = scenario.get("weather_schedule")
            if schedule is not None:
                schedule = validate_weather_schedule(schedule)
        except ValueError as exc:
            raise ValueError(f"Invalid race weather for scenario {name!r}: {exc}") from exc
        weather_overrides[name] = deepcopy(weather)
        weather_schedules[name] = deepcopy(schedule)
        control_schedules[name] = validate_control_schedule(scenario.get("control_schedule"))
        supplied_weights[name] = _weight_number(
            scenario["weight"], f"weight for rival scenario {name!r}",
        )
        plans = scenario["pit_plans"]
        if not isinstance(plans, Mapping):
            raise ValueError(f"pit_plans for rival scenario {name!r} must be a mapping")
        for driver_id, instructions in plans.items():
            if not isinstance(driver_id, str) or not driver_id.strip():
                raise ValueError(
                    f"rival pit plan IDs for scenario {name!r} must be nonempty strings",
                )
            if instructions is not None and not isinstance(instructions, list):
                raise ValueError(
                    f"rival plan for driver {driver_id!r} must be a list or null",
                )
            if instructions is not None:
                # Check all input-independent instruction rules before dashboard
                # capacity admission. Loaded roster, distance, and inventory are
                # checked later against every weather runner.
                _validate_pit_plans({driver_id: instructions}, None,
                                    total_laps=None, tire_inventory=None)
        rival_plans[name] = deepcopy(dict(plans))

    scale = max(supplied_weights.values())
    scaled = [float(supplied_weights[name] / scale) for name in names]
    if any(value == 0 for value in scaled):
        raise ValueError("rival scenario weights are too far apart to normalize safely")
    total = fsum(scaled)
    if not isfinite(total) or total <= 0:
        raise ValueError("rival scenario weights could not be normalized")
    normalized = {name: scaled[index] / total for index, name in enumerate(names)}
    if any(value == 0 for value in normalized.values()):
        raise ValueError("rival scenario weights are too far apart to normalize safely")
    return names, supplied_weights, {"plans": rival_plans, "weights": normalized,
                                    "weather": weather_overrides,
                                    "weather_schedule": weather_schedules,
                                    "control_schedule": control_schedules}


def validate_rival_pit_plan_selection_request(rival_scenarios: Mapping) -> None:
    """Validate rival labels, weights, and override shapes without loaded inputs."""
    _validate_rival_scenarios(rival_scenarios)


def _apply_rival_overrides(
    runner,
    overrides: Mapping,
    *,
    target_members: set[str],
    runnable_ids: set[str],
) -> dict[str, list[dict]]:
    saved_plans = deepcopy(getattr(runner, "pit_plans", None) or {})
    for driver_id, instructions in overrides.items():
        if not isinstance(driver_id, str) or driver_id not in runnable_ids:
            raise ValueError(f"Unknown or nonrunnable rival driver ID: {driver_id!r}")
        if driver_id in target_members:
            raise ValueError(f"rival pit plans cannot override target driver {driver_id!r}")
        if instructions is not None and not isinstance(instructions, list):
            raise ValueError(
                f"rival plan for driver {driver_id!r} must be a list or null",
            )
        if instructions is None:
            saved_plans.pop(driver_id, None)
        else:
            saved_plans[driver_id] = deepcopy(instructions)
    canonical = _validate_pit_plans(
        saved_plans or None,
        [driver.id for driver in runner.drivers],
        total_laps=runner.track.total_laps,
        tire_inventory=runner.tire_inventory,
    )
    return canonical or {}


def _check_scenario_cohort(
    scenario_results: Mapping[str, Mapping[str, SimulationResults]],
    labels: list[str],
    target_members: list[str],
    *,
    phase: str,
    weather_contexts: Mapping[str, dict] | None = None,
    frozen_qualifying: dict | None = None,
) -> None:
    """Require equal non-plan inputs and qualifying outcomes across scenarios."""
    baseline_result = next(iter(scenario_results.values()))[labels[0]]
    baseline_snapshot = _snapshot(baseline_result)
    baseline_input = baseline_result.input_snapshot
    baseline_opening = (
        baseline_input.get("starting_tires", {}),
        baseline_input.get("starting_tire_ages", {}),
    )
    baseline_qualifying = baseline_result.qualifying_results
    if baseline_snapshot is None:
        raise ValueError(f"{phase} scenario has invalid saved input evidence")
    def comparable_snapshot(result, name):
        snapshot = _snapshot(result)
        if snapshot is None or weather_contexts is None:
            return snapshot
        expected = weather_contexts[name]
        values = snapshot[0]
        if (values["weather"] != expected["weather"]
                or values["weather_schedule"] != expected["weather_schedule"]):
            raise ValueError(f"{phase} {name!r} used different frozen race weather inputs")
        if values["qualifying_weather"] != frozen_qualifying:
            raise ValueError(f"{phase} {name!r} used different frozen qualifying weather inputs")
        if ("control_schedule" in expected
                and values["control_schedule"] != expected["control_schedule"]):
            raise ValueError(f"{phase} {name!r} used different frozen control schedule inputs")
        varied = {"weather", "weather_schedule"}
        if "control_schedule" in expected:
            varied.add("control_schedule")
        return ({key: value for key, value in values.items()
                 if key not in varied}, *snapshot[1:])

    baseline_snapshot = comparable_snapshot(
        baseline_result, next(iter(scenario_results)),
    )
    for scenario_name, variants in scenario_results.items():
        result = variants[labels[0]]
        if comparable_snapshot(result, scenario_name) != baseline_snapshot:
            raise ValueError(
                f"{phase} scenarios have incompatible non-plan inputs or runtime provenance",
            )
        raw_input = result.input_snapshot
        if (
            raw_input.get("starting_tires", {}),
            raw_input.get("starting_tire_ages", {}),
        ) != baseline_opening:
            raise ValueError(f"{phase} scenarios have different opening tyre inputs")
        if result.qualifying_results != baseline_qualifying:
            raise ValueError(
                f"qualifying results do not match across {phase} rival scenarios "
                f"(including {scenario_name!r})",
            )
        for label in labels:
            if label not in variants:
                continue
            if len(variants[label].qualifying_results) != len(baseline_qualifying):
                raise ValueError(f"{phase} {scenario_name!r}/{label!r} has incomplete qualifying")
    if not target_members:
        raise ValueError("the selected target has no runnable saved member")


def _reported_number(value: int | float | Fraction) -> int | float:
    """Convert exact weighted arithmetic only at the JSON reporting boundary."""
    return _json_number(float(value) if isinstance(value, Fraction) else value)


def _weighted_seed_points(
    point_rows: Mapping[str, Mapping[str, list[int | float]]],
    label: str,
    weights: Mapping[str, Fraction],
    count: int,
) -> list[Fraction]:
    return [
        sum(weight * Fraction(point_rows[name][label][trial])
            for name, weight in weights.items())
        for trial in range(count)
    ]


def _score_table(
    labels: list[str], point_rows: Mapping[str, list[int | float | Fraction]], count: int,
) -> list[dict]:
    return [
        {
            "label": label,
            "total_points": _reported_number(sum(point_rows[label])),
            "mean_points": _reported_number(mean(point_rows[label])),
            "trials": count,
        }
        for label in labels
    ]


def _training_regrets(labels, scenario_values, count):
    """Compare scenario means exactly, without weighting or per-trial hindsight."""
    scenario_means = {
        name: {
            label: sum((Fraction(value) for value in values[label]), Fraction()) / count
            for label in labels
        }
        for name, values in scenario_values.items()
    }
    best_means = {name: max(means.values()) for name, means in scenario_means.items()}
    regrets = {
        label: {name: best_means[name] - means[label] for name, means in scenario_means.items()}
        for label in labels
    }
    maximum_regrets = {label: max(values.values()) for label, values in regrets.items()}
    table = [
        {
            "label": label,
            "maximum_regret": _reported_number(maximum_regrets[label]),
            "worst_scenarios": [name for name, value in regrets[label].items()
                                if value == maximum_regrets[label]],
            "scenarios": {
                name: {
                    "mean_score": _reported_number(scenario_means[name][label]),
                    "best_candidate_mean_score": _reported_number(best_means[name]),
                    "regret": _reported_number(value),
                }
                for name, value in regrets[label].items()
            },
            "trials_per_scenario": count,
        }
        for label in labels
    ]
    return maximum_regrets, table


def _paired_summary(
    reference_values: list[int | float | Fraction],
    selected_values: list[int | float | Fraction],
    *,
    differences: list | None = None,
):
    if differences is None:
        differences = [selected - reference for reference, selected in zip(
            reference_values, selected_values,
        )]
    elif len(differences) != len(reference_values) or len(reference_values) != len(
        selected_values,
    ):
        raise ValueError("paired summary values must have matching seed cohorts")
    mean_difference = mean(differences)
    if differences and isinstance(differences[0], Fraction):
        mean_difference = float(mean_difference)
    standard_error = None
    if len(differences) > 1:
        standard_error = stdev(differences) / sqrt(len(differences))
    return {
        "reference_mean_points": _reported_number(mean(reference_values)),
        "selected_mean_points": _reported_number(mean(selected_values)),
        "mean_points_difference": _json_number(mean_difference),
        "points_difference_standard_error": (
            None if standard_error is None else _json_number(standard_error)
        ),
        "paired_races": len(differences),
    }


def prepare_rival_pit_plan_selection(
    runner,
    source_simulations: int,
    plans: Mapping[str, Any],
    reference_label: str,
    rival_scenarios: Mapping,
    *,
    driver_id: str | None = None,
    constructor_id: str | None = None,
    objective: str = "points",
    selection_method: str = "weighted_mean",
    training_simulations: int = 100,
    validation_simulations: int = 100,
    rng_policy: str | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Freeze all rival assumptions and phase runners before spending trials."""
    selection_method = validate_rival_selection_method(selection_method)
    source_count = _positive_int(source_simulations, "source_simulations")
    training_count = _positive_int(training_simulations, "training_simulations")
    validation_count = _positive_int(validation_simulations, "validation_simulations")
    if rng_policy is not None:
        rng_policy = validate_rng_policy(rng_policy)
    labels = validate_pit_plan_selection_request(
        plans, reference_label, driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_count, validation_simulations=validation_count,
        objective=objective,
    )
    scenario_names, supplied_weights, scenario_data = _validate_rival_scenarios(
        rival_scenarios,
    )
    normalized_weights = scenario_data["weights"]
    scenario_plans = scenario_data["plans"]
    weather_overrides = scenario_data["weather"]
    schedule_overrides = scenario_data["weather_schedule"]
    control_overrides = scenario_data["control_schedule"]
    control_assumptions = any(control_overrides[name] is not None for name in scenario_names)
    weather_assumptions = any(
        weather_overrides[name] or schedule_overrides[name] is not None
        for name in scenario_names
    )
    # Race uncertainty must not silently change the qualifying experiment.
    frozen_qualifying = effective_qualifying_weather(
        runner.weather, getattr(runner, "qualifying_weather", None),
    ) if weather_assumptions or control_assumptions else None
    weather_contexts = {} if weather_assumptions or control_assumptions else None
    candidate_definitions = deepcopy(dict(plans))

    if driver_id is not None:
        target = next((driver for driver in runner.drivers if driver.id == driver_id), None)
        if target is None:
            raise ValueError(f"Unknown driver ID: {driver_id}")
        if target.team_id not in runner.cars:
            raise ValueError(f"No saved car available for driver ID: {driver_id}")
        target_members = [driver_id]
    else:
        if not isinstance(constructor_id, str) or not constructor_id:
            raise ValueError("constructor_id must be an exact saved constructor ID")
        target_members = [
            driver.id for driver in runner.drivers if driver.team_id == constructor_id
        ]
        if constructor_id not in runner.cars:
            if target_members:
                raise ValueError(f"No saved car available for constructor ID: {constructor_id}")
            raise ValueError(f"Unknown constructor ID: {constructor_id}")
        if not target_members:
            raise ValueError(f"No runnable saved drivers for constructor ID: {constructor_id}")

    target_member_set = set(target_members)
    runnable_ids = {
        driver.id for driver in runner.drivers if driver.team_id in runner.cars
    }
    ranges = _validate_seed_ranges(
        int(runner.base_seed), source_count, training_count, validation_count,
    )
    train_start = ranges["training"]["first_seed"]
    valid_start = ranges["validation"]["first_seed"]

    # Validate each override against this loaded roster, target, track, and
    # inventory, then freeze every candidate and phase runner before any run.
    base_variants = {}
    plans_by_rival_scenario = {}
    for scenario_name in scenario_names:
        _check_cancelled(cancel_requested)
        assumption_runner = runner
        if weather_assumptions or control_assumptions:
            weather = validate_qualifying_weather({"Q1": {
                **runner.weather.model_dump(mode="json"),
                **(weather_overrides[scenario_name] or {}),
            }})["Q1"]
            schedule = validate_weather_schedule(
                getattr(runner, "weather_schedule", None)
                if schedule_overrides[scenario_name] is None else schedule_overrides[scenario_name],
                total_laps=runner.track.total_laps,
            )
            canonical_weather = Weather.model_validate(weather)
            assumption_runner = _runner_variant(
                runner, weather=runner.weather.model_copy(deep=True, update={
                    key: getattr(canonical_weather, key) for key in Weather.model_fields
                }),
                qualifying_weather=frozen_qualifying, weather_schedule=schedule,
            )
            weather_contexts[scenario_name] = {"weather": weather, "weather_schedule": schedule}
            if control_assumptions:
                control = validate_control_schedule(
                    getattr(runner, "control_schedule", None)
                    if control_overrides[scenario_name] is None
                    else control_overrides[scenario_name],
                    total_laps=runner.track.total_laps,
                )
                assumption_runner = _runner_variant(assumption_runner, control_schedule=control)
                weather_contexts[scenario_name]["control_schedule"] = control
        base_plans = _apply_rival_overrides(
            assumption_runner, scenario_plans[scenario_name],
            target_members=target_member_set, runnable_ids=runnable_ids,
        )
        variants = _build_pit_plan_variant_runners(
            assumption_runner, candidate_definitions,
            driver_id=driver_id,
            constructor_id=constructor_id,
            rng_policy=rng_policy,
            base_pit_plans=base_plans,
        )
        base_variants[scenario_name] = variants
        plans_by_rival_scenario[scenario_name] = {
            label: deepcopy(getattr(variant, "pit_plans", None) or {})
            for label, variant in variants.items()
        }

    training_runners = {
        scenario_name: {
            label: _runner_variant(variant, seed=train_start)
            for label, variant in base_variants[scenario_name].items()
        }
        for scenario_name in scenario_names
    }
    validation_candidates = {
        scenario_name: {
            label: _runner_variant(variant, seed=valid_start)
            for label, variant in base_variants[scenario_name].items()
        }
        for scenario_name in scenario_names
    }
    _check_cancelled(cancel_requested)
    return {
        "labels": labels,
        "reference_label": reference_label,
        "driver_id": driver_id,
        "constructor_id": constructor_id,
        "objective": objective,
        "selection_method": selection_method,
        "target_members": target_members,
        "training_count": training_count,
        "validation_count": validation_count,
        "train_start": train_start,
        "valid_start": valid_start,
        "ranges": ranges,
        "scenario_names": scenario_names,
        "supplied_weights": supplied_weights,
        "normalized_weights": normalized_weights,
        "weather_contexts": weather_contexts,
        "control_assumptions": control_assumptions,
        "frozen_qualifying_weather": frozen_qualifying,
        "scenario_plans": scenario_plans,
        "plans": candidate_definitions,
        "plans_by_rival_scenario": plans_by_rival_scenario,
        "training_runners": training_runners,
        "validation_candidates": validation_candidates,
    }


def evaluate_prepared_rival_pit_plan_selection(
    prepared: Mapping[str, Any],
    *,
    parallel: bool = False,
    max_workers: int | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen rivals with shared seed cohorts and disjoint validation."""
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    _check_cancelled(cancel_requested)
    labels = prepared["labels"]
    reference_label = prepared["reference_label"]
    driver_id = prepared["driver_id"]
    constructor_id = prepared["constructor_id"]
    objective = _validate_objective(prepared.get("objective", "points"))
    selection_method = validate_rival_selection_method(
        prepared.get("selection_method", "weighted_mean"),
    )
    target_members = prepared["target_members"]
    training_count = prepared["training_count"]
    validation_count = prepared["validation_count"]
    train_start = prepared["train_start"]
    valid_start = prepared["valid_start"]
    ranges = prepared["ranges"]
    scenario_names = prepared["scenario_names"]
    supplied_weights = prepared["supplied_weights"]
    normalized_weights = prepared["normalized_weights"]
    weather_contexts = prepared.get("weather_contexts")
    exact_weights = {name: Fraction(normalized_weights[name]) for name in scenario_names}
    scenario_plans = prepared["scenario_plans"]
    training_runners = prepared["training_runners"]
    validation_candidates = prepared["validation_candidates"]

    training_results = {}
    for scenario_name in scenario_names:
        _check_cancelled(cancel_requested)
        training_results[scenario_name] = _run_variants(
            training_runners[scenario_name], training_count,
            parallel=parallel, max_workers=max_workers,
            cancel_requested=cancel_requested,
        )
    _check_cancelled(cancel_requested)

    training_points = {}
    training_coverage = {}
    for scenario_name in scenario_names:
        _check_cancelled(cancel_requested)
        training_points[scenario_name], training_coverage[scenario_name] = _phase_points(
            training_results[scenario_name], labels, reference_label,
            expected_runner=training_runners[scenario_name][reference_label],
            expected_seed=train_start, expected_count=training_count,
            target_members=target_members,
        )
    _check_cancelled(cancel_requested)
    _check_scenario_cohort(
        training_results, labels, target_members, phase="training",
        weather_contexts=weather_contexts,
        frozen_qualifying=prepared.get("frozen_qualifying_weather"),
    )

    weighted_training_points = {
        label: _weighted_seed_points(training_points, label, exact_weights, training_count)
        for label in labels
    }
    training_values = {
        name: _phase_scores(
            training_results[name], training_points[name], target_members, objective,
        )
        for name in scenario_names
    }
    weighted_training_values = {
        label: _weighted_seed_points(training_values, label, exact_weights, training_count)
        for label in labels
    }
    scenario_score_tables = {}
    for name in scenario_names:
        scenario_score_tables[name] = {
            "weight": supplied_weights[name],
            "normalized_weight": normalized_weights[name],
            "scores": _score_table(labels, training_points[name], training_count),
        }
        for row in scenario_score_tables[name]["scores"]:
            values = training_values[name][row["label"]]
            row.update(total_score=_reported_number(sum(values)),
                       mean_score=_reported_number(mean(values)))

    training_scores = {
        label: mean(values)
        for label, values in weighted_training_values.items()
    }
    best_score = max(training_scores.values())
    regret_table = None
    if selection_method == "minimax_regret":
        maximum_regrets, regret_table = _training_regrets(labels, training_values, training_count)
        best_regret = min(maximum_regrets.values())
        tied_labels = [label for label in labels if maximum_regrets[label] == best_regret]
        for row in regret_table:
            row["tied_for_best"] = row["label"] in tied_labels
    else:
        tied_labels = [label for label in labels if training_scores[label] == best_score]
    if len(tied_labels) == 1:
        selected_label = tied_labels[0]
        tiebreak = ("unique_lowest_training_maximum_regret" if regret_table is not None
                    else "unique_highest_weighted_training_mean")
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
        name: {
            label: validation_candidates[name][label]
            for label in validation_labels
        }
        for name in scenario_names
    }
    validation_results = {}
    for scenario_name in scenario_names:
        _check_cancelled(cancel_requested)
        validation_results[scenario_name] = _run_variants(
            validation_runners[scenario_name], validation_count,
            parallel=parallel, max_workers=max_workers,
            cancel_requested=cancel_requested,
        )
    _check_cancelled(cancel_requested)

    validation_points = {}
    validation_coverage = {}
    for name in scenario_names:
        _check_cancelled(cancel_requested)
        validation_points[name], validation_coverage[name] = _phase_points(
            validation_results[name], validation_labels, reference_label,
            expected_runner=validation_runners[name][reference_label],
            expected_seed=valid_start, expected_count=validation_count,
            target_members=target_members,
        )
    _check_cancelled(cancel_requested)
    _check_scenario_cohort(
        validation_results, validation_labels, target_members, phase="validation",
        weather_contexts=weather_contexts,
        frozen_qualifying=prepared.get("frozen_qualifying_weather"),
    )

    no_change = selected_label == reference_label
    weighted_reference = _weighted_seed_points(
        validation_points, reference_label, exact_weights, validation_count,
    )
    if no_change:
        target_metrics = {
            "reference_mean_points": _reported_number(mean(weighted_reference)),
            "selected_mean_points": _reported_number(mean(weighted_reference)),
            "mean_points_difference": 0,
            "points_difference_standard_error": None,
            "paired_races": validation_count,
            "points_outcome_profile": None,
            "comparison": "identity; no separate alternative was estimated",
        }
        validation_status = "no_change"
    else:
        weighted_selected = _weighted_seed_points(
            validation_points, selected_label, exact_weights, validation_count,
        )
        weighted_differences = [
            selected - reference
            for reference, selected in zip(weighted_reference, weighted_selected)
        ]
        target_metrics = _paired_summary(
            weighted_reference, weighted_selected, differences=weighted_differences,
        )
        target_metrics["points_outcome_profile"] = (
            _points_outcome_profile_from_differences(weighted_differences)
        )
        target_metrics["comparison"] = (
            "selected plan minus fixed reference, weighted within seed across rival scenarios"
        )
        validation_status = "evaluated"

    validation_values = {
        name: _phase_scores(
            validation_results[name], validation_points[name], target_members, objective,
        )
        for name in scenario_names
    }
    target_metrics.update(_score_metrics(
        _weighted_seed_points(validation_values, reference_label, exact_weights, validation_count),
        _weighted_seed_points(validation_values, selected_label, exact_weights, validation_count),
        identity=no_change,
    ))

    scenario_validation_metrics = {}
    for name in scenario_names:
        reference_values = validation_points[name][reference_label]
        if no_change:
            scenario_validation_metrics[name] = {
                "reference_mean_points": _json_number(mean(reference_values)),
                "selected_mean_points": _json_number(mean(reference_values)),
                "mean_points_difference": 0,
                "points_difference_standard_error": None,
                "paired_races": validation_count,
                "points_outcome_profile": None,
                "comparison": "identity; no separate alternative was estimated",
            }
        else:
            scenario_validation_metrics[name] = _paired_summary(
                reference_values, validation_points[name][selected_label],
            )
            scenario_validation_metrics[name]["points_outcome_profile"] = (
                _points_outcome_profile(
                    reference_values, validation_points[name][selected_label],
                )
            )
            scenario_validation_metrics[name]["comparison"] = (
                "selected plan minus fixed reference, paired by seed within this scenario"
            )
        scenario_validation_metrics[name].update(_score_metrics(
            validation_values[name][reference_label], validation_values[name][selected_label],
            identity=no_change,
        ))

    training_table = _score_table(labels, weighted_training_points, training_count)
    for row in training_table:
        score = training_scores[row["label"]]
        values = weighted_training_values[row["label"]]
        row["total_score"] = _reported_number(sum(values))
        row["mean_score"] = _reported_number(score)
        row["mean_score_behind_selected"] = _reported_number(
            training_scores[selected_label] - score,
        )
        row["mean_points_behind_selected"] = _reported_number(
            mean(weighted_training_points[selected_label])
            - mean(weighted_training_points[row["label"]]),
        )
        row["tied_for_best"] = row["label"] in tied_labels
    selection = {
        "schema_version": 1,
        "method": "weighted_rival_scenario_training_then_disjoint_seed_validation",
        "objective": objective,
        "selection_method": selection_method,
        "objective_description": _objective_description(objective, constructor_id is not None),
        "score_unit": "points" if objective == "points" else "probability",
        "target_mode": "driver" if driver_id is not None else "constructor",
        "target_id": driver_id if driver_id is not None else constructor_id,
        "target_member_ids": target_members,
        "candidate_order": labels,
        "reference_label": reference_label,
        "selected_label": selected_label,
        "selection_status": "no_change" if no_change else "selected",
        "selection_rule": (
            f"highest mean of per-seed target {objective} score weighted across rival scenarios; "
            "exact ties prefer the reference, then candidate mapping order"
        ),
        "tiebreak_applied": tiebreak,
        "rival_scenarios": [
            {
                "name": name,
                "weight": supplied_weights[name],
                "normalized_weight": normalized_weights[name],
                "rival_pit_plans": deepcopy(scenario_plans[name]),
            }
            for name in scenario_names
        ],
        "training_score_table": training_table,
        "training_scenario_score_tables": scenario_score_tables,
        "training_coverage_by_scenario": training_coverage,
        "validation_coverage_by_scenario": validation_coverage,
        "seed_ranges": ranges,
        "validation_status": validation_status,
        "validation_target_metrics": target_metrics,
        "validation_scenario_metrics": scenario_validation_metrics,
        "methodology_limits": [
            "Scenario weights are supplied assumptions, not probabilities learned from data.",
            "This evaluates outcomes under saved simulator inputs and does not establish "
            "real-world calibration, causality, or a globally optimal strategy.",
            "Matching qualifying and seed records do not freeze later race events across plans.",
            "Repeating the same request reuses the same validation seed range.",
        ],
    }
    if regret_table is not None:
        selection["schema_version"] = 2
        selection["selection_rule"] = (
            f"lowest maximum across scenarios of the best candidate mean target {objective} "
            "score minus this candidate's mean; scenario weights do not affect selection; "
            "exact ties prefer the reference, then candidate mapping order"
        )
        selection["training_regret_table"] = regret_table
        selection["methodology_limits"].append(
            "Maximum regret uses training scenario means over the supplied candidates only. "
            "It is not a bound on individual races, unseen scenarios or sampling uncertainty. "
            "Adding candidates or scenarios can change the choice. Weights apply to reported "
            "aggregate scores and validation, not to this selection criterion.",
        )
    if weather_contexts is not None:
        selection["method"] = "weighted_weather_and_rival_training_then_disjoint_seed_validation"
        selection["frozen_qualifying_weather"] = deepcopy(prepared["frozen_qualifying_weather"])
        for scenario in selection["rival_scenarios"]:
            scenario.update(deepcopy(weather_contexts[scenario["name"]]))
        selection["methodology_limits"].append(
            "Race weather and rival plans are supplied joint assumptions. Qualifying weather "
            "is shared from the source; their weights are not learned weather forecasts.",
        )
    if regret_table is not None:
        selection["method"] = (
            "minimax_regret_weather_and_rival_training_then_disjoint_seed_validation"
            if weather_contexts is not None else
            "minimax_regret_rival_training_then_disjoint_seed_validation"
        )
    if prepared.get("control_assumptions"):
        selection["method"] = (
            "minimax_regret_race_scenario_training_then_disjoint_seed_validation"
            if regret_table is not None else
            "weighted_race_scenario_training_then_disjoint_seed_validation"
        )
        selection["methodology_limits"].append(
            "SC/VSC schedules are supplied scenario assumptions. Strategies observe each "
            "announcement only at its crossing; future deployments are not known to forecasts. "
            "Red flags can suppress requests, and a race can finish before later requests.",
        )
    return {
        "selection": selection,
        "training_results": training_results,
        "validation_results": validation_results,
    }


def evaluate_saved_rival_pit_plan_selection(
    path,
    plans: Mapping[str, Any],
    reference_label: str,
    rival_scenarios: Mapping,
    *,
    driver_id: str | None = None,
    constructor_id: str | None = None,
    objective: str = "points",
    selection_method: str = "weighted_mean",
    scenario: str | None = None,
    training_simulations: int = 100,
    validation_simulations: int = 100,
    parallel: bool = False,
    max_workers: int | None = None,
    rng_policy: str | None = None,
) -> dict[str, Any]:
    """Load saved inputs, prepare every assumption, and evaluate the frozen set."""
    selection_method = validate_rival_selection_method(selection_method)
    validate_pit_plan_selection_request(
        plans, reference_label, driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_simulations,
        validation_simulations=validation_simulations,
        objective=objective,
    )
    validate_rival_pit_plan_selection_request(rival_scenarios)
    if max_workers is not None:
        max_workers = _positive_int(max_workers, "max_workers")
    if rng_policy is not None:
        rng_policy = validate_rng_policy(rng_policy)
    runner, saved_count = _load_saved_runner(path, scenario)
    prepared = prepare_rival_pit_plan_selection(
        runner, saved_count, plans, reference_label, rival_scenarios,
        driver_id=driver_id, constructor_id=constructor_id,
        training_simulations=training_simulations,
        validation_simulations=validation_simulations,
        objective=objective,
        rng_policy=rng_policy,
        selection_method=selection_method,
    )
    return evaluate_prepared_rival_pit_plan_selection(
        prepared, parallel=parallel, max_workers=max_workers,
    )
