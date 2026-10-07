"""Winner probabilities and multiclass scores for Monte Carlo race results.

Winner probabilities include a distinct ``no_classified_winner`` outcome. The
Wilson bounds describe Monte Carlo sampling error for each outcome; they are
not confidence bounds on the simulator's real-world accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import fsum, isclose, isfinite, sqrt
from numbers import Integral, Real
from typing import Any

from f1sim.analysis.montecarlo import wilson_interval
from f1sim.simulation.race import result_is_classified

_PROBABILITY_SUM_TOLERANCE = 1e-9
_INTERVAL_METADATA = {
    "confidence": 0.95,
    "method": "wilson",
    "scope": "monte_carlo_sampling",
    "units": "fraction",
}


def summarize_winner_counts(wins, no_winner_count):
    """Summarize already aggregated native counts with the same trial contract."""
    score_winner_counts(wins, no_winner_count, None)
    trials = sum(wins.values()) + no_winner_count
    return {
        "trials": int(trials),
        "drivers": {driver: {"wins": int(count), "probability": count / trials,
            "mc_sampling_interval_95": _fraction_wilson_interval(count, trials)}
            for driver, count in wins.items()},
        "no_classified_winner": {"count": int(no_winner_count),
            "probability": no_winner_count / trials,
            "mc_sampling_interval_95": _fraction_wilson_interval(no_winner_count, trials)},
        "interval_metadata": dict(_INTERVAL_METADATA),
    }


def summarize_winner_trials(
    race_results: Iterable[Iterable[Any]],
    driver_ids: Iterable[str],
    expected_trials: int,
) -> dict[str, Any]:
    """Summarize complete race trials into winner-category probabilities.

    A winner is a result at position 1 for which
    :func:`result_is_classified` returns true. Each trial must contain the full
    driver roster exactly once, with positions forming a permutation from 1 to
    the roster size. A trial without a classified position-1 result contributes
    to the separate ``no_classified_winner`` category.

    The returned probability values are fractions. Every category also has an
    individual 95% Wilson interval for Monte Carlo sampling error.

    Raises:
        ValueError: If the roster, trial count, or any race trial is invalid.
    """
    if (
        isinstance(expected_trials, bool)
        or not isinstance(expected_trials, Integral)
        or expected_trials <= 0
    ):
        raise ValueError("expected_trials must be a positive integer")
    trial_count = int(expected_trials)

    roster = _materialize_iterable(driver_ids, "driver_ids")
    if not roster:
        raise ValueError("driver_ids must contain at least one driver")
    for driver_id in roster:
        _validate_driver_id(driver_id, "driver_ids")
    if len(set(roster)) != len(roster):
        raise ValueError("driver_ids must be unique")
    roster_set = set(roster)

    trials = _materialize_exact_length(race_results, trial_count, "race_results")

    wins = dict.fromkeys(roster, 0)
    no_winner_count = 0
    legal_positions = set(range(1, len(roster) + 1))

    for trial_index, race in enumerate(trials):
        rows = _materialize_exact_length(race, len(roster), f"race_results[{trial_index}]")

        seen_drivers: set[str] = set()
        seen_positions: set[int] = set()
        classified_winners: list[str] = []
        for row_index, result in enumerate(rows):
            location = f"race_results[{trial_index}][{row_index}]"
            try:
                driver_id = result.driver_id
            except Exception as exc:
                raise ValueError(f"{location} must have a driver_id") from exc
            _validate_driver_id(driver_id, f"{location}.driver_id")
            if driver_id not in roster_set:
                raise ValueError(f"{location}.driver_id is not in driver_ids")
            if driver_id in seen_drivers:
                raise ValueError(f"trial {trial_index} contains duplicate driver {driver_id!r}")
            seen_drivers.add(driver_id)

            try:
                position = result.position
            except Exception as exc:
                raise ValueError(f"{location} must have a legal position") from exc
            if (
                isinstance(position, bool)
                or not isinstance(position, Integral)
                or position not in legal_positions
            ):
                raise ValueError(f"{location}.position must be an integer from 1 to {len(roster)}")
            position = int(position)
            if position in seen_positions:
                raise ValueError(f"trial {trial_index} contains duplicate position {position}")
            seen_positions.add(position)

            try:
                classified = result_is_classified(result)
            except Exception as exc:
                raise ValueError(f"{location} has invalid classification data") from exc
            if not isinstance(classified, bool):
                raise ValueError(f"{location} has invalid classification data")
            if position == 1 and classified:
                classified_winners.append(driver_id)

        if seen_drivers != roster_set:
            raise ValueError(f"trial {trial_index} does not match driver_ids")
        if seen_positions != legal_positions:
            # The position count and uniqueness checks above normally imply this.
            raise ValueError(f"trial {trial_index} does not use every legal position")
        if len(classified_winners) > 1:
            raise ValueError(f"trial {trial_index} has multiple classified winners")
        if classified_winners:
            winner_id = classified_winners[0]
            wins[winner_id] += 1
        else:
            no_winner_count += 1

    return {
        "trials": trial_count,
        "drivers": {
            driver_id: {
                "wins": win_count,
                "probability": win_count / trial_count,
                "mc_sampling_interval_95": _fraction_wilson_interval(win_count, trial_count),
            }
            for driver_id, win_count in wins.items()
        },
        "no_classified_winner": {
            "count": no_winner_count,
            "probability": no_winner_count / trial_count,
            "mc_sampling_interval_95": _fraction_wilson_interval(no_winner_count, trial_count),
        },
        "interval_metadata": dict(_INTERVAL_METADATA),
    }


def score_winner_probabilities(
    driver_probabilities: Mapping[str, Real],
    no_winner_probability: Real,
    observed_winner_id: str | None,
) -> dict[str, float]:
    """Return multiclass Brier score and its uniform-roster baseline delta.

    ``observed_winner_id=None`` represents the no-classified-winner outcome.
    The baseline assigns equal probability to each driver and zero probability
    to no winner. Its Brier score is evaluated against the same observed
    outcome as the model score. Probabilities are validated and scored as
    supplied; this function never renormalizes them.

    Raises:
        ValueError: If any category, probability, or observed outcome is invalid.
    """
    if not isinstance(driver_probabilities, Mapping) or not driver_probabilities:
        raise ValueError("driver_probabilities must be a non-empty mapping")

    validated_probabilities: dict[str, float] = {}
    for driver_id, probability in driver_probabilities.items():
        _validate_driver_id(driver_id, "driver_probabilities key")
        validated_probabilities[driver_id] = _validate_probability(
            probability, f"probability for {driver_id!r}"
        )

    no_winner = _validate_probability(no_winner_probability, "no_winner_probability")
    total_probability = fsum((*validated_probabilities.values(), no_winner))
    if not isclose(
        total_probability,
        1.0,
        rel_tol=_PROBABILITY_SUM_TOLERANCE,
        abs_tol=_PROBABILITY_SUM_TOLERANCE,
    ):
        raise ValueError("winner probabilities must sum to 1")

    if observed_winner_id is not None:
        _validate_driver_id(observed_winner_id, "observed_winner_id")
        if observed_winner_id not in validated_probabilities:
            raise ValueError("observed_winner_id must be a modeled driver or None")

    brier_score = fsum(
        (probability - float(driver_id == observed_winner_id)) ** 2
        for driver_id, probability in validated_probabilities.items()
    )
    brier_score += (no_winner - float(observed_winner_id is None)) ** 2
    driver_count = len(validated_probabilities)
    baseline_driver_probability = 1.0 / driver_count
    baseline_brier_score = fsum(
        (baseline_driver_probability - float(driver_id == observed_winner_id)) ** 2
        for driver_id in validated_probabilities
    )
    baseline_brier_score += float(observed_winner_id is None)

    return {
        "brier_score": brier_score,
        "uniform_baseline_brier_score": baseline_brier_score,
        "delta_from_uniform_baseline": brier_score - baseline_brier_score,
    }


def score_winner_counts(
    driver_wins: Mapping[str, Integral],
    no_winner_count: Integral,
    observed_winner_id: str | None,
) -> dict[str, Any]:
    """Score counts with a finite-trial correction and conditional sampling SE.

    The adjustment estimates the underlying distribution's Brier loss for a
    fixed observed outcome, assuming independent identically distributed trials.
    Its standard error substitutes empirical category frequencies into the
    exact variance of the order-two U statistic. It is an estimate, not a
    confidence interval or an unbiased estimator of the variance. Categories
    include no classified winner without colliding with any driver identifier.
    """
    if not isinstance(driver_wins, Mapping) or not driver_wins:
        raise ValueError("driver_wins must be a non-empty mapping")
    wins = {}
    for driver_id, count in driver_wins.items():
        _validate_driver_id(driver_id, "driver_wins key")
        wins[driver_id] = _validate_count(count, f"wins for {driver_id!r}")
    no_winner = _validate_count(no_winner_count, "no_winner_count")
    counts = [*wins.values(), no_winner]
    trials = sum(counts)
    if trials == 0:
        raise ValueError("winner counts must contain at least one trial")
    score = score_winner_probabilities(
        {driver_id: count / trials for driver_id, count in wins.items()},
        no_winner / trials,
        observed_winner_id,
    )
    adjustment = {
        "method": "finite_ensemble_multiclass_brier_v1",
        "trials": trials,
        "scope": "monte_carlo_sampling_with_fixed_observed_outcome",
        "trial_assumption": "independent_identically_distributed_winner_categories",
        "status": "unavailable",
        "reason": "at_least_two_trials_required",
        "estimated_empirical_score_bias": None,
        "adjusted_brier_score": None,
        "adjusted_delta_from_uniform_baseline": None,
        "mc_standard_error": None,
        "mc_standard_error_method": "multinomial_plugin_u_statistic_v1",
        "mc_standard_error_reason": "at_least_two_trials_required",
        "observed_categories": sum(count > 0 for count in counts),
    }
    score["mc_adjustment"] = adjustment
    if trials == 1:
        return score

    observed_count = no_winner if observed_winner_id is None else wins[observed_winner_id]
    other_counts = [count for driver_id, count in wins.items()
                    if driver_id != observed_winner_id]
    if observed_winner_id is not None:
        other_counts.append(no_winner)
    wrong = trials - observed_count
    # Average pair kernels directly in integer counts. This avoids subtracting
    # two almost equal floating-point scores when the corrected loss is zero.
    pair_sum = wrong * (wrong - 1) + sum(count * (count - 1) for count in other_counts)
    adjusted_score = pair_sum / (trials * (trials - 1))
    bias = (trials * trials - sum(count * count for count in counts)) / (
        trials * trials * (trials - 1)
    )
    adjustment.update({
        "status": "available",
        "reason": None,
        "estimated_empirical_score_bias": bias,
        "adjusted_brier_score": adjusted_score,
        "adjusted_delta_from_uniform_baseline": (
            adjusted_score - score["uniform_baseline_brier_score"]
        ),
        "mc_standard_error_reason": None,
    })
    if adjustment["observed_categories"] < 2:
        adjustment["mc_standard_error_reason"] = "single_observed_category"
        return score

    # h(X,Z) = 1[X=Z] - 1[X=y] - 1[Z=y] + 1. Its values are
    # zero if either outcome is y, two for equal non-y outcomes, and one
    # otherwise. Covariances between overlapping pairs supply zeta_one.
    probabilities = [(observed_count / trials, observed_count / trials - 1)]
    probabilities.extend((count / trials, count / trials) for count in other_counts)
    q_mean = fsum(probability * q for probability, q in probabilities)
    zeta_one = fsum(probability * (q - q_mean) ** 2 for probability, q in probabilities)
    square_total = sum(count * count for count in other_counts)
    squared_trials = trials * trials
    p_zero = observed_count * (2 * trials - observed_count) / squared_trials
    p_one = (wrong * wrong - square_total) / squared_trials
    p_two = square_total / squared_trials
    kernel_mean = (wrong * wrong + square_total) / squared_trials
    zeta_two = fsum((p_zero * kernel_mean**2,
                     p_one * (1 - kernel_mean)**2,
                     p_two * (2 - kernel_mean)**2))
    variance = (4 * (trials - 2) / (trials * (trials - 1)) * zeta_one
                + 2 / (trials * (trials - 1)) * zeta_two)
    adjustment["mc_standard_error"] = sqrt(variance)
    return score


def _validate_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _materialize_iterable(value: Any, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must be an iterable sequence")
    try:
        return tuple(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an iterable sequence") from exc


def _materialize_exact_length(value: Any, expected_length: int, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(f"{name} must contain exactly {expected_length} items")
    try:
        iterator = iter(value)
    except Exception as exc:
        raise ValueError(f"{name} must contain exactly {expected_length} items") from exc

    items = []
    try:
        for _ in range(expected_length + 1):
            try:
                items.append(next(iterator))
            except StopIteration:
                break
    except Exception as exc:
        raise ValueError(f"{name} could not be read completely") from exc
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain exactly {expected_length} items")
    return tuple(items)


def _validate_driver_id(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")


def _validate_probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite probability")
    try:
        probability = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not isfinite(probability) or probability < 0 or probability > 1:
        raise ValueError(f"{name} must be a finite probability between 0 and 1")
    return probability


def _fraction_wilson_interval(successes: int, trials: int) -> dict[str, float]:
    percent_bounds = wilson_interval(successes, trials)
    return {
        "lower": percent_bounds["lower"] / 100,
        "upper": percent_bounds["upper"] / 100,
    }
