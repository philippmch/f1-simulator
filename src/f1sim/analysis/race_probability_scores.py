"""Winner probabilities and multiclass scores for Monte Carlo race results.

Winner probabilities include a distinct ``no_classified_winner`` outcome. The
Wilson bounds describe Monte Carlo sampling error for each outcome; they are
not confidence bounds on the simulator's real-world accuracy.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import fsum, isclose, isfinite
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
