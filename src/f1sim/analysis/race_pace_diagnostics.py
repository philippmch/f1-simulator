"""Offline pace-component diagnostics, independent of forecast execution.

Race-pace labels are measurements or estimates, not causal ground truth. These
scores cannot establish winner-forecast accuracy or authorize a deployment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from statistics import median


def _validated_values(values, name, field):
    if not isinstance(values, Mapping):
        raise ValueError(f"{name} must map driver identities to pace percentages")
    if any(key not in field for key in values):
        raise ValueError(f"{name} contains a driver outside the starting grid")
    result = {}
    for key, value in values.items():
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"{name} must contain finite numeric pace percentages")
        result[key] = float(value)
    return result


def _pair_error(predicted, observed, codes):
    pairs = list(combinations(codes, 2))
    if not pairs:
        return None
    return sum(
        abs((predicted[a] - predicted[b]) - (observed[a] - observed[b]))
        for a, b in pairs
    ) / len(pairs)


def _metrics(predicted, observed, cohort, front):
    if len(cohort) < 2:
        return {
            "whole_field_centered_mae_percent": None,
            "front_field_centered_mae_percent": None,
            "front_pair_mae_percent": None,
        }
    predicted_center = median(predicted[c] for c in cohort)
    observed_center = median(observed[c] for c in cohort)
    errors = {
        c: abs(predicted[c] - predicted_center - observed[c] + observed_center)
        for c in cohort
    }
    return {
        "whole_field_centered_mae_percent": sum(errors.values()) / len(cohort),
        "front_field_centered_mae_percent": (
            sum(errors[c] for c in front) / len(front) if front else None
        ),
        "front_pair_mae_percent": _pair_error(predicted, observed, front),
    }


def score_race_pace_components(
    starting_grid: Sequence[str],
    baseline_pace_percent: Mapping[str, float],
    candidate_pace_percent: Mapping[str, float],
    observed_pace_percent: Mapping[str, float],
) -> dict:
    """Compare a matched cohort and the grid's first four entrants.

    Inputs use percentage points of relative lap time, with higher values meaning
    faster pace. Each map may have its own arbitrary additive offset. Omit missing
    measurements rather than replacing them with zero. The front group is selected
    from the supplied pre-race grid before inspecting any performance map.

    Partial front scores remain visible, but front improvement is unreported when
    any selected entrant lacks a baseline, candidate, or observed measurement.
    """
    if (isinstance(starting_grid, (str, bytes))
            or not isinstance(starting_grid, Sequence)):
        raise ValueError("starting_grid must be a sequence of driver identities")
    grid = list(starting_grid)
    if (len(grid) < 2 or any(not isinstance(c, str) or not c.strip() for c in grid)
            or len(grid) != len(set(grid))):
        raise ValueError("starting_grid must contain at least two unique driver identities")
    field = set(grid)
    selected_front = grid[:4]
    baseline = _validated_values(baseline_pace_percent, "baseline_pace_percent", field)
    candidate = _validated_values(candidate_pace_percent, "candidate_pace_percent", field)
    observed = _validated_values(observed_pace_percent, "observed_pace_percent", field)
    cohort = [c for c in grid if c in baseline and c in candidate and c in observed]
    front = [c for c in selected_front if c in cohort]
    full_support = len(cohort) == len(grid)
    front_support = len(front) == len(selected_front)
    expected_pairs = len(selected_front) * (len(selected_front) - 1) // 2
    scored_pairs = len(front) * (len(front) - 1) // 2
    old, new = _metrics(baseline, observed, cohort, front), _metrics(
        candidate, observed, cohort, front
    )
    whole_gain = (
        old["whole_field_centered_mae_percent"] - new["whole_field_centered_mae_percent"]
        if old["whole_field_centered_mae_percent"] is not None else None
    )
    front_gain = (
        old["front_pair_mae_percent"] - new["front_pair_mae_percent"]
        if front_support and scored_pairs else None
    )
    return {
        "kind": "offline_race_pace_component_diagnostic",
        "pace_units": "percentage points of relative lap time; higher means faster",
        "front_selection": "first four supplied pre-race starting-grid entrants",
        "cohort": {
            "grid_drivers": len(grid), "scored_drivers": len(cohort),
            "scored_driver_ids": cohort,
            "missing_baseline_ids": [c for c in grid if c not in baseline],
            "missing_candidate_ids": [c for c in grid if c not in candidate],
            "missing_observed_ids": [c for c in grid if c not in observed],
            "selected_front_ids": selected_front, "scored_front_ids": front,
            "front_observed_drivers": sum(c in observed for c in selected_front),
            "front_expected_pairs": expected_pairs, "front_scored_pairs": scored_pairs,
            "front_missing_pairs": expected_pairs - scored_pairs,
            "full_field_support_complete": full_support,
            "front_support_complete": front_support,
            "front_comparison_status": (
                "complete" if front_support else "partial" if scored_pairs else "unavailable"
            ),
        },
        "baseline": old, "candidate": new,
        "comparison": {
            "matched_cohort_whole_field_mae_gain_percent": whole_gain,
            "complete_front_pair_mae_gain_percent": front_gain,
        },
        "limitations": [
            "Observed race pace may be estimated and confounded by tyres, weather, or traffic.",
            "Matched-cohort scores do not describe entrants without supported measurements.",
            "A lower whole-field error can coexist with worse pace gaps between front rivals.",
            "Component scores alone do not establish winner forecast or deployment quality.",
        ],
    }
