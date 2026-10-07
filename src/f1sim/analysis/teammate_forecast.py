"""Allocate native constructor win probabilities using strictly earlier race points.

This changes a reported forecast, never a simulated race or its win counts.
Sampling intervals and score corrections condition on the frozen point history;
they do not quantify uncertainty in the allocation policy or real-world outcomes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from math import fsum, isfinite, sqrt
from numbers import Real

from f1sim.analysis.race_probability_scores import (
    score_winner_counts,
    score_winner_probabilities,
)

POLICY = "teammate_race_points_v1"
PRIOR_POINTS = 25.0
MINIMUM_ENTRIES = 2
MAX_HISTORY_ROWS = 10_000


def _identity(value):
    return isinstance(value, str) and bool(value) and value == value.strip()


def _points(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (ValueError, OverflowError):
        return None
    return value if isfinite(value) and value >= 0 else None


def build_teammate_allocation(entrant_teams, prior_race_points, *, cutoff_round):
    """Freeze a fixed allocation, with native fallbacks for sparse/invalid history.

    Rows require ``round``, ``driver_id``, ``team_id`` and ``points``. A null or
    invalid point observation disables the affected current constructor rather
    than becoming zero. Exact duplicates do not add coverage; conflicting
    driver/round observations disable that driver's current constructor.
    """
    if type(cutoff_round) is not int or cutoff_round < 0:
        raise ValueError("Allocation cutoff_round must be a nonnegative integer")
    if (
        not isinstance(entrant_teams, Mapping)
        or not entrant_teams
        or any(
            not _identity(driver) or not _identity(team) for driver, team in entrant_teams.items()
        )
    ):
        raise ValueError("Allocation requires identified entrants and constructors")
    if not isinstance(prior_race_points, list) or len(prior_race_points) > MAX_HISTORY_ROWS:
        raise ValueError("Allocation requires a bounded list of earlier race points")
    roster = dict(sorted(entrant_teams.items()))
    unique = {}
    source_rows = {}
    invalid_teams = set()
    for row in prior_race_points:
        if (
            not isinstance(row, Mapping)
            or type(row.get("round")) is not int
            or not 1 <= row["round"] <= cutoff_round
            or not _identity(row.get("driver_id"))
            or not _identity(row.get("team_id"))
        ):
            raise ValueError("Point history requires identified strictly earlier race entries")
        driver, team = row["driver_id"], row["team_id"]
        value = _points(row.get("points"))
        canonical = {"round": row["round"], "driver_id": driver, "team_id": team, "points": value}
        source_rows[(row["round"], driver, team, value)] = canonical
        key = row["round"], driver
        if key in unique and unique[key] != canonical:
            if driver in roster:
                invalid_teams.add(roster[driver])
        else:
            unique[key] = canonical
        if driver in roster and team == roster[driver] and value is None:
            invalid_teams.add(team)

    totals, rounds = defaultdict(list), defaultdict(set)
    for row in unique.values():
        driver = row["driver_id"]
        if driver in roster and row["team_id"] == roster[driver] and row["points"] is not None:
            totals[driver].append(row["points"])
            rounds[driver].add(row["round"])
    members = defaultdict(list)
    for driver, team in roster.items():
        members[team].append(driver)
    teams = {}
    for team, drivers in sorted(members.items()):
        covered = team not in invalid_teams and all(
            len(rounds[driver]) >= MINIMUM_ENTRIES for driver in drivers
        )
        try:
            points = {driver: fsum(totals[driver]) for driver in drivers}
            denominator = fsum(PRIOR_POINTS + points[driver] for driver in drivers)
        except OverflowError as error:
            raise ValueError("Earlier point totals must remain finite") from error
        if any(not isfinite(value) for value in points.values()) or not isfinite(denominator):
            raise ValueError("Earlier point totals must remain finite")
        weights = (
            {driver: (PRIOR_POINTS + points[driver]) / denominator for driver in drivers}
            if covered
            else None
        )
        teams[team] = {
            "status": "calibrated" if covered else "native_fallback",
            "reason": (
                None
                if covered
                else "invalid_or_conflicting_points"
                if team in invalid_teams
                else "insufficient_earlier_entries"
            ),
            "points": points,
            "entries": {driver: len(rounds[driver]) for driver in drivers},
            "training_rounds": {driver: sorted(rounds[driver]) for driver in drivers},
            "weights": weights,
        }
    return {
        "policy": POLICY,
        "prior_points_per_driver": PRIOR_POINTS,
        "minimum_entries_per_driver": MINIMUM_ENTRIES,
        "cutoff_round": cutoff_round,
        "entrant_teams": roster,
        "prior_race_points": sorted(
            source_rows.values(),
            key=lambda row: (
                row["round"],
                row["driver_id"],
                row["team_id"],
                -1.0 if row["points"] is None else row["points"],
            ),
        ),
        "teams": teams,
    }


def validate_teammate_allocation(value, entrant_teams=None):
    """Rebuild a saved allocation instead of trusting its weights or cutoffs."""
    if not isinstance(value, Mapping) or value.get("policy") != POLICY:
        raise ValueError("Unsupported teammate allocation policy")
    rebuilt = build_teammate_allocation(
        value.get("entrant_teams"),
        value.get("prior_race_points"),
        cutoff_round=value.get("cutoff_round"),
    )
    try:
        saved = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        expected = json.dumps(rebuilt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("Teammate allocation must contain finite JSON evidence") from error
    if saved != expected:
        raise ValueError("Teammate allocation must match its recorded earlier point history")
    if entrant_teams is not None and rebuilt["entrant_teams"] != dict(entrant_teams):
        raise ValueError("Teammate allocation must match the simulation roster and constructors")
    return deepcopy(rebuilt)


def _native_counts(native):
    if (
        not isinstance(native, Mapping)
        or type(native.get("trials")) is not int
        or native["trials"] <= 0
        or not isinstance(native.get("drivers"), Mapping)
    ):
        raise ValueError("Native forecast requires a positive trial count and driver counts")
    try:
        wins = {driver: row["wins"] for driver, row in native["drivers"].items()}
        no_winner = native["no_classified_winner"]["count"]
        score_winner_counts(wins, no_winner, next(iter(wins)))
        if sum(wins.values()) + no_winner != native["trials"]:
            raise ValueError("Native winner counts must match the trial count")
        for driver, row in native["drivers"].items():
            if _points(row["probability"]) != wins[driver] / native["trials"]:
                raise ValueError("Native probabilities must match their win counts")
        if _points(native["no_classified_winner"]["probability"]) != no_winner / native["trials"]:
            raise ValueError("Native no-winner probability must match its count")
    except (KeyError, TypeError, StopIteration) as error:
        raise ValueError("Native forecast contains invalid counts or probabilities") from error
    return wins, no_winner


def _columns(native, allocation):
    categories = [*native["drivers"], None]
    columns = {driver: {driver: 1.0} for driver in categories}
    for driver, team in allocation["entrant_teams"].items():
        weights = allocation["teams"][team]["weights"]
        if weights is not None:
            columns[driver] = weights
    probabilities = {driver: row["probability"] for driver, row in native["drivers"].items()}
    probabilities[None] = native["no_classified_winner"]["probability"]
    calibrated = {
        driver: fsum(probabilities[key] * columns[key].get(driver, 0.0) for key in categories)
        for driver in categories
    }
    return categories, probabilities, columns, calibrated


def teammate_winner_forecast(native, allocation):
    """Return calibrated estimates alongside unchanged native wins/probabilities."""
    from f1sim.analysis.montecarlo import wilson_interval

    wins, no_winner = _native_counts(native)
    allocation = validate_teammate_allocation(allocation)
    if set(wins) != set(allocation["entrant_teams"]):
        raise ValueError("Native forecast must match its allocation roster")
    categories, probabilities, _, calibrated = _columns(native, allocation)
    rows = {}
    for driver in native["drivers"]:
        team = allocation["entrant_teams"][driver]
        weights = allocation["teams"][team]["weights"]
        if weights is None:
            count, scale = wins[driver], 1.0
        else:
            count = sum(wins[key] for key in weights)
            scale = weights[driver]
        interval = wilson_interval(count, native["trials"])
        rows[driver] = {
            "probability": calibrated[driver],
            "native_win_count": wins[driver],
            "native_probability": probabilities[driver],
            "mc_sampling_interval_95": {
                key: scale * value / 100 for key, value in interval.items()
            },
        }
    score_winner_probabilities(
        {driver: calibrated[driver] for driver in categories if driver is not None},
        calibrated[None],
        None,
    )
    return {
        "policy": POLICY,
        "trials": native["trials"],
        "drivers": rows,
        "no_classified_winner_probability": no_winner / native["trials"],
        "interval_metadata": {
            "confidence": 0.95,
            "method": "wilson_fixed_teammate_allocation_v1",
            "scope": "monte_carlo_sampling_with_frozen_point_history",
            "units": "fraction",
        },
        "allocation": allocation,
    }


def score_teammate_forecast(native, allocation, observed_winner):
    """Score the frozen linear forecast with its own finite-ensemble correction."""
    forecast = teammate_winner_forecast(native, allocation)
    score = score_winner_probabilities(
        {driver: row["probability"] for driver, row in forecast["drivers"].items()},
        forecast["no_classified_winner_probability"],
        observed_winner,
    )
    categories, probabilities, columns, calibrated = _columns(native, forecast["allocation"])
    n = native["trials"]
    adjustment = {
        "method": "finite_ensemble_linear_allocation_brier_v1",
        "trials": n,
        "scope": "monte_carlo_sampling_with_frozen_point_history_and_outcome",
        "trial_assumption": "independent_identically_distributed_native_winner_categories",
        "status": "unavailable" if n < 2 else "available",
        "reason": "at_least_two_trials_required" if n < 2 else None,
        "estimated_empirical_score_bias": None,
        "adjusted_brier_score": None,
        "adjusted_delta_from_uniform_baseline": None,
        "mc_standard_error": None,
        "mc_standard_error_method": "multinomial_plugin_linear_allocation_u_statistic_v1",
        "mc_standard_error_reason": "at_least_two_trials_required" if n < 2 else None,
        "observed_categories": sum(value > 0 for value in probabilities.values()),
    }
    score["mc_adjustment"] = adjustment
    if n < 2:
        return score
    expected_squared_length = fsum(
        probabilities[key] * fsum(value**2 for value in columns[key].values()) for key in categories
    )
    bias = max(0.0, expected_squared_length - fsum(value**2 for value in calibrated.values())) / (
        n - 1
    )
    adjustment.update(
        estimated_empirical_score_bias=bias,
        adjusted_brier_score=score["brier_score"] - bias,
        adjusted_delta_from_uniform_baseline=score["delta_from_uniform_baseline"] - bias,
    )
    if adjustment["observed_categories"] < 2:
        adjustment["mc_standard_error_reason"] = "single_observed_category"
        return score
    kernels = {}
    for a in categories:
        for b in categories:
            kernels[a, b] = (
                fsum(columns[a].get(key, 0.0) * columns[b].get(key, 0.0) for key in categories)
                - columns[a].get(observed_winner, 0.0)
                - columns[b].get(observed_winner, 0.0)
                + 1.0
            )
    conditional = {
        a: fsum(probabilities[b] * kernels[a, b] for b in categories) for a in categories
    }
    kernel_mean = fsum(probabilities[a] * conditional[a] for a in categories)
    zeta_one = fsum(probabilities[a] * (conditional[a] - kernel_mean) ** 2 for a in categories)
    zeta_two = fsum(
        probabilities[a] * probabilities[b] * (kernels[a, b] - kernel_mean) ** 2
        for a in categories
        for b in categories
    )
    variance = 4 * (n - 2) / (n * (n - 1)) * zeta_one + 2 / (n * (n - 1)) * zeta_two
    adjustment["mc_standard_error"] = sqrt(max(0.0, variance))
    return score
