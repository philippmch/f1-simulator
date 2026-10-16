"""Conditional learned forecasts from complete, unchanged native race paths."""

from collections import Counter
from copy import deepcopy
from math import isclose, isfinite
from numbers import Integral, Real
from typing import Mapping

from f1sim.analysis.learned_afterqual import (
    AfterQualifyingFeatures,
    FrozenAfterQualifyingModel,
    canonical_digest,
    load_frozen_afterqual_model,
    predict_after_qualifying,
)
from f1sim.simulation.race import DriverStatus, result_is_classified

FORECAST_POLICY = "conditional_learned_winner_reverse_kl_v1"
NULL_WINNER = "__NO_CLASSIFIED_WINNER__"
PRIMARY_TIE_POLICY = "official_grid_rank_then_driver_id_null_last_v1"


def _probabilities(value, name):
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a nonempty probability mapping")
    checked = {}
    for key, number in value.items():
        if (not isinstance(key, str) or not key or isinstance(number, bool)
                or not isinstance(number, Real) or not isfinite(number)
                or not 0 <= number <= 1):
            raise ValueError(f"invalid {name} probability")
        checked[key] = float(number)
    if not isclose(sum(checked.values()), 1., abs_tol=1e-10):
        raise ValueError(f"{name} probabilities must sum to one")
    return checked


def project_winner_probabilities(native, analytic) -> tuple[dict, dict]:
    """Frozen KL(native||q) rule, including identity, ties and unsupported strata.

    The null stratum is fixed.  No pseudocounts, alternative tie thresholds,
    positive margin floors or changes to the learned ratios are introduced.
    """
    native = _probabilities(native, "native")
    analytic = _probabilities(analytic, "analytic")
    if NULL_WINNER not in native or set(native) - {NULL_WINNER} != set(analytic):
        raise ValueError("native and analytic driver fields must match, with one native null")
    drivers = sorted(analytic)
    learned_top = [driver for driver in drivers if analytic[driver] == max(analytic.values())]
    native_top = sorted(driver for driver in native if native[driver] == max(native.values()))
    diagnostic = {"native_top": native_top, "learned_top": learned_top, "active": []}

    def unsupported(reason):
        return dict(native), {**diagnostic, "status": "unsupported", "fallback": reason,
                              "identity": False}

    if len(learned_top) != 1:
        return unsupported("nonunique_analytic_top")
    j = learned_top[0]
    diagnostic["j"] = j
    diagnostic["native_j_minus_strongest_other_margin"] = native[j] - max(
        native[driver] for driver in native if driver != j
    )
    if j in native_top:
        return dict(native), {**diagnostic, "status": "identity", "identity": True,
                              "fallback": None}
    if native[j] == 0:
        return unsupported("native_j_zero")
    ratios = {driver: analytic[driver] / analytic[j] for driver in drivers if driver != j}
    if any(ratios[driver] == 0 and native[driver] > 0 for driver in ratios):
        return unsupported("zero_ratio_positive_native")
    ordered = sorted((driver for driver in ratios if ratios[driver] > 0),
                     key=lambda driver: (-native[driver] / ratios[driver], driver))
    active, mass, ratio_sum, t = [], native[j], 0., native[j]
    for driver in ordered:
        if native[driver] <= ratios[driver] * t:
            break
        active.append(driver)
        mass += native[driver]
        ratio_sum += ratios[driver]
        t = mass / (1 + ratio_sum)
    q = dict(native)
    q[j] = t
    for driver in active:
        q[driver] = ratios[driver] * t
    if (not isclose(sum(q.values()), sum(native.values()), abs_tol=1e-12)
            or q[NULL_WINNER] != native[NULL_WINNER]):
        raise ValueError("conditional projection failed its mass checks")
    multipliers = {driver: native[driver] / q[driver] - 1 for driver in active}
    stationarity = -native[j] / q[j] + 1 - sum(
        ratios[driver] * multipliers[driver] for driver in active
    )
    return q, {**diagnostic, "status": "projected", "fallback": None, "identity": False,
               "active": active, "ratios": ratios, "active_multipliers": multipliers,
               "j_stationarity_residual": stationarity}


def _joint_rows(races, codes):
    """Validate portable complete ordinal paths, retaining classification separately."""
    if not isinstance(races, list) or not races:
        raise ValueError("complete native joint outcomes are required")
    n = len(codes)
    output = []
    for race in races:
        if not isinstance(race, list) or len(race) != n:
            raise ValueError("every native trial must contain the complete driver field")
        rows = []
        for row in race:
            if not isinstance(row, Mapping):
                raise ValueError("native joint rows must be objects")
            position = row.get("position")
            if (isinstance(position, bool) or not isinstance(position, Integral)
                    or not 1 <= position <= n):
                raise ValueError("native joint positions must be complete ordinal ranks")
            if not isinstance(row.get("classified"), bool):
                raise ValueError("native joint classification must be explicit")
            status = row.get("status")
            if status not in {item.value for item in DriverStatus}:
                raise ValueError("invalid native joint operational status")
            rows.append({"driver_id": row.get("driver_id"), "position": int(position),
                         "status": status, "classified": row["classified"]})
        if (Counter(row["driver_id"] for row in rows) != Counter(codes)
                or {row["position"] for row in rows} != set(range(1, n + 1))):
            raise ValueError("native joint identities and positions must each be permutations")
        # A canonical row order makes aggregate replay independent of caller list order.
        by_code = {row["driver_id"]: row for row in rows}
        output.append([by_code[code] for code in codes])
    return output


def _build_from_joint(joint, features, artifact, eligibility):
    analytic = predict_after_qualifying(features, artifact)
    codes = features.codes
    n, trials = len(codes), len(joint)
    winners = [next((row["driver_id"] for row in race
                     if row["position"] == 1 and row["classified"]), NULL_WINNER)
               for race in joint]
    counts = {code: winners.count(code) for code in (*codes, NULL_WINNER)}
    native = {code: count / trials for code, count in counts.items()}
    q, diagnostic = project_winner_probabilities(native, analytic["winner_probabilities"])
    weights = [q[winner] / counts[winner] for winner in winners]
    if not isclose(sum(weights), 1., abs_tol=1e-10):
        raise ValueError("native path weights must sum to one")
    position_distributions = {code: {str(p): 0. for p in range(1, n + 1)} for code in codes}
    top_n = {code: {str(p): 0. for p in range(1, n + 1)} for code in codes}
    nonfinish = dict.fromkeys(codes, 0.)
    for race, weight in zip(joint, weights, strict=True):
        for row in race:
            code, position = row["driver_id"], row["position"]
            position_distributions[code][str(position)] += weight
            if row["classified"]:
                for cutoff in range(position, n + 1):
                    top_n[code][str(cutoff)] += weight
            if row["status"] != DriverStatus.FINISHED.value:
                nonfinish[code] += weight
    expected = {code: sum(int(position) * probability for position, probability
                          in position_distributions[code].items()) for code in codes}
    if not isclose(sum(expected.values()), n * (n + 1) / 2, abs_tol=1e-8):
        raise ValueError("weighted full ordinal expectations failed conservation")
    sum_squares = sum(weight * weight for weight in weights)
    maximum_weight = max(weights)
    projection_mass = sorted((q[code] for code in codes), reverse=True)
    code_by_id = dict(zip(features.ids, features.codes, strict=True))
    grid_rank = {code_by_id[identity]: rank for rank, identity in enumerate(features.grid_ids)}
    winner_top_set = sorted(
        (code for code, probability in q.items() if probability == max(q.values())),
        key=lambda code: (grid_rank.get(code, len(grid_rank)), code),
    )
    estimate = {
        "schema_version": 1, "policy": FORECAST_POLICY, "status": diagnostic["status"],
        "eligibility": deepcopy(eligibility), "num_simulations": trials,
        "artifact": artifact.to_dict(), "feature_snapshot": features.to_dict(),
        "feature_snapshot_sha256": canonical_digest(features.to_dict()),
        "analytic": analytic, "native_winner_counts": counts,
        "native_win_probabilities": native, "projection": diagnostic,
        "win_probabilities": {code: q[code] for code in codes},
        "no_classified_winner_probability": q[NULL_WINNER],
        "winner_top_set": winner_top_set, "primary_winner": winner_top_set[0],
        "primary_tie_policy": PRIMARY_TIE_POLICY,
        "podium_probabilities": {code: top_n[code][str(min(3, n))] for code in codes},
        "classified_top_n_probabilities": top_n,
        "position_distributions": position_distributions,
        "expected_positions": expected, "nonfinish_probabilities": nonfinish,
        "joint_outcomes": joint, "trial_weights": weights,
        "effective_sample_size": 1 / sum_squares,
        "maximum_normalized_trial_weight": maximum_weight,
        "maximum_importance_ratio": max(q[code] / native[code]
                                        for code in q if native[code] > 0),
        "minimum_positive_native_stratum_count": min(count for count in counts.values() if count),
        "minimum_reweighted_stratum_count": min(q[code] * trials for code in q if q[code] > 0),
        "projected_top_two_probability_margin": projection_mass[0] - projection_mass[1],
        "uncertainty": "point_estimates_only_no_calibrated_intervals",
        "native_paths_changed": False,
    }
    estimate["content_sha256"] = canonical_digest(estimate)
    return estimate


def build_learned_forecast(result, snapshot, artifact=None, *, eligibility=None) -> dict:
    """Build one coherent estimate without mutating native outcomes or counts.

    The public caller must establish the supported confirmed-grid stock scenario
    before running Monte Carlo.  This pure layer can check a supplied positive
    eligibility proof, but cannot infer that proof from a numerical matrix.
    """
    if getattr(result, "winner_allocation", None) is not None:
        raise ValueError("legacy winner allocation cannot be combined with the learned forecast")
    if not isinstance(eligibility, Mapping) or eligibility.get("eligible") is not True:
        raise ValueError("forecast eligibility proof must be explicitly positive")
    eligibility = deepcopy(dict(eligibility))
    features = (AfterQualifyingFeatures.from_mapping(snapshot.to_dict())
                if isinstance(snapshot, AfterQualifyingFeatures)
                else AfterQualifyingFeatures.from_mapping(snapshot))
    if artifact is None:
        artifact = load_frozen_afterqual_model()
    else:
        artifact = FrozenAfterQualifyingModel.from_mapping(
            artifact.to_dict() if isinstance(artifact, FrozenAfterQualifyingModel) else artifact
        )
    trials = getattr(result, "num_simulations", None)
    if isinstance(trials, bool) or not isinstance(trials, Integral) or trials <= 0:
        raise ValueError("a learned forecast requires a positive complete trial count")
    if len(result.race_results) != trials or set(result.driver_stats) != set(features.codes):
        raise ValueError("complete native joint trials and the same driver field are required")
    joint = _joint_rows([
        [{"driver_id": row.driver_id, "position": row.position,
          "status": getattr(row.status, "value", row.status),
          "classified": result_is_classified(row)} for row in race]
        for race in result.race_results
    ], features.codes)
    estimate = _build_from_joint(joint, features, artifact, eligibility)
    if estimate["status"] == "unsupported":
        raise ValueError("learned forecast is unsupported: " + estimate["projection"]["fallback"])
    for code, stats in result.driver_stats.items():
        if (stats.wins != estimate["native_winner_counts"][code]
                or len(stats.positions) != trials):
            raise ValueError("native driver counts do not match the saved complete joint paths")
    return estimate


build_confirmed_grid_forecast = build_learned_forecast


def validate_saved_learned_forecast(estimate) -> dict:
    """Reconstruct offline, allowing only numerical drift in derived floats.

    The saved seal and all input evidence remain exact.  Small differences in
    NumPy integration or matrix operations across supported environments may
    affect derived floating-point values.  Discrete favourites, trigger states,
    support and active sets must still agree exactly.  Return the original
    validated point values rather than silently replacing the recorded forecast.
    """
    if not isinstance(estimate, Mapping):
        raise ValueError("saved learned forecast must be an object")
    value = deepcopy(dict(estimate))
    if (value.get("policy") != FORECAST_POLICY or value.get("schema_version") != 1
            or value.get("status") not in ("identity", "projected")
            or value.get("content_sha256") != canonical_digest(
                {k: v for k, v in value.items() if k != "content_sha256"}
            )):
        raise ValueError("saved learned forecast policy or content seal is invalid")
    features = AfterQualifyingFeatures.from_mapping(value.get("feature_snapshot"))
    artifact = FrozenAfterQualifyingModel.from_mapping(value.get("artifact"))
    eligibility = value.get("eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("eligible") is not True:
        raise ValueError("saved forecast requires an explicit positive eligibility context")
    joint = _joint_rows(value.get("joint_outcomes"), features.codes)
    rebuilt = _build_from_joint(joint, features, artifact, eligibility)
    raw_keys = {
        "schema_version", "policy", "status", "eligibility", "num_simulations", "artifact",
        "feature_snapshot", "feature_snapshot_sha256", "native_winner_counts", "joint_outcomes",
        "uncertainty", "native_paths_changed",
    }
    if (set(value) != set(rebuilt)
            or canonical_digest({key: value[key] for key in raw_keys})
            != canonical_digest({key: rebuilt[key] for key in raw_keys})
            or not _same_numerical_values(
                {key: item for key, item in value.items()
                 if key not in raw_keys and key != "content_sha256"},
                {key: item for key, item in rebuilt.items()
                 if key not in raw_keys and key != "content_sha256"},
            )):
        raise ValueError("saved forecast does not match its native joint replay")
    return value


def _same_numerical_values(saved, rebuilt):
    if isinstance(saved, dict) and isinstance(rebuilt, dict):
        return (set(saved) == set(rebuilt)
                and all(_same_numerical_values(saved[key], rebuilt[key]) for key in saved))
    if isinstance(saved, list) and isinstance(rebuilt, list):
        return (len(saved) == len(rebuilt)
                and all(_same_numerical_values(a, b) for a, b in zip(saved, rebuilt, strict=True)))
    if (type(saved) is float and type(rebuilt) is float
            and isfinite(saved) and isfinite(rebuilt)):
        return isclose(saved, rebuilt, rel_tol=1e-12, abs_tol=1e-12)
    return type(saved) is type(rebuilt) and saved == rebuilt


recompute_saved_learned_forecast = validate_saved_learned_forecast
