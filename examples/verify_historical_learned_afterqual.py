"""Verify the compact historical development evidence, offline and without races.

Run from a checkout or an environment with f1sim installed.  This reconstructs
the frozen per-event inference and forecast weighting; it never fetches feeds,
trains models or simulates races.  The archive is retrospective development
evidence with logical information cutoffs, not authenticated contemporary
forecasts, untouched validation or prospective accuracy evidence.
"""

import argparse
import gzip
import json
from collections import Counter
from math import isclose, log
from pathlib import Path

import numpy as np

from f1sim.analysis.learned_afterqual import canonical_digest, predict_after_qualifying
from f1sim.analysis.learned_forecast import (
    NULL_WINNER,
    PRIMARY_TIE_POLICY,
    build_learned_forecast,
    project_winner_probabilities,
)
from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.simulation.race import DriverStatus, RaceResult

KIND = "historical_development_learned_afterqual_v1"
PREFIXES = (32, 128, 400)
TOLERANCES = {"position_mae": .05, "podium_brier": .002, "nonfinish_brier": .002}
LIMITATION = (
    "Historical development evidence with logical postqualifying cutoffs; not recorded before "
    "the actual events, not untouched validation, and not prospective forecasting skill."
)


def close_values(actual, expected, path="value"):
    """Keep discrete evidence exact; allow only tight numerical float drift."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{path}: fields differ")
        for key in expected:
            close_values(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"{path}: list length differs")
        for index, (a, b) in enumerate(zip(actual, expected, strict=True)):
            close_values(a, b, f"{path}[{index}]")
    elif type(expected) is float:
        if type(actual) not in (int, float) or not isclose(
            actual, expected, abs_tol=1e-12, rel_tol=1e-12,
        ):
            raise ValueError(f"{path}: numerical value differs: {actual!r} versus {expected!r}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path}: discrete value differs")


def aggregate_paths(paths, codes):
    """Count only supplied paths, keeping operational status and classification distinct."""
    count = len(paths)
    wins = dict.fromkeys(codes, 0)
    marginals = {code: {"position_sum": 0, "podium_count": 0, "nonfinish_count": 0}
                 for code in codes}
    null, categories = 0, []
    for race in paths:
        if (len(race) != len(codes) or {row["driver_id"] for row in race} != set(codes)
                or sorted(row["position"] for row in race) != list(range(1, len(codes) + 1))):
            raise ValueError("archived joint path is not a complete field permutation")
        winner = next((row["driver_id"] for row in race
                       if row["position"] == 1 and row["classified"]), NULL_WINNER)
        categories.append(winner)
        if winner == NULL_WINNER:
            null += 1
        else:
            wins[winner] += 1
        for row in race:
            if type(row["classified"]) is not bool:
                raise ValueError("minimal joint rows require explicit classification")
            value = marginals[row["driver_id"]]
            value["position_sum"] += row["position"]
            value["podium_count"] += int(row["position"] <= 3 and row["classified"])
            value["nonfinish_count"] += int(row["status"] != "finished")
    return {"trials": count, "wins": wins, "no_classified_winner_count": null,
            "marginals": marginals}, categories


def _probabilities(aggregate):
    n = aggregate["trials"]
    wins, null, marginals = (aggregate["wins"], aggregate["no_classified_winner_count"],
                             aggregate["marginals"])
    if (type(n) is not int or n <= 0 or not wins or set(wins) != set(marginals)
            or any(type(value) is not int or not 0 <= value <= n
                   for value in [*wins.values(), null])
            or sum(wins.values()) + null != n):
        raise ValueError("archived native winner counts must be complete integer counts")
    for value in marginals.values():
        if (type(value["position_sum"]) is not int
                or not n <= value["position_sum"] <= n * len(wins)
                or any(type(value[key]) is not int or not 0 <= value[key] <= n
                       for key in ("podium_count", "nonfinish_count"))):
            raise ValueError("archived native marginals must be valid integer sums")
    if sum(value["position_sum"] for value in marginals.values()) != (
        n * len(wins) * (len(wins) + 1) // 2
    ):
        raise ValueError("archived native ordinal position sums must be conserved")
    return {**{code: count / n for code, count in aggregate["wins"].items()},
            NULL_WINNER: aggregate["no_classified_winner_count"] / n}


def _native_marginals(aggregate):
    n = aggregate["trials"]
    return {code: {"expected_position": value["position_sum"] / n,
                   "podium_probability": value["podium_count"] / n,
                   "nonfinish_probability": value["nonfinish_count"] / n}
            for code, value in aggregate["marginals"].items()}


def forecast_from_paths(paths, features, artifact):
    """Exercise the production public forecast builder on archived native paths."""
    codes = features["codes"]
    aggregate, categories = aggregate_paths(paths, codes)
    races = [[RaceResult(
        row["driver_id"], row["driver_id"], "archived", row["position"], 0., 0., 0, 0.,
        DriverStatus(row["status"]), classified=row["classified"],
    ) for row in race] for race in paths]
    stats = {code: DriverStatistics(
        code, code, "archived", wins=aggregate["wins"][code],
        positions=[next(row["position"] for row in race if row["driver_id"] == code)
                   for race in paths],
    ) for code in codes}
    native = SimulationResults(len(paths), "Historical audit", stats, races, [], parallel=False,
                               race_engine="chronological")
    estimate = build_learned_forecast(
        native, features, artifact,
        eligibility={"eligible": True, "scope": "historical_production_equivalence_only",
                     "contemporary_availability_claimed": False},
    )
    # Independently reconstruct the extra complete-position and classified
    # top-N marginals that were not retained in the original study report.
    for code in codes:
        rows = [next(row for row in race if row["driver_id"] == code) for race in paths]
        positions = np.asarray([row["position"] for row in rows])
        weights = np.asarray(estimate["trial_weights"])
        distribution = np.bincount(positions, weights=weights, minlength=len(codes) + 1)
        close_values(estimate["position_distributions"][code],
                     {str(p): float(distribution[p]) for p in range(1, len(codes) + 1)})
        classified = np.asarray([row["classified"] for row in rows])
        close_values(estimate["classified_top_n_probabilities"][code], {
            str(p): float(weights[(positions <= p) & classified].sum())
            for p in range(1, len(codes) + 1)
        })
    q = {**estimate["win_probabilities"],
         NULL_WINNER: estimate["no_classified_winner_probability"]}
    mapping = dict(zip(features["ids"], features["codes"], strict=True))
    rank = {mapping[identity]: index for index, identity in enumerate(features["grid_ids"])}
    top = sorted((code for code, probability in q.items() if probability == max(q.values())),
                 key=lambda code: (rank.get(code, len(rank)), code))
    close_values(estimate["winner_top_set"], top)
    close_values(estimate["primary_winner"], top[0])
    close_values(estimate["primary_tie_policy"], PRIMARY_TIE_POLICY)
    return aggregate, {
        "winner_probabilities": q,
        "marginals": {code: {"expected_position": estimate["expected_positions"][code],
                              "podium_probability": estimate["podium_probabilities"][code],
                              "nonfinish_probability": estimate["nonfinish_probabilities"][code]}
                      for code in codes},
        "diagnostic": {key: estimate["projection"].get(key) for key in
                       ("identity", "fallback", "j", "active")},
        "normalized_trial_weights": estimate["trial_weights"],
        "trial_winner_categories": categories,
        "ESS": estimate["effective_sample_size"],
        "maximum_normalized_trial_weight": estimate["maximum_normalized_trial_weight"],
        "minimum_positive_winner_stratum_count": estimate["minimum_positive_native_stratum_count"],
        "unsupported": estimate["status"] == "unsupported", "joint_rows_available": True,
    }


def forecast_identity_aggregate(aggregate, analytic):
    """Verify identity from known aggregates; never fabricate missing joint paths."""
    p = _probabilities(aggregate)
    q, diagnostic = project_winner_probabilities(p, analytic)
    if diagnostic["status"] != "identity" or q != p:
        raise ValueError("an archived aggregate-only forecast must remain exact identity")
    n = aggregate["trials"]
    return {
        "winner_probabilities": q, "marginals": _native_marginals(aggregate),
        "diagnostic": {key: diagnostic.get(key) for key in ("identity", "fallback", "j", "active")},
        "ESS": float(n), "maximum_normalized_trial_weight": 1 / n,
        "minimum_positive_winner_stratum_count": min(count for count in
            [*aggregate["wins"].values(), aggregate["no_classified_winner_count"]] if count),
        "unsupported": False, "joint_rows_available": False,
        "uniform_weight_formula_only": f"1/{n}; original joint{n} rows were not retained",
    }


def prefix_diagnostic(aggregate, guarded, analytic, grid):
    p, q = _probabilities(aggregate), guarded["winner_probabilities"]
    j = max(analytic, key=analytic.get)
    ranks = {driver: index for index, driver in enumerate(grid)}
    sets = {name: sorted(code for code, value in probabilities.items()
                         if value == max(probabilities.values()))
            for name, probabilities in (("native", p), ("guard", q))}
    picks = {name: min(top, key=lambda code: (ranks.get(code, len(ranks)), code))
             for name, top in sets.items()}
    return {
        "learned_j": j, "native_j_count": aggregate["wins"][j],
        "native_j_minus_strongest_other_probability": p[j] - max(
            value for code, value in p.items() if code != j),
        "native_j_in_argmax": j in sets["native"], "top_sets": sets, "primary_picks": picks,
        "identity": guarded["diagnostic"]["identity"],
        "fallback": guarded["diagnostic"]["fallback"],
        "q": q, "ESS": guarded["ESS"],
        "maximum_importance_ratio": max(q[code] / value for code, value in p.items() if value),
        "maximum_normalized_trial_weight": guarded["maximum_normalized_trial_weight"],
        "minimum_positive_winner_stratum_count": guarded["minimum_positive_winner_stratum_count"],
        "unsupported": guarded["unsupported"],
    }


def _score(probabilities, marginals, observed, grid):
    winner, labels = observed["winner"], observed["drivers"]
    top = sorted(code for code, value in probabilities.items()
                 if value == max(probabilities.values()))
    ranks = {code: index for index, code in enumerate(grid)}
    pick = min(top, key=lambda code: (ranks.get(code, len(ranks)), code))
    return {
        "winner_brier": sum(value ** 2 for value in probabilities.values())
        + 1 - 2 * probabilities[winner],
        "winner_log_loss": -log(probabilities[winner]) if probabilities[winner] else None,
        "top_set": top, "primary_pick": pick, "correct": int(pick == winner),
        "position_mae": float(np.mean([abs(value["expected_position"] - labels[code]["position"])
                                        for code, value in marginals.items()])),
        "podium_brier": float(np.mean([(value["podium_probability"] - labels[code]["podium"]) ** 2
                                        for code, value in marginals.items()])),
        "nonfinish_brier": float(np.mean([
            (value["nonfinish_probability"] - labels[code]["dnf"]) ** 2
            for code, value in marginals.items()])),
    }


def summarize_cohort(rows):
    summaries = {}
    for year in (None, 2025, 2026):
        selected = [row for row in rows if year is None or row["year"] == year]
        models = {name: {
            "correct": sum(row[name]["correct"] for row in selected),
            **{metric: float(np.mean([row[name][metric] for row in selected]))
               for metric in ("winner_brier", *TOLERANCES)},
            "tied_events": sum(len(row[name]["top_set"]) > 1 for row in selected),
        } for name in ("native", "guard")}
        summaries[str(year or "all")] = {
            "events": len(selected), "models": models,
            "grid18": {"correct": sum(row["grid_correct"] for row in selected),
                       "winner_brier": float(np.mean([row["grid18_brier"] for row in selected]))},
            "minimum_ESS": min(row["ESS"] for row in selected),
            "unsupported": sum(row["unsupported"] for row in selected),
        }
    native, guard = [summaries["all"]["models"][name] for name in ("native", "guard")]
    changes = {metric: guard[metric] - native[metric] for metric in TOLERANCES}
    gates = {
        "top1_at_least28": guard["correct"] >= 28,
        "winnerBrier_below_native_and_grid": guard["winner_brier"] < min(
            native["winner_brier"], summaries["all"]["grid18"]["winner_brier"]),
        "neither_year_winnerBrier_worse_native": all(
            summaries[str(year)]["models"]["guard"]["winner_brier"]
            <= summaries[str(year)]["models"]["native"]["winner_brier"] for year in (2025, 2026)),
        **{metric + "_within_fixed_materiality": changes[metric] <= tolerance
           for metric, tolerance in TOLERANCES.items()},
        "no_unsupported_strata_or_fallback": summaries["all"]["unsupported"] == 0,
    }
    return {"summaries": summaries, "gates": gates,
            "guard_minus_native_auxiliary_changes": changes,
            "pre_execution_materiality_tolerances": dict(TOLERANCES),
            "passes_later400_gate": all(gates.values())}


def replay_saved_count_draws(draws, categories, aggregate, analytic, point_guard, grid):
    """Replay saved resample counts through the frozen production projection.

    This generates no new samples and no simulated race paths.  Unsupported
    resamples retain the frozen diagnostic fallback, separate from projections.
    """
    if categories != sorted(aggregate["wins"]) + [NULL_WINNER] or not draws:
        raise ValueError("saved bootstrap categories must match the complete native histogram")
    native_counts = {**aggregate["wins"], NULL_WINNER: aggregate["no_classified_winner_count"]}
    n = aggregate["trials"]
    qs, identities, fallbacks, top_sets, picks = [], [], Counter(), Counter(), Counter()
    ranks = {driver: index for index, driver in enumerate(grid)}
    for counts in draws:
        if (len(counts) != len(categories) or sum(counts) != n
                or any(type(count) is not int or count < 0 for count in counts)
                or any(count and not native_counts[code]
                       for code, count in zip(categories, counts, strict=True))):
            raise ValueError("saved bootstrap draws must be legal integer histogram resamples")
        sampled = dict(zip(categories, (count / n for count in counts), strict=True))
        q, diagnostic = project_winner_probabilities(sampled, analytic)
        qs.append([q[code] for code in categories])
        identities.append(diagnostic.get("identity", False))
        if diagnostic["fallback"] is not None:
            fallbacks[diagnostic["fallback"]] += 1
        top = tuple(sorted(code for code, value in q.items() if value == max(q.values())))
        top_sets[top] += 1
        picks[min(top, key=lambda code: (ranks.get(code, len(ranks)), code))] += 1
    array = np.asarray(qs)
    point = np.asarray([point_guard["winner_probabilities"][code] for code in categories])
    deviations = np.max(np.abs(array - point), axis=1)
    identity_count = sum(identities)
    nonidentity = len(draws) - identity_count
    actual_projection = nonidentity - sum(fallbacks.values())
    return {
        "identity_count": identity_count, "projection_count": actual_projection,
        "identity_frequency": float(np.mean(identities)),
        "point_identity": point_guard["diagnostic"]["identity"],
        "sampled_trigger_discontinuity": 0 < identity_count < len(draws),
        "fallback_counts": dict(fallbacks),
        "exact_top_sets": [{"drivers": list(top), "replicates": count}
                           for top, count in sorted(top_sets.items())],
        "q_quantiles": {code: np.quantile(array[:, index], [0, .025, .5, .975, 1]).tolist()
                        for index, code in enumerate(categories)},
        "q_mean": dict(zip(categories, array.mean(axis=0).tolist(), strict=True)),
        "maximum_absolute_q_deviation_quantiles": np.quantile(
            deviations, [0, .025, .5, .975, 1]).tolist(),
        "not_a_retuned_forecast": True, "nonidentity_count": nonidentity,
        "actual_projection_count": actual_projection,
        "primary_pick_counts": dict(sorted(picks.items())),
    }


def verify_capsule(document):
    if (document.get("kind") != KIND or document.get("schema_version") != 1
            or document.get("content_sha256") != canonical_digest(
                {key: value for key, value in document.items() if key != "content_sha256"})):
        raise ValueError("historical evidence format or content seal is invalid")
    if (document.get("scope_limitation") != LIMITATION or len(document["events"]) != 45
            or document.get("fixed_auxiliary_tolerances") != TOLERANCES):
        raise ValueError("historical evidence scope or frozen gate specification differs")
    bootstrap_receipt = _verify_receipt_chain(document)
    all_rows = {str(prefix): {} for prefix in PREFIXES}
    bootstrap_summaries = {}
    joint400, aggregate400 = 0, 0
    for event in document["events"]:
        identity = event["id"]
        features = document["features"][event["feature_key"]]
        artifact = document["artifacts"][event["artifact_key"]]
        analytic = predict_after_qualifying(features, artifact)["winner_probabilities"]
        close_values(analytic, event["saved_analytic"], identity + ".analytic")
        paths = event["minimal_joint_paths"]
        full = event["joint_path_count"] == 400
        if len(paths) != (400 if full else 128):
            raise ValueError("minimal joint path count disagrees with its explicit scope")
        joint400 += full
        aggregate400 += not full
        for count in PREFIXES:
            if count <= len(paths):
                aggregate, guarded = forecast_from_paths(paths[:count], features, artifact)
            else:
                aggregate = event["archived_identity400_aggregates"]
                if aggregate["trials"] != 400:
                    raise ValueError("aggregate-only native forecast must contain400 trials")
                guarded = forecast_identity_aggregate(aggregate, analytic)
            expected = event["expected_prefixes"][str(count)]
            close_values(aggregate, expected["native_aggregates"], identity + f".{count}.native")
            close_values(guarded, expected["guard"], identity + f".{count}.guard")
            diagnostic = prefix_diagnostic(aggregate, guarded, analytic, event["grid"])
            close_values(diagnostic, expected["stability"], identity + f".{count}.stability")
            if count == 400:
                bootstrap = document["saved_count_bootstrap"][identity]
                if len(bootstrap["native_count_draws"]) != bootstrap_receipt["configuration"][
                    "replicates_per_model_event"
                ]:
                    raise ValueError("bootstrap count differs from its frozen configuration")
                summary = replay_saved_count_draws(
                    bootstrap["native_count_draws"], bootstrap["categories"], aggregate,
                    analytic, guarded, event["grid"],
                )
                close_values(summary, bootstrap_receipt["summaries"][identity],
                             identity + ".bootstrap")
                bootstrap_summaries[identity] = summary
            observed = document["observations"][event["observation_key"]]
            grid = event["grid"]
            weights = np.exp(-18 * np.arange(len(grid)) / (len(grid) - 1))
            gridp = dict(zip(grid, weights / weights.sum(), strict=True))
            all_rows[str(count)][identity] = {
                "id": identity, "year": features["year"], "round": features["round"],
                "native": _score(_probabilities(aggregate), _native_marginals(aggregate),
                                 observed, grid),
                "guard": _score(guarded["winner_probabilities"], guarded["marginals"],
                                observed, grid),
                "grid18_brier": sum(value ** 2 for value in gridp.values())
                + 1 - 2 * gridp[observed["winner"]],
                "grid_correct": int(grid[0] == observed["winner"]),
                "ESS": guarded["ESS"], "unsupported": guarded["unsupported"],
            }
    if (joint400, aggregate400) != (11, 34):
        raise ValueError("evidence must retain eleven joint400 and34 aggregate-only400 events")
    results = {}
    for count in PREFIXES:
        rows = all_rows[str(count)]
        original = sorted(
            (row for identity, row in rows.items() if identity.startswith("original-")),
            key=lambda row: (row["year"], row["round"]),
        )
        corrected = [rows.get(row["id"].replace("original-", "corrected-"), row)
                     for row in original]
        if len(original) != 40 or len(corrected) != 40:
            raise ValueError("both complete cohorts must contain forty events")
        results[str(count)] = {name: summarize_cohort(cohort) for name, cohort in
                               (("original40", original), ("corrected40", corrected))}
    close_values(results, document["expected_cohort_results"], "cohort_results")
    unstable = [identity for identity, summary in bootstrap_summaries.items()
                if summary["sampled_trigger_discontinuity"]]
    return {"verified": True, "events": 45, "full_joint400_events": joint400,
            "aggregate_only_identity400_events": aggregate400,
            "prefixes": list(PREFIXES), "cohorts": results,
            "both_complete400_cohorts_pass": all(
                result["passes_later400_gate"] for result in results["400"].values()),
            "bootstrap_stability": {
                "saved_count_draws_replayed": sum(
                    len(row["native_count_draws"])
                    for row in document["saved_count_bootstrap"].values()),
                "new_resampling": 0, "events_with_trigger_discontinuity": unstable,
                "maximum_event_97_5_percentile_probability_deviation": max(
                    row["maximum_absolute_q_deviation_quantiles"][3]
                    for row in bootstrap_summaries.values()),
                "per_event": {identity: {key: row[key] for key in (
                    "identity_count", "actual_projection_count", "fallback_counts",
                    "primary_pick_counts", "maximum_absolute_q_deviation_quantiles",
                )} for identity, row in bootstrap_summaries.items()},
                "scope": "sampling variation conditional on the saved native histogram; "
                         "not learned/source uncertainty or calibrated real-race intervals",
            },
            "races_run": 0, "fits": 0, "network_requests": 0,
            "scope_limitation": LIMITATION, "evidence_sha256": document["content_sha256"]}


def _verify_receipt_chain(document):
    receipts = document.get("source_receipts")
    if not isinstance(receipts, dict) or not receipts:
        raise ValueError("historical evidence requires its frozen source receipt chain")
    for name, receipt in receipts.items():
        value = receipt["document"]
        if receipt["canonical_document_sha256"] != canonical_digest(value):
            raise ValueError(f"receipt content mismatch: {name}")
        if "content_sha256" in value and value["content_sha256"] != canonical_digest(
            {key: item for key, item in value.items() if key != "content_sha256"}
        ):
            raise ValueError(f"receipt seal mismatch: {name}")

    def named(suffix):
        matches = [receipt["document"] for name, receipt in receipts.items()
                   if name.endswith("/" + suffix)]
        if len(matches) != 1:
            raise ValueError(f"expected one frozen receipt: {suffix}")
        return matches[0]

    plan = named("plan-stability-reporting-before-continuation.json")
    manifest = named("all45-before-scoring.json")
    parity = named("six-original400-parity.json")
    preflight = named("independent-preflight.json")
    bootstrap = named("all45-bootstrap-before-scoring.json")
    production = named("production-equivalence.json")
    assessment = named("assessment.json")
    independent = named("independent-joint-audit.json")
    delivery = named("delivery-recommendation.json")
    stability = named("stability-and-grid-summary.json")
    if (not preflight["pass"] or not parity["pass"] or not production["pass"]
            or manifest["scoring_performed"] is not False
            or bootstrap["scoring_performed"] is not False or len(bootstrap["files"]) != 45
            or preflight["plan_sha256"] != plan["content_sha256"]
            or manifest["plan_sha256"] != plan["content_sha256"]
            or manifest["six_original400_parity_sha256"] != parity["content_sha256"]
            or bootstrap["all45_checkpoint_sha256"] != manifest["content_sha256"]
            or production["all45_checkpoint_sha256"] != manifest["content_sha256"]
            or assessment["all45_checkpoint_sha256"] != manifest["content_sha256"]
            or document["assessment_sha256"] != assessment["content_sha256"]
            or independent["assessment_sha256"] != assessment["content_sha256"]
            or not independent["pass"] or not delivery["pass"]
            or delivery["audit_sha256"] != independent["content_sha256"]
            or delivery["production_equivalence_sha256"] != production["content_sha256"]
            or stability["assessment_sha256"] != assessment["content_sha256"]
            or stability["bootstrap_sha256"] != bootstrap["content_sha256"]
            or stability["independent_audit_sha256"] != independent["content_sha256"]
            or document["production_equivalence_sha256"] != production["content_sha256"]
            or document["bootstrap_checkpoint_sha256"] != bootstrap["content_sha256"]):
        raise ValueError("frozen receipt lineage differs")
    entries = {row["id"]: row for row in manifest["forecasts"]}
    if len(entries) != 45 or set(entries) != {row["id"] for row in document["events"]}:
        raise ValueError("portable event membership differs from the complete frozen manifest")
    for row in document["events"]:
        source = entries[row["id"]]
        if (source["content_sha256"] != row["original_forecast_content_sha256"]
                or source["byte_sha256"] != row["original_forecast_byte_sha256"]):
            raise ValueError("portable forecast source seal differs from its frozen manifest")
    if set(document.get("saved_count_bootstrap", {})) != set(entries):
        raise ValueError("saved bootstrap membership must match the complete frozen manifest")
    for source in bootstrap["files"]:
        saved = document["saved_count_bootstrap"][source["id"]]
        if (saved["original_content_sha256"] != source["content_sha256"]
                or saved["original_byte_sha256"] != source["byte_sha256"]):
            raise ValueError("saved count source differs from the complete bootstrap manifest")
    expected = {}
    keys = ("summaries", "gates", "guard_minus_native_auxiliary_changes",
            "pre_execution_materiality_tolerances", "passes_later400_gate")
    for prefix, row in assessment["prefixes"].items():
        expected[prefix] = {}
        for name, cohort in row["cohorts"].items():
            selected = {key: cohort[key] for key in keys}
            selected["summaries"] = {
                split: {key: value for key, value in summary.items()
                        if key != "paired_native_minus_guard"}
                for split, summary in cohort["summaries"].items()
            }
            expected[prefix][name] = selected
    close_values(document["expected_cohort_results"], expected, "assessment_lineage")
    return bootstrap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=(
        Path(__file__).resolve().parents[1] / "evidence"
        / "historical-learned-afterqual-2026-10-16.json.gz"
    ))
    args = parser.parse_args()
    with gzip.open(args.path, "rt", encoding="utf-8") as stream:
        document = json.load(stream)
    result = verify_capsule(document)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
