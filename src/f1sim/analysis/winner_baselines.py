"""Fixed historical references, separate from simulated winner probabilities."""

from collections.abc import Mapping
from copy import deepcopy
from math import fsum, isfinite

from f1sim.analysis.race_probability_scores import score_winner_probabilities

BASELINE_POLICIES = {
    "constructor_points_share": "constructor_points_share_v1",
    "prior_race_wins_share": "prior_race_wins_share_v1",
}


def build_winner_baselines(entrant_teams, constructor_points, prior_outcomes, *, cutoff_round):
    """Freeze references using earlier evidence, without fitting a transformation.

    Constructor points are divided equally among modeled teammates. Prior win
    frequencies are conditional on resolved historical winners in this roster.
    Missing history is unavailable, rather than a uniform or zero forecast.
    """
    if type(cutoff_round) is not int or cutoff_round < 0:
        raise ValueError("baseline cutoff_round must be a nonnegative integer")
    if (not isinstance(entrant_teams, Mapping) or not entrant_teams
            or any(not isinstance(key, str) or not key.strip()
                   or not isinstance(team, str) or not team.strip()
                   for key, team in entrant_teams.items())):
        raise ValueError("baselines require an identified entrant roster and teams")
    if not isinstance(constructor_points, Mapping):
        raise ValueError("constructor_points must be a mapping")
    points = {}
    for team, value in constructor_points.items():
        if not isinstance(team, str) or not team or isinstance(value, bool):
            raise ValueError("constructor points must identify teams and finite nonnegative points")
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("constructor points must be finite and nonnegative") from exc
        if not isfinite(value) or value < 0:
            raise ValueError("constructor points must be finite and nonnegative")
        points[team] = value
    roster = dict(sorted(entrant_teams.items()))
    members = {team: sum(value == team for value in roster.values())
               for team in sorted(set(roster.values()))}
    if not isinstance(prior_outcomes, list):
        raise ValueError("prior_outcomes must be a list")
    outcomes, rounds = [], set()
    wins = dict.fromkeys(roster, 0)
    for outcome in prior_outcomes:
        if (not isinstance(outcome, Mapping) or type(outcome.get("round")) is not int
                or not 1 <= outcome["round"] <= cutoff_round or outcome["round"] in rounds
                or outcome.get("status") not in ("observed", "excluded")):
            raise ValueError("prior outcomes require unique rounds before the target")
        rounds.add(outcome["round"])
        if outcome["status"] == "observed":
            if outcome.get("winner_id") not in roster:
                raise ValueError("resolved prior winners must belong to the modeled roster")
            wins[outcome["winner_id"]] += 1
        outcomes.append(deepcopy(dict(outcome)))
    outcomes.sort(key=lambda item: item["round"])
    evidence = {"entrant_teams": roster, "constructor_points": dict(sorted(points.items())),
                "prior_outcomes": outcomes}

    def forecast(name, probabilities, reason):
        return {"policy": BASELINE_POLICIES[name], "cutoff_round": cutoff_round,
                "status": "available" if probabilities is not None else "unavailable",
                "reason": reason, "probabilities": probabilities,
                "no_classified_winner_probability": 0. if probabilities is not None else None,
                "evidence": deepcopy(evidence), "score": None}

    point_probabilities, point_reason = None, "no_prior_constructor_standings"
    if points:
        if not set(members) <= points.keys():
            point_reason = "missing_modeled_constructor_standings"
        elif (total := fsum(points[team] for team in members)) > 0:
            point_probabilities = {key: points[team] / total / members[team]
                                   for key, team in roster.items()}
            point_reason = None
        else:
            point_reason = "no_positive_constructor_points"
    win_total = sum(wins.values())
    win_probabilities = ({key: count / win_total for key, count in wins.items()}
                         if win_total else None)
    return {
        "constructor_points_share": forecast("constructor_points_share", point_probabilities,
                                              point_reason),
        "prior_race_wins_share": forecast("prior_race_wins_share", win_probabilities,
                                          None if win_total else "no_resolved_prior_winners"),
    }


def score_saved_winner_baselines(baselines, entrant_ids, *, target_round, observed_winner):
    """Rebuild recorded references and scores; never trust an edited saved score."""
    if not isinstance(baselines, Mapping) or set(baselines) != set(BASELINE_POLICIES):
        raise ValueError("winner baselines require both supported historical references")
    result = {}
    for name, saved in baselines.items():
        if (not isinstance(saved, Mapping) or saved.get("policy") != BASELINE_POLICIES[name]
                or type(saved.get("cutoff_round")) is not int
                or saved["cutoff_round"] != target_round - 1
                or not isinstance(saved.get("evidence"), Mapping)):
            raise ValueError("winner baseline policy and cutoff must match the target")
        evidence = saved["evidence"]
        if (not isinstance(evidence.get("entrant_teams"), Mapping)
                or set(evidence["entrant_teams"]) != set(entrant_ids)):
            raise ValueError("winner baseline evidence must match the forecast roster")
        rebuilt = build_winner_baselines(
            evidence["entrant_teams"], evidence.get("constructor_points"),
            evidence.get("prior_outcomes"), cutoff_round=saved["cutoff_round"],
        )[name]
        if any(saved.get(key) != rebuilt[key] for key in (
                "status", "reason", "probabilities", "no_classified_winner_probability")):
            raise ValueError("winner baseline probabilities must match recorded earlier evidence")
        if rebuilt["status"] == "available" and observed_winner is not None:
            rebuilt["score"] = score_winner_probabilities(
                rebuilt["probabilities"], rebuilt["no_classified_winner_probability"],
                observed_winner,
            )
        result[name] = rebuilt
    return result


def summarize_baseline_comparisons(folds):
    """Compare each reference and the simulator on identical scored events."""
    result = {}
    for name, policy in BASELINE_POLICIES.items():
        pairs = [(fold["score"], fold["baselines"][name]["score"]) for fold in folds
                 if fold.get("status") == "scored" and isinstance(fold.get("baselines"), Mapping)
                 and fold["baselines"].get(name, {}).get("score") is not None]
        count = len(pairs)
        adjusted = [(model["mc_adjustment"], baseline) for model, baseline in pairs
                    if model["mc_adjustment"]["status"] == "available"]
        result[name] = {
            "policy": policy, "selected_events": len(folds), "scored_events": count,
            "unpaired_events": len(folds) - count,
            "mean_model_brier_score": fsum(model["brier_score"] for model, _ in pairs) / count
            if count else None,
            "mean_baseline_brier_score": fsum(baseline["brier_score"] for _, baseline in pairs)
            / count if count else None,
            "mean_model_minus_baseline": fsum(
                model["brier_score"] - baseline["brier_score"] for model, baseline in pairs
            ) / count if count else None,
            "adjusted_events": len(adjusted),
            "mean_adjusted_model_minus_baseline": fsum(
                model["adjusted_brier_score"] - baseline["brier_score"]
                for model, baseline in adjusted) / len(adjusted) if adjusted else None,
            "event_weighting": "equal_weight_per_common_scored_event",
        }
    return result
