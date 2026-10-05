"""Historical comparisons enforce cutoffs and share the model's scored cohort."""

from copy import deepcopy

import pytest

from f1sim.analysis.race_probability_scores import score_winner_counts
from f1sim.analysis.winner_baselines import (
    build_winner_baselines,
    score_saved_winner_baselines,
    summarize_baseline_comparisons,
)


def references():
    return build_winner_baselines(
        {"A": "a", "B": "b"}, {"a": 30., "b": 10.},
        [{"round": 1, "status": "observed", "winner_id": "A"}], cutoff_round=1,
    )


def test_constructor_share_divides_team_points_between_modeled_members():
    teams = {"A2": "a", "B": "b", "A1": "a"}
    points = {"a": "60", "b": "40", "absent": "900"}
    prior = [{"round": 2, "status": "observed", "winner_id": "A1"},
             {"round": 1, "status": "observed", "winner_id": "B"},
             {"round": 3, "status": "excluded", "reason": "winner_outside_entrant_roster"}]
    before = deepcopy((teams, points, prior))
    result = build_winner_baselines(teams, points, prior, cutoff_round=3)
    assert (teams, points, prior) == before
    assert result["constructor_points_share"]["probabilities"] == {"A1": .3, "A2": .3, "B": .4}
    assert result["prior_race_wins_share"]["probabilities"] == {"A1": .5, "A2": 0., "B": .5}
    assert [row["round"] for row in result["prior_race_wins_share"]["evidence"]["prior_outcomes"]
            ] == [1, 2, 3]
    result["constructor_points_share"]["evidence"]["constructor_points"]["a"] = 999.
    assert result["prior_race_wins_share"]["evidence"]["constructor_points"]["a"] == 60.


@pytest.mark.parametrize("points,reason", [
    ({}, "no_prior_constructor_standings"),
    ({"a": 30.}, "missing_modeled_constructor_standings"),
    ({"a": 0., "b": 0.}, "no_positive_constructor_points"),
])
def test_missing_or_zero_history_is_unavailable_without_fabricating_a_forecast(points, reason):
    result = build_winner_baselines({"A": "a", "B": "b"}, points, [], cutoff_round=0)
    assert result["constructor_points_share"]["reason"] == reason
    for item in result.values():
        assert item["status"] == "unavailable" and item["probabilities"] is None
        assert item["no_classified_winner_probability"] is None and item["score"] is None


@pytest.mark.parametrize("points", [{"a": True}, {"a": float("nan")},
                                      {"a": float("inf")}, {"a": -1.}, {"a": "bad"}])
def test_invalid_points_cannot_enter_a_baseline(points):
    with pytest.raises(ValueError, match="points"):
        build_winner_baselines({"A": "a"}, points, [], cutoff_round=1)


@pytest.mark.parametrize("prior", [
    [{"round": 2, "status": "observed", "winner_id": "A"}],
    [{"round": True, "status": "observed", "winner_id": "A"}],
    [{"round": 1, "status": "observed", "winner_id": "outside"}],
    [{"round": 1, "status": "observed", "winner_id": "A"},
     {"round": 1, "status": "observed", "winner_id": "A"}],
])
def test_target_round_duplicate_or_unresolved_history_is_rejected(prior):
    with pytest.raises(ValueError):
        build_winner_baselines({"A": "a", "B": "b"}, {}, prior, cutoff_round=1)


def test_offline_scoring_rebuilds_scores_from_frozen_evidence():
    baseline = references()
    baseline["constructor_points_share"]["score"] = {"brier_score": 999.}
    before = deepcopy(baseline)
    result = score_saved_winner_baselines(baseline, ["A", "B"],
                                         target_round=2, observed_winner="B")
    assert baseline == before
    assert result["constructor_points_share"]["score"]["brier_score"] == 1.125
    assert result["prior_race_wins_share"]["score"]["brier_score"] == 2.
    unobserved = score_saved_winner_baselines(baseline, ["A", "B"],
                                             target_round=2, observed_winner=None)
    assert all(item["score"] is None for item in unobserved.values())


@pytest.mark.parametrize("mutation", ["policy", "cutoff", "probabilities", "points", "history"])
def test_saved_baselines_cannot_silently_change_the_reference_or_leak_target_labels(mutation):
    baseline = references()
    item = baseline["constructor_points_share"]
    if mutation == "policy":
        item["policy"] = "unknown_v2"
    elif mutation == "cutoff":
        item["cutoff_round"] = 2
    elif mutation == "probabilities":
        item["probabilities"] = {"A": .5, "B": .5}
    elif mutation == "points":
        item["evidence"]["constructor_points"]["a"] = 10.
    else:
        item["evidence"]["prior_outcomes"][0]["round"] = 2
    with pytest.raises(ValueError):
        score_saved_winner_baselines(baseline, ["A", "B"],
                                    target_round=2, observed_winner="A")


def test_aggregate_uses_identical_events_and_keeps_adjusted_mc_comparison_separate():
    baselines = score_saved_winner_baselines(references(), ["A", "B"],
                                            target_round=2, observed_winner="A")
    model = score_winner_counts({"A": 3, "B": 1}, 0, "A")
    folds = [{"status": "scored", "score": model, "baselines": baselines},
             {"status": "scored", "score": score_winner_counts({"A": 0, "B": 4}, 0, "A")},
             {"status": "excluded", "score": None, "baselines": baselines}]
    result = summarize_baseline_comparisons(folds)
    for item in result.values():
        assert item["selected_events"] == 3 and item["scored_events"] == 1
        assert item["unpaired_events"] == 2 and item["mean_model_brier_score"] == .125
        assert item["adjusted_events"] == 1
    assert result["constructor_points_share"]["mean_model_minus_baseline"] == 0.
    assert result["constructor_points_share"]["mean_adjusted_model_minus_baseline"] == -.125
    assert result["prior_race_wins_share"]["mean_model_minus_baseline"] == .125
    empty = summarize_baseline_comparisons([folds[1]])
    assert all(item["scored_events"] == 0 and item["mean_model_minus_baseline"] is None
               for item in empty.values())
