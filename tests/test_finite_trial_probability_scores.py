"""Exact categorical sampling oracles for finite-trial winner score evidence."""

import copy
import importlib.util
import json
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from f1sim.analysis.race_probability_evaluation import rescore_saved_winner_evaluation
from f1sim.analysis.race_probability_scores import score_winner_counts, score_winner_probabilities

CATEGORIES = ("A", "B", None)


def pair_loss(outcomes, observed):
    """Independent average of actual vector inner products over unordered pairs."""
    losses = []
    for first, second in combinations(outcomes, 2):
        first_vector = [int(first == name) - int(observed == name) for name in CATEGORIES]
        second_vector = [int(second == name) - int(observed == name) for name in CATEGORIES]
        losses.append(sum(a * b for a, b in zip(first_vector, second_vector)))
    return Fraction(sum(losses), len(losses))


def sequence_probability(outcomes, probabilities):
    probability = Fraction(1)
    for outcome in outcomes:
        probability *= probabilities[CATEGORIES.index(outcome)]
    return probability


@pytest.mark.parametrize("trials", (2, 3, 5))
@pytest.mark.parametrize("observed", ("A", None))
@pytest.mark.parametrize("probabilities", (
    (Fraction(3, 5), Fraction(3, 10), Fraction(1, 10)),
    (Fraction(0), Fraction(1, 2), Fraction(1, 2)),
    (Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)),
))
def test_exact_repeated_sampling_removes_bias_including_unseen_observed_winner(
    trials, observed, probabilities,
):
    empirical_expectation = 0.0
    adjusted_expectation = 0.0
    expected_loss = sum((p - int(name == observed))**2
                        for name, p in zip(CATEGORIES, probabilities))
    expected_bias = (1 - sum(p * p for p in probabilities)) / trials
    for outcomes in product(CATEGORIES, repeat=trials):
        counts = Counter(outcomes)
        score = score_winner_counts({"A": counts["A"], "B": counts["B"]},
                                    counts[None], observed)
        weight = float(sequence_probability(outcomes, probabilities))
        adjustment = score["mc_adjustment"]
        assert adjustment["adjusted_brier_score"] == pytest.approx(
            float(pair_loss(outcomes, observed)),
        )
        assert 0 <= adjustment["adjusted_brier_score"] <= 2
        assert score["brier_score"] - adjustment["adjusted_brier_score"] == pytest.approx(
            adjustment["estimated_empirical_score_bias"], abs=1e-14,
        )
        empirical_expectation += weight * score["brier_score"]
        adjusted_expectation += weight * adjustment["adjusted_brier_score"]
    assert adjusted_expectation == pytest.approx(float(expected_loss), abs=1e-12)
    assert empirical_expectation == pytest.approx(float(expected_loss + expected_bias), abs=1e-12)


@pytest.mark.parametrize("counts", ((1, 1, 0), (2, 1, 0), (0, 2, 2), (1, 2, 1), (0, 1, 3)))
@pytest.mark.parametrize("observed", ("A", None))
def test_sampling_se_matches_exhaustive_variance_of_empirical_multinomial(counts, observed):
    trials = sum(counts)
    probabilities = tuple(Fraction(count, trials) for count in counts)
    expected_loss = sum((p - int(name == observed))**2
                        for name, p in zip(CATEGORIES, probabilities))
    exact_variance = sum(
        sequence_probability(outcomes, probabilities)
        * (pair_loss(outcomes, observed) - expected_loss)**2
        for outcomes in product(CATEGORIES, repeat=trials)
    )
    score = score_winner_counts({"A": counts[0], "B": counts[1]}, counts[2], observed)
    assert score["mc_adjustment"]["mc_standard_error"] == pytest.approx(sqrt(float(exact_variance)))


def test_second_order_variance_survives_zero_first_order_variance():
    # Observed winner never appears, and the other categories have equal mass.
    # The first-order delta method would report zero, although pair matches vary.
    score = score_winner_counts({"A": 0, "B": 2}, 2, "A")
    assert score["mc_adjustment"]["mc_standard_error"] == pytest.approx(sqrt(1 / 24))


def test_hand_computable_score_correction_and_no_category_name_collision():
    score = score_winner_counts({"A": 6, "no_classified_winner": 3}, 1, "A")
    adjustment = score["mc_adjustment"]
    assert score["brier_score"] == pytest.approx(0.26)
    assert adjustment["estimated_empirical_score_bias"] == pytest.approx(0.06)
    assert adjustment["adjusted_brier_score"] == pytest.approx(0.2)
    assert adjustment["adjusted_delta_from_uniform_baseline"] == pytest.approx(-0.3)
    assert adjustment["mc_standard_error"] == pytest.approx(sqrt(0.04584))
    assert json.loads(json.dumps(score, allow_nan=False)) == score
    assert score_winner_probabilities({"A": 0.6, "no_classified_winner": 0.3}, 0.1, "A") == {
        key: value for key, value in score.items() if key != "mc_adjustment"
    }


@pytest.mark.parametrize("counts, observed, expected", (
    ((4, 0, 0), "A", 0), ((0, 4, 0), "A", 2), ((0, 0, 4), None, 0),
))
def test_single_observed_category_does_not_claim_zero_sampling_uncertainty(
    counts, observed, expected,
):
    adjustment = score_winner_counts(
        {"A": counts[0], "B": counts[1]}, counts[2], observed,
    )["mc_adjustment"]
    assert adjustment["adjusted_brier_score"] == expected
    assert adjustment["estimated_empirical_score_bias"] == 0
    assert adjustment["mc_standard_error"] is None
    assert adjustment["mc_standard_error_reason"] == "single_observed_category"


def test_one_trial_retains_empirical_score_but_adjustment_is_unavailable():
    score = score_winner_counts({"A": 1, "B": 0}, 0, "A")
    assert score["brier_score"] == 0
    assert score["mc_adjustment"]["status"] == "unavailable"
    assert score["mc_adjustment"]["reason"] == "at_least_two_trials_required"
    assert score["mc_adjustment"]["adjusted_brier_score"] is None
    assert score["mc_adjustment"]["mc_standard_error"] is None


@pytest.mark.parametrize("wins, no_winner, observed", (
    ({}, 1, None), ({" A ": 1}, 0, " A "), ({False: 1}, 0, None),
    ({"A": True}, 0, "A"), ({"A": -1}, 2, "A"), ({"A": 1.0}, 0, "A"),
    ({"A": float("nan")}, 0, "A"), ({"A": float("inf")}, 0, "A"),
    ({"A": None}, 0, "A"), ({"A": 1}, False, "A"), ({"A": 1}, -1, "A"),
    ({"A": 1}, 0.0, "A"), ({"A": 0}, 0, None), ({"A": 1}, 0, "B"),
))
def test_invalid_counts_or_observed_category_fail(wins, no_winner, observed):
    with pytest.raises(ValueError):
        score_winner_counts(wins, no_winner, observed)


def test_numpy_integral_counts_are_normalized_to_json_numbers():
    score = score_winner_counts({"A": np.int64(6), "B": np.uint64(3)}, np.int64(1), "A")
    assert json.loads(json.dumps(score, allow_nan=False)) == score


def saved_report():
    return {
        "year": 2026,
        "evaluation": "round_holdout_race_winner_probabilities",
        "race_engine": "chronological",
        "provenance": {"urls": ["original.example"], "fetched_at": "original-date"},
        "folds": [{
            "round": 1, "status": "scored", "entrant_ids": ["A", "B"],
            "observed_outcome": {"status": "observed", "winner_id": "A"},
            "simulation_inputs": {"source_fingerprint": "original", "seed": 72},
            "forecast": {
                "trials": 10,
                "drivers": {"A": {"wins": 6, "probability": 0.6},
                            "B": {"wins": 3, "probability": 0.3}},
                "no_classified_winner": {"count": 1, "probability": 0.1},
            },
            "score": {"brier_score": 0.26},
        }, {"round": 2, "status": "excluded", "reason": "missing_position_one_result",
            "forecast": None, "score": None}],
    }


def test_offline_rescoring_preserves_inputs_and_unscored_events_without_mutation():
    original = saved_report()
    before = copy.deepcopy(original)
    report = rescore_saved_winner_evaluation(original)
    assert original == before
    assert report["folds"][0]["forecast"] == original["folds"][0]["forecast"]
    assert report["folds"][0]["simulation_inputs"] == original["folds"][0]["simulation_inputs"]
    assert report["provenance"] == original["provenance"]
    assert report["folds"][1] == original["folds"][1]
    assert report["aggregate"]["selected_events"] == 2
    assert report["aggregate"]["scored_events"] == report["aggregate"]["adjusted_events"] == 1
    assert report["aggregate"]["mean_adjusted_brier_score"] == pytest.approx(0.2)
    assert report["aggregate"]["mean_estimated_finite_trial_bias"] == pytest.approx(0.06)
    assert report["rescoring"]["additional_simulation_trials"] == 0


@pytest.mark.parametrize("mutation", (
    lambda report: report.update(evaluation="different"),
    lambda report: report["folds"].append(copy.deepcopy(report["folds"][0])),
    lambda report: report["folds"][0].update(status="unexpected"),
    lambda report: report["folds"][0].update(entrant_ids=["A", "A"]),
    lambda report: report["folds"][0]["forecast"].update(trials=11),
    lambda report: report["folds"][0]["forecast"]["drivers"]["A"].update(wins=5),
    lambda report: report["folds"][0]["forecast"]["drivers"]["A"].update(probability=0.5),
    lambda report: report["folds"][0]["observed_outcome"].update(winner_id="unknown"),
    lambda report: report["folds"][0].update(forecast=None),
    lambda report: report["folds"][1].update(score={"brier_score": 0}),
))
def test_corrupted_saved_evidence_is_rejected(mutation):
    report = saved_report()
    mutation(report)
    with pytest.raises(ValueError):
        rescore_saved_winner_evaluation(report)


def load_cli():
    path = Path(__file__).resolve().parents[1] / "examples/rescore_race_probabilities.py"
    spec = importlib.util.spec_from_file_location("rescore_probability_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_cli_outputs_complete_json_and_preserves_source_bytes(tmp_path, capsys):
    source = tmp_path / "saved.json"
    original = json.dumps(saved_report()).encode()
    source.write_bytes(original)
    load_cli().main([str(source)])
    output = capsys.readouterr()
    report = json.loads(output.out)
    assert output.err == ""
    assert source.read_bytes() == original
    assert report["rescoring"]["source_report_filename"] == "saved.json"
    assert len(report["rescoring"]["source_report_sha256"]) == 64
    assert report["aggregate"]["mean_adjusted_brier_score"] == pytest.approx(0.2)


@pytest.mark.parametrize("contents", ('{"evaluation":', '{"year":2026,"year":2027}', '{}'))
def test_cli_input_failure_emits_no_partial_json(tmp_path, capsys, contents):
    source = tmp_path / "invalid.json"
    source.write_text(contents)
    with pytest.raises(SystemExit) as exc:
        load_cli().main([str(source)])
    assert exc.value.code == 2
    assert capsys.readouterr().out == ""
