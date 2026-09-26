"""Winner probabilities retain no-winner mass and score all outcome classes."""

import json

import pytest

from f1sim.analysis.montecarlo import wilson_interval
from f1sim.analysis.race_probability_scores import (
    score_winner_probabilities,
    summarize_winner_trials,
)
from f1sim.simulation.race import DriverStatus, RaceResult


def _result(
    driver_id: str,
    position: int,
    status: DriverStatus = DriverStatus.FINISHED,
    *,
    classified: bool | None = None,
) -> RaceResult:
    return RaceResult(
        driver_id,
        f"Driver {driver_id}",
        "Team",
        position,
        90.0 + position,
        float(position - 1),
        0,
        91.0,
        status,
        classified=classified,
    )


def test_summary_counts_classified_winners_and_preserves_no_winner_mass():
    races = [
        # Classification, rather than the status name, controls whether P1 wins.
        [
            _result("A", 1, DriverStatus.DNF, classified=True),
            _result("B", 2),
        ],
        [
            _result("A", 2),
            _result("B", 1, DriverStatus.FINISHED, classified=False),
        ],
        [_result("A", 1), _result("B", 2)],
    ]

    summary = summarize_winner_trials(races, ["A", "B"], expected_trials=3)

    assert summary["trials"] == 3
    assert summary["drivers"]["A"]["wins"] == 2
    assert summary["drivers"]["A"]["probability"] == pytest.approx(2 / 3)
    assert summary["drivers"]["B"]["wins"] == 0
    assert summary["no_classified_winner"]["count"] == 1
    assert summary["no_classified_winner"]["probability"] == pytest.approx(1 / 3)
    assert summary["drivers"]["A"]["mc_sampling_interval_95"] == {
        key: value / 100 for key, value in wilson_interval(2, 3).items()
    }
    assert summary["interval_metadata"] == {
        "confidence": 0.95,
        "method": "wilson",
        "scope": "monte_carlo_sampling",
        "units": "fraction",
    }
    assert json.loads(json.dumps(summary)) == summary


def test_no_winner_category_does_not_collide_with_driver_id():
    summary = summarize_winner_trials(
        [[_result("no_classified_winner", 1), _result("B", 2)]],
        ["no_classified_winner", "B"],
        expected_trials=1,
    )

    assert summary["drivers"]["no_classified_winner"]["wins"] == 1
    assert summary["no_classified_winner"]["count"] == 0


@pytest.mark.parametrize(
    ("race_results", "driver_ids", "expected_trials"),
    [
        ([[_result("A", 1), _result("B", 2)]], ["A", "B"], 2),
        ([[_result("A", 1)]], ["A", "B"], 1),
        ([[_result("A", 1), _result("A", 2)]], ["A", "B"], 1),
        ([[_result("A", 1), _result("B", 1)]], ["A", "B"], 1),
        ([[_result("A", 1), _result("B", 3)]], ["A", "B"], 1),
        ([[_result("A", 1), _result("C", 2)]], ["A", "B"], 1),
        ([[_result("A", 1), _result("B", 2)]], ["A", "A"], 1),
        ([[_result("A", 1), _result("B", 2)]], ["A", " "], 1),
        ([[_result("A", 1), _result("B", 2)]], [], 1),
        ([[_result("A", 1), _result("B", 2)]], ["A", "B"], True),
        ([[_result("A", 1), _result("B", 2)]], ["A", "B"], 0),
    ],
)
def test_malformed_or_incomplete_trials_and_rosters_are_rejected(
    race_results, driver_ids, expected_trials
):
    with pytest.raises(ValueError):
        summarize_winner_trials(race_results, driver_ids, expected_trials)


def test_invalid_race_result_objects_raise_value_error():
    class MissingPosition:
        driver_id = "A"

    with pytest.raises(ValueError):
        summarize_winner_trials([[MissingPosition()]], ["A"], 1)


def test_multiclass_brier_and_driver_uniform_baseline_are_hand_computable():
    score = score_winner_probabilities(
        {"A": 0.6, "B": 0.3},
        no_winner_probability=0.1,
        observed_winner_id="A",
    )

    # (0.6 - 1)^2 + (0.3 - 0)^2 + (0.1 - 0)^2 = 0.26.
    # The two-driver baseline is (0.5, 0.5, 0), with Brier score 0.5.
    assert score["brier_score"] == pytest.approx(0.26)
    assert score["uniform_baseline_brier_score"] == pytest.approx(0.5)
    assert score["delta_from_uniform_baseline"] == pytest.approx(-0.24)


def test_perfect_wrong_and_no_winner_observations_are_scored_as_categories():
    perfect = score_winner_probabilities({"A": 1.0, "B": 0.0}, 0.0, "A")
    wrong = score_winner_probabilities({"A": 0.0, "B": 1.0}, 0.0, "A")
    no_winner = score_winner_probabilities({"A": 0.2, "B": 0.3}, 0.5, None)

    assert perfect["brier_score"] == 0
    assert perfect["delta_from_uniform_baseline"] == pytest.approx(-0.5)
    assert wrong["brier_score"] == 2
    assert wrong["delta_from_uniform_baseline"] == pytest.approx(1.5)
    assert no_winner["brier_score"] == pytest.approx(0.38)
    assert no_winner["uniform_baseline_brier_score"] == pytest.approx(1.5)
    assert no_winner["delta_from_uniform_baseline"] == pytest.approx(-1.12)


@pytest.mark.parametrize(
    ("driver_probabilities", "no_winner_probability", "observed_winner_id"),
    [
        ({}, 1.0, None),
        ({"A": True}, 0.0, "A"),
        ({"A": float("nan")}, 0.0, "A"),
        ({"A": float("inf")}, 0.0, "A"),
        ({"A": -0.1}, 1.1, "A"),
        ({"A": 1.1}, 0.0, "A"),
        ({"A": 0.5}, float("nan"), "A"),
        ({"A": 0.5, "B": 0.5}, 0.1, "A"),
        ({"A": 1.0}, 0.0, "B"),
        ({" A ": 1.0}, 0.0, " A "),
        ({"A": 1.0}, False, "A"),
    ],
)
def test_invalid_probabilities_or_observed_category_are_rejected(
    driver_probabilities, no_winner_probability, observed_winner_id
):
    with pytest.raises(ValueError):
        score_winner_probabilities(
            driver_probabilities,
            no_winner_probability,
            observed_winner_id,
        )


def test_small_probability_sum_rounding_error_is_tolerated_without_renormalizing():
    score = score_winner_probabilities(
        {"A": 0.1, "B": 0.2},
        0.7000000001,
        "A",
    )

    expected = (0.1 - 1) ** 2 + 0.2**2 + 0.7000000001**2
    assert score["brier_score"] == pytest.approx(expected)
