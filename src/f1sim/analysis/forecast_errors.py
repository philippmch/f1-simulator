"""Attribute probability-score losses without treating attribution as causation."""

import math

from f1sim.analysis.race_probability_scores import score_winner_probabilities


def compare_winner_errors(
    probabilities,
    reference,
    observed_winner,
    *,
    no_winner_probability=0.0,
    reference_no_winner_probability=0.0,
):
    """Separate underweighting the winner from probability mass on other outcomes.

    Positive contributions identify where the model loses Brier score against
    the fixed reference. They do not establish a physical cause of that loss.
    """
    if set(probabilities) != set(reference):
        raise ValueError("Winner error comparisons require the same entrant field")
    model_score = score_winner_probabilities(probabilities, no_winner_probability, observed_winner)
    reference_score = score_winner_probabilities(
        reference,
        reference_no_winner_probability,
        observed_winner,
    )
    rows = []
    for key in probabilities:
        outcome = float(key == observed_winner)
        rows.append(
            {
                "driver": key,
                "observed_win": bool(outcome),
                "model_probability": probabilities[key],
                "reference_probability": reference[key],
                "brier_difference": (probabilities[key] - outcome) ** 2
                - (reference[key] - outcome) ** 2,
            }
        )
    no_winner_difference = no_winner_probability**2 - reference_no_winner_probability**2
    winner_difference = (1 - probabilities[observed_winner]) ** 2 - (
        1 - reference[observed_winner]
    ) ** 2
    other_difference = (
        math.fsum(row["brier_difference"] for row in rows if not row["observed_win"])
        + no_winner_difference
    )
    return {
        "model_brier": model_score["brier_score"],
        "reference_brier": reference_score["brier_score"],
        "model_minus_reference_brier": model_score["brier_score"] - reference_score["brier_score"],
        "winner_probability_loss_difference": winner_difference,
        "other_outcome_loss_difference": other_difference,
        "no_winner_loss_difference": no_winner_difference,
        "drivers": sorted(rows, key=lambda row: (-row["brier_difference"], row["driver"])),
        "interpretation": "score attribution; physical causes require separate controlled tests",
    }
