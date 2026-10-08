"""The prospective reference is fixed before results and reproducible after them."""

from copy import deepcopy

import pytest

from f1sim.analysis.forecast_errors import compare_winner_errors
from f1sim.analysis.practice_winner_reference import (
    build_practice_winner_reference,
    practice_rank_probabilities,
    score_saved_practice_reference,
)


def practice():
    return {
        "year": 2026,
        "round": 17,
        "session_number": 1,
        "practice_started_at": "2026-10-09T08:30:00Z",
        "fetched_at": "2026-10-09T09:35:00Z",
        "source_url": "https://example.org/practice/1",
        "rows": [
            {"driver": "A", "position": 1, "lap_seconds": 90.0, "laps": 20},
            {"driver": "B", "position": 2, "lap_seconds": 91.0, "laps": 19},
        ],
    }


def build(evidence=None):
    return build_practice_winner_reference(
        ["A", "B", "C"],
        evidence if evidence is not None else practice(),
        year=2026,
        target_round=17,
        recorded_at="2026-10-09T09:40:00Z",
        qualifying_starts_at="2026-10-09T12:30:00Z",
    )


def test_reference_matches_fixed_historical_formula_and_missing_middle_rank():
    before = practice()
    saved = build(before)
    assert before == practice()
    assert saved["probabilities"]["B"] == saved["probabilities"]["C"]
    assert saved["probabilities"]["A"] / saved["probabilities"]["B"] == pytest.approx(403.42879349)
    assert sum(saved["probabilities"].values()) == pytest.approx(1.0)
    before["rows"][0]["position"] = 3
    assert saved["evidence"]["rows"][0]["position"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("year", 2025),
        ("round", 16),
        ("session_number", True),
        ("fetched_at", "2026-10-09T09:00:00Z"),
        ("fetched_at", "2026-10-09T13:00:00Z"),
        ("practice_started_at", "2026-10-09T09:00:00Z"),
    ],
)
def test_wrong_event_or_unfinished_or_later_practice_is_rejected(field, value):
    source = practice()
    source[field] = value
    with pytest.raises(ValueError):
        build(source)


@pytest.mark.parametrize(
    "field,value",
    [
        ("driver", "outside"),
        ("driver", "A"),
        ("position", 1),
        ("position", True),
        ("lap_seconds", float("nan")),
        ("lap_seconds", True),
        ("laps", -1),
    ],
)
def test_ambiguous_or_invalid_practice_observations_are_rejected(field, value):
    source = practice()
    source["rows"][1][field] = value
    with pytest.raises(ValueError):
        build(source)


def test_missing_practice_never_becomes_a_fabricated_uniform_reference():
    with pytest.raises(ValueError, match="half"):
        practice_rank_probabilities(["A", "B", "C"], [])


def test_scoring_rebuilds_the_saved_reference_and_ignores_edited_saved_scores():
    saved = build()
    saved["score"] = {"brier_score": 999.0}
    before = deepcopy(saved)
    scored = score_saved_practice_reference(
        saved,
        ["A", "B", "C"],
        year=2026,
        target_round=17,
        recorded_at="2026-10-09T09:40:00Z",
        qualifying_starts_at="2026-10-09T12:30:00Z",
        observed_winner="B",
    )
    assert saved == before
    assert scored["score"]["brier_score"] != 999.0
    saved["probabilities"]["A"] -= 0.01
    with pytest.raises(ValueError, match="probabilities"):
        score_saved_practice_reference(
            saved,
            ["A", "B", "C"],
            year=2026,
            target_round=17,
            recorded_at="2026-10-09T09:40:00Z",
            qualifying_starts_at="2026-10-09T12:30:00Z",
        )


def test_score_attribution_separates_winner_mass_and_wrong_favorite_and_handles_empty_races():
    result = compare_winner_errors(
        {"A": 0.1, "B": 0.8}, {"A": 0.4, "B": 0.6}, "A", no_winner_probability=0.1
    )
    assert result["winner_probability_loss_difference"] == pytest.approx(0.45)
    assert result["other_outcome_loss_difference"] == pytest.approx(0.29)
    assert result["model_minus_reference_brier"] == pytest.approx(0.74)
    assert result["drivers"][0]["driver"] == "A"
    with pytest.raises(ValueError, match="field"):
        compare_winner_errors({"A": 1.0}, {"B": 1.0}, "A")
