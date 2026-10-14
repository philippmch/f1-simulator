"""Pace metric invariants and the failure hidden by full-field averages."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from f1sim.analysis.race_pace_diagnostics import score_race_pace_components

GRID = ["A", "B", "C", "D", "E", "F"]
OBSERVED = {"A": 3., "B": 2., "C": 1., "D": 0., "E": -3., "F": -5.}
BASELINE = {"A": 3., "B": 2., "C": 1., "D": 0., "E": -20., "F": -30.}
CANDIDATE = {"A": 4., "B": 1., "C": 1., "D": 0., "E": -3., "F": -5.}


def test_wholefield_gain_does_not_hide_front_rival_regression():
    result = score_race_pace_components(GRID, BASELINE, CANDIDATE, OBSERVED)
    assert result["comparison"]["matched_cohort_whole_field_mae_gain_percent"] > 0
    assert result["baseline"]["front_pair_mae_percent"] == 0
    assert result["candidate"]["front_pair_mae_percent"] > 0
    assert result["comparison"]["complete_front_pair_mae_gain_percent"] < 0
    assert result["cohort"]["front_scored_pairs"] == 6


def test_independent_additive_offsets_do_not_change_scores():
    original = score_race_pace_components(GRID, BASELINE, CANDIDATE, OBSERVED)
    shifted = score_race_pace_components(
        GRID, {c: v + 17. for c, v in BASELINE.items()},
        {c: v - 8. for c, v in CANDIDATE.items()},
        {c: v + 92. for c, v in OBSERVED.items()},
    )
    assert shifted == original


def test_front_group_comes_from_grid_before_observed_performance():
    observed = {**OBSERVED, "E": 40., "F": 50.}
    result = score_race_pace_components(GRID, BASELINE, CANDIDATE, observed)
    assert result["cohort"]["selected_front_ids"] == ["A", "B", "C", "D"]
    assert result["cohort"]["scored_front_ids"] == ["A", "B", "C", "D"]


def test_missing_front_label_reports_attrition_without_improvement_claim():
    observed = {c: v for c, v in OBSERVED.items() if c != "C"}
    result = score_race_pace_components(GRID, BASELINE, CANDIDATE, observed)
    assert result["cohort"]["missing_observed_ids"] == ["C"]
    assert result["cohort"]["front_comparison_status"] == "partial"
    assert result["cohort"]["front_observed_drivers"] == 3
    assert result["cohort"]["front_scored_pairs"] == 3
    assert result["cohort"]["front_missing_pairs"] == 3
    assert result["baseline"]["front_pair_mae_percent"] is not None
    assert result["comparison"]["complete_front_pair_mae_gain_percent"] is None


def test_missing_prediction_is_not_replaced_by_a_neutral_pace():
    candidate = {c: v for c, v in CANDIDATE.items() if c != "B"}
    result = score_race_pace_components(GRID, BASELINE, candidate, OBSERVED)
    assert result["cohort"]["missing_candidate_ids"] == ["B"]
    assert result["cohort"]["scored_drivers"] == 5
    assert result["cohort"]["front_support_complete"] is False
    assert result["comparison"]["complete_front_pair_mae_gain_percent"] is None


def test_one_observed_driver_cannot_produce_a_zero_error_success():
    result = score_race_pace_components(GRID, BASELINE, CANDIDATE, {"A": 3.})
    assert result["baseline"]["whole_field_centered_mae_percent"] is None
    assert result["candidate"]["front_pair_mae_percent"] is None
    assert result["cohort"]["front_comparison_status"] == "unavailable"
    assert result["cohort"]["front_missing_pairs"] == 6


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), "fast"])
def test_non_numeric_or_non_finite_labels_are_rejected(invalid):
    with pytest.raises(ValueError, match="finite numeric"):
        score_race_pace_components(GRID, BASELINE, CANDIDATE, {**OBSERVED, "A": invalid})


def test_grid_identity_mismatch_is_rejected():
    with pytest.raises(ValueError, match="outside the starting grid"):
        score_race_pace_components(GRID, BASELINE, CANDIDATE, {**OBSERVED, "OTHER": 1.})
    with pytest.raises(ValueError, match="unique driver"):
        score_race_pace_components(["A", "A"], BASELINE, CANDIDATE, OBSERVED)


def test_offline_cli_reads_explicit_json_and_writes_diagnostic(tmp_path):
    source = tmp_path / "pace.json"
    destination = tmp_path / "diagnostic.json"
    source.write_text(json.dumps({
        "starting_grid": GRID, "baseline_pace_percent": BASELINE,
        "candidate_pace_percent": CANDIDATE, "observed_pace_percent": OBSERVED,
    }))
    script = Path(__file__).resolve().parents[1] / "examples/evaluate_race_pace_components.py"
    result = subprocess.run(
        [sys.executable, str(script), str(source), "--output", str(destination)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    output = json.loads(destination.read_text())
    assert output["comparison"]["complete_front_pair_mae_gain_percent"] < 0
    assert output["cohort"]["front_support_complete"] is True
