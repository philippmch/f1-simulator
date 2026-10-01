"""Focused checks for the standalone weighted rival selection report."""

import re

import pytest

from f1sim.output.comparison import (
    _selection_report_standard_error,
    render_rival_strategy_selection_report,
)


def _manifest(*, identity=False):
    selected = "reference" if identity else "selected <script>alert(1)</script>"
    scenario_name = "../cautious <rival>"
    outcome_profile = None if identity else {
        "paired_races": 1,
        "more_points_races": 1,
        "equal_points_races": 0,
        "fewer_points_races": 0,
        "mean_points_gain_when_ahead": 2,
        "mean_points_loss_when_behind": None,
    }
    target_plans = {"reference": None}
    if not identity:
        target_plans[selected] = [{"lap": 4, "compound": "hard"}]
    return {
        "manifest_filename": "../manifest.json",
        "report_context": {"track_name": "Montréal <track>", "race_engine": "Chronological"},
        "target_plans": target_plans,
        "rival_scenarios": {
            scenario_name: {
                "training_comparison_html": "../training.html",
                "validation_comparison_html": "rival_selection_abc_scenario_00_validation.html",
                "training_comparison_json": "rival_selection_abc_scenario_00_training.json",
                "validation_comparison_json": "javascript:alert(1)",
            },
        },
        "selection": {
            "target_mode": "constructor",
            "target_id": "Team <T>",
            "target_member_ids": ["A<&", "B"],
            "reference_label": "reference",
            "selected_label": selected,
            "validation_status": "no_change" if identity else "evaluated",
            "rival_scenarios": [{
                "name": scenario_name,
                "weight": 2,
                "normalized_weight": 1,
                "rival_pit_plans": {"C": None, "D": [], "E": [{"lap": 3, "compound": "soft"}]},
            }],
            "training_score_table": [
                {"label": "reference", "mean_points": 8.5, "trials": 2},
                {"label": selected, "mean_points": 12, "trials": 2},
            ],
            "training_scenario_score_tables": {
                scenario_name: {
                    "scores": [
                        {"label": "reference", "mean_points": 8.5, "trials": 2},
                        {"label": selected, "mean_points": 12, "trials": 2},
                    ],
                },
            },
            "validation_target_metrics": {
                "reference_mean_points": 8,
                "selected_mean_points": 10,
                "mean_points_difference": 2,
                "points_difference_standard_error": None,
                "paired_races": 1,
                "points_outcome_profile": outcome_profile,
            },
            "validation_scenario_metrics": {
                scenario_name: {
                    "reference_mean_points": 8,
                    "selected_mean_points": 10,
                    "mean_points_difference": 2,
                    "points_difference_standard_error": None,
                    "paired_races": 1,
                    "points_outcome_profile": outcome_profile,
                },
            },
            "seed_ranges": {
                "training": {"first_seed": 101, "last_seed": 102, "trials": 2},
                "validation": {"first_seed": 103, "last_seed": 103, "trials": 1},
            },
            "methodology_limits": ["Methodology <limit>"],
        },
    }


def test_report_shows_frozen_constructor_training_and_heldout_evidence_safely():
    report = render_rival_strategy_selection_report(_manifest())

    assert "Selected and frozen" in report
    assert "Target pit plan" in report
    assert "constructor" in report
    assert "A&lt;&amp;, B" in report
    assert "Montréal &lt;track&gt;" in report
    assert "Chronological" in report
    assert "Training scores" in report
    assert "Per-rival held-out changes" in report
    assert "selected minus reference" in report.lower()
    assert "within-seed cross-scenario covariance" in report
    assert "Not estimated (1 paired race); this is not zero uncertainty" in report
    assert "Paired points outcome profile" in report
    assert 'aria-label="Held-out points outcome profile"' in report
    assert "Mean gain when ahead (points)" in report
    assert "Mean loss when behind (points)" in report
    assert "2.000" in report
    assert "not calibrated win probabilities" in report
    assert "101–102 inclusive" in report
    assert "103–103 inclusive" in report
    assert "Methodology &lt;limit&gt;" in report
    assert "../cautious &lt;rival&gt;" in report
    assert "<script>alert(1)</script>" not in report
    hrefs = re.findall(r'href="([^"]+)"', report)
    assert hrefs == ["rival_selection_abc_scenario_00_validation.html",
                     "rival_selection_abc_scenario_00_training.json"]
    assert all("/" not in href and "\\" not in href and ":" not in href for href in hrefs)


@pytest.mark.parametrize("gap,tied,expected", [
    (0, True, "0.000 (exact tie)"),
    (0, False, "Below numeric reporting precision"),
    (7e-20, False, "7.000e-20"),
    (None, None, "Not recorded"),
    (0, 1, "Not recorded"),
    (True, False, "Not recorded"),
    (float("nan"), False, "Not recorded"),
    (-1, False, "Not recorded"),
    (1, True, "Not recorded"),
])
def test_report_training_shortfall_uses_exact_flag_and_rejects_invalid_evidence(
    gap, tied, expected,
):
    manifest = _manifest()
    selection = manifest["selection"]
    selection["training_score_table"] = [{
        "label": "reference <unsafe>", "mean_points": 25, "trials": 1,
        "mean_points_behind_selected": gap, "tied_for_best": tied,
    }, {
        "label": selection["selected_label"], "mean_points": 25, "trials": 1,
        "mean_points_behind_selected": 0, "tied_for_best": True,
    }]
    selection["tiebreak_applied"] = "unique_highest_weighted_training_mean"
    report = render_rival_strategy_selection_report(manifest)
    assert "Mean points behind selected" in report
    assert f"<td>{expected}</td>" in report
    assert "0.000 (selected)" in report
    assert "unique highest weighted training mean" in report
    assert "reference &lt;unsafe&gt;" in report


@pytest.mark.parametrize("reason,expected", [
    ("reference_preferred_on_exact_tie", "An exact training tie preferred the reference."),
    ("first_plan_order_on_exact_tie",
     "An exact training tie used the first candidate in plan order."),
    ("unknown <script>", "Selection reason not recorded."),
    (None, "Selection reason not recorded."),
])
def test_report_selection_reason_and_legacy_rows(reason, expected):
    manifest = _manifest()
    manifest["selection"]["tiebreak_applied"] = reason
    report = render_rival_strategy_selection_report(manifest)
    assert expected in report
    assert "<td>Not recorded</td>" in report
    assert "unknown <script>" not in report


def test_report_keeps_small_weights_validation_differences_and_gains_visible():
    manifest = _manifest()
    selection = manifest["selection"]
    selection["rival_scenarios"][0]["normalized_weight"] = 1e-20
    metrics = selection["validation_target_metrics"]
    metrics["mean_points_difference"] = -7e-20
    metrics["points_difference_standard_error"] = 2e-20
    metrics["points_outcome_profile"]["mean_points_gain_when_ahead"] = 7e-20
    report = render_rival_strategy_selection_report(manifest)
    assert "<td>1.000e-20</td>" in report
    assert "<td>-7.000e-20</td>" in report
    assert "2.000e-20 sample SE" in report
    assert "<td>7.000e-20</td>" in report


def test_identity_report_explains_zero_by_definition_without_an_alternative_se():
    report = render_rival_strategy_selection_report(_manifest(identity=True))

    assert "difference is zero by definition" in report
    assert "No independent alternative estimate" in report
    assert "0.000 (identity by definition)" in report
    assert "Not estimated (1 paired race)" not in report
    assert "no separate alternative outcome profile exists" in report


def test_non_numeric_nonfinite_and_oversized_profile_values_are_not_formatted():
    manifest = _manifest()
    selection = manifest["selection"]
    selection["training_score_table"][0]["mean_points"] = True
    selection["training_scenario_score_tables"]["../cautious <rival>"]["scores"][0][
        "mean_points"
    ] = float("nan")
    selection["validation_target_metrics"]["reference_mean_points"] = float("inf")
    selection["validation_target_metrics"]["points_outcome_profile"][
        "mean_points_gain_when_ahead"
    ] = 10**1000

    report = render_rival_strategy_selection_report(manifest)

    assert "<td>Not recorded</td>" in report
    assert ">nan<" not in report.lower()
    assert ">inf<" not in report.lower()
    assert re.search(
        r'<th scope="row">Weighted across rival scenarios</th><td>1</td><td>1</td>'
        r'<td>0</td><td>0</td><td>Not recorded</td>'
        r'<td>None \(no fewer-points seeds\)</td>',
        report,
    )


def test_missing_or_malformed_outcome_profiles_are_not_recorded_and_labels_are_escaped():
    manifest = _manifest()
    scenario = "../cautious <rival>"
    del manifest["selection"]["validation_target_metrics"]["points_outcome_profile"]
    manifest["selection"]["validation_scenario_metrics"][scenario][
        "points_outcome_profile"
    ] = {
        "paired_races": 2,
        "more_points_races": 1,
        "equal_points_races": 0,
        "fewer_points_races": 0,
        "mean_points_gain_when_ahead": 3,
        "mean_points_loss_when_behind": None,
    }

    report = render_rival_strategy_selection_report(manifest)

    assert "<th scope=\"row\">Weighted across rival scenarios</th><td>Not recorded" in report
    assert "<th scope=\"row\">../cautious &lt;rival&gt;</th>" in report


def test_missing_standard_error_requires_an_integer_single_race_count():
    assert "1 paired race" in _selection_report_standard_error(None, 1)
    assert _selection_report_standard_error(None, True) == "Not estimated"
    assert _selection_report_standard_error(None, 1.5) == "Not estimated"
