"""Focused checks for the standalone weighted rival selection report."""

import re

from f1sim.output.comparison import (
    _selection_report_standard_error,
    render_rival_strategy_selection_report,
)


def _manifest(*, identity=False):
    selected = "reference" if identity else "selected <script>alert(1)</script>"
    scenario_name = "../cautious <rival>"
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
            },
            "validation_scenario_metrics": {
                scenario_name: {
                    "reference_mean_points": 8,
                    "selected_mean_points": 10,
                    "mean_points_difference": 2,
                    "points_difference_standard_error": None,
                    "paired_races": 1,
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
    assert "101–102 inclusive" in report
    assert "103–103 inclusive" in report
    assert "Methodology &lt;limit&gt;" in report
    assert "../cautious &lt;rival&gt;" in report
    assert "<script>alert(1)</script>" not in report
    hrefs = re.findall(r'href="([^"]+)"', report)
    assert hrefs == ["rival_selection_abc_scenario_00_validation.html",
                     "rival_selection_abc_scenario_00_training.json"]
    assert all("/" not in href and "\\" not in href and ":" not in href for href in hrefs)


def test_identity_report_explains_zero_by_definition_without_an_alternative_se():
    report = render_rival_strategy_selection_report(_manifest(identity=True))

    assert "difference is zero by definition" in report
    assert "No independent alternative estimate" in report
    assert "0.000 (identity by definition)" in report
    assert "Not estimated (1 paired race)" not in report


def test_non_numeric_boolean_and_nonfinite_metrics_are_not_formatted_as_points():
    manifest = _manifest()
    selection = manifest["selection"]
    selection["training_score_table"][0]["mean_points"] = True
    selection["training_scenario_score_tables"]["../cautious <rival>"]["scores"][0][
        "mean_points"
    ] = float("nan")
    selection["validation_target_metrics"]["reference_mean_points"] = float("inf")

    report = render_rival_strategy_selection_report(manifest)

    assert "<td>Not recorded</td>" in report
    assert ">nan<" not in report.lower()
    assert ">inf<" not in report.lower()


def test_missing_standard_error_requires_an_integer_single_race_count():
    assert "1 paired race" in _selection_report_standard_error(None, 1)
    assert _selection_report_standard_error(None, True) == "Not estimated"
    assert _selection_report_standard_error(None, 1.5) == "Not estimated"
