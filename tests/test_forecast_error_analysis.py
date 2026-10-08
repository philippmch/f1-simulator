"""The error analysis reproduces both sealed forecasts on a common complete field."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def analyzer():
    spec = importlib.util.spec_from_file_location(
        "forecast_error_example",
        ROOT / "examples/analyze_forecast_errors.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_error_analysis_reproduces_original_full_field_native_and_simple_reference():
    result = analyzer().analyze(
        ROOT / "evidence/practice-native-winner-2026.json",
        ROOT / "evidence/practice-winner-reference-2026.json",
    )
    assert result["summary"]["events"] == 16
    assert result["summary"]["native_brier"] == pytest.approx(0.68326640625, abs=1e-14)
    assert result["summary"]["practice_reference_brier"] == pytest.approx(
        0.6818858310201636,
        abs=1e-14,
    )
    assert result["events"][0]["round"] == 11
    assert result["events"][0]["winner_practice_rank"] == 1
    assert result["events"][0]["winner_modeled_clean_race_pace_rank"] == 5
    assert result["events"][0]["winner_probability"] == 0.085
    assert result["independent_untouched_test"] is False
    assert result["production_accuracy_improved"] is False


@pytest.mark.parametrize("mutation", ["seal", "cohort", "session", "year"])
def test_incompatible_or_tampered_reference_cannot_enter_the_comparison(mutation, tmp_path):
    path = ROOT / "evidence/practice-winner-reference-2026.json"
    document = json.loads(path.read_text())
    if mutation == "cohort":
        document["events"].append(document["events"][0])
    elif mutation == "session":
        document["events"][0]["session_number"] = 1
    elif mutation == "year":
        document["year"] = 2025
    else:
        document["content_sha256"] = "edited"
    if mutation != "seal":
        body = {key: value for key, value in document.items() if key != "content_sha256"}
        document["content_sha256"] = hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
        ).hexdigest()
    edited = tmp_path / "reference.json"
    edited.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        analyzer().analyze(ROOT / "evidence/practice-native-winner-2026.json", edited)


def test_controlled_experiments_reproduce_failures_and_keep_hindsight_separate():
    module = analyzer()
    report = module.verify_experiments(
        ROOT / "evidence/forecast-error-experiments-2026.json",
        module.read_sealed(ROOT / "evidence/practice-native-winner-2026.json"),
    )
    uncertainty = report["persistent_race_pace_uncertainty"]
    assert uncertainty["baseline_brier"] == pytest.approx(0.6522285714285714)
    assert uncertainty["candidate_brier"] == pytest.approx(0.6692857142857143)
    hindsight = report["observed_race_pace_sensitivity"]
    assert hindsight["target_race_observations_used"] is True
    assert hindsight["events_detail"][0]["candidate_winner_probability"] == 0.41
    longrun = report["pre_qualifying_longrun_pace"]
    assert longrun["events"] == 16
    assert longrun["target_race_observations_used"] is False
    assert longrun["baseline_brier"] == pytest.approx(0.6825375)
    assert longrun["candidate_brier"] == pytest.approx(0.68715)
    assert all(experiment["production_changed"] is False for experiment in report.values())


@pytest.mark.parametrize("mutation", ["qualifying", "hindsight_scope", "winner", "counts"])
def test_resealed_controlled_receipts_must_preserve_pairing_and_information_scope(
    mutation, tmp_path
):
    module = analyzer()
    document = module.read_sealed(ROOT / "evidence/forecast-error-experiments-2026.json")
    experiment = document["experiments"][0]
    row = experiment["records"][0]
    if mutation == "qualifying":
        row["variants"]["on"]["qualifying_mean_positions"]["RUS"] += 1
    elif mutation == "hindsight_scope":
        document["experiments"][1]["target_race_observations_used"] = False
    elif mutation == "winner":
        row["observed_winner"] = "ANT"
    else:
        row["variants"]["on"]["wins"]["RUS"] += 1
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    edited = tmp_path / "experiment.json"
    edited.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        module.verify_experiments(
            edited, module.read_sealed(ROOT / "evidence/practice-native-winner-2026.json")
        )
