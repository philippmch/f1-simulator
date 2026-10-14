"""Verify original forecast scores and reject altered outcomes or publication claims."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "recorded_scores", ROOT / "examples/verify_recorded_forecast_scores.py",
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture(scope="module")
def evidence():
    return json.loads(verifier.EVIDENCE.read_text(encoding="utf-8"))


def reseal(record):
    record["content_sha256"] = verifier.digest(
        {key: value for key, value in record.items() if key != "content_sha256"},
    )
    return record


def test_original_scores_reproduce_offline_with_preserved_scope(evidence, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("Offline score verification must not fetch data")

    monkeypatch.setattr(verifier.CurrentSeasonDataLoader, "_fetch_paginated", no_network)
    result = verifier.assess(evidence)
    assert result["events"] == 1
    assert result["forecast_stage"] == "pre_qualifying"
    assert result["after_qualifying_grid_reference_evidence"] is False
    assert result["observed_winner"] == "VER"
    assert [row["winner_brier"] for row in result["scores"]] == pytest.approx(
        [.5926, .5912, .5427525017101849],
    )
    assert result["scores"][2]["native_winner_brier"] == pytest.approx(.5912)
    assert result["scores"][2]["winner_policy"] == "teammate_race_points_v1"


@pytest.mark.parametrize("change", ["score", "winner", "field", "qualifying", "blob",
                                    "late_publication", "duplicate", "stage", "receipt_head",
                                    "receipt_event", "receipt_repository", "receipt_run",
                                    "early_observation"])
def test_resealed_altered_evidence_is_rejected(evidence, change):
    changed = deepcopy(evidence)
    row = changed["records"][0]
    if change == "score":
        row["score_report"]["winner_score"]["brier_score"] = .01
    elif change == "winner":
        changed["observed_winner"] = "ANT"
    elif change == "field":
        changed["observed_results"].pop()
    elif change == "qualifying":
        changed["observed_qualifying"][0]["position"] = "22"
    elif change == "blob":
        row["publication_receipt"]["blob_sha"] = "0" * 40
    elif change == "late_publication":
        row["publication_receipt"]["created_at"] = "2026-10-11T12:00:00Z"
    elif change == "duplicate":
        changed["records"][1] = deepcopy(row)
    elif change == "receipt_head":
        row["publication_receipt"]["receipt_head_sha"] = "0" * 40
    elif change == "receipt_event":
        row["publication_receipt"]["receipt_event"] = "workflow_dispatch"
    elif change == "receipt_repository":
        receipt = row["publication_receipt"]
        receipt["receipt_api_url"] = receipt["receipt_api_url"].replace(
            "philippmch/f1-simulator", "another-owner/f1-simulator",
        )
        receipt["receipt_url"] = receipt["receipt_url"].replace(
            "philippmch/f1-simulator", "another-owner/f1-simulator",
        )
    elif change == "receipt_run":
        row["publication_receipt"]["receipt_url"] += "0"
    elif change == "early_observation":
        row["score_report"]["observations_provenance"]["fetched_at"] = "2026-10-06T12:00:00Z"
    else:
        changed["forecast_stage"] = "post_qualifying"
    with pytest.raises(ValueError):
        verifier.assess(reseal(changed))


def test_seal_detects_unresealed_change(evidence):
    changed = deepcopy(evidence)
    changed["observed_winner"] = "ANT"
    with pytest.raises(ValueError, match="seal"):
        verifier.assess(changed)
