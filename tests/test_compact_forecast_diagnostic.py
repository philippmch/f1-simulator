"""Compact development evidence preserves literal models and frozen allocation."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "diagnostic_verifier", ROOT / "examples/verify_post_qualifying_diagnostic.py",
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def evidence():
    return json.loads((ROOT / "evidence/post-qualifying-winner-diagnostic-2026.json").read_text())


def test_compact_evidence_expands_without_mutation_and_reproduces_all_scores():
    saved = evidence()
    before = deepcopy(saved)
    expanded = VERIFIER.expanded_events(saved)
    assert saved == before
    assert len(expanded) == 16
    assert all(len(event["drivers"]) == 22 for event in expanded)
    assert all(event["weather"]["rain_intensity"] == 0 for event in expanded)
    report = VERIFIER.verify(
        ROOT / "evidence/post-qualifying-winner-diagnostic-2026.json",
        ROOT / "evidence/published-grid-coverage-2026.json",
    )
    assert report["native_brier"] == pytest.approx(.4935125)
    assert report["references"]["grid_18"] == pytest.approx(.49049912077947705)


@pytest.mark.parametrize("index", [True, -1, 10000, "1"])
def test_invalid_shared_history_reference_cannot_change_the_frozen_allocation(index):
    saved = evidence()
    saved["events"][2]["point_history_indices"][0] = index
    with pytest.raises(ValueError, match="invalid reference"):
        VERIFIER.expanded_events(saved)


def test_changed_shared_point_history_is_detected_against_each_original_allocation():
    saved = evidence()
    saved["point_history"][0]["points"] += 1
    with pytest.raises(ValueError, match="original frozen allocation"):
        VERIFIER.expanded_events(saved)


def test_unsupported_compact_definition_cannot_refit_the_allocation():
    saved = evidence()
    saved["allocation_definition"]["prior_points_per_driver"] = 50
    with pytest.raises(ValueError, match="original fixed allocation"):
        VERIFIER.expanded_events(saved)
