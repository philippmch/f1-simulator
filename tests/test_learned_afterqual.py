import gzip
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from f1sim.analysis.learned_afterqual import (
    AfterQualifyingFeatures,
    FrozenAfterQualifyingModel,
    canonical_digest,
    load_frozen_afterqual_model,
    predict_after_qualifying,
    stable_sigmoid,
)


@pytest.fixture(scope="module")
def parity_cases():
    path = Path(__file__).parent / "fixtures" / "learned_afterqual_parity.json.gz"
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        fixture = json.load(stream)
    assert fixture["content_sha256"] == canonical_digest(
        {key: value for key, value in fixture.items() if key != "content_sha256"}
    )
    return fixture["cases"]


def test_sealed_inference_parity_without_scipy_or_fitting(parity_cases):
    for case in parity_cases:
        snapshot = deepcopy(case["snapshot"])
        artifact = deepcopy(case["artifact"])
        original = deepcopy((snapshot, artifact))
        actual = predict_after_qualifying(snapshot, artifact)
        assert (snapshot, artifact) == original
        for key, expected in case["expected"].items():
            np.testing.assert_allclose(actual[key], expected, atol=1e-12, rtol=0)
        assert actual["codes"] == snapshot["codes"]
        assert actual["driver_ids"] == snapshot["ids"]
        assert sum(actual["winner_probabilities"].values()) == pytest.approx(1)


def test_packaged_artifact_is_the_fixed_reviewed_candidate():
    artifact = load_frozen_afterqual_model()
    assert artifact.document["source_model_sha256"] == (
        "51bff87cebd961a6b1ecd80b6dd4b2b31db632c515476f6774ff881e4ed71136"
    )
    assert artifact.document["training_boundary"] == [2026, 15]
    assert len(artifact.model["trees"]) == 100
    assert artifact.model["training_event_keys"][-1] == [2026, 15]


def test_target_must_follow_every_training_event(parity_cases):
    snapshot = deepcopy(parity_cases[-1]["snapshot"])
    snapshot["year"], snapshot["round"] = 2026, 15
    with pytest.raises(ValueError, match="strictly later"):
        predict_after_qualifying(snapshot)


@pytest.mark.parametrize("mutation", ["duplicate_code", "wrong_grid", "nonfinite", "boolean",
                                    "risk", "delta", "width", "x_mismatch"])
def test_typed_features_reject_malformed_inputs(parity_cases, mutation):
    snapshot = deepcopy(parity_cases[0]["snapshot"])
    if mutation == "duplicate_code":
        snapshot["codes"][0] = snapshot["codes"][1]
    elif mutation == "wrong_grid":
        snapshot["grid_ids"][0], snapshot["grid_ids"][1] = (
            snapshot["grid_ids"][1], snapshot["grid_ids"][0]
        )
    elif mutation == "nonfinite":
        snapshot["tree_x"][0][3] = float("nan")
    elif mutation == "boolean":
        snapshot["tree_x"][0][3] = True
    elif mutation == "risk":
        snapshot["risk"][0][0] = 0
    elif mutation == "delta":
        snapshot["tree_x"][0][15] += .1
    elif mutation == "width":
        snapshot["tree_x"][0].pop()
    elif mutation == "x_mismatch":
        snapshot["x"][0][3] += .1
    with pytest.raises(ValueError):
        AfterQualifyingFeatures.from_mapping(snapshot)


def test_feature_roundtrip_retains_source_context(parity_cases):
    snapshot = deepcopy(parity_cases[0]["snapshot"])
    snapshot["source_context"] = {"available_at": "2025-06-14T20:00:00+00:00"}
    typed = AfterQualifyingFeatures.from_mapping(snapshot)
    snapshot["source_context"]["available_at"] = "changed"
    roundtrip = AfterQualifyingFeatures.from_mapping(typed.to_dict())
    assert roundtrip.to_dict() == typed.to_dict()
    assert typed.context["source_context"]["available_at"] != "changed"


def test_negative_qualifying_rank_for_a_smaller_race_roster_is_preserved(parity_cases):
    snapshot = deepcopy(parity_cases[0]["snapshot"])
    # Pick a back-row driver so the frozen rank difference remains in [-1,1].
    index = snapshot["ids"].index(snapshot["grid_ids"][-1])
    snapshot["tree_x"][index][1] = -.05
    snapshot["tree_x"][index][15] = -.05 - snapshot["tree_x"][index][0]
    snapshot["x"][index][1] = -.05
    assert AfterQualifyingFeatures.from_mapping(snapshot).tree_x[index][1] == -.05


def test_artifact_seal_and_fixed_specification_are_checked():
    artifact = load_frozen_afterqual_model().to_dict()
    artifact["model"]["risk_beta"][0] += .1
    with pytest.raises(ValueError, match="seal"):
        FrozenAfterQualifyingModel.from_mapping(artifact)
    artifact["model"]["learning_rate"] = .2
    artifact["content_sha256"] = canonical_digest(
        {key: value for key, value in artifact.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="frozen inference specification"):
        FrozenAfterQualifyingModel.from_mapping(artifact)


def test_stable_sigmoid_handles_extreme_logits_without_overflow():
    with np.errstate(over="raise", invalid="raise"):
        result = stable_sigmoid(np.array([-1000., -2., 0., 2., 1000.]))
    np.testing.assert_allclose(result, [0., 0.11920292202211755, .5,
                                      0.8807970779778823, 1.], atol=1e-16, rtol=0)
