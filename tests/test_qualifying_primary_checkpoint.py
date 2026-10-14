"""Verify compact development evidence and reject altered accuracy claims."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qualifying_checkpoint",
    ROOT / "examples/verify_qualifying_primary_checkpoint.py",
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture(scope="module")
def evidence():
    return json.loads(verifier.EVIDENCE.read_text(encoding="utf-8"))


def reseal(value):
    value["content_sha256"] = verifier.digest(
        {k: v for k, v in value.items() if k != "content_sha256"},
    )
    return value


def test_complete_screen_reconstructs_offline_without_simulation(evidence, monkeypatch):
    from f1sim.analysis.montecarlo import MonteCarloRunner
    from f1sim.data.current import CurrentSeasonDataLoader

    def forbidden(*args, **kwargs):
        raise AssertionError("Compact verifier cannot fetch or simulate")

    monkeypatch.setattr(MonteCarloRunner, "run", forbidden)
    monkeypatch.setattr(CurrentSeasonDataLoader, "_fetch_paginated", forbidden)
    result = verifier.assess(evidence)
    assert result["events"] == 40
    assert result["historical_distance_faults"] == 2
    assert result["production_promoted"] is False
    assert result["authentic_historical_inputs"] is False
    overall = result["summaries"]["all"]["winner_metrics"]
    assert overall["native"]["brier"] == pytest.approx(0.503662109375)
    assert overall["candidate"]["brier"] == pytest.approx(0.484912109375)
    assert overall["candidate"]["log_loss_infinite"] is True
    assert result["gates"]["positive_paired_gain_over_reference"] is False


@pytest.mark.parametrize("change", ["counts", "stage", "promotion", "distance", "score"])
def test_resealed_inaccurate_data_or_claim_is_rejected(evidence, change):
    value = deepcopy(evidence)
    if change == "counts":
        wins = value["events"][0]["candidate"]["wins"]
        donor = max(wins, key=wins.get)
        receiver = next(c for c in wins if c != donor)
        wins[donor] -= 1
        wins[receiver] += 1
    elif change == "stage":
        value["evaluation_stage"] = "prospective_post_qualifying_forecast"
    elif change == "promotion":
        value["production_promoted"] = True
    elif change == "distance":
        value["historical_distance_faults"][0]["scheduled_laps"] = 57
    else:
        value["winner_summaries"]["all"]["winner_metrics"]["candidate"]["brier"] = 0.1
    with pytest.raises(ValueError):
        verifier.assess(reseal(value))


def test_unsealed_count_change_is_rejected(evidence):
    value = deepcopy(evidence)
    value["events"][0]["candidate"]["wins"]["VER"] += 1
    with pytest.raises(ValueError, match="seal"):
        verifier.assess(value)
