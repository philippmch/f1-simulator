"""Reproduce historical tradeoffs and reject edited data/validation claims."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "qualifying_led_verifier",
    ROOT / "examples/verify_historical_qualifying_led.py",
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


def test_complete_scores_and_tradeoffs_reproduce_without_network_or_simulation(
    evidence, monkeypatch
):
    import socket

    from f1sim.analysis.montecarlo import MonteCarloRunner
    from f1sim.data.current import CurrentSeasonDataLoader

    def forbidden(*args, **kwargs):
        raise AssertionError("Historical verifier must remain offline and never simulate")

    monkeypatch.setattr(MonteCarloRunner, "run", forbidden)
    monkeypatch.setattr(CurrentSeasonDataLoader, "_fetch_paginated", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    result = verifier.assess(evidence)
    overall = result["summaries"]["all"]
    assert result["events"] == 40
    assert overall["models"]["native"]["winner_brier"] == pytest.approx(0.5207434375)
    assert overall["models"]["qualifying_led"]["winner_brier"] == pytest.approx(0.4650146875)
    assert overall["grid_references"]["18"]["winner_brier"] == pytest.approx(0.5075666393061634)
    assert overall["winner_brier_reduction_percent_against_native"] == pytest.approx(10.701767125)
    assert overall["winner_brier_reduction_percent_against_best_reference"] == pytest.approx(
        8.383520214
    )
    assert overall["models"]["native"]["winner_log_loss_infinite"] is True
    assert overall["models"]["qualifying_led"]["winner_log_loss"] == pytest.approx(0.995945029958)
    assert (
        overall["models"]["qualifying_led"]["position_mae"]
        > overall["models"]["native"]["position_mae"]
    )
    assert overall["qualifying_led_gain_against_native"]["position_mae"]["gain_95"][1] < 0
    assert result["robust_gate_passed"] is False
    assert result["robust_gate_checks"]["positive_paired_winner_gain_over_reference"] is False
    assert result["original_decision"] == "rejected_for_deployment"
    assert result["prospective_skill_established"] is False


@pytest.mark.parametrize(
    "change",
    [
        "winner_counts",
        "position_counts",
        "nonfinish_counts",
        "observed_position",
        "grid",
        "input_identity",
        "year_coverage",
        "gate_claim",
        "prospective_claim",
        "distance",
        "summary",
        "bootstrap",
    ],
)
def test_resealed_tampering_is_rejected(evidence, change):
    value = deepcopy(evidence)
    first = value["events"][0]
    if change == "winner_counts":
        counts = first["qualifying_led"]["wins"]
        donor = max(counts, key=counts.get)
        receiver = next(d for d in counts if d != donor)
        counts[donor] -= 1
        counts[receiver] += 1
    elif change == "position_counts":
        next(iter(first["qualifying_led"]["marginals"].values()))["position_sum"] += 400
    elif change == "nonfinish_counts":
        next(iter(first["qualifying_led"]["marginals"].values()))["nonfinish_count"] += 1
    elif change == "observed_position":
        first["observed_drivers"][first["observed_winner"]]["position"] = 20
    elif change == "grid":
        first["starting_grid"].reverse()
    elif change == "input_identity":
        first["native"]["input_sha256"] = "0" * 64
    elif change == "year_coverage":
        value["events"].pop()
    elif change == "gate_claim":
        value["robust_gate_passed"] = True
    elif change == "prospective_claim":
        value["prospective_skill_established"] = True
    elif change == "distance":
        value["historical_distance_faults"][0]["literal_simulation_laps"] = 50
    elif change == "bootstrap":
        value["bootstrap"]["resamples"] = 100
    else:
        value["summaries"]["all"]["models"]["qualifying_led"]["winner_brier"] = 0.1
    with pytest.raises(ValueError):
        verifier.assess(reseal(value))


def test_unsealed_edit_is_rejected(evidence):
    value = deepcopy(evidence)
    value["events"][0]["native"]["wins"]["VER"] += 1
    with pytest.raises(ValueError, match="seal"):
        verifier.assess(value)
