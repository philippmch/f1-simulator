"""Production pace inputs match the fixed historical sensitivity, without racing."""
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_postqual_pace_inputs", ROOT / "examples/verify_postqual_pace_inputs.py",
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def reseal(document):
    document["content_sha256"] = VERIFIER.digest(
        {k: v for k, v in document.items() if k != "content_sha256"})
    return document


def test_both_production_policies_rebuild_all_historical_inputs_without_simulation(monkeypatch):
    from f1sim.analysis.montecarlo import MonteCarloRunner
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *a, **k: pytest.fail("Race execution"))
    result = VERIFIER.verify_data(VERIFIER.read())
    assert result["verified_archived_400_trial_input_hashes"] == 80
    assert result["strategy_identity_checks"] == 832
    assert result["pre_qualifying_unchanged"] is True


def test_resealed_target_result_in_history_is_rejected():
    document = deepcopy(VERIFIER.read())
    event = next(e for e in document["events"] if e["builder_args"]["race_rows"])
    event["builder_args"]["race_rows"][0]["round"] = event["round"]
    with pytest.raises(ValueError, match="Target/future"):
        VERIFIER.verify_data(reseal(document), expected_seal=None)


def test_resealed_car_change_cannot_reuse_measured_results():
    document = deepcopy(VERIFIER.read())
    cars = document["events"][0]["policies"]["qualifying_led"]["cars"]
    cars[next(iter(cars))]["base_pace"] -= .01
    with pytest.raises(ValueError, match="Historical car inputs changed"):
        VERIFIER.verify_data(reseal(document), expected_seal=None)


def test_resealed_context_change_cannot_reuse_input_hashes():
    document = deepcopy(VERIFIER.read())
    document["events"][0]["race_context"]["seed"] += 1
    with pytest.raises(ValueError, match="400-trial model input hash"):
        VERIFIER.verify_data(reseal(document), expected_seal=None)
