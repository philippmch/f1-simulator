"""The input correction receipt cannot certify future or incomplete forecasts."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
spec = importlib.util.spec_from_file_location(
    "input_correction_verifier", ROOT / "examples/verify_post_qualifying_input_correction.py",
)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def test_frozen_correction_receipt_recomputes_all_40_pairs():
    result = verifier.verify()
    assert result["events"] == 40
    assert result["candidate_brier"] == pytest.approx(0.52114)
    assert result["against_native"]["gain_95"][0] > 0
    assert result["candidate_brier"] > result["grid18_brier"]


@pytest.mark.parametrize("alteration", ["future_claim", "missing_count", "weaker_reference"])
def test_resealed_false_claim_or_incomplete_field_is_rejected(tmp_path, alteration):
    record = json.loads((ROOT / "evidence/post-qualifying-input-correction.json").read_text())
    if alteration == "future_claim":
        record["prospectively_recorded"] = True
    elif alteration == "missing_count":
        record["events"][0]["updated_wins"].pop("ALB")
    else:
        record["reference_scales"] = [6]
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    record["content_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()
    path = tmp_path / "modified.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        verifier.verify(path)
