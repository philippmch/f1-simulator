"""Published winner-policy evidence must reproduce and reject re-sealed tampering."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "practice_winner_verifier", ROOT / "examples/verify_practice_winner_policy.py",
)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)
EVIDENCE = ROOT / "evidence/practice-native-winner-2026.json"


def test_sealed_full_field_production_choice_and_scores_reproduce():
    result = VERIFIER.verify(EVIDENCE)
    assert result["events"] == 16 and result["trials"] == 400
    assert result["relative_gain"] > .05
    assert result["paired_bootstrap_95_gain"][0] > 0
    assert result["gain_without_three_best"] > 0
    assert result["independent_untouched_test"] is False
    assert result["broader_winner_goal_achieved"] is False


@pytest.mark.parametrize("mutation", ["missing_entrant", "future_history", "changed_policy"])
def test_resealed_invalid_evidence_is_rejected(tmp_path, mutation):
    body = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    body.pop("content_sha256")
    event = body["events"][2]
    if mutation == "missing_entrant":
        event["drivers"].pop()
    elif mutation == "future_history":
        event["training_rounds"].append(event["round"])
    else:
        event["practice_forecast"]["policy"] = "earlier_team_q1"
    body["content_sha256"] = VERIFIER.digest(body)
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError):
        VERIFIER.verify(path)
