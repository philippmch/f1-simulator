"""Keep the complete paired forecast measurements reproducible and tamper evident."""

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
spec = importlib.util.spec_from_file_location(
    "accuracy_checkpoint", ROOT / "examples/verify_post_qualifying_accuracy_checkpoint.py",
)
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)


@pytest.fixture(scope="module")
def record():
    return checkpoint.sealed(checkpoint.CHECKPOINT)


def reseal(record):
    record["content_sha256"] = checkpoint.digest(
        {key: value for key, value in record.items() if key != "content_sha256"},
    )
    return record


def test_complete_checkpoint_reproduces_scores_and_exposes_position_regression(record):
    result = checkpoint.assess(record)
    all_events = result["summaries"]["all"]
    assert all_events["events"] == 40
    assert all_events["models"]["current"]["winner_brier"] == pytest.approx(.5207434375)
    assert all_events["models"]["candidate"]["winner_brier"] == pytest.approx(.4650146875)
    assert all_events["reference_brier"]["18"] == pytest.approx(.5075666393061634)
    assert all_events["models"]["current"]["top_choices_correct"] == 24
    assert all_events["pole_choices_correct"] == 27
    assert all_events["other_metric_gains"]["position_mae"]["gain_95"][1] < 0
    assert all_events["confidence"]["fixed_native_confidence"]["winner_brier"] == pytest.approx(
        .520719533227903,
    )
    assert result["candidate_deployed"] is False
    assert result["scope"]["prospective_forecasts"] is False


@pytest.mark.parametrize("change", ["duplicate", "missing", "input", "field", "count", "truth"])
def test_resealed_corrupted_checkpoint_cannot_silently_become_valid(record, change):
    changed = deepcopy(record)
    row = changed["events"][0]
    if change == "duplicate":
        changed["events"].append(deepcopy(row))
    elif change == "missing":
        changed["events"].pop()
    elif change == "input":
        row["candidate_car_overrides"][next(iter(row["candidate_car_overrides"]))]["base_pace"] = .1
    elif change == "field":
        row["candidate_wins"].pop(next(iter(row["candidate_wins"])))
    elif change == "count":
        row["current_wins"][next(iter(row["current_wins"]))] += 1
    else:
        driver = next(iter(row["observed_drivers"]))
        row["observed_drivers"][driver]["dnf"] = not row["observed_drivers"][driver]["dnf"]
    with pytest.raises(ValueError):
        checkpoint.assess(reseal(changed))
