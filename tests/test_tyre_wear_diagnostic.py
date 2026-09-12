"""Executed schedule evidence for wear beyond the numerical grip floor."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def diagnostic():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / "examples" / "check_tyre_wear.py"))


@pytest.fixture(scope="module")
def rows(diagnostic):
    return diagnostic["check_cases"]()


def test_native_policies_match_independently_executed_schedules(rows):
    assert len(rows) == 8
    for row in rows:
        assert row["schedules_checked"] >= 16
        assert row["completed_schedules"] == row["schedules_checked"]
        assert row["selected"]["status"] == "finished"
        assert row["selected"]["laps_completed"] == 6
        assert row["difference_seconds"] == pytest.approx(0., abs=1e-8)
        # The observed optimum must include a replacement: retaining these
        # old rain tyres forever was attractive under the former pace cap.
        assert row["best"]["stops"]
        assert len(row["selected"]["stops"]) <= row["inputs"]["max_enumerated_stops"]
    json.dumps(rows, allow_nan=False)


def test_engines_agree_and_finite_opening_wear_is_preserved(rows):
    for name in {row["case"] for row in rows}:
        matching = [row for row in rows if row["case"] == name]
        assert len({round(row["selected"]["total_time"], 8) for row in matching}) == 1
        for row in matching:
            if row["inventory"] != "finite":
                continue
            history = row["selected"]["sets"]
            assert history[0]["set_id"] == "opening"
            assert history[0]["age_at_fit"] == row["inputs"]["age"]
            assert sum(stint["laps_used"] for stint in history) == 6
            assert all(stint["age_at_end"] == stint["age_at_fit"] + stint["laps_used"]
                       for stint in history)


def test_cases_cross_cliff_while_grip_remains_bounded(rows):
    for row in rows:
        samples = row["wear_samples"]
        increments = [b["penalty"] - a["penalty"] for a, b in zip(samples, samples[1:])]
        assert min(increments) > 0
        assert max(increments) > min(increments) * 1.5
        assert samples[-1]["grip"] == .5


def test_finite_schedule_enumeration_allows_physical_set_reuse(diagnostic):
    case = diagnostic["CASES"]["rain_cliff"]
    schedules = list(diagnostic["_schedules"](case, True))
    assert () in schedules
    assert ((2, "wet"), (4, "opening")) in schedules
    assert ((2, "opening"),) not in schedules
    assert ((2, "wet"), (4, "wet")) not in schedules
