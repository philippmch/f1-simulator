"""Window coverage and service timing use complete, matching recorded histories."""

from copy import deepcopy

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output.comparison import _pit_plan_history_html, _pit_plan_statistics_html
from f1sim.simulation.race import DriverStatus, RaceResult

PLAN = {"lap": 6, "compound": "hard", "earliest_lap": 3, "trigger": "safety_car"}


def history(status="executed", actual_lap=4, actual_compound="hard"):
    return [PLAN | {"status": status, "reason": "user_plan", "actual_lap": actual_lap,
                    "actual_compound": actual_compound, "actual_set_id": "set-H"}]


def results(histories):
    return SimulationResults(
        100, "Window", {}, [[RaceResult("A", "A", "A", 1, 90, 0, 1, 90,
                                       DriverStatus.FINISHED, pit_plan_history=value)]
                            for value in histories], [],
        input_snapshot={"pit_plans": {"A": [PLAN]}},
    )


def test_window_service_laps_and_statuses_have_separate_recorded_denominators():
    run = results([history(), history(actual_lap=6), history("overridden", 6, "wet"),
                   history("not_reached", None, None), None])
    summary = run.get_pit_plan_statistics()
    assert summary["recorded_trials"] == 5
    driver = summary["drivers"][0]
    assert driver["valid_histories"] == 4 and driver["missing_histories"] == 1
    assert driver["instructions"] == [PLAN | {"executed": 2, "overridden": 1, "skipped": 0,
                                              "not_reached": 1, "service_laps": [
                                                  {"lap": 4, "count": 1},
                                                  {"lap": 6, "count": 2}]}]
    report = _pit_plan_statistics_html(run, "Window")
    assert "3–6 own laps (SC; deadline 6)" in report
    assert "Service laps: L4 × 1, L6 × 2" in report
    assert "4 valid, 1 missing, 0 invalid of 5 recorded trials" in report
    detail = _pit_plan_history_html(run, "Window")
    assert "hard (service own lap 4)" in detail


@pytest.mark.parametrize("changes", [
    {"earliest_lap": 3.0}, {"earliest_lap": 2}, {"trigger": "vsc"},
    {"actual_lap": True}, {"actual_lap": 2}, {"actual_lap": 7}, {"actual_lap": None},
    {"actual_compound": []}, {"actual_compound": "wet"},
    {"status": "not_reached"}, {"status": "overridden", "actual_lap": 4},
])
def test_malformed_window_histories_do_not_add_status_or_service_counts(changes):
    bad = deepcopy(history())
    bad[0].update(changes)
    summary = results([bad]).get_pit_plan_statistics()["drivers"][0]
    assert summary["invalid_histories"] == 1 and summary["valid_histories"] == 0
    instruction = summary["instructions"][0]
    assert instruction["service_laps"] == []
    assert all(instruction[name] == 0
               for name in ("executed", "overridden", "skipped", "not_reached"))


def test_incomplete_window_metadata_remains_invalid_evidence():
    bad = history()
    del bad[0]["actual_lap"]
    assert results([bad]).get_pit_plan_statistics()["drivers"][0]["invalid_histories"] == 1
