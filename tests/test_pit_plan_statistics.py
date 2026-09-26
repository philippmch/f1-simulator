import json

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output import Exporter
from f1sim.output.comparison import render_comparison_report
from f1sim.simulation.race import DriverStatus, RaceResult
from f1sim.web import server


def _row(driver_id, history):
    return RaceResult(
        driver_id, driver_id, "Team", 1, 90.0, 0.0, 0, 90.0,
        DriverStatus.FINISHED, pit_plan_history=history,
    )


def _results(plans, races, *, snapshot=True):
    input_snapshot = {"pit_plans": plans} if snapshot else None
    return SimulationResults(
        num_simulations=100,
        track_name="Track",
        driver_stats={},
        race_results=races,
        qualifying_results=[],
        seed=17,
        input_snapshot=input_snapshot,
    )


def _history(lap1_status="executed", lap2_status="overridden"):
    return [
        {"lap": 2, "compound": "hard", "status": lap1_status},
        {"lap": 5, "compound": "medium", "status": lap2_status},
    ]


def test_aggregate_counts_only_complete_valid_histories_per_driver():
    results = _results(
        {
            "A": [
                {"lap": 2, "compound": "hard"},
                {"lap": 5, "compound": "medium"},
            ],
            "B": [],
        },
        [
            [_row("A", _history()), _row("B", [])],
            [_row("A", _history("skipped", "not_reached"))],
            [_row("A", None), _row("B", [])],
            [_row("A", _history("executed", "unknown")), _row("B", None)],
        ],
    )

    summary = results.get_pit_plan_statistics()
    assert summary["status"] == "available"
    assert summary["recorded_trials"] == 4
    assert summary["drivers"] == [
        {
            "driver_id": "A",
            "no_elective_stops": False,
            "valid_histories": 2,
            "missing_histories": 1,
            "invalid_histories": 1,
            "instructions": [
                {
                    "lap": 2, "compound": "hard", "executed": 1,
                    "overridden": 0, "skipped": 1, "not_reached": 0,
                },
                {
                    "lap": 5, "compound": "medium", "executed": 0,
                    "overridden": 1, "skipped": 0, "not_reached": 1,
                },
            ],
        },
        {
            "driver_id": "B",
            "no_elective_stops": True,
            "valid_histories": 2,
            "missing_histories": 2,
            "invalid_histories": 0,
            "instructions": [],
        },
    ]
    for driver in summary["drivers"]:
        assert (driver["valid_histories"] + driver["missing_histories"]
                + driver["invalid_histories"] == summary["recorded_trials"])


@pytest.mark.parametrize(
    "history, duplicate_row",
    [
        ("malformed", False),
        ([], False),
        ([{"lap": 2, "compound": "hard", "status": "executed"}], False),
        ([
            {"lap": 2, "compound": "hard", "status": "executed"},
            {"lap": 2, "compound": "medium", "status": "skipped"},
        ], False),
        ([
            {"lap": 5, "compound": "medium", "status": "executed"},
            {"lap": 2, "compound": "hard", "status": "skipped"},
        ], False),
        ([
            {"lap": 2, "compound": "hard", "status": []},
            {"lap": 5, "compound": "medium", "status": "skipped"},
        ], False),
        (_history(), True),
    ],
)
def test_malformed_and_duplicate_driver_histories_are_invalid(history, duplicate_row):
    rows = [_row("A", history)]
    if duplicate_row:
        rows.append(_row("A", _history()))
    result = _results(
        {"A": [
            {"lap": 2, "compound": "hard"},
            {"lap": 5, "compound": "medium"},
        ]},
        [rows],
    )
    driver = result.get_pit_plan_statistics()["drivers"][0]
    assert driver["invalid_histories"] == 1
    assert driver["missing_histories"] == 0
    assert driver["valid_histories"] == 0
    assert all(
        instruction[key] == 0
        for instruction in driver["instructions"]
        for key in ("executed", "overridden", "skipped", "not_reached")
    )


def test_malformed_saved_plan_and_trial_container_do_not_crash_or_break_coverage():
    for raw_plans in (None, {"A": [{"lap": True, "compound": "hard"}]}):
        malformed_plan = _results(raw_plans, [[], []])
        assert malformed_plan.get_pit_plan_statistics() == {
            "status": "invalid", "recorded_trials": 2, "drivers": [],
        }

    malformed_trials = _results({"A": []}, [None, "not a race"])
    summary = malformed_trials.get_pit_plan_statistics()
    assert summary["status"] == "available"
    assert summary["drivers"][0]["invalid_histories"] == 2
    assert summary["drivers"][0]["valid_histories"] == 0
    assert summary["drivers"][0]["missing_histories"] == 0


def test_empty_plan_requires_an_empty_list_history_and_legacy_is_not_recorded():
    empty = _results({"A": []}, [[_row("A", [])], [_row("A", None)], [_row("A", [{}])]])
    driver = empty.get_pit_plan_statistics()["drivers"][0]
    assert driver["no_elective_stops"] is True
    assert (driver["valid_histories"], driver["missing_histories"],
            driver["invalid_histories"]) == (1, 1, 1)
    assert driver["instructions"] == []

    legacy = SimulationResults(9, "Track", {}, [], [])
    assert legacy.get_pit_plan_statistics() == {
        "status": "not_recorded", "recorded_trials": 0, "drivers": [],
    }
    no_saved_plans = _results({}, [[]], snapshot=False)
    assert no_saved_plans.get_pit_plan_statistics()["status"] == "not_recorded"


def test_dashboard_and_both_json_exports_include_aggregate(tmp_path):
    result = _results(
        {"A": [{"lap": 2, "compound": "hard"}]},
        [[_row("A", [{"lap": 2, "compound": "hard", "status": "executed"}])]],
    )
    dashboard = server._summarize_scenario_results({"custom": result})
    expected = result.get_pit_plan_statistics()
    assert dashboard["scenarios"]["custom"]["pit_plan_statistics"] == expected

    exporter = Exporter(tmp_path)
    single = json.loads(exporter.export_statistics_json(result).read_text(encoding="utf-8"))
    comparison = json.loads(
        exporter.export_scenario_comparison_json({"custom": result}).read_text(encoding="utf-8")
    )
    assert single["pit_plan_statistics"] == expected
    assert comparison["scenarios"]["custom"]["pit_plan_statistics"] == expected


def test_aggregate_html_precedes_details_and_escapes_saved_driver_labels(tmp_path):
    hostile = "<img src=x onerror=alert(1)>"
    result = _results(
        {hostile: [{"lap": 2, "compound": "hard"}]},
        [[_row(hostile, [{"lap": 2, "compound": "hard", "status": "executed"}])]],
    )
    exporter = Exporter(tmp_path)
    single = exporter.export_report_html(result).read_text(encoding="utf-8")
    comparison = render_comparison_report({"<script>": result})
    for report in (single, comparison):
        assert "History coverage" in report
        assert "1 valid, 0 missing, 0 invalid of 1 recorded trial" in report
        assert "Executed" in report
        assert hostile not in report
        assert "&lt;img" in report
    assert single.index("aggregate custom pit-plan outcomes") < single.index(
        "Requested and executed custom pit-plan history"
    )
    assert comparison.index("aggregate custom pit-plan outcomes") < comparison.index(
        "Trial-by-trial pit-plan history"
    )
    assert "<script>" not in comparison
    assert "&lt;script&gt;" in comparison


def test_comparison_report_handles_malformed_status_types():
    result = _results(
        {"A": [{"lap": 2, "compound": "hard"}]},
        [[_row("A", [{"lap": 2, "compound": "hard", "status": ["unexpected"]}])]],
    )
    assert "invalid of 1 recorded trial" in render_comparison_report({"bad": result})


def test_trial_report_does_not_call_empty_malformed_histories_no_stop_plan():
    result = _results(
        {
            "A": [{"lap": 2, "compound": "hard"}],
            "B": [],
            "C": [],
        },
        [[_row("A", []), _row("B", ""), _row("C", [])]],
    )
    report = render_comparison_report({"custom": result})
    assert (
        '<td>1</td><td>A</td><td colspan="6">Incomplete history; '
        'requested instructions are missing</td>'
    ) in report
    assert '<td>1</td><td>B</td><td colspan="6">Malformed history</td>' in report
    assert '<td>1</td><td>C</td><td colspan="6">Explicit no elective stops</td>' in report
