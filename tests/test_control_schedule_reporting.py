"""Incomplete global histories never become evidence of zero deployments."""

import csv
import json
from copy import deepcopy

import pytest
from test_control_schedule_replay import SCHEDULE
from test_qualifying_session_weather import runner

from f1sim.output import ConsoleOutput, Exporter
from f1sim.output.comparison import render_comparison_report, render_rival_strategy_selection_report
from f1sim.output.control_schedule_context import control_schedule_context


@pytest.mark.parametrize("bad", [[], {}, "<script>",
    [{"lap": True, "control": "safety_car", "duration_laps": 1,
      "status": "applied", "reason": "scheduled_announcement"}],
    [item | {"status": "applied", "reason": "unknown"} for item in SCHEDULE],
    [item | {"status": [], "reason": "scheduled_announcement"} for item in SCHEDULE],
    [item | {"status": "applied", "reason": "scheduled_announcement", "extra": 0}
     for item in SCHEDULE]])
def test_whole_malformed_trial_is_excluded(tmp_path, bad):
    result = runner(control_schedule=SCHEDULE).run(1, parallel=False)
    good = deepcopy(result.control_schedule_histories[0])
    result.race_results *= 3
    result.num_simulations = 4
    result.control_schedule_histories = [good, bad, None]
    stats = result.get_control_schedule_statistics()
    assert stats["valid_history_races"] == stats["missing_history_races"] == 1
    assert stats["invalid_history_races"] == stats["unrecorded_races"] == 1
    assert stats["applied"] + stats["suppressed"] + stats["not_reached"] == len(SCHEDULE)
    rows = list(csv.DictReader(Exporter(tmp_path).export_control_schedule_history_csv(
        result,
    ).open(encoding="utf-8")))
    assert rows[-2]["history_status"] == "invalid" and rows[-1]["history_status"] == "missing"
    assert len(rows) == len(SCHEDULE) + 2


@pytest.mark.parametrize("histories", [None, [], [None], {}, "invalid"])
def test_missing_or_invalid_ledger_keeps_coverage_unknown(tmp_path, histories):
    result = runner(control_schedule=[]).run(1, parallel=False)
    result.control_schedule_histories = histories
    stats = result.get_control_schedule_statistics()
    assert stats["valid_history_races"] == 0
    assert stats["invalid_history_races"] + stats["missing_history_races"] == 1
    assert stats["entries"] == []
    rows = list(csv.DictReader(Exporter(tmp_path).export_control_schedule_history_csv(
        result,
    ).open(encoding="utf-8")))
    assert rows[0]["history_status"] == (
        "invalid" if histories is not None and not isinstance(histories, list) else "missing"
    )


def test_empty_control_source_has_complete_zero_request_history(tmp_path):
    result = runner(control_schedule=[]).run(1, parallel=False)
    stats = result.get_control_schedule_statistics()
    assert stats["source"] == "controlled" and stats["valid_history_races"] == 1
    assert result.control_schedule_histories == [[]]
    result.control_schedule_histories.append([])
    assert result.get_control_schedule_statistics()["unexpected_histories"] == 1
    rows = list(csv.DictReader(Exporter(tmp_path).export_control_schedule_history_csv(
        result,
    ).open(encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["status"] == "no_requests"
    automatic = runner().run(1, parallel=False)
    assert automatic.get_control_schedule_statistics()["source"] == "automatic"
    automatic.input_snapshot = None
    assert automatic.get_control_schedule_statistics()["source"] == "not_recorded"


def test_reports_and_downloads_use_completed_evidence_not_mutable_configuration(tmp_path, capsys):
    configured = runner(control_schedule=SCHEDULE)
    result = configured.run(1, parallel=False)
    context = control_schedule_context(result.input_snapshot)
    configured.control_schedule.clear()
    assert result.input_snapshot["control_schedule"] == SCHEDULE
    ConsoleOutput.print_monte_carlo_summary(result)
    ConsoleOutput.print_scenario_comparison({"controlled": result})
    text = capsys.readouterr().out
    assert context in text and "1 complete, 0 missing, 0 invalid" in text
    exporter = Exporter(tmp_path)
    for html in (exporter.export_report_html(result).read_text(encoding="utf-8"),
                 render_comparison_report({"controlled": result})):
        assert context in html and "1 complete, 0 missing, 0 invalid" in html
        assert "SC/VSC requests in complete trial histories" in html
    saved = json.loads(exporter.export_statistics_json(result).read_text())
    assert saved["control_schedule_histories"] == result.control_schedule_histories
    assert saved["control_schedule_statistics"] == result.get_control_schedule_statistics()
    comparison = json.loads(exporter.export_scenario_comparison_json(
        {"controlled": result},
    ).read_text())
    assert comparison["scenarios"]["controlled"]["control_schedule_histories"] == (
        result.control_schedule_histories
    )
    assert "control_schedule_csv" in exporter.export_all(result)


@pytest.mark.parametrize("change", [{"schema_version": 10}, {"schema_version": 11.0},
    {"control_schedule_policy": "<img src=x>"}, {"control_schedule": "<script>"},
    {"control_schedule": None}])
def test_invalid_context_is_not_echoed_or_treated_as_automatic(change):
    result = runner(control_schedule=[]).run(1, parallel=False)
    result.input_snapshot.update(change)
    text = control_schedule_context(result.input_snapshot)
    assert text == "SC/VSC scenario: invalid saved schedule or policy."
    assert result.get_control_schedule_statistics()["source"] == "invalid"


def test_selection_context_is_escaped():
    html = render_rival_strategy_selection_report({
        "report_context": {"control_schedule_context": "SC/VSC: <img src=x>"},
    })
    assert "SC/VSC: &lt;img src=x&gt;" in html and "SC/VSC: <img src=x>" not in html
