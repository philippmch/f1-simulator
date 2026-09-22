"""API, serializers, exports and CLI retain custom-plan semantics."""

import argparse
import csv
import importlib.util
import json
import runpy

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult
from f1sim.web import server


def _result():
    row = RaceResult(
        "A", "A", "A", 1, 90, 0, 1, 90, DriverStatus.FINISHED,
        strategy=["medium", "hard"], pit_laps=[2],
        pit_stop_details=[{
            "lap": 2, "from_compound": "medium", "to_compound": "hard",
            "tire_age": 1, "condition": "dry", "rain_intensity": 0,
            "track_wetness": 0, "control": "green", "lane_loss": 1,
            "service_time": 2, "queue_time": 0, "total_loss": 3,
            "decision_reason": "user_plan",
        }],
        pit_plan_history=[{
            "lap": 2, "compound": "hard", "status": "executed",
            "reason": "user_plan", "actual_compound": "hard", "actual_set_id": None,
        }],
    )
    return SimulationResults(
        1, "Track", {}, [[row]], [],
        input_snapshot={
            "schema_version": 5,
            "pit_plans": {"A": [{"lap": 2, "compound": "hard"}]},
        },
    )


def test_serialized_and_exported_plan_history(tmp_path):
    result = _result()
    payload = server._serialize_race_result(result.race_results[0][0])
    assert payload["pit_plan_history"][0]["reason"] == "user_plan"
    exporter = Exporter(tmp_path)
    csv_path = exporter.export_race_results_csv(result)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert json.loads(row["pit_plan_history"])[0]["actual_compound"] == "hard"
    saved = json.loads(exporter.export_statistics_json(result).read_text(encoding="utf-8"))
    assert saved["pit_plan_histories"][0]["pit_plan_history"] == (
        result.race_results[0][0].pit_plan_history
    )
    report = exporter.export_scenario_comparison_html({"<img>": result}).read_text(encoding="utf-8")
    assert "Custom pit-plan execution" in report
    assert "<img>" not in report
    assert "&lt;img&gt;" in report
    assert "2 (own lap)" in report


def test_dashboard_rejects_malformed_plan_before_live_io(monkeypatch):
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading"))
    request = server.DashboardRunRequest(
        pit_plans={"A": [{"lap": True, "compound": "hard"}]},
    )
    with pytest.raises(ValueError, match="pit plan"):
        server.run_dashboard_simulation(request)


def test_cli_parser_and_duplicate_plan_json(tmp_path):
    module = runpy.run_path("examples/simulate_race.py")
    parse = module["_pit_plans"]
    assert parse("A=18:medium;B=none") == {
        "A": [{"lap": 18, "compound": "medium"}], "B": [],
    }
    with pytest.raises(argparse.ArgumentTypeError):
        parse("A=1:soft")

    path = tmp_path / "plans.json"
    path.write_text('{"a": null, "a": []}', encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "compare_pit_plan_entrypoint", "examples/compare_pit_plans.py",
    )
    command = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(command)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        command._load_plans(path)
