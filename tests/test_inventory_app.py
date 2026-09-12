"""Finite set inputs stay strict and their audit records survive exports."""

import argparse
import csv
import json
import runpy

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult
from f1sim.web import server


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_finite_dashboard_real_runner(monkeypatch, engine):
    loader = runpy.run_path("tests/test_engine_entrypoints.py")["SyntheticLoader"]
    monkeypatch.setattr(server, "_get_loader", loader)
    pool = {"A": [{"compound": "hard", "age": 5}, {"compound": "soft"},
                  {"compound": "intermediate"}, {"compound": "wet"}]}
    payload = server.run_dashboard_simulation(server.DashboardRunRequest(
        simulations=10, scenarios="dry,light_rain", seed=7, parallel=False,
        race_engine=engine, starting_tires={"A": "hard"}, starting_tire_ages={"A": 5},
        tire_inventory=pool,
    ))
    assert payload["request"]["tire_inventory"]["A"][0] == {
        "id": "set-1", "compound": "hard", "age": 5,
    }
    assert 'id' not in pool['A'][0]
    for scenario in payload["scenarios"].values():
        assert (scenario["simulation_inputs"]["tire_inventory"]
                == payload["request"]["tire_inventory"])
        result = scenario["sample_race"][0]
        assert result["tire_set_history"][0]["age_at_fit"] == 5
        assert {item["id"] for item in result["tire_inventory"]} == {
            "set-1", "set-2", "set-3", "set-4",
        }
    assert 'Final race set pool' in payload['comparison_report_html']


@pytest.mark.parametrize("inventory", [[], {"A": []}, {"A": ["soft"]},
    *({"A": [{"compound": "soft", "age": age}]} for age in [True, 1.0, "1", -1, 1001]),
    {"A": [{"compound": "soft", "extra": 0}]}])
def test_http_inventory_rejected_before_live_io(monkeypatch, tmp_path, inventory):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_get_loader", lambda **kw: pytest.fail("live loading"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={"tire_inventory": inventory})
    assert response.status_code == 400
    assert "tire_inventory" in response.text


def test_cli_inventory_parser():
    parse = runpy.run_path("examples/simulate_race.py")["_tire_inventory"]
    assert parse("A=soft@5,hard")["A"] == [
        {"id": "set-1", "compound": "soft", "age": 5},
        {"id": "set-2", "compound": "hard", "age": 0},
    ]
    for text in ("A=soft@1.5", "A=soft;A=hard", "A=", "A=soft@1001"):
        with pytest.raises(argparse.ArgumentTypeError):
            parse(text)


def test_inventory_serialization_and_escaped_ledger(tmp_path):
    result = RaceResult("A", "A", "A", 1, 90, 0, 0, 90, DriverStatus.FINISHED)
    assert server._serialize_race_result(result)["tire_set_history"] is None
    legacy = RaceResult("B", "B", "B", 2, 91, 1, 0, 91, DriverStatus.FINISHED)
    exporter = Exporter(tmp_path)
    legacy_results = SimulationResults(1, "T", {}, [[legacy]], [])
    with exporter.export_race_results_csv(legacy_results).open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames[-1] == 'points_awarded'
        assert 'tire_inventory' not in reader.fieldnames
    result.tire_set_history = [{"lap": 1, "kind": "start", "set_id": '<img src=x>',
        "compound": "soft", "age_at_fit": 5, "age_at_end": 6, "laps_used": 1}]
    result.tire_inventory = [{"id": '<img src=x>', "compound": "soft", "age": 6,
                             "current": True, "available": False, "unavailable": False}]
    saved = SimulationResults(1, "T", {}, [[result, legacy]], [])
    serialized = server._serialize_race_result(result)
    serialized["tire_set_history"][0]["age_at_end"] = 99
    assert result.tire_set_history[0]["age_at_end"] == 6
    exporter = Exporter(tmp_path)
    html = exporter._tire_set_ledger_html(saved)
    assert '<img' not in html and '&lt;img src=x&gt;' in html
    assert 'Final race set pool' in html and 'Physical set fittings' in html
    rows = list(csv.DictReader(exporter.export_race_results_csv(saved).open(encoding="utf-8")))
    assert json.loads(rows[0]["tire_inventory"]) == result.tire_inventory
    assert rows[1]["tire_inventory"] == rows[1]["tire_set_history"] == ''
    payload = json.loads(exporter.export_statistics_json(saved).read_text(encoding="utf-8"))
    assert payload["tire_set_ledgers"][0]["tire_set_history"] == result.tire_set_history
    saved.input_snapshot = {"tire_inventory": {"A": [
        {"id": "<img src=x>", "compound": "soft", "age": 5},
    ]}}
    for path in (exporter.export_report_html(saved),
                 exporter.export_scenario_comparison_html({"finite": saved})):
        report = path.read_text(encoding="utf-8")
        assert '<img' not in report and '&lt;img src=x&gt;' in report
        assert 'Input race set pools' in report and 'Final race set pool' in report
