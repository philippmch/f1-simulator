"""Strict SC/VSC scenarios reach HTTP, CLI and every dashboard workflow."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_dashboard_plan_comparison import CountingLoader, MultiDriverLoader

from f1sim.web import server

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = [{"lap": 1, "control": "vsc", "duration_laps": 1}]


@pytest.mark.parametrize("text", ["2:sc:true", "2.0:sc:2", "2:SC:2", "0:sc:2", "2:vsc:7",
                                  "2:sc:2,4:vsc:1", "[]", "automatic", "", "2:sc:1,"])
def test_cli_invalid_schedule_precedes_live_io(text):
    result = subprocess.run([sys.executable, str(ROOT / "examples/simulate_race.py"),
                             "--control-schedule", text], capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode == 2 and "--control-schedule" in result.stderr
    assert "Fetching the current calendar" not in result.stdout


@pytest.mark.parametrize("value", [True, {}, "none", [{"lap": True, "control": "vsc",
    "duration_laps": 1}], [{"lap": 1, "control": "vsc", "duration_laps": "1"}],
    [{"lap": 1, "control": "sc", "duration_laps": 1}],
    [{"lap": 1, "control": "vsc", "duration_laps": 1, "extra": 0}]])
def test_dashboard_and_http_invalid_source_precedes_live_io(monkeypatch, value):
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live I/O attempted"))
    with pytest.raises(ValueError, match="control_schedule"):
        server.run_dashboard_simulation(server.DashboardRunRequest(control_schedule=value))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={"control_schedule": value})
    assert response.status_code == 400 and "control_schedule" in response.text


@pytest.mark.parametrize("schedule", [[], SCHEDULE])
@pytest.mark.parametrize("mode", ["ordinary", "comparison", "selection", "rivals"])
def test_dashboard_retains_source_and_global_evidence_in_every_workflow(
    monkeypatch, schedule, mode,
):
    monkeypatch.setattr(server, "_get_loader", lambda: MultiDriverLoader())
    values = dict(simulations=10, parallel=False, scenarios="dry", control_schedule=schedule)
    if mode == "comparison":
        values.update(compare_automatic=True, pit_plans={"A": []})
    if mode in ("selection", "rivals"):
        selection = dict(driver_id="A", plans={"auto": None, "hold": []}, reference_label="auto",
                         training_simulations=2, validation_simulations=2)
        if mode == "rivals":
            selection["rival_scenarios"] = {"source": {"weight": 1, "pit_plans": {"B": []}}}
        values["pit_plan_selection"] = selection
    payload = server.run_dashboard_simulation(server.DashboardRunRequest(**values))
    assert payload["request"]["control_schedule"] == schedule

    def summaries(value):
        if isinstance(value, dict):
            if "control_schedule_statistics" in value:
                yield value
            for child in value.values():
                yield from summaries(child)
        elif isinstance(value, list):
            for child in value:
                yield from summaries(child)

    saved = list(summaries(payload))
    assert len(saved) >= (1 if mode == "ordinary" else 2)
    for row in saved:
        assert row["simulation_inputs"]["schema_version"] == 11
        assert row["simulation_inputs"]["control_schedule"] == schedule
        stats = row["control_schedule_statistics"]
        assert stats["source"] == "controlled"
        assert stats["valid_history_races"] == len(row["control_schedule_histories"])
        assert "observe announcements only at their crossings" in row["control_schedule_context"]


def test_rival_http_control_override_is_strict_and_keeps_empty_list(monkeypatch):
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live I/O attempted"))
    selection = {"driver_id": "A", "plans": {"auto": None, "hold": []}, "reference_label": "auto",
                 "training_simulations": 2, "validation_simulations": 2,
                 "rival_scenarios": {"none": {"weight": 1, "pit_plans": {},
                                               "control_schedule": []}}}
    valid = server.DashboardPitPlanSelectionRequest.model_validate(selection)
    assert valid.rival_scenarios["none"].control_schedule == []
    selection["rival_scenarios"]["none"]["control_schedule"] = "none"
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={"pit_plan_selection": selection})
    assert response.status_code == 422 and "control_schedule" in response.text


@pytest.mark.parametrize("text,schedule", [("none", []), ("1:vsc:1", SCHEDULE)])
def test_cli_forwards_source_to_all_scenarios_and_preflights_distance(monkeypatch, text, schedule):
    spec = importlib.util.spec_from_file_location("control_cli", ROOT / "examples/simulate_race.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "CurrentSeasonDataLoader", lambda **kw: CountingLoader())
    captured = []
    original = cli.MonteCarloRunner

    def capture(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(cli, "MonteCarloRunner", capture)
    argv = ["simulate_race", "-n", "1", "--no-parallel", "--scenarios", "dry,light_rain",
            "--control-schedule", text]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    assert len(captured) == 2 and all(row["control_schedule"] == schedule for row in captured)
    captured.clear()
    monkeypatch.setattr(sys, "argv", argv[:-1] + ["4:vsc:1"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2 and captured == []


def test_dashboard_distance_checked_before_trials(monkeypatch):
    monkeypatch.setattr(server, "_get_loader", lambda: MultiDriverLoader())
    monkeypatch.setattr(server, "MonteCarloRunner", lambda **kw: pytest.fail("trial attempted"))
    with pytest.raises(ValueError, match="control_schedule"):
        server.run_dashboard_simulation(server.DashboardRunRequest(
            control_schedule=[{"lap": 4, "control": "vsc", "duration_laps": 1}],
        ))
