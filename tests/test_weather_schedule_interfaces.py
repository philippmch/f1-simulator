"""Strict rainfall inputs fail before live I/O and reach all dashboard workflows."""
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from test_dashboard_plan_comparison import CountingLoader, MultiDriverLoader

from f1sim.web import server

ROOT = Path(__file__).resolve().parents[1]
SCHEDULE = [{"lap": 2, "rain_intensity": .8}, {"lap": 3, "rain_intensity": 0}]


@pytest.mark.parametrize("value", ["1=0", "2=true", "2=NaN", "2=Infinity", "2=1.1",
                                  "2.0=0", "2=0:wet", "2=0:DRY", '2="0"'])
def test_cli_strict_parsing_before_io(value):
    result = subprocess.run([sys.executable, str(ROOT / "examples/simulate_race.py"),
                             "--rainfall-step", value], capture_output=True, text=True,
                            timeout=10, check=False)
    assert result.returncode == 2
    assert "rainfall step" in result.stderr
    assert "Fetching the current calendar" not in result.stdout


def test_cli_duplicate_steps_before_io():
    result = subprocess.run([sys.executable, str(ROOT / "examples/simulate_race.py"),
                             "--rainfall-step", "2=0", "--rainfall-step", "2=1"],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 2 and "weather_schedule" in result.stderr
    assert "Fetching the current calendar" not in result.stdout


@pytest.mark.parametrize("value", [True, {}, [{"lap": 1, "rain_intensity": 0}],
    [{"lap": 2, "rain_intensity": True}], [{"lap": 2, "rain_intensity": "0"}],
    [{"lap": 2, "rain_intensity": float("inf")}], [{"lap": 2, "rain_intensity": 0, "extra": 1}]])
def test_dashboard_invalid_schedule_precedes_live_io(monkeypatch, value):
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading attempted"))
    with pytest.raises(ValueError, match="weather_schedule"):
        server.run_dashboard_simulation(server.DashboardRunRequest(weather_schedule=value))


def test_http_invalid_schedule_precedes_live_io(monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading attempted"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={
            "weather_schedule": [{"lap": 2, "rain_intensity": True}],
        })
    assert response.status_code == 400 and "weather_schedule" in response.text


@pytest.mark.parametrize("mode", ["ordinary", "comparison", "selection", "rivals"])
def test_dashboard_schedule_reaches_every_saved_workflow(monkeypatch, mode):
    monkeypatch.setattr(server, "_get_loader", lambda: MultiDriverLoader())
    values = dict(simulations=10, parallel=False, scenarios="dry,light_rain",
                  weather_schedule=SCHEDULE)
    if mode == "comparison":
        values.update(compare_automatic=True, pit_plans={"A": []})
    if mode in ("selection", "rivals"):
        selection = dict(driver_id="A", plans={"auto": None, "hold": []},
                         reference_label="auto", training_simulations=2,
                         validation_simulations=2)
        if mode == "rivals":
            selection["rival_scenarios"] = {"hold": {"weight": 1, "pit_plans": {"B": []}}}
        values["pit_plan_selection"] = selection
    payload = server.run_dashboard_simulation(server.DashboardRunRequest(**values))
    assert payload["request"]["weather_schedule"] == SCHEDULE
    def snapshots(value):
        if isinstance(value, dict):
            if isinstance(value.get("simulation_inputs"), dict):
                yield value["simulation_inputs"]
            for child in value.values():
                yield from snapshots(child)
        elif isinstance(value, list):
            for child in value:
                yield from snapshots(child)
    saved = list(snapshots(payload))
    assert len(saved) >= (2 if mode == "ordinary" else 4)
    assert all(row["weather_schedule"] == SCHEDULE and row["schema_version"] == 8 for row in saved)
    assert all("known to strategy" in row["weather_schedule_context"]
               for row in payload["scenarios"].values())


def test_cli_valid_schedule_forwarded_and_distance_checked(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "schedule_cli", ROOT / "examples/simulate_race.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "CurrentSeasonDataLoader", lambda **kwargs: CountingLoader())
    captured = []
    original = cli.MonteCarloRunner
    def capture(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(cli, "MonteCarloRunner", capture)
    argv = ["simulate_race", "-n", "1", "--no-parallel", "--scenarios", "dry,light_rain",
            "--rainfall-step", "2=0.8", "--rainfall-step", "3=0"]
    monkeypatch.setattr(sys, "argv", argv)
    assert cli.main() == 0
    assert len(captured) == 2
    assert all(kwargs["weather_schedule"] == SCHEDULE for kwargs in captured)
    captured.clear()
    monkeypatch.setattr(sys, "argv", argv[:-1] + ["4=0"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert captured == []


def test_dashboard_distance_checked_before_trials(monkeypatch):
    monkeypatch.setattr(server, "_get_loader", lambda: MultiDriverLoader())
    monkeypatch.setattr(server, "MonteCarloRunner", lambda **kwargs: pytest.fail("trial attempted"))
    with pytest.raises(ValueError, match="weather_schedule"):
        server.run_dashboard_simulation(server.DashboardRunRequest(
            weather_schedule=[{"lap": 4, "rain_intensity": 0}],
        ))
