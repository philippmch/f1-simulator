"""Qualifying session weather at the CLI and dashboard boundaries."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from test_dashboard_plan_comparison import CountingLoader, MultiDriverLoader

from f1sim.models import Weather
from f1sim.simulation.qualifying_weather import validate_qualifying_weather
from f1sim.web import server

ROOT = Path(__file__).resolve().parents[1]
PROFILE = {"Q1": {"condition": "heavy_rain", "rain_intensity": 0.8, "track_wetness": 0.8}}


@pytest.mark.parametrize("value", ["not json", "[]", '{"q1":{}}',
    '{"Q1":{"rain_intensity":true}}', '{"Q2":{"track_wetness":"0.4"}}',
    '{"Q3":{"rain_intensity":NaN}}', '{"Q1":{"unknown":0}}'])
def test_cli_invalid_weather_precedes_live_loading(value):
    result = subprocess.run([sys.executable, str(ROOT / "examples/simulate_race.py"),
                             "--qualifying-weather", value], capture_output=True,
                            text=True, timeout=10, check=False)
    assert result.returncode == 2
    assert "qualifying weather" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Fetching the current calendar" not in result.stdout


@pytest.mark.parametrize("value", [[], True, {"q1": {}}, {"Q1": []},
    {"Q2": {"rain_intensity": True}}, {"Q3": {"track_wetness": "0.4"}},
    {"Q1": {"rain_intensity": float("inf")}}, {"Q1": {"extra": 0}}])
def test_dashboard_invalid_weather_precedes_live_loading(monkeypatch, value):
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading attempted"))
    with pytest.raises(ValueError, match="qualifying_weather"):
        server.run_dashboard_simulation(server.DashboardRunRequest(qualifying_weather=value))


@pytest.mark.parametrize("profile", [None, {}, PROFILE])
def test_dashboard_runner_optional_weather_and_isolation(monkeypatch, profile):
    captured = []
    monkeypatch.setattr(server, "MonteCarloRunner", lambda **kwargs: captured.append(kwargs))
    request = server.DashboardRunRequest(qualifying_weather=profile)
    for weather in [Weather(), Weather(condition="light_rain", rain_intensity=0.3)]:
        server._dashboard_runner(drivers=[], cars={}, track=None, weather=weather,
            seed=7, request=request, tire_inventory=None, starting_tires={},
            starting_tire_ages={}, pit_plans=None, copy_inputs=True)
    if profile:
        expected = validate_qualifying_weather(profile)
        assert captured[0]["qualifying_weather"] == expected == captured[1]["qualifying_weather"]
        captured[0]["qualifying_weather"]["Q1"]["rain_intensity"] = 0
        assert captured[1]["qualifying_weather"] == expected
        assert profile == PROFILE
    else:
        assert all("qualifying_weather" not in kwargs for kwargs in captured)


@pytest.mark.parametrize("mode", ["ordinary", "comparison", "selection", "rivals"])
def test_dashboard_weather_reaches_all_variants(monkeypatch, mode):
    monkeypatch.setattr(server, "_get_loader", lambda: MultiDriverLoader())
    values = dict(simulations=10, parallel=False, scenarios="dry,light_rain",
                  qualifying_weather=PROFILE)
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
    assert payload["request"]["qualifying_weather"] == validate_qualifying_weather(PROFILE)
    for name, scenario in payload["scenarios"].items():
        snapshot = scenario["simulation_inputs"]
        assert snapshot["qualifying_weather"] == validate_qualifying_weather(PROFILE)
        assert "Q1:" in scenario["qualifying_weather_context"]
        assert "Q2:" in scenario["qualifying_weather_context"]
        assert "heavy" in scenario["qualifying_weather_context"].lower()
        if mode == "comparison":
            reference = payload["automatic_reference"]["scenarios"][name]
            assert (reference["simulation_inputs"]["qualifying_weather"]
                    == snapshot["qualifying_weather"])


    def saved_inputs(value):
        if isinstance(value, dict):
            if isinstance(value.get("simulation_inputs"), dict):
                yield value["simulation_inputs"]
            for child in value.values():
                yield from saved_inputs(child)
        elif isinstance(value, list):
            for child in value:
                yield from saved_inputs(child)
    snapshots = list(saved_inputs(payload))
    assert len(snapshots) >= (2 if mode == "ordinary" else 4)
    assert all(snapshot["qualifying_weather"] == validate_qualifying_weather(PROFILE)
               for snapshot in snapshots)
    contexts = [scenario["qualifying_weather_context"]
                for scenario in payload["scenarios"].values()]
    assert "Q2: dry" in contexts[0]
    assert "Q2: light rain" in contexts[1]


@pytest.mark.parametrize("value", [[], {"Q1": {"rain_intensity": True}},
    {"Q3": {"track_wetness": "0.4"}}, {"Q2": {"extra": 0}}])
def test_http_weather_validation_precedes_live_io(monkeypatch, value):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading attempted"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={"qualifying_weather": value})
    assert response.status_code == 400
    assert "qualifying_weather" in response.text


def test_cli_valid_weather_shared_across_race_scenarios(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "qualifying_cli", ROOT / "examples/simulate_race.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    monkeypatch.setattr(cli, "CurrentSeasonDataLoader", lambda **kwargs: CountingLoader())
    captured = []
    original = cli.MonteCarloRunner
    def runner(**kwargs):
        captured.append(kwargs)
        return original(**kwargs)
    monkeypatch.setattr(cli, "MonteCarloRunner", runner)
    monkeypatch.setattr(sys, "argv", ["simulate_race", "-n", "1", "--no-parallel",
        "--scenarios", "dry,light_rain", "--qualifying-weather", json.dumps(PROFILE)])
    assert cli.main() == 0
    assert len(captured) == 2
    assert captured[0]["weather"].condition != captured[1]["weather"].condition
    assert all(kwargs["qualifying_weather"] == validate_qualifying_weather(PROFILE)
               for kwargs in captured)
