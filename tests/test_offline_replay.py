"""Saved inputs reproduce individual seeded trials without a data provider."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def run(engine="standard", seed=71, count=2):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id="t", name="Saved circuit", country="T", total_laps=5, base_lap_time=90)
    return MonteCarloRunner(drivers, cars, track, Weather(change_probability=0),
                            seed=seed, race_engine=engine).run(count, parallel=False)


@pytest.fixture
def saved(tmp_path):
    return Exporter(tmp_path).export_statistics_json(run())


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_roundtrip_second_trial_and_events(tmp_path, engine, monkeypatch):
    original = run(engine)
    path = Exporter(tmp_path).export_statistics_json(original)

    def no_network(*args, **kwargs):
        raise AssertionError("Replay must not contact any provider")

    monkeypatch.setattr("socket.socket.connect", no_network)
    replay = replay_saved_simulation(path, simulation=2)
    expected = run(engine, seed=72, count=1)
    assert replay.race_results == [original.race_results[1]]
    assert replay.qualifying_results == [original.qualifying_results[1]]
    assert replay.event_stats == expected.event_stats
    assert replay.seed == 72
    assert replay.race_engine == engine
    assert replay.num_simulations == 1
    assert replay.parallel is False


@pytest.mark.parametrize("index", [True, False, 0, -1, 3, 1.0, "1", None])
def test_invalid_index(saved, index):
    with pytest.raises(ValueError, match="simulation"):
        replay_saved_simulation(saved, index)


@pytest.mark.parametrize("field", ["base_lap_time", "pit_lane_delta", "sector_time"])
@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_overflowing_track_times_fail_before_execution(saved, field, engine, monkeypatch):
    data = json.loads(saved.read_text(encoding="utf-8"))
    data["metadata"]["race_engine"] = engine
    track = data["simulation_inputs"]["track"]
    if field == "sector_time":
        track["sectors"] = [{"number": 1, "base_time": float("inf")}]
    else:
        track[field] = float("inf")
    # 1e400 is valid JSON numeric syntax but overflows a Python float.
    saved.write_text(json.dumps(data).replace("Infinity", "1e400"), encoding="utf-8")

    def unexpected_run(*args, **kwargs):
        pytest.fail("Invalid timing input reached simulation execution")

    monkeypatch.setattr(MonteCarloRunner, "run", unexpected_run)
    with pytest.raises(ValueError, match="finite number"):
        replay_saved_simulation(saved)


@pytest.mark.parametrize("field,value", [
    ("seed", True), ("seed", -1), ("seed", None), ("seed", 1.0),
    ("num_simulations", False), ("num_simulations", 0),
    ("race_engine", "unknown"),
])
def test_invalid_metadata(saved, field, value):
    data = json.loads(saved.read_text(encoding="utf-8"))
    data["metadata"][field] = value
    saved.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=field):
        replay_saved_simulation(saved)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 3), ("schema_version", 1.0),
    ("drivers", {}), ("drivers", [None]), ("cars", []), ("cars", {"0": None}),
    ("track", []), ("weather", None), ("runtime", []),
    ("track", {"id": "t", "name": "T", "country": "T", "total_laps": -1}),
])
def test_invalid_inputs(saved, field, value):
    data = json.loads(saved.read_text(encoding="utf-8"))
    data["simulation_inputs"][field] = value
    saved.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_saved_simulation(saved)


def test_legacy_file_has_clear_error(saved):
    data = json.loads(saved.read_text(encoding="utf-8"))
    data.pop("simulation_inputs")
    saved.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="legacy exports cannot replay"):
        replay_saved_simulation(saved)


def test_dashboard_scenario_selection(saved):
    expected = replay_saved_simulation(saved, 2)
    data = json.loads(saved.read_text(encoding="utf-8"))
    scenario = {**data["metadata"], "simulation_inputs": data["simulation_inputs"]}
    saved.write_text(json.dumps({"scenarios": {"dry": scenario}}), encoding="utf-8")
    assert replay_saved_simulation(saved, 2) == expected
    saved.write_text(json.dumps({"scenarios": {"dry": scenario, "wet": scenario}}),
                     encoding="utf-8")
    with pytest.raises(ValueError, match="Multiple saved scenarios"):
        replay_saved_simulation(saved)
    with pytest.raises(ValueError, match="Unknown saved scenario"):
        replay_saved_simulation(saved, scenario="missing")
    assert replay_saved_simulation(saved, 2, "wet") == expected


def test_statistics_rejects_named_scenario(saved):
    with pytest.raises(ValueError, match="no named scenarios"):
        replay_saved_simulation(saved, scenario="dry")


def test_duplicate_driver_rejected(saved):
    data = json.loads(saved.read_text(encoding="utf-8"))
    inputs = data["simulation_inputs"]
    inputs["drivers"].append(inputs["drivers"][0])
    saved.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        replay_saved_simulation(saved)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("case", ["empty", "partial", "mismatched_key"])
def test_permitted_runner_inputs_roundtrip(tmp_path, engine, case):
    drivers = [] if case == "empty" else [
        Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)
    ]
    cars = {} if case == "empty" else {
        "0": Car(team_id="other" if case == "mismatched_key" else "0", team_name="Car")
    }
    track = Track(id="t", name="T", country="T", total_laps=3, base_lap_time=90)
    original = MonteCarloRunner(drivers, cars, track, Weather(change_probability=0),
                               seed=71, race_engine=engine).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    replay = replay_saved_simulation(path)
    assert replay == original


def test_cli_named_scenario(saved):
    data = json.loads(saved.read_text(encoding="utf-8"))
    scenario = {**data["metadata"], "simulation_inputs": data["simulation_inputs"]}
    saved.write_text(json.dumps({"scenarios": {"dry": scenario, "wet": scenario}}),
                     encoding="utf-8")
    script = Path(__file__).parents[1] / "examples" / "replay_simulation.py"
    result = subprocess.run([sys.executable, str(script), str(saved), "--scenario", "wet"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "effective seed: 71" in result.stdout


def test_cli_displays_actual_results_and_exports_unique_bundle(saved, tmp_path):
    script = Path(__file__).parents[1] / "examples" / "replay_simulation.py"
    output = tmp_path / "replays"
    command = [sys.executable, str(script), str(saved), "--simulation", "2",
               "--export", "--output-dir", str(output)]
    first = subprocess.run(command, capture_output=True, text=True, check=False)
    assert first.returncode == 0, first.stderr
    assert "effective seed: 72" in first.stdout
    assert "QUALIFYING RESULTS" in first.stdout
    assert "RACE RESULTS" in first.stdout
    second = subprocess.run(command, capture_output=True, text=True, check=False)
    assert second.returncode == 0, second.stderr
    assert len(list(output.glob("replay_*statistics.json"))) == 2


@pytest.mark.parametrize("content", ["{broken", "[]", "{}"])
def test_cli_invalid_file_is_clean_error(tmp_path, content):
    path = tmp_path / "invalid.json"
    path.write_text(content, encoding="utf-8")
    script = Path(__file__).parents[1] / "examples" / "replay_simulation.py"
    result = subprocess.run([sys.executable, str(script), str(path)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
