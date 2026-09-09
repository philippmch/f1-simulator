"""Weather draw policies survive exports, process workers and legacy replay."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.analysis.strategy_comparison import (
    compare_saved_race_engines,
    compare_saved_starting_tires,
)
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter
from f1sim.output.comparison import render_comparison_report


def runner(engine="standard", policy="isolated_weather_v1"):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)]
    return MonteCarloRunner(
        drivers, {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers},
        Track(id="t", name="Saved", country="T", total_laps=8, base_lap_time=90),
        Weather(change_probability=1), seed=37, race_engine=engine, rng_policy=policy,
    )


def save(tmp_path, result, legacy=False):
    path = Exporter(tmp_path).export_statistics_json(result)
    if legacy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["simulation_inputs"]["schema_version"] = 1
        payload["simulation_inputs"].pop("rng_policy")
        path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("policy", ["shared_v1", "isolated_weather_v1"])
def test_serial_process_and_second_trial_replay(tmp_path, monkeypatch, engine, policy):
    source = runner(engine, policy)
    serial = source.run(3, parallel=False)
    parallel = source.run(3, parallel=True, max_workers=2)
    assert parallel.race_results == serial.race_results
    assert parallel.qualifying_results == serial.qualifying_results
    assert parallel.event_stats == serial.event_stats
    assert parallel.weather_histories == serial.weather_histories
    assert parallel.input_snapshot == serial.input_snapshot
    assert serial.input_snapshot["rng_policy"] == policy
    assert serial.input_snapshot["schema_version"] == 2
    path = save(tmp_path, serial, legacy=policy == "shared_v1")
    before = path.read_bytes()

    def no_network(*args, **kwargs):
        pytest.fail("Saved-input replay accessed a provider")

    monkeypatch.setattr("socket.socket.connect", no_network)
    replay = replay_saved_simulation(path, 2)
    assert replay.race_results == [serial.race_results[1]]
    assert replay.qualifying_results == [serial.qualifying_results[1]]
    assert replay.weather_histories == [serial.weather_histories[1]]
    assert replay.input_snapshot["rng_policy"] == policy
    assert path.read_bytes() == before


@pytest.mark.parametrize("comparison", ["tyres", "engines"])
@pytest.mark.parametrize("policy", ["shared_v1", "isolated_weather_v1"])
@pytest.mark.parametrize("override", [None, "isolated_weather_v1"])
def test_comparison_policy_inheritance_override_and_replay(
    tmp_path, monkeypatch, comparison, policy, override,
):
    path = save(tmp_path, runner(policy=policy).run(1, parallel=False),
                legacy=policy == "shared_v1")
    before = path.read_bytes()
    monkeypatch.setattr("socket.socket.connect", lambda *a: pytest.fail("network accessed"))
    arguments = {"num_simulations": 2, "rng_policy": override}
    results = (
        compare_saved_starting_tires(path, "0", ["soft", "hard"], **arguments)
        if comparison == "tyres" else compare_saved_race_engines(path, **arguments)
    )
    effective = override or policy
    assert {r.input_snapshot["rng_policy"] for r in results.values()} == {effective}
    first, second = results.values()
    assert first.qualifying_results == second.qualifying_results
    if effective == "isolated_weather_v1":
        for a, b in zip(first.weather_histories, second.weather_histories, strict=True):
            count = min(len(a), len(b))
            assert count > 1
            assert a[:count] == b[:count]
    comparison_path = Exporter(tmp_path).export_scenario_comparison_json(results)
    for label, result in results.items():
        replay = replay_saved_simulation(comparison_path, simulation=2, scenario=label)
        assert replay.input_snapshot["rng_policy"] == effective
        assert replay.race_results == [result.race_results[1]]
        assert replay.weather_histories == [result.weather_histories[1]]
    assert path.read_bytes() == before


@pytest.mark.parametrize("value", [None, True, 1, [], {}, "unknown"])
def test_invalid_saved_policy_is_rejected_before_execution(tmp_path, monkeypatch, value):
    path = save(tmp_path, runner().run(1, parallel=False))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["simulation_inputs"]["rng_policy"] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *a, **k: pytest.fail("ran invalid input"))
    with pytest.raises(ValueError, match="rng_policy"):
        replay_saved_simulation(path)


def test_new_schema_requires_explicit_policy(tmp_path, monkeypatch):
    path = save(tmp_path, runner().run(1, parallel=False))
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["simulation_inputs"].pop("rng_policy")
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *a, **k: pytest.fail("ran invalid input"))
    with pytest.raises(ValueError, match="rng_policy"):
        replay_saved_simulation(path)


@pytest.mark.parametrize("value", [True, 1, [], {}, "unknown"])
def test_invalid_comparison_override_precedes_loading(monkeypatch, value):
    monkeypatch.setattr(
        "f1sim.analysis.strategy_comparison._load_saved_runner",
        lambda *a: pytest.fail("loaded invalid input"),
    )
    for function, args in ((compare_saved_race_engines, ("unused",)),
                           (compare_saved_starting_tires, ("unused", "0"))):
        with pytest.raises(ValueError, match="rng_policy"):
            function(*args, rng_policy=value)


@pytest.mark.parametrize("command", ["compare_starting_tyres.py", "compare_race_engines.py"])
def test_cli_upgrades_legacy_weather_and_exports_replayable_inputs(tmp_path, command):
    path = save(tmp_path, runner(policy="shared_v1").run(1, parallel=False), legacy=True)
    script = Path(__file__).parents[1] / "examples" / command
    output = tmp_path / "comparison"
    args = [sys.executable, str(script), str(path), "--simulations", "1",
            "--independent-weather", "--export", "--output-dir", str(output)]
    if command == "compare_starting_tyres.py":
        args.extend(["--driver", "0", "--compounds", "soft,hard"])
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "Weather draws: independent of race decisions" in completed.stdout
    exported = list(output.glob("*statistics.json"))
    assert len(exported) == 2
    for saved in exported:
        payload = json.loads(saved.read_text(encoding="utf-8"))
        assert payload["simulation_inputs"]["rng_policy"] == "isolated_weather_v1"
        replay = replay_saved_simulation(saved)
        assert replay.weather_histories == payload["weather_histories"]


def test_html_context_identifies_weather_randomness_and_legacy():
    result = runner().run(1, parallel=False)
    assert "weather draws independent of race decisions" in render_comparison_report({"x": result})
    result.input_snapshot.pop("rng_policy")
    report = render_comparison_report({"x": result})
    assert "weather draws shared with race events (legacy)" in report
    result.input_snapshot["rng_policy"] = {"bad": "policy"}
    assert "weather draws not recorded" in render_comparison_report({"x": result})
