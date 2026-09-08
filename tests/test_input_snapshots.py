"""Saved inputs describe the actual run and cannot drift with caller mutation."""

import copy
import json

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.export import Exporter


def test_snapshot_uses_run_time_inputs_and_is_detached_from_later_changes(tmp_path):
    driver = Driver(id="A", name="A", team_id="T")
    car = Car(team_id="T", team_name="T")
    track = Track(id="t", name="T", country="T", total_laps=3, base_lap_time=90)
    weather = Weather(change_probability=0)
    runner = MonteCarloRunner([driver], {"T": car}, track, weather, seed=7)
    car.base_pace = .95
    results = runner.run(1, parallel=False)
    snapshot = copy.deepcopy(results.input_snapshot)
    assert snapshot["cars"]["T"]["base_pace"] == .95
    assert snapshot["drivers"][0]["name"] == "A"
    assert set(snapshot["runtime"]) == {
        "f1sim", "python", "numpy", "pydantic", "simulation_source_sha256",
    }
    assert len(snapshot["runtime"]["simulation_source_sha256"]) == 64
    driver.name = "Changed"
    car.base_pace = .1
    track.total_laps = 4
    weather.rain_intensity = .9
    assert results.input_snapshot == snapshot
    path = Exporter(tmp_path).export_statistics_json(results)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["simulation_inputs"] == json.loads(json.dumps(snapshot))
    assert saved["metadata"]["seed"] == 7
    next_result = runner.run(1, parallel=False)
    assert next_result.input_snapshot["track"]["total_laps"] == 4
    next_result.input_snapshot["cars"]["T"]["base_pace"] = .2
    assert car.base_pace == .1
    assert results.input_snapshot == snapshot
