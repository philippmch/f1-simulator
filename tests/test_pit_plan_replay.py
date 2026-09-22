"""Saved custom pit plans remain replayable and strict across schema boundaries."""

import json

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def _runner(*, pit_plans=None, engine="standard"):
    driver = Driver(id="A", name="A", team_id="T")
    return MonteCarloRunner(
        [driver], {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Saved", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), seed=41, starting_tires={"A": "medium"},
        race_engine=engine,
        pit_plans=pit_plans,
    )


def test_custom_plan_snapshot_and_second_trial_replay(tmp_path):
    result = _runner(pit_plans={"A": [{"lap": 2, "compound": "hard"}]}).run(
        2, parallel=False,
    )
    assert result.input_snapshot["schema_version"] == 5
    assert result.input_snapshot["pit_plans"] == {
        "A": [{"lap": 2, "compound": "hard"}],
    }
    assert result.race_results[0][0].pit_plan_history[0]["status"] in {
        "executed", "overridden", "skipped",
    }
    path = Exporter(tmp_path).export_statistics_json(result)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.seed == 42
    assert replay.race_results == [result.race_results[1]]
    assert replay.input_snapshot["pit_plans"] == result.input_snapshot["pit_plans"]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_custom_plan_serial_and_process_trials_match(engine):
    plan = {"A": [{"lap": 2, "compound": "hard"}]}
    serial = _runner(pit_plans=plan, engine=engine).run(2, parallel=False)
    process = _runner(pit_plans=plan, engine=engine).run(
        2, parallel=True, max_workers=1,
    )
    assert process.race_results == serial.race_results
    assert process.qualifying_results == serial.qualifying_results


def test_schema_one_to_four_reject_any_plan_key(tmp_path):
    result = _runner().run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    data = json.loads(path.read_text(encoding="utf-8"))
    for version in (1, 2, 3, 4):
        candidate = json.loads(json.dumps(data))
        candidate["simulation_inputs"]["schema_version"] = version
        candidate["simulation_inputs"]["pit_plans"] = {}
        if version < 4:
            candidate["simulation_inputs"].pop("tire_inventory", None)
        if version < 3:
            candidate["simulation_inputs"].pop("starting_tire_ages", None)
        else:
            candidate["simulation_inputs"]["starting_tire_ages"] = {}
        if version >= 4:
            candidate["simulation_inputs"]["tire_inventory"] = {
                "A": [{"id": "set-1", "compound": "medium", "age": 0}],
            }
        # Use the same JSON path so the malformed payload is actually loaded.
        malformed = path.with_name(f"schema-{version}.json")
        malformed.write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ValueError, match="pit_plans"):
            _load_saved_runner(malformed)


def test_schema_five_malformed_track_is_controlled_validation_error(tmp_path):
    result = _runner(pit_plans={"A": []}).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["simulation_inputs"]["track"].pop("total_laps")
    malformed = tmp_path / "missing-laps.json"
    malformed.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="track|total_laps|Field required"):
        _load_saved_runner(malformed)
