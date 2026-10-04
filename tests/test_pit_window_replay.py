"""Pit windows survive native workers, replay and paired alternatives with older inputs."""

import csv
import json
from copy import deepcopy

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.strategy_comparison import compare_saved_pit_plans
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter
from f1sim.simulation.pit_plans import PIT_PLAN_WINDOW_POLICY
from f1sim.web.server import _serialize_race_result

PLAN = [{"lap": 4, "compound": "hard", "earliest_lap": 2, "trigger": "neutralized"}]


def runner(engine="standard", extras=False):
    return MonteCarloRunner(
        [Driver(id="A", name="A", team_id="A")], {"A": Car(team_id="A", team_name="A")},
        Track(id="T", name="Window", country="T", total_laps=6, base_lap_time=90),
        Weather(change_probability=0), seed=41, starting_tires={"A": "medium"},
        race_engine=engine, pit_plans={"A": PLAN}, rng_policy="isolated_race_v1",
        **({"tire_inventory": {"A": [
            {"id": "M", "compound": "medium", "remaining_laps": 4},
            {"id": "H", "compound": "hard", "remaining_laps": 6},
        ]}, "tire_warmup": {"hard": 1.0},
            "qualifying_weather": {"Q1": {"track_wetness": .1}},
            "weather_schedule": [{"lap": 3, "rain_intensity": 0.0}]}
           if extras else {}),
    )


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("extras", [False, True])
def test_workers_replay_and_paired_fixed_reference_keep_all_window_options(
    tmp_path, engine, extras,
):
    configured = runner(engine, extras)
    serial = configured.run(2, parallel=False)
    workers = configured.run(2, parallel=True, max_workers=1)
    assert serial.race_results == workers.race_results
    assert serial.qualifying_results == workers.qualifying_results
    assert serial.input_snapshot == workers.input_snapshot
    assert serial.input_snapshot["schema_version"] == 10
    assert serial.input_snapshot["pit_plan_policy"] == PIT_PLAN_WINDOW_POLICY
    assert serial.input_snapshot["pit_plans"] == {"A": PLAN}
    exporter = Exporter(tmp_path)
    path = exporter.export_statistics_json(serial)
    recorded = json.loads(path.read_text(encoding="utf-8"))["pit_plan_histories"]
    assert [row["pit_plan_history"] for row in recorded] == [
        race[0].pit_plan_history for race in serial.race_results]
    assert _serialize_race_result(serial.race_results[0][0])["pit_plan_history"] == (
        serial.race_results[0][0].pit_plan_history)
    with exporter.export_race_results_csv(serial).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [json.loads(row["pit_plan_history"]) for row in rows] == [
        race[0].pit_plan_history for race in serial.race_results]
    for trial in (1, 2):
        replay = replay_saved_simulation(path, simulation=trial)
        assert replay.race_results == [serial.race_results[trial - 1]]
        assert replay.input_snapshot == serial.input_snapshot
    variants = compare_saved_pit_plans(
        path, "A", {"window": PLAN, "fixed": [{"lap": 4, "compound": "hard"}], "auto": None},
        num_simulations=2,
    )
    assert variants["window"].race_results == serial.race_results
    paired = paired_comparison_statistics(variants, "window")
    assert all(row["status"] == "paired" for row in paired["variants"].values())
    assert variants["fixed"].input_snapshot["schema_version"] == (9 if extras else 5)
    assert "pit_plan_policy" not in variants["fixed"].input_snapshot
    for option in ("tire_warmup", "qualifying_weather", "weather_schedule", "tire_usage_policy"):
        if extras:
            assert all(result.input_snapshot[option] == serial.input_snapshot[option]
                       for result in variants.values())


@pytest.mark.parametrize("change", [
    {"pit_plan_policy": "unknown"}, {"pit_plan_policy": None},
    {"pit_plans": {}}, {"pit_plans": {"A": [{"lap": 4, "compound": "hard"}]}},
    {"schema_version": 5}, {"schema_version": 9},
    {"pit_plans": {"A": [PLAN[0] | {"trigger": "green"}]}},
])
def test_malformed_or_downgraded_windows_are_not_replayed_or_paired(tmp_path, change):
    valid = runner().run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(valid)
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["simulation_inputs"].update(change)
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_saved_runner(path)
    invalid = deepcopy(valid)
    invalid.input_snapshot.update(change)
    assert paired_comparison_statistics({"valid": valid, "invalid": invalid}, "valid")[
        "variants"]["invalid"]["status"] == "unavailable"
