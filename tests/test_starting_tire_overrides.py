"""Explicit opening tyres survive workers and saved-input replay."""

import json

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, _run_single_simulation
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.output import Exporter
from f1sim.simulation.execution import validate_starting_tires


def runner(engine="standard", overrides=None):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id="t", name="T", country="T", total_laps=5, base_lap_time=90)
    return MonteCarloRunner(drivers, cars, track, Weather(change_probability=0),
                            seed=71, race_engine=engine, starting_tires=overrides)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_partial_override_parallel_and_qualifying_parity(engine):
    default = runner(engine).run(2, parallel=False)
    sequential = runner(engine, {"0": TireCompound.HARD}).run(2, parallel=False)
    parallel = runner(engine, {"0": "hard"}).run(2, parallel=True, max_workers=2)
    assert sequential.race_results == parallel.race_results
    assert sequential.qualifying_results == parallel.qualifying_results
    assert sequential.event_stats == parallel.event_stats
    assert sequential.qualifying_results == default.qualifying_results
    for race in sequential.race_results:
        assert next(row for row in race if row.driver_id == "0").strategy[0] == "hard"
        assert next(row for row in race if row.driver_id == "1").strategy


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_empty_override_preserves_seeded_defaults_and_legacy_worker(engine):
    baseline = runner(engine).run(2, parallel=False)
    empty = runner(engine, {}).run(2, parallel=False)
    assert baseline == empty
    snapshot = baseline.input_snapshot
    args = tuple(snapshot[key] for key in ("drivers", "cars", "track", "weather")) + (71,)
    six = _run_single_simulation(args + (engine,))
    seven = _run_single_simulation(args + (engine, {}))
    assert six == seven
    assert six[0] == baseline.race_results[0]
    if engine == "standard":
        assert _run_single_simulation(args) == six


@pytest.mark.parametrize("value", [[], "soft", 1, False, {"": "soft"}, {" ": "soft"},
                                  {1: "soft"}, {"0": None}, {"0": 1}, {"0": "SOFT"},
                                  {"0": "unknown"}, {"missing": "soft"}])
def test_invalid_overrides_rejected(value):
    with pytest.raises(ValueError, match="starting_tires"):
        runner(overrides=value)


def test_shared_validation_and_caller_isolation():
    original = {"0": TireCompound.SOFT}
    normalized = validate_starting_tires(original)
    assert normalized == {"0": "soft"}
    assert type(normalized["0"]) is str
    simulation = runner(overrides=original)
    original["0"] = "wet"
    results = simulation.run(1, parallel=False)
    assert results.input_snapshot["starting_tires"] == {"0": "soft"}
    selected = next(row for row in results.race_results[0] if row.driver_id == "0")
    assert selected.strategy[0] == "soft"
    simulation.starting_tires["0"] = "medium"
    assert results.input_snapshot["starting_tires"] == {"0": "soft"}
    assert original == {"0": "wet"}


def test_roster_revalidated_before_run():
    simulation = runner(overrides={"0": "soft"})
    simulation.drivers.pop(0)
    with pytest.raises(ValueError, match="Unknown starting_tires driver ID"):
        simulation.run(1, parallel=False)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_override_snapshot_replays_second_trial(tmp_path, engine):
    original = runner(engine, {"0": "wet", "1": "intermediate"}).run(2, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [original.race_results[1]]
    assert replay.qualifying_results == [original.qualifying_results[1]]
    assert replay.input_snapshot["starting_tires"] == {"0": "wet", "1": "intermediate"}


def test_legacy_snapshot_without_override_replays(tmp_path):
    original = runner().run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = json.loads(path.read_text(encoding="utf-8"))
    del saved["simulation_inputs"]["starting_tires"]
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert replay_saved_simulation(path).race_results == original.race_results


@pytest.mark.parametrize("override", [[], {"missing": "soft"}, {"0": "unknown"}])
def test_invalid_saved_override_fails_cleanly(tmp_path, override):
    path = Exporter(tmp_path).export_statistics_json(runner().run(1, parallel=False))
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["simulation_inputs"]["starting_tires"] = override
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="starting_tires"):
        replay_saved_simulation(path)
