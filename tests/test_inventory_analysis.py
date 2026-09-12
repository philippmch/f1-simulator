"""Finite tyre pools survive analysis, replay and controlled comparisons."""

import json
from copy import deepcopy

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, _run_single_simulation
from f1sim.analysis.paired_comparison import _snapshot
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.strategy_comparison import (
    compare_saved_race_engines,
    compare_saved_starting_tires,
)
from f1sim.models import Car, Driver, Track, Weather


def make_runner(engine="standard", **kwargs):
    return MonteCarloRunner(
        [Driver(id="A", name="A", team_id="A")],
        {"A": Car(team_id="A", team_name="A")},
        Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), seed=42, race_engine=engine, **kwargs,
    )


def save_result(tmp_path, result):
    path = tmp_path / "saved.json"
    path.write_text(json.dumps({"metadata": {"seed": result.seed,
        "num_simulations": result.num_simulations, "race_engine": result.race_engine},
        "simulation_inputs": result.input_snapshot}), encoding="utf-8")
    return path


def pool():
    return {"A": [{"id": "s", "compound": "soft", "age": 5},
                  {"id": "m", "compound": "medium", "age": 0}]}


def test_constructor_copies_and_revalidates_pool():
    original = pool()
    runner = make_runner(tire_inventory=original)
    original["A"][0]["age"] = 99
    assert runner.tire_inventory["A"][0]["age"] == 5
    runner.tire_inventory["A"][0]["age"] = True
    with pytest.raises(ValueError, match="ages"):
        runner.run(1, parallel=False)
    with pytest.raises(ValueError, match="exact opening"):
        make_runner(tire_inventory=pool(), starting_tires={"A": "soft"})


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_inventory_parallel_reuse_replay_and_qualifying(tmp_path, engine):
    runner = make_runner(engine, tire_inventory=pool(), starting_tires={"A": "soft"},
                         starting_tire_ages={"A": 5})
    serial = runner.run(2, parallel=False)
    parallel = runner.run(2, parallel=True, max_workers=2)
    repeated = runner.run(2, parallel=False)
    unlimited = make_runner(engine).run(2, parallel=False)
    assert serial.race_results == parallel.race_results == repeated.race_results
    assert serial.weather_histories == parallel.weather_histories
    assert serial.event_stats == parallel.event_stats
    assert serial.qualifying_results == unlimited.qualifying_results
    assert runner.tire_inventory == pool()
    assert serial.input_snapshot["schema_version"] == 4
    assert serial.input_snapshot["tire_inventory"] == pool()
    path = save_result(tmp_path, serial)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results[0] == serial.race_results[1]
    assert replay.input_snapshot["tire_inventory"] == pool()
    serial.input_snapshot["tire_inventory"]["A"][0]["age"] = 100
    assert runner.tire_inventory == pool()


@pytest.mark.parametrize("length", range(5, 11))
def test_worker_legacy_shapes(length):
    runner = make_runner()
    args = ([d.model_dump() for d in runner.drivers],
            {k: v.model_dump() for k, v in runner.cars.items()},
            runner.track.model_dump(), runner.weather.model_dump(), 42,
            "standard", {}, "shared_v1", {}, {})
    assert _run_single_simulation(args[:length]) == _run_single_simulation(args[:5])


def test_inventory_snapshot_and_variant_boundaries(tmp_path):
    result = make_runner(tire_inventory=pool()).run(1, parallel=False)
    path = save_result(tmp_path, result)
    variants = compare_saved_starting_tires(path, "A", ["automatic", "soft@5", "medium"],
                                           num_simulations=1)
    assert all(r.input_snapshot["tire_inventory"] == pool() for r in variants.values())
    assert variants["automatic"].input_snapshot["starting_tires"] == {}
    with pytest.raises(ValueError, match="exact opening"):
        compare_saved_starting_tires(path, "A", ["soft"], num_simulations=1)
    engines = compare_saved_race_engines(path, num_simulations=1)
    assert all(r.input_snapshot["tire_inventory"] == pool() for r in engines.values())
    changed = deepcopy(result)
    changed.input_snapshot["tire_inventory"]["A"][0]["age"] += 1
    assert _snapshot(result)[0] != _snapshot(changed)[0]
    unlimited = make_runner().run(1, parallel=False)
    empty = deepcopy(unlimited)
    empty.input_snapshot.update(schema_version=4, tire_inventory={})
    assert _snapshot(unlimited)[0] == _snapshot(empty)[0]


@pytest.mark.parametrize("version,inventory", [(1, pool()), (2, pool()), (3, pool()),
                                               (4, None), (4, []), (4, "bad")])
def test_replay_rejects_wrong_schema_inventory(tmp_path, version, inventory):
    result = make_runner().run(1, parallel=False)
    path = save_result(tmp_path, result)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"].update(schema_version=version, tire_inventory=inventory)
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="tire_inventory"):
        _load_saved_runner(path)
    result.input_snapshot.update(schema_version=version, tire_inventory=inventory)
    assert _snapshot(result) is None



def test_worker_copies_inventory_before_native_execution(monkeypatch):
    from f1sim.simulation.race import RaceSimulator

    runner = make_runner()
    original = pool()
    seen = []

    def simulate(self, **kwargs):
        seen.append(deepcopy(kwargs["tire_inventory"]))
        kwargs["tire_inventory"]["A"][0]["age"] = 999
        return []

    monkeypatch.setattr(RaceSimulator, "simulate_race", simulate)
    args = ([d.model_dump() for d in runner.drivers],
            {k: v.model_dump() for k, v in runner.cars.items()},
            runner.track.model_dump(), runner.weather.model_dump(), 42,
            "standard", {}, "shared_v1", {}, original)
    _run_single_simulation(args)
    _run_single_simulation(args)
    assert seen == [pool(), pool()]
    assert original == pool()


def test_starting_variants_preserve_other_driver_ages_and_inventory(tmp_path, monkeypatch):
    runner = make_runner(tire_inventory=pool(), starting_tires={"A": "soft"},
                         starting_tire_ages={"A": 5})
    runner.drivers.append(Driver(id="B", name="B", team_id="A"))
    runner.starting_tires["B"] = "hard"
    runner.starting_tire_ages["B"] = 7
    # Capture constructor inputs without requiring the native inventory planner.
    source = make_runner().run(1, parallel=False)
    source.input_snapshot.update(schema_version=4, tire_inventory=pool(),
        drivers=[d.model_dump() for d in runner.drivers],
        starting_tires=runner.starting_tires, starting_tire_ages=runner.starting_tire_ages)
    path = save_result(tmp_path, source)
    captured = []

    def run(self, *args, **kwargs):
        captured.append((self.starting_tires, self.starting_tire_ages, self.tire_inventory))
        return None

    monkeypatch.setattr(MonteCarloRunner, "run", run)
    compare_saved_starting_tires(path, "A", ["automatic", "soft@5"], num_simulations=1)
    assert captured == [({"B": "hard"}, {"B": 7}, pool()),
                        ({"A": "soft", "B": "hard"}, {"A": 5, "B": 7}, pool())]
