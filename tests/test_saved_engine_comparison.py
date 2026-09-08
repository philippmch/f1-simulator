"""Offline engine comparisons retain inputs, ordering and per-engine replay."""
import json

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.strategy_comparison import compare_saved_race_engines
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def make_saved(tmp_path, engine="standard"):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)]
    runner = MonteCarloRunner(
        drivers, {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers},
        Track(id="t", name="Saved", country="T", total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=81, race_engine=engine,
        starting_tires={"0": "hard", "1": "medium"},
    )
    return Exporter(tmp_path).export_statistics_json(runner.run(2, parallel=False))


@pytest.mark.parametrize("saved_engine", ["standard", "chronological"])
def test_ordered_variants_match_direct_runs_and_replay(tmp_path, monkeypatch, saved_engine):
    path = make_saved(tmp_path, saved_engine)
    before = path.read_bytes()
    runner, _ = _load_saved_runner(path)

    def no_network(*args, **kwargs):
        raise AssertionError("offline comparison contacted network")

    monkeypatch.setattr("socket.socket.connect", no_network)
    variants = compare_saved_race_engines(
        path, iter(["chronological", "standard"]), num_simulations=np.int64(2),
        max_workers=np.int64(1),
    )
    assert list(variants) == ["chronological", "standard"]
    assert path.read_bytes() == before
    for label, result in variants.items():
        expected = MonteCarloRunner(
            runner.drivers, runner.cars, runner.track, runner.weather,
            seed=81, race_engine=label, starting_tires=runner.starting_tires,
        ).run(2, parallel=False)
        assert result.race_engine == label
        assert result.race_results == expected.race_results
        assert result.qualifying_results == expected.qualifying_results
        assert result.event_stats == expected.event_stats
        assert result.input_snapshot == expected.input_snapshot
        assert result.input_snapshot["starting_tires"] == {"0": "hard", "1": "medium"}
        assert result.seed == 81
        assert result.qualifying_results == variants["standard"].qualifying_results
    exported = Exporter(tmp_path).export_scenario_comparison_json(variants)
    for label, result in variants.items():
        replay = replay_saved_simulation(exported, simulation=2, scenario=label)
        assert replay.race_engine == label
        assert replay.race_results == [result.race_results[-1]]
        assert replay.qualifying_results == [result.qualifying_results[-1]]


@pytest.mark.parametrize("kwargs", [
    {"engines": []}, {"engines": "standard"}, {"engines": b"standard"},
    {"engines": None}, {"engines": ["standard", "standard"]},
    {"engines": ["standard", "invalid"]}, {"engines": ["Standard"]},
    {"engines": [None]}, {"engines": [{}]},
    {"num_simulations": True}, {"num_simulations": 0}, {"num_simulations": -1},
    {"num_simulations": 1.5}, {"max_workers": False}, {"max_workers": 0},
    {"max_workers": -1}, {"max_workers": 1.5},
])
def test_invalid_arguments_before_loading_or_simulation(monkeypatch, kwargs):
    monkeypatch.setattr(
        "f1sim.analysis.strategy_comparison._load_saved_runner",
        lambda *a: pytest.fail("loaded invalid input"),
    )
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *a, **k: pytest.fail("ran invalid input"))
    with pytest.raises(ValueError):
        compare_saved_race_engines("unused.json", **kwargs)


def test_named_scenario_and_default_order(tmp_path, monkeypatch):
    path = make_saved(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = {**payload["metadata"], "simulation_inputs": payload["simulation_inputs"]}
    path.write_text(json.dumps({"scenarios": {"dry": entry, "wet": entry}}), encoding="utf-8")
    variants = compare_saved_race_engines(path, scenario="dry", num_simulations=1)
    assert list(variants) == ["standard", "chronological"]
    assert all(result.seed == 81 for result in variants.values())
    monkeypatch.setattr(
        MonteCarloRunner, "run", lambda *a, **k: pytest.fail("ran invalid scenario"),
    )
    with pytest.raises(ValueError, match="Multiple saved scenarios"):
        compare_saved_race_engines(path)
    with pytest.raises(ValueError, match="Unknown saved scenario"):
        compare_saved_race_engines(path, scenario="missing")


def test_variants_deep_copy_all_models_and_forward_execution(tmp_path, monkeypatch):
    path = make_saved(tmp_path)
    runner, _ = _load_saved_runner(path)
    monkeypatch.setattr(
        "f1sim.analysis.strategy_comparison._load_saved_runner", lambda *a: (runner, 2),
    )

    def snapshot(value):
        return (
            [d.model_dump() for d in value.drivers],
            {key: car.model_dump() for key, car in value.cars.items()},
            value.track.model_dump(), value.weather.model_dump(), value.starting_tires.copy(),
        )

    original = snapshot(runner)
    seen = []

    def inspect_and_mutate(self, count, **kwargs):
        seen.append(snapshot(self))
        assert count == 3 and type(count) is int
        assert kwargs == {"parallel": True, "max_workers": 2}
        self.drivers[0].name = "changed"
        self.cars["0"].team_name = "changed"
        self.track.name = "changed"
        self.weather.change_probability = 0.5
        self.starting_tires.clear()
        return None

    monkeypatch.setattr(MonteCarloRunner, "run", inspect_and_mutate)
    compare_saved_race_engines(
        path, num_simulations=np.int64(3), parallel=True, max_workers=np.int64(2),
    )
    assert seen == [original, original]
    assert snapshot(runner) == original
