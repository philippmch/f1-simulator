"""Offline counterfactual opening tyres preserve saved context and replayability."""
import json

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.strategy_comparison import compare_saved_starting_tires
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.output import Exporter


def make_saved(tmp_path, engine='standard'):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(2)]
    runner = MonteCarloRunner(
        drivers, {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers},
        Track(id='t', name='Saved', country='T', total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=81, race_engine=engine,
        starting_tires={'0': 'hard', '1': 'medium'},
    )
    return Exporter(tmp_path).export_statistics_json(runner.run(2, parallel=False))


@pytest.mark.parametrize('engine', ['standard', 'chronological'])
def test_variants_match_explicit_runs_and_replay(tmp_path, monkeypatch, engine):
    path = make_saved(tmp_path, engine)
    before = path.read_bytes()
    runner, count = _load_saved_runner(path)
    assert count == 2

    def no_network(*args, **kwargs):
        raise AssertionError('offline comparison contacted network')

    monkeypatch.setattr('socket.socket.connect', no_network)
    variants = compare_saved_starting_tires(
        path, '0', ['automatic', TireCompound.SOFT, 'wet'], num_simulations=np.int64(2),
    )
    assert list(variants) == ['automatic', 'soft', 'wet']
    assert path.read_bytes() == before
    assert runner.starting_tires == {'0': 'hard', '1': 'medium'}
    for label, result in variants.items():
        overrides = {'1': 'medium'}
        if label != 'automatic':
            overrides['0'] = label
        expected = MonteCarloRunner(
            runner.drivers, runner.cars, runner.track, runner.weather,
            seed=81, race_engine=engine, starting_tires=overrides,
        ).run(2, parallel=False)
        assert result.race_results == expected.race_results
        assert result.qualifying_results == expected.qualifying_results
        assert result.event_stats == expected.event_stats
        assert result.input_snapshot['starting_tires'] == overrides
        assert result.seed == 81
        assert result.qualifying_results == variants['automatic'].qualifying_results
    exported = Exporter(tmp_path).export_scenario_comparison_json(variants)
    for label, result in variants.items():
        replay = replay_saved_simulation(exported, simulation=2, scenario=label)
        assert replay.race_results == [result.race_results[1]]
        assert replay.qualifying_results == [result.qualifying_results[1]]


@pytest.mark.parametrize('kwargs', [
    {'compounds': []}, {'compounds': 'soft'}, {'compounds': None},
    {'compounds': ['soft', 'soft']}, {'compounds': ['soft', TireCompound.SOFT]},
    {'compounds': ['soft', 'invalid']}, {'compounds': [None]},
    {'driver_id': 'missing'}, {'num_simulations': True}, {'num_simulations': 0},
    {'num_simulations': 1.5}, {'max_workers': False}, {'max_workers': -1},
    {'max_workers': 1.5},
])
def test_invalid_arguments_before_any_simulation(tmp_path, monkeypatch, kwargs):
    path = make_saved(tmp_path)
    monkeypatch.setattr(MonteCarloRunner, 'run', lambda *a, **k: pytest.fail('ran invalid input'))
    arguments = {'driver_id': '0', 'num_simulations': 1, **kwargs}
    with pytest.raises(ValueError):
        compare_saved_starting_tires(path, **arguments)


def test_missing_car_rejected_before_simulation(tmp_path, monkeypatch):
    path = make_saved(tmp_path)
    payload = json.loads(path.read_text())
    del payload['simulation_inputs']['cars']['0']
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(MonteCarloRunner, 'run', lambda *a, **k: pytest.fail('ran invalid input'))
    with pytest.raises(ValueError, match='No saved car'):
        compare_saved_starting_tires(path, '0')


def test_named_scenario_and_loader_does_not_run(tmp_path, monkeypatch):
    path = make_saved(tmp_path)
    payload = json.loads(path.read_text())
    entry = {**payload['metadata'], 'simulation_inputs': payload['simulation_inputs']}
    path.write_text(json.dumps({'scenarios': {'dry': entry, 'wet': entry}}))
    expected = compare_saved_starting_tires(path, '0', ['hard'], scenario='dry', num_simulations=1)
    assert expected['hard'].seed == 81
    monkeypatch.setattr(
        MonteCarloRunner, 'run', lambda *a, **k: pytest.fail('loader ran simulation'),
    )
    loaded, count = _load_saved_runner(path, 'dry')
    assert loaded.base_seed == 81 and count == 2
    with pytest.raises(ValueError, match='Multiple saved scenarios'):
        compare_saved_starting_tires(path, '0')
    with pytest.raises(ValueError, match='Unknown saved scenario'):
        compare_saved_starting_tires(path, '0', scenario='missing')


def test_variant_models_are_isolated(tmp_path, monkeypatch):
    path = make_saved(tmp_path)
    runner, _ = _load_saved_runner(path)
    original = runner.track.model_dump()
    monkeypatch.setattr(
        'f1sim.analysis.strategy_comparison._load_saved_runner', lambda *a: (runner, 2),
    )
    seen = []

    def inspect_and_mutate(self, *args, **kwargs):
        seen.append(self.track.model_dump())
        self.track.name = 'mutated'
        self.drivers.clear()
        self.cars.clear()
        return None

    monkeypatch.setattr(MonteCarloRunner, 'run', inspect_and_mutate)
    compare_saved_starting_tires(path, '0', ['soft', 'hard'], num_simulations=1)
    assert seen == [original, original]
    assert runner.track.model_dump() == original
    assert len(runner.drivers) == len(runner.cars) == 2
