"""Versioned weather streams keep strategy choices from changing rainfall draws."""

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, _run_single_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.randomness import weather_rng_for_trial


def inputs():
    return ([Driver(id="A", name="A", team_id="A")],
            {"A": Car(team_id="A", team_name="A")},
            Track(id="t", name="T", country="T", total_laps=12, base_lap_time=90),
            Weather(change_probability=1))


def serialized():
    drivers, cars, track, weather = inputs()
    return ([d.model_dump() for d in drivers],
            {key: car.model_dump() for key, car in cars.items()},
            track.model_dump(), weather.model_dump())


@pytest.mark.parametrize("policy", [None, True, 1, [], {}, "", "isolated", "SHARED_V1"])
def test_invalid_policy_rejected_before_execution(policy):
    with pytest.raises(ValueError, match="rng_policy"):
        MonteCarloRunner(*inputs(), rng_policy=policy)
    runner = MonteCarloRunner(*inputs())
    runner.rng_policy = policy
    with pytest.raises(ValueError, match="rng_policy"):
        runner.run(1, parallel=False)
    with pytest.raises(ValueError, match="rng_policy"):
        _run_single_simulation((*serialized(), 17, "standard", {}, policy))


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_race_draw_perturbation_cannot_change_isolated_weather(engine):
    def run(policy, extra_draws):
        race_rng = np.random.default_rng(17)
        simulator = RaceSimulator(
            rng=race_rng, weather_rng=weather_rng_for_trial(17, race_rng, policy),
        )

        def process_lap(*args, **kwargs):
            race_rng.random(extra_draws)
            return []

        simulator.event_manager.process_lap = process_lap
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        execute(*inputs(), ["A"])
        return simulator.weather_history

    original = run("isolated_weather_v1", 0)
    perturbed = run("isolated_weather_v1", 37)
    assert len(original) == len(perturbed) == 12
    assert original == perturbed
    assert run("shared_v1", 0) != run("shared_v1", 37)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_starting_tyres_share_weather_prefix_and_keep_qualifying(engine):
    trials = [MonteCarloRunner(*inputs(), seed=17, race_engine=engine,
                              starting_tires={"A": compound}).run(2, parallel=False)
              for compound in ("soft", "medium", "hard")]
    for result in trials:
        assert result.input_snapshot["rng_policy"] == "isolated_weather_v1"
        assert result.qualifying_results == trials[0].qualifying_results
        for observed, reference in zip(result.weather_histories, trials[0].weather_histories):
            length = min(len(observed), len(reference))
            assert length > 1
            assert observed[:length] == reference[:length]
    assert trials[0].weather_histories[0] != trials[0].weather_histories[1]
    shared = MonteCarloRunner(*inputs(), seed=17, race_engine=engine,
                             rng_policy="shared_v1").run(2, parallel=False)
    assert shared.qualifying_results == trials[0].qualifying_results


@pytest.mark.parametrize("suffix", [(), ("standard",), ("standard", {}),
                                    ("chronological",), ("chronological", {})])
def test_legacy_worker_tuples_preserve_shared_policy(suffix):
    engine = suffix[0] if suffix else "standard"
    assert _run_single_simulation((*serialized(), 17, *suffix)) == _run_single_simulation(
        (*serialized(), 17, engine, {}, "shared_v1"),
    )


def test_direct_simulator_defaults_to_shared_rng_and_namespace_is_stable():
    rng = np.random.default_rng(17)
    simulator = RaceSimulator(rng=rng)
    assert simulator.weather_rng is rng
    assert weather_rng_for_trial(17, rng, "shared_v1") is rng
    before = rng.bit_generator.state
    weather_rng = weather_rng_for_trial(17, rng, "isolated_weather_v1")
    expected = np.random.default_rng(np.random.SeedSequence(17, spawn_key=(0x57454154,)))
    assert np.array_equal(weather_rng.random(10), expected.random(10))
    assert rng.bit_generator.state == before
