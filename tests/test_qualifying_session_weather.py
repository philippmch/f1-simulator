"""Fixed phase inputs are strict, isolated and shared by tyre projection and attempts."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.qualifying import QualifyingSimulator
from f1sim.simulation.qualifying_weather import (
    effective_qualifying_weather,
    validate_qualifying_weather,
)


def runner(qualifying_weather=None, engine="standard", **options):
    return MonteCarloRunner(
        [Driver(id="A", name="A", team_id="A"), Driver(id="B", name="B", team_id="B")],
        {team: Car(team_id=team, team_name=team) for team in ("A", "B")},
        Track(id="t", name="T", country="T", total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=19, race_engine=engine,
        qualifying_weather=qualifying_weather, **options,
    )


@pytest.mark.parametrize("value", [
    [], True, 1, "wet", {"q1": {}}, {"Q4": {}}, {1: {}}, {"Q1": None}, {"Q1": []},
    {"Q1": True}, {"Q1": {"wetness": .5}}, {"Q2": {"rain_intensity": True}},
    {"Q2": {"rain_intensity": "0.5"}}, {"Q3": {"humidity": float("nan")}},
    {"Q1": {"air_temperature": float("inf")}}, {"Q1": {"track_wetness": -.1}},
    {"Q1": {"rain_intensity": 1.1}}, {"Q1": {"condition": "HEAVY_RAIN"}},
])
def test_invalid_inputs_fail_before_qualifying_or_trials(monkeypatch, value):
    with pytest.raises(ValueError, match="qualifying_weather"):
        validate_qualifying_weather(value)
    with pytest.raises(ValueError, match="qualifying_weather"):
        runner(value)
    simulation = QualifyingSimulator(np.random.default_rng(19))
    state = deepcopy(simulation.rng.bit_generator.state)
    monkeypatch.setattr(simulation, "_simulate_session", lambda *a, **k: pytest.fail("session ran"))
    with pytest.raises(ValueError, match="qualifying_weather"):
        simulation.simulate_qualifying([], {}, runner().track, Weather(), qualifying_weather=value)
    assert simulation.rng.bit_generator.state == state


def test_defaults_are_independent_of_race_fields_and_all_containers_are_fresh():
    override_weather = Weather(track_wetness=.8)
    overrides = {"Q1": {"rain_intensity": .5}, "Q3": override_weather}
    canonical = validate_qualifying_weather(overrides)
    assert canonical["Q1"]["track_wetness"] == 0
    assert canonical["Q1"]["change_probability"] == .1
    race = Weather(track_wetness=.2, humidity=.9, change_probability=0)
    effective = effective_qualifying_weather(race, overrides)
    assert effective["Q1"] == canonical["Q1"]
    assert effective["Q2"] == race.model_dump(mode="json")
    assert effective["Q3"] == override_weather.model_dump(mode="json")
    effective["Q1"]["rain_intensity"] = 0
    canonical["Q3"]["track_wetness"] = 0
    assert overrides["Q1"] == {"rain_intensity": .5}
    assert override_weather.track_wetness == .8
    fallback = effective_qualifying_weather(race, {})
    assert fallback["Q1"] is not fallback["Q2"]
    fallback["Q1"]["humidity"] = 0
    assert fallback["Q2"]["humidity"] == race.humidity == .9


def test_phase_dispatch_compound_choice_classification_and_input_isolation(monkeypatch):
    configured = runner()
    drivers = [configured.drivers[0].model_copy(update={"id": str(index)}) for index in range(12)]
    overrides = {"Q1": {"track_wetness": .85, "rain_intensity": .9},
                 "Q3": Weather(track_wetness=.35, rain_intensity=.4)}
    simulation = QualifyingSimulator(np.random.default_rng(19))
    before = deepcopy((drivers, configured.cars, configured.weather, overrides))
    session = simulation._simulate_session
    calculate = simulation.lap_simulator.calculate_qualifying_lap
    sessions, attempts, projections = [], [], []

    def observe_session(drivers, cars, track, weather, **kwargs):
        sessions.append((len(drivers), weather))
        return session(drivers, cars, track, weather, **kwargs)

    def observe_lap(**kwargs):
        phase = len(sessions) - 1
        assert kwargs["weather"] is sessions[phase][1]
        row = (phase, kwargs["tire"].compound)
        (attempts if kwargs.get("sample_variation", True) else projections).append(row)
        return calculate(**kwargs)

    monkeypatch.setattr(simulation, "_simulate_session", observe_session)
    monkeypatch.setattr(simulation.lap_simulator, "calculate_qualifying_lap", observe_lap)
    results = simulation.simulate_qualifying(drivers, configured.cars, configured.track,
                                           configured.weather, qualifying_weather=overrides)
    assert [size for size, _ in sessions] == [12, 11, 10]
    assert sessions[1][1] is configured.weather
    assert sessions[0][1].track_wetness == .85
    assert sessions[2][1].track_wetness == .35
    for phase, (_, weather) in enumerate(sessions):
        expected = min(TireCompound, key=lambda compound: LapSimulator().calculate_qualifying_lap(
            drivers[0], configured.cars["A"], configured.track, TIRE_COMPOUNDS[compound], weather,
            sample_variation=False,
        ))
        assert all(compound is expected for number, compound in attempts if number == phase)
        assert [compound for number, compound in projections if number == phase][:5] == list(
            TireCompound,
        )
    assert [result.position for result in results] == list(range(1, 13))
    assert sum(result.eliminated_in == "Q1" for result in results) == 1
    assert sum(result.eliminated_in == "Q2" for result in results) == 1
    assert (drivers, configured.cars, configured.weather, overrides) == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_empty_settings_preserve_seeded_results_snapshots_and_worker_shapes(monkeypatch, engine):
    from f1sim.analysis import montecarlo

    worker = montecarlo._run_single_simulation
    lengths = []

    def observe(args):
        lengths.append(len(args))
        return worker(args)

    monkeypatch.setattr(montecarlo, "_run_single_simulation", observe)
    baseline = runner(engine=engine).run(1, parallel=False)
    empty = runner({}, engine=engine).run(1, parallel=False)
    assert empty == baseline
    assert lengths == [11, 11]
    assert baseline.input_snapshot["schema_version"] == 2
    assert "qualifying_weather" not in baseline.input_snapshot
    runner({"Q1": {}}, engine=engine).run(1, parallel=False)
    assert lengths[-1] == 13
    runner({}, engine=engine, tire_warmup={"soft": .5}).run(1, parallel=False)
    assert lengths[-1] == 12


def test_runner_isolates_constructor_input_and_revalidates_mutations(monkeypatch):
    source = {"Q2": {"track_wetness": .5}}
    configured = runner(source)
    source["Q2"]["track_wetness"] = .9
    assert configured.qualifying_weather["Q2"]["track_wetness"] == .5
    configured.qualifying_weather["Q2"]["track_wetness"] = "0.5"
    monkeypatch.setattr("f1sim.analysis.montecarlo._run_single_simulation",
                        lambda *args: pytest.fail("invalid weather reached a trial"))
    with pytest.raises(ValueError, match="qualifying_weather"):
        configured.run(1, parallel=False)
