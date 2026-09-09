"""Prior wear affects opening physics without inventing race tyre use."""

import copy
import json

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.strategy_comparison import (
    compare_saved_race_engines,
    compare_saved_starting_tires,
)
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.output import Exporter
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.execution import parse_starting_tire_spec, validate_starting_tire_ages
from f1sim.simulation.pit_strategy import plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def runner(engine="standard", ages=None):
    drivers = [Driver(id=d, name=d, team_id=d) for d in "AB"]
    return MonteCarloRunner(
        drivers, {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers},
        Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), seed=42, race_engine=engine,
        starting_tires={"A": "medium", "B": "hard"}, starting_tire_ages=ages,
    )


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_used_opening_physics_ages_and_replacement_reset(monkeypatch, engine):
    model = runner(engine)
    sim = RaceSimulator(np.random.default_rng(7))
    monkeypatch.setattr(sim.event_manager, "process_lap", lambda *a, **k: [])
    monkeypatch.setattr(sim.event_manager, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(sim.event_manager, "_check_mechanical_failure", lambda *a, **k: None)
    monkeypatch.setattr(sim, "_should_pit", lambda state, states, track, lap, *a, **k: lap == 3)
    observed = []
    actual = sim.lap_simulator.calculate_lap_time

    def lap(driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        observed.append((lap_number, driver.current_tire_laps))
        kwargs["sample_variation"] = False
        return actual(driver, car, track, tire, weather, lap_number, total_laps, **kwargs)

    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", lap)
    run = sim.simulate_race if engine == "standard" else ChronologicalRace(sim).run
    results = run(model.drivers[:1], {"A": model.cars["A"]}, model.track, model.weather, ["A"],
                  starting_tires={"A": TireCompound.MEDIUM}, starting_tire_ages={"A": 7})
    assert observed == [(1, 7), (2, 8), (3, 0), (4, 1)]
    assert results[0].laps_completed == 4
    assert results[0].pit_stop_details[0]["tire_age"] == 9


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_immediate_refit_does_not_count_prior_wet_usage(monkeypatch, engine):
    model = runner(engine)
    sim = RaceSimulator(np.random.default_rng(9))
    monkeypatch.setattr(sim.event_manager, "process_lap", lambda *a, **k: [])
    monkeypatch.setattr(sim.event_manager, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(sim.event_manager, "_check_mechanical_failure", lambda *a, **k: None)
    observed = []
    original = sim._fit_tire

    def fit(state, compound):
        observed.append((sim._has_used_wet_compound(state), state.tire_laps,
                         state.prior_tire_laps))
        original(state, compound)

    monkeypatch.setattr(sim, "_fit_tire", fit)
    run = sim.simulate_race if engine == "standard" else ChronologicalRace(sim).run
    results = run(model.drivers[:1], {"A": model.cars["A"]}, model.track, model.weather, ["A"],
                  starting_tires={"A": "wet"}, starting_tire_ages={"A": 12})
    assert observed[0] == (False, 12, 12)
    assert "wet" not in results[0].strategy
    assert len(set(results[0].strategy)) >= 2


def test_prior_wear_does_not_satisfy_dry_rule_before_running():
    model = runner()
    state = DriverRaceState(model.drivers[0], model.cars["A"], 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM],
                            tire_laps=20, prior_tire_laps=20)
    assert not RaceSimulator._actually_used_compounds(state)
    decision = plan_dry_stop(state.driver, state.car, model.track, state.current_tire,
                             20, 1, 1, set(), current_set_used=False)
    assert decision.pit_now_cost == float("inf")
    state.tire_laps += 1
    assert RaceSimulator._actually_used_compounds(state) == {TireCompound.MEDIUM}
    decision = plan_dry_stop(state.driver, state.car, model.track, state.current_tire,
                             21, 1, 1, {TireCompound.MEDIUM}, current_set_used=True)
    assert decision.pit_now_cost < float("inf")


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_serial_parallel_zero_parity_and_schema3_replay(tmp_path, engine):
    baseline = runner(engine).run(2, parallel=False)
    zero = runner(engine, {"A": 0}).run(2, parallel=False)
    assert baseline.race_results == zero.race_results
    assert baseline.event_stats == zero.event_stats
    assert baseline.input_snapshot["schema_version"] == 2
    serial = runner(engine, {"A": 7, "B": 3}).run(2, parallel=False)
    parallel = runner(engine, {"A": 7, "B": 3}).run(2, parallel=True, max_workers=2)
    assert serial.race_results == parallel.race_results
    assert serial.event_stats == parallel.event_stats
    assert serial.weather_histories == parallel.weather_histories
    assert serial.qualifying_results == baseline.qualifying_results
    assert serial.input_snapshot["schema_version"] == 3
    path = Exporter(tmp_path).export_statistics_json(serial)
    replay = replay_saved_simulation(path, 2)
    assert replay.race_results == [serial.race_results[1]]


@pytest.mark.parametrize("value", [{"A": True}, {"A": -1}, {"A": 1001}, {"A": 1.5},
                                  {"A": "5"}, {"C": 2}, [], False])
def test_invalid_ages_rejected_before_running(value):
    with pytest.raises(ValueError, match="starting_tire"):
        runner(ages=value)


@pytest.mark.parametrize("value", ["soft@-1", "soft@1.5", "soft@1001", "soft@@1", "soft@",
                                  "soft@+2", "soft@ 2", "SOFT@2"])
def test_invalid_specs(value):
    with pytest.raises(ValueError):
        parse_starting_tire_spec(value)


def test_spec_validation_isolated_and_requires_compound():
    assert parse_starting_tire_spec("soft") == ("soft", 0)
    assert parse_starting_tire_spec("soft@5") == ("soft", 5)
    assert parse_starting_tire_spec("wet@1000") == ("wet", 1000)
    with pytest.raises(ValueError):
        validate_starting_tire_ages({"A": 2}, {})
    values = {"A": 2}
    model = runner(ages=values)
    values["A"] = 9
    assert model.starting_tire_ages == {"A": 2}


def test_saved_variants_preserve_other_driver_ages_and_engine_context(tmp_path):
    result = runner(ages={"A": 7, "B": 3}).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    variants = compare_saved_starting_tires(path, "A", ["automatic", "soft", "soft@5"],
                                           num_simulations=1)
    assert variants["automatic"].input_snapshot["starting_tire_ages"] == {"B": 3}
    assert variants["soft"].input_snapshot["starting_tire_ages"] == {"B": 3}
    assert variants["soft@5"].input_snapshot["starting_tire_ages"] == {"A": 5, "B": 3}
    engines = compare_saved_race_engines(path, num_simulations=1)
    assert all(r.input_snapshot["starting_tire_ages"] == {"A": 7, "B": 3}
               for r in engines.values())
    saved = json.loads(path.read_text())
    original = copy.deepcopy(saved)
    saved["simulation_inputs"]["schema_version"] = 2
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Legacy"):
        _load_saved_runner(path)
    original["simulation_inputs"].pop("starting_tire_ages")
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match="requires"):
        _load_saved_runner(path)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_red_flag_free_fit_resets_opening_prior_wear(monkeypatch, engine):
    model = runner(engine)
    sim = RaceSimulator(np.random.default_rng(8))
    sim.event_manager.set_forced_red_flag(1)
    monkeypatch.setattr(sim.event_manager, "_deploy_safety_measure", lambda *a, **k: None)
    monkeypatch.setattr(sim.event_manager, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(sim.event_manager, "_check_mechanical_failure", lambda *a, **k: None)
    monkeypatch.setattr(sim, "_should_pit", lambda *a, **k: False)
    observed = []

    def lap(driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        observed.append((lap_number, driver.current_tire_laps))
        return 90

    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", lap)
    run = sim.simulate_race if engine == "standard" else ChronologicalRace(sim).run
    result = run(model.drivers[:1], {"A": model.cars["A"]}, model.track, model.weather, ["A"],
                 starting_tires={"A": "medium"}, starting_tire_ages={"A": 7})
    assert observed == [(1, 7), (2, 0), (3, 1), (4, 2)]
    assert result[0].pit_stops == 0
    assert result[0].laps_completed == 4


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_invalid_age_precedes_direct_engine_state_reset(engine):
    model = runner(engine)
    model.drivers[0].current_tire_laps = 23
    model.drivers[0].dnf = True
    sim = RaceSimulator(np.random.default_rng(10))
    before = copy.deepcopy(sim.rng.bit_generator.state)
    run = sim.simulate_race if engine == "standard" else ChronologicalRace(sim).run
    with pytest.raises(ValueError):
        run(model.drivers, model.cars, model.track, model.weather, ["A", "B"],
            starting_tires={"A": "soft"}, starting_tire_ages={"A": -1})
    assert model.drivers[0].current_tire_laps == 23
    assert model.drivers[0].dnf
    assert sim.rng.bit_generator.state == before
