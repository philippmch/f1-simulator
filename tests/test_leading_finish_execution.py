"""Late native stops are verified against complete mean-physics race execution."""

from copy import deepcopy

import numpy as np
import pytest
from test_custom_pit_replacements import snapshot

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def inputs(lap, age, stress, lane, warmup):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A", tire_degradation_factor=1.5,
              pit_stop_avg=2.75, pit_stop_std=.1)
    track = Track(id="timed", name="timed", country="test", total_laps=90,
                  base_lap_time=100., tire_stress=stress, pit_lane_delta=lane)
    physics = LapSimulator()
    prefix = 0.
    fit_lap = lap - age
    for number in range(1, lap + 1):
        compound = TireCompound.HARD if number < fit_lap else TireCompound.SOFT
        driver.current_tire_laps = number - 1 if number < fit_lap else number - fit_lap
        prefix += physics.calculate_lap_time(driver, car, track, TIRE_COMPOUNDS[compound],
                                             Weather(), number, 90, sample_variation=False)
    # Choose a reference pace that places the retained first crossing just
    # before expiry. This exposes the finish phase without changing tyre curves.
    fitting = warmup.get("soft", 0.)
    track.base_lap_time = (
        7200. - .5 - lane - expected_stationary_time(car) - fitting
    ) * 100 / prefix
    driver.reset_race_state()
    return driver, car, track


def run(monkeypatch, engine_name, finite, case, warmup, *, guarded):
    lap, age, stress, lane = case
    driver, car, track = inputs(lap, age, stress, lane, warmup)
    simulator = RaceSimulator(np.random.default_rng(21), tire_warmup=warmup)
    engine = ChronologicalRace(simulator)
    native = simulator._should_pit
    physics = simulator.lap_simulator.calculate_lap_time
    observations = {}
    with monkeypatch.context() as patch:
        patch.setattr(simulator, "_infer_team_strategy",
                      lambda *args: TeamStrategyArchetype.BALANCED)
        patch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
        patch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *args: None)
        patch.setattr(simulator.event_manager, "_check_random_incident",
                      lambda *args, **kwargs: None)
        patch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                      lambda car: expected_stationary_time(car))

        def mean_running(*args, **kwargs):
            kwargs.pop("sample_variation", None)
            return physics(*args, sample_variation=False, **kwargs)

        patch.setattr(simulator.lap_simulator, "calculate_lap_time", mean_running)

        def decide(state, states, planning, number, *args, **kwargs):
            if number < lap:
                if number == lap - age:
                    if finite:
                        state.inventory_pit_proposal = (number, "S")
                    else:
                        state.dry_pit_proposal = (number, TireCompound.SOFT)
                    return True
                return False
            stop = native(state, states, planning, number, *args, **kwargs)
            if number == lap:
                observations.update(native_stop=bool(stop), now=state.total_time,
                                    horizon=planning.total_laps,
                                    proposal=state.inventory_pit_proposal if finite
                                    else state.dry_pit_proposal)
                retained = state.driver.model_copy(deep=True)
                retained.current_tire_laps = state.tire_laps
                retained_time = LapSimulator().calculate_lap_time(
                    retained, state.car, track, state.current_tire, Weather(), number, 90,
                    sample_variation=False,
                )
                observations["retained_first"] = state.total_time + retained_time
            return stop

        patch.setattr(simulator, "_should_pit", decide)
        if engine_name == "standard":
            protect = simulator._protect_leading_finish_distance
            target, name = simulator, "_protect_leading_finish_distance"
        else:
            protect = engine._protect_elective_finish_distance
            target, name = engine, "_protect_elective_finish_distance"

        def inspect(state, *args, **kwargs):
            before = snapshot(state, simulator), deepcopy(track), deepcopy(engine.weather) if (
                engine_name == "chronological") else None
            result = protect(state, *args, **kwargs) if guarded else False
            assert before == (snapshot(state, simulator), track, engine.weather if (
                engine_name == "chronological") else None)
            if state.laps_completed + 1 == lap:
                observations["veto"] = result
            return result

        patch.setattr(target, name, inspect)
        inventory = [{"id": identifier, "compound": compound, "age": 0}
                     for identifier, compound in (("H", "hard"), ("S", "soft"),
                                                   ("S2", "soft"), ("M", "medium"))]
        execute = simulator.simulate_race if engine_name == "standard" else engine.run
        result = execute([driver], {"A": car}, track, Weather(change_probability=0.), ["A"],
                         starting_tires={"A": TireCompound.HARD},
                         tire_inventory={"A": inventory} if finite else None)[0]
    return result, observations


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", [(45, 20, .4, 8.), (65, 30, 1., 20.)])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "hard": 1.}])
def test_late_native_choice_keeps_the_additional_executed_lap(
    monkeypatch, engine, finite, case, warmup,
):
    guarded, decision = run(monkeypatch, engine, finite, case, warmup, guarded=True)
    baseline, original = run(monkeypatch, engine, finite, case, warmup, guarded=False)
    lap = case[0]
    assert decision["native_stop"] and original["native_stop"]
    assert decision["retained_first"] == pytest.approx(7199.5, abs=1.e-8)
    assert decision["veto"] and not original["veto"]
    assert lap not in guarded.pit_laps and lap in baseline.pit_laps
    assert guarded.laps_completed == baseline.laps_completed + 1
    assert guarded.laps_completed == lap + 2
    assert guarded.race_time_limited and baseline.race_time_limited
