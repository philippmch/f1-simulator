"""Native queue decisions are checked against executed paid continuations."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator, TeamStrategyArchetype


def run(monkeypatch, engine, finite, lane, warmup, stop_lap=None, compound=None):
    simulator = RaceSimulator(np.random.default_rng(21), tire_warmup=warmup)
    native = simulator._should_pit
    physics = simulator.lap_simulator.calculate_lap_time
    observed = {}
    with monkeypatch.context() as patch:
        patch.setattr(simulator, "_infer_team_strategy",
                      lambda *args: TeamStrategyArchetype.BALANCED)
        patch.setattr(simulator.lap_simulator, "calculate_lap_time",
                      lambda *args, **kwargs: physics(
                          *args, **dict(kwargs, sample_variation=False)))
        patch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", expected_stationary_time)
        patch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *args: None)
        patch.setattr(simulator.event_manager, "_check_random_incident", lambda *args, **kw: None)
        # Future passing is conditional and fixed for every alternative, so a
        # failed random attack cannot masquerade as a tyre-cost improvement.
        patch.setattr(simulator.overtaking_model, "should_attempt_overtake", lambda *a, **kw: True)
        patch.setattr(simulator.overtaking_model, "attempt_overtake",
                      lambda *a, **kw: (True, False))

        def control(lap, *args, **kwargs):
            simulator.event_manager.current_lap = lap
            simulator.event_manager.safety_car_active = lap == 13
            return []

        patch.setattr(simulator.event_manager, "process_lap", control)

        def choose(state, lap, selected):
            if finite:
                state.inventory_pit_proposal = (lap, selected.value)
            else:
                state.dry_pit_proposal = (lap, selected)
            return True

        def decide(state, states, track, lap, *args, **kwargs):
            if state.driver.id == "A":
                return choose(state, lap, TireCompound.MEDIUM) if lap == 2 else False
            if lap == 4:
                return choose(state, lap, TireCompound.MEDIUM)
            if lap < 14:
                return False
            if lap == 14:
                proposed = native(state, states, track, lap, *args, **kwargs)
                observed.update(native=bool(proposed), age=state.tire_laps,
                                compound=state.current_tire.compound,
                                safety_car=simulator.event_manager.safety_car_active)
                if stop_lap is None:
                    return proposed
            if stop_lap is not None:
                return choose(state, lap, compound) if lap == stop_lap else False
            return native(state, states, track, lap, *args, **kwargs)

        patch.setattr(simulator, "_should_pit", decide)
        drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
        cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
        track = Track(id="sc", name="SC", country="test", total_laps=18,
                      base_lap_time=90., pit_lane_delta=lane, tire_stress=.5)
        inventory = [dict(id=c.value, compound=c.value, age=0)
                     for c in (TireCompound.HARD, TireCompound.MEDIUM, TireCompound.SOFT)]
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        results = execute(drivers, cars, track, Weather(change_probability=0.), list("AB"),
                          starting_tires={key: TireCompound.HARD for key in "AB"},
                          tire_inventory={key: inventory for key in "AB"} if finite else None)
    selected = next(result for result in results if result.driver_id == "B")
    assert selected.status == DriverStatus.FINISHED and selected.laps_completed == 18
    assert selected.classified and set(selected.strategy) >= {"hard", "medium"}
    return selected, observed


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("lane", [3., 8.])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1.}])
def test_native_safety_car_stop_matches_best_executed_one_stop_continuation(
    monkeypatch, engine, finite, lane, warmup,
):
    selected, observed = run(monkeypatch, engine, finite, lane, warmup)
    alternatives = [run(monkeypatch, engine, finite, lane, warmup, lap, compound)[0]
                    for lap in range(14, 19)
                    for compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)]
    retained, _ = run(monkeypatch, engine, finite, lane, warmup, -1, TireCompound.SOFT)
    assert observed == dict(native=True, age=10, compound=TireCompound.MEDIUM, safety_car=True)
    assert selected.pit_laps == [4, 14] and selected.strategy == ["hard", "medium", "soft"]
    assert selected.total_time == pytest.approx(min(result.total_time for result in alternatives),
                                              abs=1.e-8)
    assert selected.total_time < retained.total_time - .1
