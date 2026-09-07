"""High-wear full races benefit from searching beyond style-based stop caps."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def run(laps, budget=None, forced_lap=None, forced_compound=None):
    class MeanPace:
        def normal(self, mean, _std):
            return mean

    driver = Driver(id="A", name="A", team_id="team", tire_management=0.4)
    car = Car(team_id="team", team_name="Team", tire_degradation_factor=1.5)
    track = Track(id="test", name="Test", country="Test", total_laps=laps,
                  base_lap_time=110, tire_stress=1, pit_lane_delta=20)
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    simulator.event_manager.process_lap = lambda **kwargs: []
    simulator.lap_simulator.rng = MeanPace()
    simulator.lap_simulator.calculate_pit_stop_time = lambda car: expected_stationary_time(car)
    if budget is not None:
        simulator._dry_stop_budget = lambda *args: budget
    if forced_lap is not None:
        simulator._should_pit = lambda state, states, track, lap, *a, **kw: lap == forced_lap
        simulator._choose_distinct_dry_compound = lambda *args: forced_compound
    return simulator.simulate_race(
        [driver], {"team": car}, track, Weather(change_probability=0), ["A"],
        starting_tires={"A": TireCompound.SOFT},
    )[0]


def test_high_wear_two_stop_beats_every_legal_one_stop_full_race():
    selected = run(50)
    one_stop_times = [run(50, forced_lap=lap, forced_compound=compound).total_time
                      for lap in range(2, 51)
                      for compound in (TireCompound.MEDIUM, TireCompound.HARD)]
    assert selected.pit_stops == 2
    assert min(one_stop_times) - selected.total_time > 2.5
    assert run(50, budget=1).total_time == pytest.approx(min(one_stop_times), abs=1e-8)
    assert selected == run(50)


def test_long_high_wear_race_benefits_from_third_stop():
    selected, restricted = run(78), run(78, budget=2)
    assert selected.pit_stops == 3
    assert restricted.pit_stops == 2
    assert restricted.total_time - selected.total_time > 3.5
    assert selected == run(78)
