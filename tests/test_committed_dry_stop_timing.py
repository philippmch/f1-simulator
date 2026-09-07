"""Forced dry replacements are checked against actual remaining race runs."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def test_forced_dry_compound_matches_full_race_alternatives():
    class MeanPace:
        def normal(self, mean, _std):
            return mean

    def run(forced_compound=None):
        driver = Driver(id="A", name="A", team_id="team")
        car = Car(team_id="team", team_name="Team")
        track = Track(id="test", name="Test", country="Test", total_laps=60,
                      base_lap_time=90, tire_stress=0.9, pit_lane_delta=10)
        simulator = RaceSimulator(np.random.default_rng(42))
        simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
        simulator.event_manager.process_lap = lambda **kwargs: []
        simulator.lap_simulator.rng = MeanPace()
        simulator.lap_simulator.calculate_pit_stop_time = lambda car: expected_stationary_time(car)
        original_should_pit = simulator._should_pit
        original_choose = simulator._choose_committed_dry_compound
        choices = []

        def should_pit(state, states, track, lap, *args, **kwargs):
            if lap <= 10:
                return lap in (2, 10)
            return original_should_pit(state, states, track, lap, *args, **kwargs)

        def choose(state, track, lap):
            if lap == 2:
                return TireCompound.HARD
            choice = forced_compound or original_choose(state, track, lap)
            if lap == 10:
                choices.append(choice)
            return choice

        simulator._should_pit = should_pit
        simulator._choose_committed_dry_compound = choose
        result = simulator.simulate_race(
            [driver], {"team": car}, track, Weather(change_probability=0), ["A"],
            starting_tires={"A": TireCompound.MEDIUM},
        )[0]
        return result, choices[0]

    selected, compound = run()
    alternatives = {c: run(c)[0] for c in
                    (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)}
    assert compound == TireCompound.MEDIUM
    assert selected.total_time == pytest.approx(
        min(result.total_time for result in alternatives.values()), abs=1e-8,
    )
    assert alternatives[TireCompound.HARD].total_time - selected.total_time > 1.3
    assert selected.pit_stops == 3
    assert (selected, compound) == run()
