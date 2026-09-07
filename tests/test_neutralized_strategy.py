"""Current neutralization decisions agree with controlled full-race timing."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


@pytest.mark.parametrize("neutralization", ["safety_car_active", "vsc_active"])
@pytest.mark.parametrize("stress", [0.3, 0.9])
def test_current_neutralization_matches_remaining_one_stop_alternatives(neutralization, stress):
    class MeanPace:
        def normal(self, mean, _std):
            return mean

    def run(forced_lap=None, forced_compound=None):
        driver = Driver(id="A", name="A", team_id="team")
        car = Car(team_id="team", team_name="Team")
        track = Track(id="test", name="Test", country="Test", total_laps=30,
                      base_lap_time=90, tire_stress=stress)
        simulator = RaceSimulator(np.random.default_rng(42))
        simulator.lap_simulator.rng = MeanPace()
        simulator.lap_simulator.calculate_pit_stop_time = lambda car: expected_stationary_time(car)
        simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED

        # All alternatives reach the same lap-21 decision with the original
        # medium set. Only that lap is neutralized; future running is green.
        def events(lap, **kwargs):
            setattr(simulator.event_manager, neutralization, lap == 20)
            return []

        simulator.event_manager.process_lap = events
        should_pit = simulator._should_pit

        def choose(state, states, track, lap, *args, **kwargs):
            if lap < 21:
                return False
            if forced_lap is not None:
                return lap == forced_lap
            return should_pit(state, states, track, lap, *args, **kwargs)

        simulator._should_pit = choose
        if forced_compound is not None:
            simulator._choose_distinct_dry_compound = lambda *args: forced_compound
        return simulator.simulate_race(
            [driver], {"team": car}, track, Weather(change_probability=0), ["A"],
            starting_tires={"A": TireCompound.MEDIUM},
        )[0]

    selected = run()
    alternatives = [run(lap, compound).total_time for lap in range(21, 31)
                    for compound in (TireCompound.SOFT, TireCompound.HARD)]
    assert selected.total_time == pytest.approx(min(alternatives), abs=1e-8)
    assert selected.pit_stops == 1
    assert selected == run()
