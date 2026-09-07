"""Late weather calls preserve time when the tyre mismatch is survivable."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


@pytest.mark.parametrize("rain_lap", [8, 9, 10])
def test_late_damp_laps_avoid_costly_stop_in_controlled_full_race(rain_lap):
    class MeanPace:
        def normal(self, mean, _std):
            return mean

    def run(final_stop=None):
        driver = Driver(id="A", name="A", team_id="team")
        car = Car(team_id="team", team_name="Team")
        track = Track(id="test", name="Test", country="Test", total_laps=10,
                      base_lap_time=90, pit_lane_delta=22)
        simulator = RaceSimulator(np.random.default_rng(42))
        simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
        simulator.lap_simulator.rng = MeanPace()
        simulator.lap_simulator.calculate_pit_stop_time = lambda car: expected_stationary_time(car)

        def events(lap, weather, **kwargs):
            if lap == rain_lap - 1:
                weather.condition = WeatherCondition.LIGHT_RAIN
                weather.track_wetness = 0.21
                weather.rain_intensity = 0.21
            return []

        simulator.event_manager.process_lap = events
        should_pit = simulator._should_pit

        def choose(state, states, track, lap, *args, **kwargs):
            # Both alternatives run the same legal medium-to-hard strategy
            # before rain arrives near the finish.
            if lap < rain_lap:
                return lap == 3
            if final_stop is not None:
                return final_stop and lap == rain_lap
            return should_pit(state, states, track, lap, *args, **kwargs)

        simulator._should_pit = choose
        simulator._choose_committed_dry_compound = lambda *args: TireCompound.HARD
        return simulator.simulate_race(
            [driver], {"team": car}, track, Weather(change_probability=0), ["A"],
            starting_tires={"A": TireCompound.MEDIUM},
        )[0]

    selected, stay, stop = run(), run(False), run(True)
    assert selected.total_time == pytest.approx(stay.total_time, abs=1e-8)
    assert stop.total_time - selected.total_time > 15
    assert selected.pit_stops == 1
    assert stop.pit_stops == 2
    assert selected == run()
