"""A paid weather refit must not win by claiming an unavailable tyre or traffic."""

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator


def test_drying_retention_avoids_losing_paid_wet_stop_in_controlled_race(monkeypatch):
    def run(force_stop):
        simulator = RaceSimulator(np.random.default_rng(0))
        should_pit = simulator._should_pit

        def decide(state, states, track, lap, *args, **kwargs):
            # Isolate the decision under review; both alternatives retain
            # their chosen set afterwards so the full cost can be observed.
            return lap == 2 and (
                force_stop or should_pit(state, states, track, lap, *args, **kwargs)
            )

        lap_time = simulator.lap_simulator.calculate_lap_time
        monkeypatch.setattr(simulator, "_should_pit", decide)
        monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
        monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                            lambda **kwargs: lap_time(**kwargs, sample_variation=False))
        monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                            expected_stationary_time)
        result, = simulator.simulate_race(
            [Driver(id="A", name="A", team_id="A")],
            {"A": Car(team_id="A", team_name="A")},
            Track(id="t", name="T", country="T", total_laps=10,
                  base_lap_time=90, pit_lane_delta=1),
            Weather(condition=WeatherCondition.LIGHT_RAIN, track_wetness=0.76,
                    rain_intensity=0, change_probability=0),
            ["A"], starting_tires={"A": TireCompound.INTERMEDIATE},
        )
        return result

    selected, paid = run(False), run(True)
    assert selected.pit_stops == 0
    assert selected.strategy == ["intermediate"]
    assert paid.pit_laps == [2]
    assert paid.strategy == ["intermediate", "wet"]
    assert paid.total_time - selected.total_time > 5
