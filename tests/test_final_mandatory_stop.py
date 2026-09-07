"""Mandatory tyre use must not force a costlier penultimate-lap stop."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator


def run_short_race(monkeypatch, weather, forced_lap=None, laps=3):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=laps, base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(42))
    actual_lap_time = simulator.lap_simulator.calculate_lap_time
    ran = []

    def mean_lap(**kwargs):
        ran.append(kwargs["tire"].compound)
        return actual_lap_time(**kwargs, sample_variation=False)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", mean_lap)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda *args: expected_stationary_time(car))
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    if forced_lap is not None:
        monkeypatch.setattr(simulator, "_should_pit",
                            lambda state, states, track, lap, *args, **kwargs: lap == forced_lap)
    result, = simulator.simulate_race(
        [driver], {"A": car}, track, weather, ["A"], starting_tires={"A": TireCompound.SOFT},
    )
    return result, ran


@pytest.mark.parametrize("weather", [
    Weather(change_probability=0),
    Weather(track_wetness=0.1, rain_intensity=0.1, change_probability=0),
])
def test_final_lap_change_runs_distinct_set_and_avoids_costlier_early_stop(monkeypatch, weather):
    selected, ran = run_short_race(monkeypatch, weather)
    early, _ = run_short_race(monkeypatch, weather, forced_lap=2)
    final, _ = run_short_race(monkeypatch, weather, forced_lap=3)
    assert selected.pit_laps == [3]
    assert ran[:2] == [TireCompound.SOFT, TireCompound.SOFT]
    assert ran[-1] in {TireCompound.MEDIUM, TireCompound.HARD}
    assert selected.total_time == pytest.approx(final.total_time)
    assert selected.total_time < early.total_time - 0.3


def test_one_lap_race_does_not_buy_a_set_that_cannot_add_actual_compound_use(monkeypatch):
    result, ran = run_short_race(monkeypatch, Weather(change_probability=0), laps=1)
    assert result.pit_laps == []
    assert ran == [TireCompound.SOFT]
    assert result.strategy == ["soft"]
