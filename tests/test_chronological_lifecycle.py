"""Full-race lifecycle invariants with real pace, strategy and incidents."""

from collections import Counter

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import DriverStatus, RaceSimulator


@pytest.mark.parametrize("seed", [2, 17, 42])
@pytest.mark.parametrize("condition,intensity", [
    (WeatherCondition.DRY, 0), (WeatherCondition.LIGHT_RAIN, 0.35),
    (WeatherCondition.HEAVY_RAIN, 0.85),
])
def test_actual_work_matches_own_distance_and_terminal_state(
    monkeypatch, seed, condition, intensity,
):
    drivers = [Driver(id=str(i), name=str(i), team_id=str(i // 2)) for i in range(6)]
    cars = {str(i): Car(team_id=str(i), team_name=str(i), base_pace=1 - i * 0.2,
                       reliability=0.6) for i in range(3)}
    track = Track(id="lifecycle", name="Lifecycle", country="Test", total_laps=60,
                  base_lap_time=90, safety_car_probability=0.6)
    weather = Weather(condition=condition, rain_intensity=intensity,
                      track_wetness=intensity, change_probability=0.05)
    original_inputs = (
        track.model_dump(), weather.model_dump(), [c.model_dump() for c in cars.values()],
    )
    simulator = RaceSimulator(np.random.default_rng(seed))
    engine = ChronologicalRace(simulator)
    started, exposure = [], []
    actual_running = simulator.lap_simulator.calculate_lap_time
    actual_exposure = simulator.event_manager._check_mechanical_failure

    def running(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        # Instrument this simulator only. Speculative strategy physics uses
        # separate LapSimulator instances and must not count as real running.
        started.append((driver.id, lap))
        assert total_laps == 60
        assert 0 <= driver.current_tire_laps < lap
        return actual_running(driver, car, track, tire, weather, lap, total_laps, **kwargs)

    def mechanical(driver, car, track, lap, weather):
        exposure.append((driver.id, lap))
        return actual_exposure(driver, car, track, lap, weather)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", running)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", mechanical)
    results = engine.run(drivers, cars, track, weather, [d.id for d in drivers])
    assert len(results) == len(drivers)
    assert Counter(started) == Counter(exposure)
    assert all(count == 1 for count in Counter(started).values())
    assert all(a[2] <= b[2] for a, b in zip(engine.crossings, engine.crossings[1:]))
    for result in results:
        assert result.status in {DriverStatus.FINISHED, DriverStatus.DNF}
        own_starts = [lap for driver, lap in started if driver == result.driver_id]
        extra = int(result.status == DriverStatus.DNF)
        assert own_starts == list(range(1, result.laps_completed + extra + 1))
        crosses = [(lap, time) for driver, lap, time in engine.crossings
                   if driver == result.driver_id]
        assert [lap for lap, _ in crosses] == list(range(1, result.laps_completed + 1))
        assert result.total_time == (crosses[-1][1] if crosses else 0)
        exits = [lap for driver, lap, _ in engine.pit_exits if driver == result.driver_id]
        assert exits == result.pit_laps
        assert len(exits) == result.pit_stops
    assert original_inputs == (
        track.model_dump(), weather.model_dump(), [c.model_dump() for c in cars.values()],
    )
