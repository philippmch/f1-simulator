"""Automatic rain starts must not choose tyres that require an immediate refit."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def fixture():
    return (Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
            Track(id="test", name="Test", country="Test", total_laps=10, base_lap_time=90))


@pytest.mark.parametrize("condition", [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN])
@pytest.mark.parametrize("wetness,rain", [(0, 0), (0.079, 0.149)])
def test_automatic_start_does_not_choose_critical_intermediates(condition, wetness, rain):
    driver, car, track = fixture()
    simulator = RaceSimulator(np.random.default_rng(42))
    weather = Weather(condition=condition, track_wetness=wetness, rain_intensity=rain)
    assert simulator._check_tire_weather_mismatch(
        TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], weather,
    ) == "critical"
    choice = simulator._choose_starting_compound(
        TeamStrategyArchetype.BALANCED, track, weather, driver, car,
    )
    assert choice in SLICKS


@pytest.mark.parametrize("wetness,rain", [(0.08, 0), (0, 0.15), (0, 0.2)])
def test_usable_precautionary_intermediates_remain_available(wetness, rain):
    driver, car, track = fixture()
    simulator = RaceSimulator(np.random.default_rng(42))
    weather = Weather(condition=WeatherCondition.LIGHT_RAIN,
                      track_wetness=wetness, rain_intensity=rain)
    before = simulator.rng.bit_generator.state
    assert simulator._choose_starting_compound(
        TeamStrategyArchetype.BALANCED, track, weather, driver, car,
    ) == TireCompound.INTERMEDIATE
    assert simulator.rng.bit_generator.state == before


@pytest.mark.parametrize("condition", [WeatherCondition.LIGHT_RAIN, WeatherCondition.HEAVY_RAIN])
def test_rain_label_on_dry_surface_does_not_cost_an_opening_paid_stop(monkeypatch, condition):
    def run(start=None):
        driver, car, track = fixture()
        simulator = RaceSimulator(np.random.default_rng(42))
        original_lap = simulator.lap_simulator.calculate_lap_time
        monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                            lambda **kwargs: original_lap(**kwargs, sample_variation=False))
        monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                            lambda *args: expected_stationary_time(car))
        monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
        result, = simulator.simulate_race(
            [driver], {"A": car}, track, Weather(condition=condition, change_probability=0),
            ["A"], starting_tires={"A": start} if start else None,
        )
        return result

    selected = run()
    alternatives = [run(compound) for compound in SLICKS]
    forced_inter = run(TireCompound.INTERMEDIATE)
    assert 1 not in selected.pit_laps
    assert forced_inter.pit_laps[0] == 1  # Explicit overrides retain their semantics.
    assert selected.total_time == pytest.approx(min(r.total_time for r in alternatives), abs=1e-8)
    assert forced_inter.total_time > selected.total_time + 20
