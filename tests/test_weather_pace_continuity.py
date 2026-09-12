"""Numeric weather controls pace continuously across shared execution paths."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


@pytest.fixture
def physics():
    return (LapSimulator(np.random.default_rng(4)),
            Driver(id="D", name="Driver", team_id="T", wet_skill_modifier=.6),
            Car(team_id="T", team_name="Team", wet_performance=.4),
            Track(id="T", name="Track", country="Test", total_laps=20, base_lap_time=90))


@pytest.mark.parametrize("compound", list(TireCompound))
@pytest.mark.parametrize("water,rain", [(0., 0.), (.07, .1), (.6, .2), (.85, .9)])
def test_labels_cannot_change_racing_qualifying_or_relative_pace(physics, compound, water, rain):
    sim, driver, car, track = physics
    tire = TIRE_COMPOUNDS[compound]
    driver.current_tire_laps = 12
    rng_before = deepcopy(sim.rng.bit_generator.state)
    rows = []
    for label in WeatherCondition:
        weather = Weather(condition=label, track_wetness=water, rain_intensity=rain)
        rows.append((
            sim.calculate_lap_time(driver, car, track, tire, weather, 5, 20,
                                   sample_variation=False),
            sim.calculate_qualifying_lap(driver, car, track, tire, weather,
                                         sample_variation=False),
            sim.tire_weather_pace_contribution(driver, car, track, tire, 12, weather),
        ))
    assert all(row == rows[0] for row in rows)
    assert sim.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("compound,boundary", [("soft", .2), ("soft", .5),
                                               ("intermediate", .15), ("intermediate", .8),
                                               ("wet", .3), ("hard", .3)])
def test_tiny_water_changes_cannot_trigger_multi_second_pace_steps(physics, compound, boundary):
    sim, driver, car, track = physics
    tire = TIRE_COMPOUNDS[TireCompound(compound)]
    rows = []
    for delta in (-1e-8, 0., 1e-8):
        weather = Weather(condition="heavy_rain", track_wetness=boundary + delta)
        rows.append((
            sim.calculate_lap_time(driver, car, track, tire, weather, 5, 20,
                                   sample_variation=False),
            sim.calculate_qualifying_lap(driver, car, track, tire, weather,
                                         sample_variation=False),
        ))
    for column in zip(*rows, strict=True):
        assert max(column) - min(column) < 1e-4


@pytest.mark.parametrize("compound,anchors", [
    ("soft", [(0., 0.), (.2, 0.), (.5, 7.5), (1., 30.)]),
    ("intermediate", [(0., 7.), (.15, 0.), (.8, 0.), (1., 10.)]),
    ("wet", [(0., 14.), (.3, 0.), (1., 0.)]),
])
def test_continuous_mismatch_retains_declared_model_anchors(compound, anchors):
    tire = TIRE_COMPOUNDS[TireCompound(compound)]
    for water, expected in anchors:
        assert LapSimulator._tire_weather_mismatch(tire, Weather(track_wetness=water)) == (
            pytest.approx(expected)
        )


@pytest.mark.parametrize("skill", [.5, 1., 1.5])
@pytest.mark.parametrize("car_wet", [0., 1.])
def test_wet_exposure_scales_smoothly_without_a_dry_wet_skill_bonus(physics, skill, car_wet):
    sim, driver, car, _ = physics
    driver.wet_skill_modifier = skill
    car.wet_performance = car_wet
    factors = [sim.weather_pace_multiplier(driver, car, Weather(track_wetness=i / 100))
               for i in range(101)]
    assert factors[0] == 1.
    assert all(b > a for a, b in zip(factors, factors[1:]))
    assert max(b - a for a, b in zip(factors, factors[1:])) < .003


def test_surface_water_remains_after_rain_stops_and_labels_change():
    wet = Weather(condition="heavy_rain", track_wetness=.85, rain_intensity=.8)
    stopped = wet.model_copy(update={"condition": WeatherCondition.CLOUDY, "rain_intensity": 0.})
    assert stopped.wet_severity() == wet.wet_severity() == .85
    assert stopped.lap_time_multiplier() == wet.lap_time_multiplier()
    drying = stopped.project_surface()
    assert 1. < drying.lap_time_multiplier() < stopped.lap_time_multiplier()
    assert stopped.track_wetness == .85


def test_rain_contributes_before_water_accumulates_and_zero_water_is_neutral():
    assert Weather(condition="heavy_rain").lap_time_multiplier() == 1.
    factors = [Weather(track_wetness=0., rain_intensity=i / 100).lap_time_multiplier()
               for i in range(101)]
    assert factors[0] == 1.
    assert factors[-1] == pytest.approx(1.14)
    assert all(b > a for a, b in zip(factors, factors[1:]))


@pytest.mark.parametrize("water,rain", [(0., 0.), (.08, 0.), (0., .15), (.1, .1), (.35, .35)])
def test_opening_candidates_and_race_rng_do_not_depend_on_label(physics, water, rain):
    _, driver, car, track = physics
    track.total_laps = 6
    rows = []
    for label in WeatherCondition:
        sim = RaceSimulator(np.random.default_rng(42))
        compound = sim._choose_starting_compound(
            TeamStrategyArchetype.BALANCED, track,
            Weather(condition=label, track_wetness=water, rain_intensity=rain), driver, car,
        )
        rows.append((compound, deepcopy(sim.rng.bit_generator.state)))
    assert all(row == rows[0] for row in rows)
