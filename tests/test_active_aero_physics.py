"""Configured Straight Mode physics is shared by race and qualifying laps."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import ActiveAeroZone, Car, Driver, Sector, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator


def inputs(zones=3, sectors=True, speed=.8):
    track = Track(id="t", name="T", country="T", total_laps=20, base_lap_time=90,
                  overtake_difficulty=.25,
                  active_aero_zones=[ActiveAeroZone(zone_id=i + 1, sector=1,
                                                   time_gain=gain)
                                     for i, gain in enumerate((.2, .3, .4)[:zones])])
    if sectors:
        track.sectors = [Sector(number=1, base_time=15, is_high_speed=True,
                                overtake_opportunity=.8),
                         Sector(number=2, base_time=45, overtake_opportunity=.2)]
    return (Driver(id="d", name="D", team_id="t"),
            Car(team_id="t", team_name="T", straight_line_speed=speed), track)


@pytest.mark.parametrize("zones,sectors,speed", [(0, True, .8), (3, True, .6),
                                               (3, True, 1.), (3, False, .8)])
@pytest.mark.parametrize("wetness,rain", [(0., 0.), (.6, .7)])
def test_independent_gain_is_weather_scaled_in_race_and_qualifying(zones, sectors, speed,
                                                                  wetness, rain):
    driver, car, track = inputs(zones, sectors, speed)
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    tire = TIRE_COMPOUNDS[TireCompound.SOFT]  # Flat wet mismatch stays outside scaling.
    simulator = LapSimulator(np.random.default_rng(42))
    before = deepcopy((driver, car, track, weather, simulator.rng.bit_generator.state))
    opportunity = .25 * .8 + .75 * .2 if sectors else .75
    total = sum((.2, .3, .4)[:zones])
    gain = total * .8 * ((.65 + .35 * opportunity) * (.9 + .2 * speed))
    assert simulator._active_aero_gain(car, track) == gain
    severity = max(wetness, .7 * rain)
    multiplier = ((1. + .2 * severity)
                  * (1. + (1. - driver.wet_skill_modifier) * .02 * severity))
    if severity > 0:
        multiplier *= 1. + (1. - car.wet_performance) * severity * .06
    expected_difference = gain * multiplier
    enabled = simulator.calculate_qualifying_lap(driver, car, track, tire, weather,
                                                 sample_variation=False)
    disabled = simulator.calculate_qualifying_lap(driver, car, track, tire, weather,
                                                  sample_variation=False,
                                                  active_aero_enabled=False)
    assert disabled - enabled == pytest.approx(expected_difference, abs=1e-13)
    race_enabled = simulator.calculate_lap_time(driver, car, track, tire, weather, 1, 20,
                                                sample_variation=False)
    race_disabled = simulator.calculate_lap_time(driver, car, track, tire, weather, 1, 20,
                                                 sample_variation=False,
                                                 active_aero_enabled=False)
    assert race_disabled - race_enabled == pytest.approx(expected_difference, abs=1e-13)
    assert (driver, car, track, weather, simulator.rng.bit_generator.state) == before


@pytest.mark.parametrize("method", ["race", "qualifying"])
def test_public_gain_preserves_stateful_getter_order_and_profile_dispatch(monkeypatch, method):
    events = []

    class CustomTrack(Track):
        @property
        def total_active_aero_gain(self):
            events.append("track")
            return float(events.count("track"))

    def speed(self):
        events.append("car")
        return .9

    class CustomSimulator(LapSimulator):
        @classmethod
        def _track_car_delta(cls, *args):
            return 0.

        @staticmethod
        def _track_profile(track):
            events.append("profile")
            return .5, .6

    driver, car, track = inputs()
    monkeypatch.setattr(Car, "straight_line_speed", property(speed), raising=False)
    track = CustomTrack.model_validate(track.model_dump())
    simulator = CustomSimulator()
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    if method == "race":
        actual = simulator.calculate_lap_time(driver, car, track, tire, Weather(), 1, 20,
                                              sample_variation=False)
        without_gain = 90. + car.pace_delta_seconds(90.) + (1. - driver.skill_rating) * 90. * .03
        without_gain += simulator.tire_pace_contribution(driver, car, track, tire, 0) + 90. * .02
    else:
        actual = simulator.calculate_qualifying_lap(driver, car, track, tire, Weather(),
                                                    sample_variation=False)
        base = 90. * .98
        without_gain = (base + car.pace_delta_seconds(base)
                        + (1. - driver.skill_rating) * base * .012)
        without_gain -= (tire.initial_grip - 1.) * .5
    gain = 2. * .8 * ((.65 + .35 * .6) * (.9 + .2 * .9))
    assert actual == pytest.approx(without_gain - gain, abs=1e-13)
    assert events == ["track", "profile", "car", "track"]
    events.clear()
    assert simulator._active_aero_gain(car, track, total_gain=2.) == gain
    assert events == ["profile", "car"]


def test_qualifying_floor_limits_even_large_configured_gain():
    driver, car, track = inputs()
    driver.skill_rating = car.base_pace = car.straight_line_speed = 1.
    track.active_aero_zones = [ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1.)
                              for i in range(20)]
    actual = LapSimulator().calculate_qualifying_lap(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT], Weather(), sample_variation=False,
    )
    assert actual == track.base_lap_time * .93


def test_disabled_qualifying_skips_gain_dispatch():
    class NoGain(LapSimulator):
        def _active_aero_gain(self, *args, **kwargs):
            raise AssertionError("Disabled qualifying must not compute Straight Mode gain")

    driver, car, track = inputs()
    result = NoGain().calculate_qualifying_lap(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT], Weather(),
        sample_variation=False, active_aero_enabled=False,
    )
    assert result > track.base_lap_time * .93
