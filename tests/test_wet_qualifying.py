"""Qualifying uses suitable rain sets and the race's shared wet physics."""

import copy
import math

import numpy as np
import pytest

from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.qualifying import QualifyingSimulator
from f1sim.simulation.race import RaceSimulator


def fixture():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=50, base_lap_time=90)
    return driver, car, track


@pytest.mark.parametrize("wetness,rain,expected", [
    (0, 0, TireCompound.SOFT), (0.2, 0.4, TireCompound.SOFT),
    (0.20001, 0, TireCompound.INTERMEDIATE), (0, 0.40001, TireCompound.INTERMEDIATE),
    (0.7, 0, TireCompound.INTERMEDIATE), (0.70001, 0, TireCompound.WET),
])
def test_race_fresh_rain_boundaries_remain_unchanged(wetness, rain, expected):
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    assert (RaceSimulator._choose_weather_compound(weather) or TireCompound.SOFT) == expected


@pytest.mark.parametrize("wetness,rain", [(0, 0), (0, 0.5), (0.21, 0.3), (0.75, 0.8), (0.9, 0.9)])
@pytest.mark.parametrize("zones", [0, 3])
def test_session_selects_fastest_fresh_set_under_fixed_weather(monkeypatch, wetness, rain, zones):
    driver, car, track = fixture()
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    track.active_aero_zones = [ActiveAeroZone(zone_id=index + 1, sector=1, time_gain=.3)
                              for index in range(zones)]

    class MeanPace:
        def normal(self, mean, std):
            return mean

        def random(self):
            return 1.0  # No mistake in the independent measured lap.

    measured = {compound: LapSimulator(MeanPace()).calculate_qualifying_lap(
        driver, car, track, tire, weather,
    ) for compound, tire in TIRE_COMPOUNDS.items()}
    simulator = QualifyingSimulator(np.random.default_rng(42))
    original = simulator.lap_simulator.calculate_qualifying_lap
    attempts, projections = [], []

    def capture(**kwargs):
        assert "active_aero_enabled" not in kwargs
        (attempts if kwargs.get("sample_variation", True) else projections).append(
            kwargs["tire"].compound
        )
        return original(**kwargs)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_qualifying_lap", capture)
    simulator._simulate_session([driver], {"A": car}, track, weather)
    assert attempts == [min(measured, key=measured.get)] * 2
    assert projections == list(TireCompound)


def test_projection_is_rng_free_and_default_positional_calls_unchanged():
    driver, car, track = fixture()
    driver.current_tire_laps = 20
    weather = Weather(track_wetness=0.75)
    simulator = LapSimulator(np.random.default_rng(42))
    before = copy.deepcopy((driver, car, weather, simulator.rng.bit_generator.state))
    args = (driver, car, track, TIRE_COMPOUNDS[TireCompound.WET], weather, 0.9)
    simulator.calculate_qualifying_lap(*args, sample_variation=False)
    assert (driver, car, weather, simulator.rng.bit_generator.state) == before
    legacy = LapSimulator(np.random.default_rng(42))
    assert simulator.calculate_qualifying_lap(*args) == legacy.calculate_qualifying_lap(
        *args, sample_variation=True
    )
    assert simulator.rng.random() == legacy.rng.random()


def test_missing_car_does_not_project_or_draw(monkeypatch):
    driver, _, track = fixture()
    simulator = QualifyingSimulator(np.random.default_rng(42))
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_qualifying_lap",
                        lambda **kwargs: pytest.fail("Missing-car entrant timed"))
    result, = simulator.simulate_qualifying([driver], {}, track, Weather())
    assert result.eliminated_in == "Q1"
    assert math.isinf(result.best_time)
    assert simulator.rng.bit_generator.state == before


@pytest.mark.parametrize("wetness,correct,wrong", [
    (0.3, TireCompound.INTERMEDIATE, TireCompound.SOFT),
    (0.9, TireCompound.WET, TireCompound.SOFT),
    (0.9, TireCompound.WET, TireCompound.INTERMEDIATE),
])
def test_wrong_qualifying_tyres_are_slower_with_identical_rng(wetness, correct, wrong):
    driver, car, track = fixture()
    weather = Weather(track_wetness=wetness)
    fast, slow = LapSimulator(np.random.default_rng(42)), LapSimulator(np.random.default_rng(42))
    assert fast.calculate_qualifying_lap(driver, car, track, TIRE_COMPOUNDS[correct], weather) < (
        slow.calculate_qualifying_lap(driver, car, track, TIRE_COMPOUNDS[wrong], weather)
    )
    assert fast.rng.random() == slow.rng.random()


@pytest.mark.parametrize("wetness", [0, 0.3, 0.6])
def test_driver_wet_skill_only_changes_wet_qualifying(wetness):
    driver, car, track = fixture()
    weather = Weather(track_wetness=wetness)
    tire = TIRE_COMPOUNDS[weather.fresh_rain_compound() or TireCompound.SOFT]
    times = [LapSimulator(np.random.default_rng(42)).calculate_qualifying_lap(
        driver.model_copy(update={"wet_skill_modifier": skill}), car, track, tire, weather,
    ) for skill in (0.8, 1.2)]
    assert (times[0] > times[1]) if wetness > 0 else (times[0] == times[1])


@pytest.mark.parametrize("enabled,zones", [(True, 0), (False, 3)])
def test_dry_qualifying_preserves_previous_formula_and_rng(enabled, zones):
    driver, car, track = fixture()
    track.active_aero_zones = [ActiveAeroZone(zone_id=index + 1, sector=1, time_gain=.3)
                              for index in range(zones)]
    rng = np.random.default_rng(42)
    simulator = LapSimulator(np.random.default_rng(42))
    base = track.base_lap_time * 0.98
    variation = rng.normal(0, driver.lap_time_variation_std(base_std=0.15))
    assert rng.random() >= 0.03
    assert rng.random() >= 0.006
    expected = max(track.base_lap_time * 0.93, base + car.pace_delta_seconds(base)
                   + simulator._track_car_delta(car, track, base)
                   + (1 - driver.skill_rating) * base * 0.012 + variation
                   - (TIRE_COMPOUNDS[TireCompound.SOFT].initial_grip - 1) * 0.5)
    assert simulator.calculate_qualifying_lap(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT], Weather(),
        active_aero_enabled=enabled,
    ) == expected
    assert simulator.rng.random() == rng.random()


def test_full_wet_session_has_finite_order_and_preserves_inputs():
    driver, car, track = fixture()
    drivers = [driver.model_copy(update={"id": str(i), "current_tire_laps": 20}) for i in range(22)]
    weather = Weather(track_wetness=0.9, rain_intensity=0.9)
    before = copy.deepcopy((drivers, car, weather))
    simulator = QualifyingSimulator(np.random.default_rng(42))
    results = simulator.simulate_qualifying(drivers, {"A": car}, track, weather)
    assert len(results) == 22
    assert [r.position for r in results] == list(range(1, 23))
    assert all(math.isfinite(r.best_time) for r in results)
    assert sum(r.q3_time is not None for r in results) == 10
    assert sum(r.q2_time is not None for r in results) == 16
    assert (drivers, car, weather) == before


@pytest.mark.parametrize("draws", [(.001,), (.5, .001), (.5, .5)])
def test_aero_keeps_qualifying_mistake_draw_order_and_weather_scaling(draws):
    driver, car, track = fixture()
    track.active_aero_zones = [ActiveAeroZone(zone_id=1, sector=1, time_gain=.3)]
    weather = Weather(track_wetness=.6, rain_intensity=.7)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]

    class ScriptedRng:
        def __init__(self):
            self.draws = iter(draws)
            self.calls = []

        def normal(self, mean, std):
            self.calls.append(("normal", mean, std))
            return .2

        def random(self):
            self.calls.append(("random",))
            return next(self.draws)

        def uniform(self, low, high):
            self.calls.append(("uniform", low, high))
            return low

    enabled_rng, disabled_rng = ScriptedRng(), ScriptedRng()
    enabled = LapSimulator(enabled_rng).calculate_qualifying_lap(driver, car, track, tire, weather)
    disabled_simulator = LapSimulator(disabled_rng)
    disabled = disabled_simulator.calculate_qualifying_lap(driver, car, track, tire, weather,
                                                          active_aero_enabled=False)
    gain = .3 * .8 * ((.65 + .35 * (1. - track.overtake_difficulty))
                     * (.9 + .2 * car.straight_line_speed))
    multiplier = disabled_simulator.weather_pace_multiplier(driver, car, weather)
    assert disabled - enabled == pytest.approx(gain * multiplier, abs=1e-13)
    assert enabled_rng.calls == disabled_rng.calls
    names = [call[0] for call in enabled_rng.calls]
    expected = ["normal", "random"]
    if len(draws) == 2:
        expected.append("random")
    if draws[-1] == .001:
        expected.append("uniform")
    assert names == expected


def test_saturated_gain_keeps_compound_enum_tie_and_projection_rng_free(monkeypatch):
    driver, car, track = fixture()
    driver.skill_rating = car.base_pace = car.straight_line_speed = 1.
    track.active_aero_zones = [ActiveAeroZone(zone_id=index + 1, sector=1, time_gain=1.)
                              for index in range(20)]
    weather = Weather()
    simulator = QualifyingSimulator(np.random.default_rng(42))
    original = simulator.lap_simulator.calculate_qualifying_lap
    attempts = []
    before = copy.deepcopy((driver, car, track, weather))

    def capture(**kwargs):
        state = copy.deepcopy(simulator.rng.bit_generator.state)
        result = original(**kwargs)
        if kwargs.get("sample_variation", True):
            attempts.append(kwargs["tire"].compound)
        else:
            assert simulator.rng.bit_generator.state == state
        return result

    monkeypatch.setattr(simulator.lap_simulator, "calculate_qualifying_lap", capture)
    times = simulator._simulate_session([driver], {"A": car}, track, weather)
    assert attempts == [TireCompound.SOFT, TireCompound.SOFT]
    assert times == {driver.id: track.base_lap_time * .93}
    assert (driver, car, track, weather) == before
