"""Electrical Overtake Mode works independently of Active Aero configuration."""

from copy import deepcopy

import numpy as np
import pytest
from pydantic import ValidationError

from f1sim.data.current import TrackStats
from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.overtaking import OvertakingModel


@pytest.fixture
def battle():
    attacker = Driver(id="A", name="A", team_id="T")
    defender = attacker.model_copy(update={"id": "B", "name": "B"})
    car = Car(team_id="T", team_name="T", straight_line_speed=1)
    track = Track(id="test", name="Test", country="Test", total_laps=20,
                  base_lap_time=90, overtake_difficulty=.95)
    return attacker, car, defender, car, track


def effects(battle, *, weather=None, aero=True):
    attacker, car, _, _, track = battle
    physics = LapSimulator(np.random.default_rng(42))
    before = deepcopy(physics.rng.bit_generator.state)
    weather = weather if weather is not None else Weather()
    args = (attacker, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], weather, 10, 20)
    laps = [physics.calculate_lap_time(
        *args, gap_to_car_ahead=.8, sample_variation=False,
        active_aero_enabled=aero, overtake_mode_active=active,
    ) for active in (False, True)]
    probabilities = [OvertakingModel()._calculate_probability(
        *battle, .8, active, weather.is_wet(),
    ) for active in (False, True)]
    assert physics.rng.bit_generator.state == before
    return laps[0] - laps[1], probabilities[1] - probabilities[0]


@pytest.mark.parametrize("aero", [False, True])
@pytest.mark.parametrize("zone_gains", [[], [.25], [.75, .75]])
def test_mode_benefits_do_not_depend_on_aero_zones_or_permission(battle, aero, zone_gains):
    reference = effects(battle, aero=aero)
    battle[-1].active_aero_zones = [
        ActiveAeroZone(zone_id=i + 1, sector=2, time_gain=gain)
        for i, gain in enumerate(zone_gains)
    ]
    actual = effects(battle, aero=aero)
    assert actual == pytest.approx(reference)
    assert 0 < actual[0] <= .35
    assert 0 < actual[1] < .2


@pytest.mark.parametrize("effectiveness", [0, .25, .5, 1])
def test_independent_venue_effectiveness_scales_both_benefits(battle, effectiveness):
    reference = effects(battle)
    battle[-1].overtake_mode_effectiveness = effectiveness
    actual = effects(battle)
    assert actual == pytest.approx(tuple(gain * effectiveness for gain in reference))


@pytest.mark.parametrize("weather", [
    Weather(track_wetness=.31), Weather(track_wetness=.7, rain_intensity=.8),
])
def test_wet_running_suppresses_both_benefits_without_aero_zones(battle, weather):
    assert effects(battle, weather=weather) == (0, 0)


def test_mode_effects_do_not_read_active_aero_gain_when_aero_is_disabled(battle):
    class IndependentTrack(Track):
        @property
        def total_active_aero_gain(self):
            raise AssertionError("Electrical deployment must not read the aero gain")

    track = IndependentTrack(**battle[-1].model_dump())
    actual = effects((*battle[:-1], track), aero=False)
    assert actual[0] > 0 and actual[1] > 0


def test_zero_zone_mode_changes_a_pass_with_one_draw_and_respects_detection(battle):
    base = OvertakingModel()._calculate_probability(*battle, .8, False, False)
    boosted = OvertakingModel()._calculate_probability(*battle, .8, True, False)
    assert boosted > base

    class FixedRoll:
        draws = 0

        def random(self):
            self.draws += 1
            return (base + boosted) / 2

    rng = FixedRoll()
    model = OvertakingModel(rng)
    assert not model.attempt_overtake(*battle, .8)[0]
    assert model.attempt_overtake(*battle, .8, overtake_mode_active=True) == (True, False)
    assert not model.attempt_overtake(*battle, 1.01, overtake_mode_active=True)[0]
    assert rng.draws == 3
    assert model.attempt_overtake(*battle, 1.6, overtake_mode_active=True) == (False, False)
    assert rng.draws == 3


@pytest.mark.parametrize("owner", [Track, TrackStats])
@pytest.mark.parametrize("value", [-.01, 1.01, float("nan"), float("inf"), -float("inf")])
def test_effectiveness_rejects_nonfinite_or_out_of_range_inputs(owner, value):
    fields = (dict(id="t", name="T", country="T", total_laps=20, base_lap_time=90)
              if owner is Track else dict(track_id="t", track_name="T"))
    with pytest.raises(ValidationError, match="overtake_mode_effectiveness"):
        owner(**fields, overtake_mode_effectiveness=value)
