"""Mechanical reliability is allocated as cumulative hazard over own laps."""

import copy
import math

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventManager, EventType


def car(reliability):
    return Car(team_id="T", team_name="T", reliability=reliability,
               **{f"{component}_reliability": reliability
                  for component in ("engine", "gearbox", "brakes", "electrical", "cooling")})


def track(laps=53, stress=.4):
    return Track(id="T", name="T", country="T", total_laps=laps,
                 base_lap_time=90, tire_stress=stress)


@pytest.mark.parametrize("reliability", [.99, .95, .7, .1, 0, 1])
@pytest.mark.parametrize("laps", [1, 7, 53, 400])
def test_full_distance_survival_matches_combined_reliability(reliability, laps):
    manager = EventManager()
    vehicle, circuit = car(reliability), track(laps)
    survival = math.prod(1 - manager._mechanical_failure_probability(vehicle, circuit, None, lap)
                         for lap in range(1, laps + 1))
    assert survival == pytest.approx(reliability, abs=1e-14)


def test_component_combination_retains_its_existing_weights():
    vehicle = car(.9)
    vehicle.engine_reliability = .5
    components = list(vehicle.component_reliability_map().values())
    expected = .55 * .9 + .3 * sum(components) / len(components) + .15 * min(components)
    manager = EventManager()
    survival = math.prod(1 - manager._mechanical_failure_probability(vehicle, track(), None, lap)
                         for lap in range(1, 54))
    assert survival == pytest.approx(expected)


def test_shortened_exposure_preserves_original_scheduled_distance():
    manager = EventManager()
    probabilities = [manager._mechanical_failure_probability(car(.7), track(), None, lap)
                     for lap in range(1, 54)]
    survivals = [math.prod(1 - p for p in probabilities[:n]) for n in (0, 1, 10, 30, 53)]
    assert all(a > b for a, b in zip(survivals, survivals[1:]))
    assert survivals[-1] == pytest.approx(.7)
    assert survivals[2] > math.prod(
        1 - manager._mechanical_failure_probability(car(.7), track(10), None, lap)
        for lap in range(1, 11)
    )


def test_stress_heat_and_late_progress_multiply_hazard():
    manager = EventManager()
    vehicle = car(.7)
    neutral = manager._mechanical_failure_probability(vehicle, track(), None, 1)
    stressed = manager._mechanical_failure_probability(vehicle, track(stress=1), None, 1)
    hot = manager._mechanical_failure_probability(vehicle, track(stress=1),
                                                 Weather(track_temperature=60), 1)
    late = manager._mechanical_failure_probability(vehicle, track(stress=1),
                                                  Weather(track_temperature=60), 53)
    assert 0 < neutral < stressed < hot < late < 1
    survival = math.prod(1 - manager._mechanical_failure_probability(vehicle, track(stress=1),
                                                                    None, lap)
                         for lap in range(1, 54))
    assert survival == pytest.approx(.7 ** 1.15)


@pytest.mark.parametrize("reliability", [0, 1e-300, .1, np.nextafter(1., 0.), 1])
@pytest.mark.parametrize("laps,stress,temperature", [(1, 0, 10), (400, 1, 60)])
def test_extreme_valid_values_are_finite_probabilities(reliability, laps, stress, temperature):
    probability = EventManager()._mechanical_failure_probability(
        car(reliability), track(laps, stress), Weather(track_temperature=temperature), laps,
    )
    assert math.isfinite(probability)
    assert 0 <= probability <= 1
    if 0 < reliability < 1:
        assert probability > 0


def test_probability_calculation_is_pure_and_does_not_consume_rng():
    manager = EventManager(np.random.default_rng(14))
    vehicle, circuit, weather = car(.9), track(), Weather()
    before = copy.deepcopy((vehicle, circuit, weather, manager.rng.bit_generator.state))
    manager._mechanical_failure_probability(vehicle, circuit, weather, 20)
    assert (vehicle, circuit, weather, manager.rng.bit_generator.state) == before


class FailureRng:
    def __init__(self, draw):
        self.draw = draw
        self.calls = []

    def random(self):
        self.calls.append("random")
        return self.draw

    def choice(self, candidates, p):
        self.calls.append("choice")
        assert candidates == ["engine", "gearbox", "brakes", "electrical", "cooling"]
        assert p == pytest.approx([.2] * 5)
        return "engine"


@pytest.mark.parametrize("reliability,draw,failed", [(1, 0, False), (0, .999, True),
                                                    (.7, 0, True), (.7, .999, False)])
def test_failure_sampling_keeps_draw_order_and_component_selection(reliability, draw, failed):
    rng = FailureRng(draw)
    manager = EventManager(rng)
    driver = Driver(id="A", name="A", team_id="T")
    event = manager._check_mechanical_failure(driver, car(reliability), track(), 20, Weather())
    assert rng.calls == (["random", "choice"] if failed else ["random"])
    assert driver.dnf is failed
    if failed:
        assert event.event_type == EventType.MECHANICAL_FAILURE
        assert event.drivers_involved == ["A"]
        assert driver.dnf_reason == "engine failure"
    else:
        assert event is None
