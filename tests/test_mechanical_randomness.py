"""Stable per-driver mechanical random streams for the opt-in policy."""

import hashlib

import numpy as np
import pytest

import f1sim.analysis.montecarlo as montecarlo
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventManager
from f1sim.simulation.randomness import (
    mechanical_rng_factory_for_trial,
)


def test_mechanical_factory_freezes_sha256_seedsequence_layout():
    seed = 37
    driver_id = "Å-7"
    lap = 12
    factory = mechanical_rng_factory_for_trial(seed, "isolated_weather_mechanical_v1")
    assert factory is not None

    digest = hashlib.sha256(driver_id.encode("utf-8")).digest()
    words = tuple(
        int.from_bytes(digest[offset:offset + 4], "little", signed=False)
        for offset in range(0, len(digest), 4)
    )
    expected = np.random.default_rng(np.random.SeedSequence(
        seed, spawn_key=(0x4D454348, *words, lap),
    ))
    assert np.array_equal(factory(driver_id, lap).random(8), expected.random(8))


def test_old_policies_have_no_mechanical_factory():
    assert mechanical_rng_factory_for_trial(1, "shared_v1") is None
    assert mechanical_rng_factory_for_trial(1, "isolated_weather_v1") is None


def test_mechanical_streams_change_with_seed_driver_and_lap():
    factory = mechanical_rng_factory_for_trial(2026, "isolated_weather_mechanical_v1")
    assert factory is not None
    streams = [
        factory("A", 1).random(4),
        factory("A", 2).random(4),
        factory("B", 1).random(4),
        mechanical_rng_factory_for_trial(2027, "isolated_weather_mechanical_v1")("A", 1).random(4),
    ]
    assert all(not np.array_equal(streams[0], other) for other in streams[1:])


def test_nonforced_hazard_outcomes_ignore_race_draws():
    factory = mechanical_rng_factory_for_trial(91, "isolated_weather_mechanical_v1")
    assert factory is not None
    track = Track(id="T", name="T", country="T", total_laps=20, base_lap_time=90)
    car = Car(
        team_id="T", team_name="T", reliability=0.2,
        **{f"{component}_reliability": 0.2 for component in (
            "engine", "gearbox", "brakes", "electrical", "cooling",
        )},
    )

    def sample(extra_race_draws: int) -> dict[str, str | None]:
        manager = EventManager(
            np.random.default_rng(5), mechanical_rng_factory=factory,
        )
        manager.rng.random(extra_race_draws)
        outcomes = {}
        for index in range(100):
            driver = Driver(id=f"D{index}", name=f"D{index}", team_id="T")
            event = manager._check_mechanical_failure(
                driver, car, track, index % track.total_laps + 1, Weather(),
            )
            outcomes[driver.id] = event.description if event else None
        return outcomes

    original = sample(0)
    perturbed = sample(100)
    assert original == perturbed
    assert any(value is not None for value in original.values())


def test_changed_risk_can_change_a_mechanical_outcome():
    factory = mechanical_rng_factory_for_trial(2026, "isolated_weather_mechanical_v1")
    assert factory is not None
    track = Track(id="T", name="T", country="T", total_laps=1, base_lap_time=90,
                  tire_stress=1)
    car = Car(
        team_id="T", team_name="T", reliability=0.8,
        **{f"{component}_reliability": 0.8 for component in (
            "engine", "gearbox", "brakes", "electrical", "cooling",
        )},
    )

    def check(weather):
        manager = EventManager(mechanical_rng_factory=factory)
        driver = Driver(id="risk", name="risk", team_id="T")
        return manager._check_mechanical_failure(driver, car, track, 1, weather)

    assert check(Weather(track_temperature=10)) is None
    assert check(Weather(track_temperature=60)) is not None


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_worker_wires_factory_to_both_race_engines(monkeypatch, engine):
    calls = []
    original_factory = montecarlo.mechanical_rng_factory_for_trial

    def recording_factory(seed, policy):
        factory = original_factory(seed, policy)
        assert factory is not None

        def record(driver_id, lap):
            calls.append((driver_id, lap))
            return factory(driver_id, lap)

        return record

    monkeypatch.setattr(montecarlo, "mechanical_rng_factory_for_trial", recording_factory)
    drivers = [Driver(id=key, name=key, team_id="T") for key in ("A", "B")]
    cars = {
        "T": Car(
            team_id="T", team_name="T", reliability=1.0,
            **{f"{component}_reliability": 1.0 for component in (
                "engine", "gearbox", "brakes", "electrical", "cooling",
            )},
        ),
    }
    track = Track(id="T", name="T", country="T", total_laps=2, base_lap_time=90)
    weather = Weather()
    montecarlo._run_single_simulation((
        [driver.model_dump() for driver in drivers],
        {key: car.model_dump() for key, car in cars.items()},
        track.model_dump(), weather.model_dump(), 91, engine, None,
        "isolated_weather_mechanical_v1", None, None,
    ))
    assert calls
    assert {driver_id for driver_id, _ in calls} == {"A", "B"}
    assert all(lap in {1, 2} for _, lap in calls)
    assert len(calls) == len(set(calls))


def test_mechanical_checks_ignore_race_draws_and_driver_order():
    factory = mechanical_rng_factory_for_trial(91, "isolated_weather_mechanical_v1")
    assert factory is not None
    track = Track(id="T", name="T", country="T", total_laps=20, base_lap_time=90)
    car = Car(team_id="T", team_name="T", reliability=0.8)

    def sample(order: list[str], extra_race_draws: int) -> dict[str, str]:
        manager = EventManager(
            np.random.default_rng(5), mechanical_rng_factory=factory,
        )
        manager.rng.random(extra_race_draws)
        observed = {}
        for driver_id in order:
            driver = Driver(id=driver_id, name=driver_id, team_id="T")
            # Force the conditional component draw while retaining the real
            # component weighting and the per-driver/per-lap stream.
            manager._mechanical_failure_probability = lambda *args: 1.0
            event = manager._check_mechanical_failure(driver, car, track, 4, Weather())
            observed[driver_id] = event.description
        return observed

    assert sample(["A", "B"], 0) == sample(["B", "A"], 100)
