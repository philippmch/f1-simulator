"""Driver/purpose streams isolate extra calls without freezing race physics."""

import hashlib
from copy import deepcopy

import numpy as np
import pytest

import f1sim.analysis.montecarlo as montecarlo
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.qualifying import QualifyingSimulator
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.randomness import (
    DEFAULT_RNG_POLICY,
    driver_rng_factory_for_trial,
    mechanical_rng_factory_for_trial,
    weather_rng_for_trial,
)

POLICY = "isolated_race_v1"


@pytest.mark.parametrize("purpose,number", [
    ("qualifying_lap", 1), ("race_lap", 2), ("pit_service", 3),
    ("opening_choice", 4), ("strategy", 5), ("replacement_choice", 6),
    ("overtake_attempt", 7), ("collision_loss", 8), ("race_events", 9),
])
def test_versioned_driver_stream_layout(purpose, number):
    driver_id = "" if purpose == "race_events" else "Å-7"
    digest = hashlib.sha256(driver_id.encode("utf-8")).digest()
    words = tuple(int.from_bytes(digest[offset:offset + 4], "little")
                  for offset in range(0, 32, 4))
    expected = np.random.default_rng(np.random.SeedSequence(
        37, spawn_key=(0x44525652, number, *words),
    ))
    factory = driver_rng_factory_for_trial(37, POLICY)
    stream = factory(driver_id, purpose)
    assert np.array_equal(stream.random(8), expected.random(8))
    assert factory(driver_id, purpose) is stream
    assert np.array_equal(stream.random(8), expected.random(8))


@pytest.mark.parametrize("policy", [
    "shared_v1", "isolated_weather_v1", "isolated_weather_mechanical_v1",
])
def test_older_policies_keep_their_driver_draws_and_default(policy):
    assert DEFAULT_RNG_POLICY == "isolated_weather_v1"
    assert driver_rng_factory_for_trial(37, policy) is None


@pytest.mark.parametrize("driver_id,purpose", [
    (None, "race_lap"), (True, "race_lap"), ("", "race_lap"),
    ("A", None), ("A", []), ("A", "unknown"),
])
def test_invalid_stream_keys_are_rejected(driver_id, purpose):
    factory = driver_rng_factory_for_trial(37, POLICY)
    with pytest.raises(ValueError):
        factory(driver_id, purpose)


def test_other_drivers_purposes_and_creation_order_do_not_advance_a_stream():
    first = driver_rng_factory_for_trial(91, POLICY)
    second = driver_rng_factory_for_trial(91, POLICY)
    untouched = second("B", "race_lap").random(12)
    first("A", "race_lap").random(100)
    first("B", "pit_service").random(100)
    first("B", "qualifying_lap").random(100)
    first("", "race_events").random(100)
    assert np.array_equal(first("B", "race_lap").random(12), untouched)
    # Further calls for that SAME purpose do advance it; this is not a lap key.
    assert not np.array_equal(first("B", "race_lap").random(12), untouched)
    assert not np.array_equal(
        driver_rng_factory_for_trial(92, POLICY)("B", "race_lap").random(12), untouched,
    )


def test_qualifying_preserves_driver_draws_when_roster_order_changes():
    drivers = [Driver(id=f"D{i}", name=f"D{i}", team_id="T") for i in range(22)]
    cars = {"T": Car(team_id="T", team_name="T")}
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)

    def run(roster):
        simulator = QualifyingSimulator(
            np.random.default_rng(91), driver_rng_factory=driver_rng_factory_for_trial(91, POLICY),
        )
        return simulator.simulate_qualifying(roster, cars, track, Weather())

    assert run(deepcopy(drivers)) == run(deepcopy(drivers[::-1]))


def state(driver_id):
    return DriverRaceState(
        Driver(id=driver_id, name=driver_id, team_id=driver_id),
        Car(team_id=driver_id, team_name=driver_id), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
    )


def test_forecasts_never_request_driver_or_service_draws():
    requests = []

    def unexpected_draw(driver_id, purpose):
        requests.append((driver_id, purpose))
        if purpose == "race_events":
            return np.random.default_rng(91)
        pytest.fail(f"Forecast requested {driver_id}/{purpose}")

    simulator = RaceSimulator(driver_rng_factory=unexpected_draw)
    driver = state("A")
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    lap = simulator.lap_simulator
    lap.calculate_lap_time(driver.driver, driver.car, track, driver.current_tire,
                           Weather(), 1, 8, sample_variation=False)
    lap.calculate_qualifying_lap(driver.driver, driver.car, track, driver.current_tire,
                                 Weather(), sample_variation=False)
    driver.pit_plan_target = TireCompound.HARD
    simulator._execute_pit_stop(driver, track, Weather(), 3, sample_service=False)
    assert requests == [("", "race_events")]


@pytest.mark.parametrize("raises", [False, True])
def test_public_service_hook_receives_driver_stream_and_restores_rng(monkeypatch, raises):
    factory = driver_rng_factory_for_trial(91, POLICY)
    simulator = RaceSimulator(driver_rng_factory=factory)
    original_rng = simulator.lap_simulator.rng
    driver = state("A")

    def service(car):
        assert car is driver.car
        assert simulator.lap_simulator.rng is factory("A", "pit_service")
        if raises:
            raise RuntimeError("custom service failed")
        return 3.5

    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", service)
    if raises:
        with pytest.raises(RuntimeError, match="custom service failed"):
            simulator._sample_pit_service(driver)
    else:
        assert simulator._sample_pit_service(driver) == 3.5
    assert simulator.lap_simulator.rng is original_rng


def test_overtake_draws_follow_the_attacker_with_unchanged_probability():
    factory = driver_rng_factory_for_trial(91, POLICY)
    model = OvertakingModel(np.random.default_rng(4), driver_rng_factory=factory)
    attacker, defender = state("A"), state("B")
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    probability = model._calculate_probability(
        attacker.driver, attacker.car, defender.driver, defender.car, track, 0.5, False, False,
    )
    incident = model._incident_probability(0.5, track.overtake_difficulty)
    rolls = driver_rng_factory_for_trial(91, POLICY)("A", "overtake_attempt").random(25)
    # Ordinary race calls and other attackers cannot shift this attacker's rolls.
    model.rng.random(100)
    factory("B", "overtake_attempt").random(100)
    actual = [model.attempt_overtake(
        attacker.driver, attacker.car, defender.driver, defender.car, track, 0.5,
    ) for _ in rolls]
    expected = [(True, False) if roll < probability
                else (False, bool(roll < probability + incident)) for roll in rolls]
    assert actual == expected


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("policy", ["isolated_weather_v1", POLICY])
def test_extra_native_stops_do_not_shift_unrelated_driver_lap_noise(monkeypatch, engine, policy):
    """A controlled, separated field removes physical interactions as a cause."""
    drivers = [state("B").driver, state("A").driver]
    cars = {"B": Car(team_id="B", team_name="B", base_pace=1),
            "A": Car(team_id="A", team_name="A", base_pace=0.7)}
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90,
                  safety_car_probability=0)
    weather = Weather(change_probability=0)

    def run(extra_stop):
        rng = np.random.default_rng(91)
        simulator = RaceSimulator(
            rng, weather_rng=weather_rng_for_trial(91, rng, policy),
            mechanical_rng_factory=mechanical_rng_factory_for_trial(91, policy),
            driver_rng_factory=driver_rng_factory_for_trial(91, policy),
        )
        # Hold event exposure fixed to isolate the effect of service draw calls.
        monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **k: [])
        monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
        monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **k: None)
        monkeypatch.setattr(simulator, "_should_pit", lambda driver, _states, _track, lap, *a, **kw:
                            bool(extra_stop and driver.driver.id == "A" and lap == 3))
        noise = {}
        physics = simulator.lap_simulator.calculate_lap_time

        def observed_lap(*args, **kwargs):
            time = physics(*args, **kwargs)
            driver = args[0] if args else kwargs["driver"]
            lap_number = args[5] if len(args) > 5 else kwargs["lap_number"]
            if kwargs.get("sample_variation", True) and driver.id == "B":
                mean = physics(*args, **{**kwargs, "sample_variation": False})
                noise[lap_number] = time - mean
            return time

        monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", observed_lap)
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        results = execute(deepcopy(drivers), deepcopy(cars), track, weather, ["B", "A"],
                          starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM})
        return noise, {row.driver_id: row for row in results}

    original, before = run(False)
    perturbed, after = run(True)
    assert after["A"].pit_laps == [3]
    assert before["A"].pit_laps == []
    assert set(original) == set(perturbed) == set(range(1, 9))
    if policy == POLICY:
        assert perturbed == pytest.approx(original, abs=1e-12)
        assert after["B"].total_time == before["B"].total_time
    else:
        assert any(abs(perturbed[lap] - original[lap]) > 1e-8 for lap in range(4, 9))


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_worker_uses_one_factory_for_qualifying_race_and_service(monkeypatch, engine):
    requests = []
    original = montecarlo.driver_rng_factory_for_trial

    def recording_factory(seed, policy):
        factory = original(seed, policy)

        def record(driver_id, purpose):
            requests.append((driver_id, purpose))
            return factory(driver_id, purpose)

        return record

    monkeypatch.setattr(montecarlo, "driver_rng_factory_for_trial", recording_factory)
    drivers = [state(key).driver for key in ("A", "B")]
    cars = {driver.id: state(driver.id).car for driver in drivers}
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    montecarlo.MonteCarloRunner(
        drivers, cars, track, Weather(), seed=91, rng_policy=POLICY, race_engine=engine,
        pit_plans={"A": [{"lap": 3, "compound": "hard"}], "B": []},
    ).run(1, parallel=False)
    for driver_id in ("A", "B"):
        assert (driver_id, "qualifying_lap") in requests
        assert (driver_id, "race_lap") in requests
    assert ("A", "pit_service") in requests
    assert ("", "race_events") in requests
