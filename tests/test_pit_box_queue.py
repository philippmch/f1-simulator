"""A constructor services one car at a time, without strategy RNG foresight."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.pit_strategy import expected_stationary_time, plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def track(laps=60):
    return Track(id="test", name="Test", country="Test", total_laps=laps,
                 base_lap_time=90, pit_lane_delta=22, tire_stress=0.7)


def state(name, position, arrival=100, team="shared"):
    return DriverRaceState(
        Driver(id=name, name=name, team_id=team),
        Car(team_id=team, team_name=team), position,
        total_time=arrival,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
    )


@pytest.mark.parametrize("gap,shared,service,delay", [
    (1, True, 3, 2), (1, False, 3, 0), (4, True, 3, 0), (1, True, 8, 7),
])
def test_full_race_shared_box_adds_service_overlap_once(monkeypatch, gap, shared, service, delay):
    sim = RaceSimulator(np.random.default_rng(3))
    a, b = state("A", 1), state("B", 2, team="shared" if shared else "other")
    monkeypatch.setattr(sim, "_should_pit", lambda *args, **kw: args[3] == 2)
    monkeypatch.setattr(sim.event_manager, "process_lap", lambda **kw: [])
    monkeypatch.setattr(sim, "_process_overtakes", lambda *args, **kw: 0)
    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", lambda **kw:
                        90 + (gap if kw["lap_number"] == 1 and kw["driver"].id == "B" else 0))
    # A slow first service blocks the box for its actual sampled duration.
    def run():
        services = iter([service, 3])
        monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time",
                            lambda car: next(services))
        return sim.simulate_race([a.driver, b.driver], {a.car.team_id: a.car, b.car.team_id: b.car},
                                 track(2), Weather(change_probability=0), ["A", "B"])
    for _ in range(2):  # Reusing a simulator cannot preserve the preceding race's box release.
        results = {r.driver_id: r for r in run()}
        assert results["A"].total_time == pytest.approx(180 + 22 + service)
        assert results["B"].total_time == pytest.approx(180 + gap + 22 + 3 + delay)
        assert all(r.pit_stops == 1 for r in results.values())


@pytest.mark.parametrize("factor,mode", [
    (1, None), (0.55, "safety_car_active"), (0.75, "vsc_active"),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_batch_uses_frozen_arrivals_and_discounts_only_lane(monkeypatch, factor, mode, reverse):
    sim = RaceSimulator(np.random.default_rng(4))
    if mode:
        setattr(sim.event_manager, mode, True)
    # Arrival clocks, not input order or position, determine service order.
    a, b = state("A", 2, 100), state("B", 1, 101)
    states = [b, a] if reverse else [a, b]
    frozen = [replace(s) for s in states]
    a.total_time = 900  # The live clocks must not replace frozen arrival times.
    monkeypatch.setattr(sim, "_should_pit", lambda *args, **kw: True)
    services = iter([8, 3])
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", lambda car: next(services))
    assert sim._process_pit_stops(states, frozen, track(), Weather(), 10) == [a, b]
    assert a.total_time == pytest.approx(900 + 22 * factor + 8)
    assert b.total_time == pytest.approx(101 + 22 * factor + 3 + 7)


def test_decisions_use_expected_reservations_before_any_service_sample(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(4))
    a, b, c = state("A", 1), state("B", 2, 101), state("C", 3, 101, "other")
    observed = []
    samples = []
    def decide(s, *args, additional_current_stop_cost, **kw):
        assert not samples
        observed.append((s.driver.id, additional_current_stop_cost))
        return True
    def service(car):
        samples.append(car.team_id)
        return 10
    monkeypatch.setattr(sim, "_should_pit", decide)
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", service)
    rng_before = deepcopy(sim.rng.bit_generator.state)
    sim._process_pit_stops([c, b, a], [replace(s) for s in [a, b, c]], track(), Weather(), 10)
    assert [name for name, _ in observed] == ["A", "B", "C"]
    assert [delay for _, delay in observed] == pytest.approx(
        [0, expected_stationary_time(a.car) - 1, 0])
    assert sim.rng.bit_generator.state == rng_before
    assert b.total_time == pytest.approx(101 + 22 + 10 + 9)


@pytest.mark.parametrize("forced", [False, True])
def test_weather_and_forced_stops_still_queue(monkeypatch, forced):
    sim = RaceSimulator(np.random.default_rng(7))
    a, b = state("A", 1), state("B", 2, 101)
    for s in [a, b]:
        s.force_pit_next_lap = forced
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    weather = Weather(track_wetness=0.8, rain_intensity=0.8)
    assert len(sim._process_pit_stops([a, b], [replace(a), replace(b)], track(), weather, 2)) == 2
    assert b.total_time == pytest.approx(101 + 22 + 3 + 2)
    assert all(s.current_tire.compound == TireCompound.WET for s in [a, b])


def test_mandatory_distinct_compound_stop_cannot_be_deferred_by_queue(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(7))
    a, b = state("A", 1), state("B", 2, 101)
    a.tire_laps = b.tire_laps = 1  # Both have actually used their first compound.
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    pitting = sim._process_pit_stops([a, b], [replace(a), replace(b)], track(), Weather(), 60)
    assert len(pitting) == 2
    assert b.total_time == pytest.approx(128)
    assert all(s.current_tire.compound != TireCompound.MEDIUM for s in [a, b])


def test_current_queue_cost_is_uncached_and_can_change_dry_decision():
    s = state("A", 1)
    s.tire_laps = 25
    args = (s.driver, s.car, track(), s.current_tire, s.tire_laps, 10, 2,
            {TireCompound.MEDIUM})
    plain = plan_dry_stop(*args)
    # Delaying now must never increase a cached future stop's cost.
    delayed = plan_dry_stop(*args, additional_current_stop_cost=3)
    assert delayed.pit_now_cost == pytest.approx(plain.pit_now_cost + 3)
    assert delayed.wait_cost == plain.wait_cost
    assert delayed.compound == plain.compound
    assert plan_dry_stop(*args) == plain
    assert plain.should_pit()
    assert not delayed.should_pit()


def test_direct_execution_and_new_lap_have_no_stale_queue(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(4))
    monkeypatch.setattr(sim, "_should_pit", lambda *args, **kw: True)
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    for lap in [10, 11]:
        a, b = state("A", 1), state("B", 2)
        sim._process_pit_stops([b, a], [replace(a), replace(b)], track(), Weather(), lap)
        assert a.total_time == 125
        assert b.total_time == 128
    assert sim._execute_pit_stop(state("A", 1), track(), Weather(), 12) == 25


@pytest.mark.parametrize("sampled_service", [2, 10])
@pytest.mark.parametrize("shared", [False, True])
def test_expected_teammate_queue_flips_real_dry_planner(monkeypatch, sampled_service, shared):
    sim = RaceSimulator(np.random.default_rng(6))
    a = state("A", 1)
    b = state("B", 2, 101, "shared" if shared else "other")
    a.force_pit_next_lap = True
    b.tire_laps = 25
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", lambda car: sampled_service)
    pitting = sim._process_pit_stops([b, a], [replace(a), replace(b)], track(), Weather(), 51)
    assert [s.driver.id for s in pitting] == (["A"] if shared else ["A", "B"])
    assert b.pit_stops == (0 if shared else 1)


def test_red_flag_free_changes_do_not_sample_or_charge_shared_box(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(4))
    a, b = state("A", 1), state("B", 2)
    # Isolate service from the separate restart gap compression.
    monkeypatch.setattr(sim.event_manager, "bunch_field", lambda states: None)
    def unexpected_service(car):
        pytest.fail("A suspension tyre change must not sample a paid service")
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", unexpected_service)
    sim._handle_red_flag_stop([a, b], Weather(track_wetness=0.8, rain_intensity=0.8), track(), 10)
    assert [a.total_time, b.total_time] == [100, 100]
    assert [a.pit_stops, b.pit_stops] == [0, 0]
    assert all(s.current_tire.compound == TireCompound.WET for s in [a, b])
