"""Standard pit decisions price earlier committed stops in the same batch."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator


def track(*, laps=8, pit_lane_delta=20):
    return Track(
        id="test",
        name="Test",
        country="Test",
        total_laps=laps,
        base_lap_time=90,
        pit_lane_delta=pit_lane_delta,
        tire_stress=0.7,
    )


def state(name, position, *, clock=0, team=None, tire_laps=0):
    team = name if team is None else team
    driver = Driver(id=name, name=name, team_id=team)
    car = Car(
        team_id=team,
        team_name=team,
        # This model's expected clipped service is exactly 3 seconds. It
        # makes expected lane/service clocks easy to assert without changing
        # the implementation's service estimator.
        pit_stop_avg=2.75,
        pit_stop_std=0.1,
    )
    return DriverRaceState(
        driver=driver,
        car=car,
        position=position,
        total_time=clock,
        tire_laps=tire_laps,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
    )


def capture_process(monkeypatch, *, stopping=("A", "B"), services=(3, 3), states=None,
                    frozen=None, sim=None, race_track=None, lap=2):
    sim = RaceSimulator(np.random.default_rng(12)) if sim is None else sim
    states = states if states is not None else [state(name, index + 1)
                                                for index, name in enumerate("AB")]
    frozen = [replace(item) for item in states] if frozen is None else frozen
    race_track = track() if race_track is None else race_track
    snapshots = {}

    def decide(item, *args, **kwargs):
        snapshots[item.driver.id] = kwargs["traffic_snapshot"]
        return item.driver.id in stopping

    monkeypatch.setattr(sim, "_should_pit", decide)
    samples = iter(services)
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time",
                        lambda car: next(samples))
    pitting = sim._process_pit_stops(states, frozen, race_track, Weather(), lap)
    return sim, states, snapshots, pitting


def test_later_candidate_prices_committed_rival_and_candidate_merge(monkeypatch):
    sim, states, snapshots, pitting = capture_process(monkeypatch)
    assert [item.driver.id for item in pitting] == ["A", "B"]
    # A is already expected to rejoin at 23. B's stay branch therefore moves
    # ahead of A, while its own pit branch retains A/B's original tie order.
    assert snapshots["A"].current_traffic_gaps == pytest.approx((None, 23))
    assert snapshots["B"].current_traffic_gaps == pytest.approx((None, 0))
    assert snapshots["B"].rejoin_traffic_cost == pytest.approx(0.5)
    assert [item.total_time for item in states] == pytest.approx([23, 23])
    assert sim._pit_rejoin_traffic_gaps(states[0], states, track()) == (
        None,
        pytest.approx(23),
    )


def test_same_team_expected_queue_reaches_snapshot_and_execution(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(13))
    a = state("A", 1, clock=100, team="team")
    b = state("B", 2, clock=101, team="team")
    frozen = [replace(a), replace(b)]
    snapshots = {}

    def decide(item, *args, **kwargs):
        snapshots[item.driver.id] = kwargs["traffic_snapshot"]
        return True

    monkeypatch.setattr(sim, "_should_pit", decide)
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    pitting = sim._process_pit_stops([a, b], frozen, track(), Weather(), 2)

    assert [item.driver.id for item in pitting] == ["A", "B"]
    assert snapshots["B"].current_traffic_gaps == pytest.approx((None, 3))
    assert snapshots["B"].gap_ahead == pytest.approx(1)
    assert [item.total_time for item in (a, b)] == pytest.approx([123, 126])
    assert [item.pit_stop_details[0]["queue_time"] for item in (a, b)] == [0, 2]


def test_forced_inventory_selection_gets_snapshot_gaps_and_expected_queue(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(14))
    a = state("A", 1, clock=100, team="team")
    b = state("B", 2, clock=101, team="team")
    b.tire_inventory = object()
    b.force_pit_next_lap = True
    frozen = [replace(a), replace(b)]
    snapshots = {}
    prepared = {}

    def decide(item, *args, **kwargs):
        snapshots[item.driver.id] = kwargs["traffic_snapshot"]
        return item.driver.id == "A"

    def prepare(item, *args, **kwargs):
        prepared[item.driver.id] = kwargs
        return False

    monkeypatch.setattr(sim, "_should_pit", decide)
    monkeypatch.setattr(sim, "_prepare_inventory_pit", prepare)
    monkeypatch.setattr(sim, "_execute_pit_stop", lambda *args, **kwargs: 23)
    pitting = sim._process_pit_stops([a, b], frozen, track(), Weather(), 10)

    assert [item.driver.id for item in pitting] == ["A"]
    assert prepared["B"]["current_traffic_gaps"] == pytest.approx((None, 3))
    assert prepared["B"]["additional_current_stop_cost"] == pytest.approx(2)
    # Forced stops bypass the strategy callback, but preparation receives the
    # same projected gaps and queue cost that the callback would have seen.


@pytest.mark.parametrize("control", ["safety_car_active", "vsc_active", "red_flag_active"])
def test_neutralized_standard_snapshot_disables_traffic_correction(monkeypatch, control):
    sim = RaceSimulator(np.random.default_rng(15))
    setattr(sim.event_manager, control, True)
    a, b = state("A", 1), state("B", 2)
    snapshot = sim._standard_pit_traffic_snapshot(
        b,
        [replace(a), replace(b)],
        [a, b],
        track(),
        0,
        {"A": 23},
    )
    assert snapshot.gap_ahead == pytest.approx(0)
    assert snapshot.current_traffic_gaps is None
    assert snapshot.rejoin_traffic_cost == 0


def test_inventory_refusal_removes_row_and_compacts_virtual_positions(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(16))
    a = state("A", 1, clock=100)
    b = state("B", 2, clock=101)
    c = state("C", 3, clock=102)
    b.tire_inventory = object()
    frozen = [replace(a), replace(b), replace(c)]
    snapshots = {}

    def decide(item, *args, **kwargs):
        snapshots[item.driver.id] = kwargs["traffic_snapshot"]
        return item.driver.id in {"B", "C"}

    def prepare(item, *args, **kwargs):
        item.status = DriverStatus.DNF
        return False

    monkeypatch.setattr(sim, "_should_pit", decide)
    monkeypatch.setattr(sim, "_prepare_inventory_pit", prepare)
    monkeypatch.setattr(sim, "_execute_pit_stop", lambda *args, **kwargs: 23)
    pitting = sim._process_pit_stops([a, b, c], frozen, track(), Weather(), 10)

    assert [item.driver.id for item in pitting] == ["C"]
    assert snapshots["C"].gap_ahead == pytest.approx(2)
    assert snapshots["C"].current_traffic_gaps == pytest.approx((2, 25))
    assert snapshots["C"].rejoin_traffic_cost == pytest.approx(0)
    assert b.status == DriverStatus.DNF
    assert a.total_time == frozen[0].total_time
    assert b.total_time == frozen[1].total_time


def test_snapshot_is_pure_and_does_not_consume_rng():
    sim = RaceSimulator(np.random.default_rng(17))
    a, b = state("A", 1), state("B", 2)
    frozen = [replace(a), replace(b)]
    before = deepcopy((a.total_time, a.position, b.total_time, b.position,
                       frozen[0].position, frozen[1].position,
                       sim.rng.bit_generator.state))
    first = sim._standard_pit_traffic_snapshot(
        b, frozen, [a, b], track(), 0, {"A": 23}
    )
    second = sim._standard_pit_traffic_snapshot(
        b, frozen, [a, b], track(), 0, {"A": 23}
    )
    after = (a.total_time, a.position, b.total_time, b.position,
             frozen[0].position, frozen[1].position,
             sim.rng.bit_generator.state)
    assert first == second
    assert after == before


@pytest.mark.parametrize("sampled_service", [3, 80])
def test_near_tie_native_dry_optimizer_changes_when_committed_rival_is_known(
    monkeypatch, sampled_service,
):
    sim = RaceSimulator(np.random.default_rng(18))
    a = state("A", 1, tire_laps=1)
    b = state("B", 2, tire_laps=1)
    a.force_pit_next_lap = True
    frozen = [replace(a), replace(b)]
    observed = {}
    original = sim._should_pit
    # The old all-rivals-stay projection selects the extra stop at this same
    # state. Keep its mutable proposal isolated from the actual batch.
    assert original(deepcopy(b), deepcopy(frozen), track(), 2, False, weather=Weather())

    def decide(item, *args, **kwargs):
        if item.driver.id == "B":
            observed["B"] = kwargs["traffic_snapshot"]
            return original(item, *args, **kwargs)
        return False

    monkeypatch.setattr(sim, "_should_pit", decide)
    def service(car):
        assert "B" in observed  # No sampled service can inform B's decision.
        return sampled_service

    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time", service)
    pitting = sim._process_pit_stops([a, b], frozen, track(), Weather(), 2)

    assert [item.driver.id for item in pitting] == ["A"]
    assert observed["B"].current_traffic_gaps == pytest.approx((None, 0))


def test_standard_snapshot_does_not_leak_between_process_calls(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(19))
    first = [state("A", 1), state("B", 2)]
    second = [state("A", 1), state("B", 2)]
    seen = []

    def decide(item, *args, **kwargs):
        seen.append((item.driver.id, kwargs["traffic_snapshot"].current_traffic_gaps))
        return item.driver.id == "A"

    monkeypatch.setattr(sim, "_should_pit", decide)
    monkeypatch.setattr(sim, "_execute_pit_stop", lambda *args, **kwargs: 23)
    sim._process_pit_stops(first, [replace(item) for item in first], track(), Weather(), 10)
    sim._process_pit_stops(second, [replace(item) for item in second], track(), Weather(), 11)
    assert seen == [
        ("A", (None, 23)),
        ("B", (None, 0)),
        ("A", (None, 23)),
        ("B", (None, 0)),
    ]
