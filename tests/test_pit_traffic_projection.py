"""The compact native pit projection preserves the former state-copy results."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot


def _state(name, position, total_time, status=DriverStatus.RACING):
    return DriverRaceState(
        driver=Driver(id=name, name=name, team_id=f"team-{name}"),
        car=Car(team_id=f"team-{name}", team_name=f"Team {name}"),
        position=position,
        total_time=total_time,
        status=status,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
        pit_laps=[4],
        pit_stop_details=[{"lap": 4, "nested": {"service": 3.0}}],
    )


def _track():
    return Track(
        id="test",
        name="Test",
        country="Test",
        total_laps=20,
        base_lap_time=90,
        pit_lane_delta=20,
    )


def _frozen_state_copy_oracle(
    sim, state, lap_start_states, states, track, queue_delay, committed_losses,
):
    """Frozen pre-optimization implementation, retained as an independent oracle."""
    active_ids = {
        candidate.driver.id
        for candidate in states
        if candidate.status == DriverStatus.RACING
    }
    frozen = {
        candidate.driver.id: candidate
        for candidate in lap_start_states
        if candidate.status == DriverStatus.RACING
        and candidate.driver.id in active_ids
    }
    if state.driver.id not in frozen:
        return StrategyTrafficSnapshot(None, None, 0.0, None)

    base_rows = [replace(candidate) for candidate in frozen.values()]
    if sorted(candidate.position for candidate in base_rows) != list(
        range(1, len(base_rows) + 1)
    ):
        for position, candidate in enumerate(
            sorted(base_rows, key=lambda row: row.position), 1
        ):
            candidate.position = position

    observed_rows = [replace(candidate) for candidate in base_rows]
    observed_by_id = {candidate.driver.id: candidate for candidate in observed_rows}
    observed = observed_by_id[state.driver.id]
    gap_ahead = sim._get_gap_to_car_ahead(observed, observed_rows)
    gap_behind = sim._get_gap_to_car_behind(observed, observed_rows)

    if not sim.event_manager.is_active_aero_allowed():
        return StrategyTrafficSnapshot(gap_ahead, gap_behind, 0.0, None)

    def merged_gap(candidate_loss):
        projected = [replace(row) for row in base_rows]
        pitting = []
        for row in projected:
            loss = committed_losses.get(row.driver.id)
            if loss is None and row.driver.id == state.driver.id:
                loss = candidate_loss
            if loss is None:
                continue
            row.total_time += loss
            pitting.append(row)
        sim._handle_pit_batch_position_changes(pitting, projected)
        own = next(row for row in projected if row.driver.id == state.driver.id)
        return sim._get_gap_to_car_ahead(own, projected)

    expected_stay_gap = merged_gap(None)
    expected_pit_gap = merged_gap(
        track.pit_lane_delta * sim._pit_lane_factor()
        + expected_stationary_time(state.car)
        + queue_delay
    )
    traffic = sim.lap_simulator.traffic_pace_contribution
    rejoin_cost = traffic(expected_pit_gap) - traffic(expected_stay_gap)
    return StrategyTrafficSnapshot(
        gap_ahead,
        gap_behind,
        rejoin_cost,
        (expected_stay_gap, expected_pit_gap),
    )


def _snapshot(sim, candidate, frozen, current, queue_delay, losses):
    return sim._standard_pit_traffic_snapshot(
        candidate, frozen, current, _track(), queue_delay, losses,
    )


@pytest.mark.parametrize(
    ("specs", "retired", "candidate", "queue_delay", "losses", "neutral"),
    [
        # A committed pitter can move behind its staying rival. Both branches
        # still start from the original order, and a zero loss is committed.
        ([('A', 1, 100), ('B', 2, 101), ('C', 3, 102)], (), "B", 2,
         {"A": 23, "C": 0}, None),
        # A removed inventory car leaves a position hole in the frozen field;
        # active cars compact in stable old-position order before gaps resolve.
        ([('A', 1, 100), ('B', 2, 101), ('C', 4, 103), ('D', 5, 104)],
         ("B",), "C", 0, {"A": 20}, None),
        # Equal elapsed clocks use old physical position for the merge tie.
        ([('A', 1, 100), ('B', 2, 105), ('C', 3, 105), ('D', 4, 110)],
         (), "C", 7, {"A": 5, "B": 0}, None),
        # Neutralized traffic reports observed physical gaps but suppresses
        # green-running stay/rejoin projections.
        ([('A', 1, 100), ('B', 2, 100), ('C', 3, 103)], (), "B", 3,
         {"A": 25}, "safety_car_active"),
    ],
)
def test_compact_projection_matches_frozen_state_copy_oracle(
    specs, retired, candidate, queue_delay, losses, neutral,
):
    sim = RaceSimulator(np.random.default_rng(331))
    if neutral:
        setattr(sim.event_manager, neutral, True)
    frozen = [_state(*spec) for spec in specs]
    current = deepcopy(frozen)
    for item in current:
        if item.driver.id in retired:
            item.status = DriverStatus.DNF
    before = deepcopy((frozen, current, losses, sim.rng.bit_generator.state))
    candidate_state = next(item for item in current if item.driver.id == candidate)

    expected = _frozen_state_copy_oracle(
        sim, candidate_state, frozen, current, _track(), queue_delay, losses,
    )
    actual = _snapshot(sim, candidate_state, frozen, current, queue_delay, losses)

    assert actual == expected
    assert frozen == before[0]
    assert current == before[1]
    assert losses == before[2]
    assert sim.rng.bit_generator.state == before[3]


def test_native_projection_and_oracle_agree_for_each_active_candidate():
    sim = RaceSimulator(np.random.default_rng(332))
    frozen = [
        _state("A", 1, 100),
        _state("B", 3, 101),
        _state("C", 5, 105),
        _state("D", 6, 104),
    ]
    current = deepcopy(frozen)
    current[1].status = DriverStatus.DNF
    losses = {"A": 18.0, "C": 0.0}

    for candidate in (current[0], current[2], current[3]):
        expected = _frozen_state_copy_oracle(
            sim, candidate, frozen, current, _track(), 4.5, losses,
        )
        actual = _snapshot(sim, candidate, frozen, current, 4.5, losses)
        assert actual == expected


def test_overridden_gap_helper_keeps_full_driver_state_views():
    seen = []

    class InspectingSimulator(RaceSimulator):
        def _get_gap_to_car_ahead(self, state, all_states):
            assert isinstance(state, DriverRaceState)
            assert isinstance(all_states[0], DriverRaceState)
            assert hasattr(state, "pit_stop_details")
            seen.append(state.driver.id)
            return super()._get_gap_to_car_ahead(state, all_states)

    sim = InspectingSimulator(np.random.default_rng(333))
    frozen = [_state("A", 1, 100), _state("B", 2, 102)]
    current = deepcopy(frozen)
    candidate = current[1]

    expected = _frozen_state_copy_oracle(
        sim, candidate, frozen, current, _track(), 1.0, {"A": 20.0},
    )
    seen.clear()
    actual = _snapshot(sim, candidate, frozen, current, 1.0, {"A": 20.0})

    assert actual == expected
    assert seen


def test_instance_instrumentation_keeps_full_driver_state_views(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(334))
    seen = []
    original = sim._get_gap_to_car_ahead

    def inspect(state, all_states):
        assert isinstance(state, DriverRaceState)
        assert isinstance(all_states[0], DriverRaceState)
        assert hasattr(state, "pit_stop_details")
        seen.append(state.driver.id)
        return original(state, all_states)

    monkeypatch.setattr(sim, "_get_gap_to_car_ahead", inspect)
    frozen = [_state("A", 1, 100), _state("B", 2, 102)]
    current = deepcopy(frozen)
    candidate = current[1]

    expected = _frozen_state_copy_oracle(
        sim, candidate, frozen, current, _track(), 1.0, {"A": 20.0},
    )
    seen.clear()
    actual = _snapshot(sim, candidate, frozen, current, 1.0, {"A": 20.0})

    assert actual == expected
    assert seen


def test_class_instrumentation_keeps_full_driver_state_views(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(335))
    seen = []
    original = RaceSimulator._NATIVE_PIT_TRAFFIC_GAP_AHEAD

    def inspect(self, state, all_states):
        assert isinstance(state, DriverRaceState)
        assert isinstance(all_states[0], DriverRaceState)
        assert hasattr(state, "pit_stop_details")
        seen.append(state.driver.id)
        return original(self, state, all_states)

    monkeypatch.setattr(RaceSimulator, "_get_gap_to_car_ahead", inspect)
    frozen = [_state("A", 1, 100), _state("B", 2, 102)]
    current = deepcopy(frozen)
    candidate = current[1]

    expected = _frozen_state_copy_oracle(
        sim, candidate, frozen, current, _track(), 1.0, {"A": 20.0},
    )
    seen.clear()
    actual = _snapshot(sim, candidate, frozen, current, 1.0, {"A": 20.0})

    assert actual == expected
    assert seen
