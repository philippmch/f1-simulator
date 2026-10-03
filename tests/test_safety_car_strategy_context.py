"""Strategy queues use frozen public observations and fail closed when unresolved."""

from copy import deepcopy

import numpy as np
import pytest
from test_safety_car_strategy_costs import setup

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.strategy_neutralization import SafetyCarBranch, StrategySafetyCarSnapshot


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("queue_delay", [0., 5.])
def test_standard_snapshot_matches_mean_queue_after_frozen_committed_merge(
    index, committed, queue_delay,
):
    simulator = RaceSimulator(np.random.default_rng(11))
    simulator.event_manager.safety_car_active = True
    track = Track(id="s", name="SC", country="test", total_laps=6,
                  base_lap_time=90., pit_lane_delta=8.)
    weather = Weather()
    states = [DriverRaceState(Driver(id=key, name=key, team_id="T"),
                              Car(team_id="T", team_name="T"), position,
                              total_time=float(position * 3), tire_laps=position * 4)
              for position, key in enumerate("ABC", 1)]
    own = states[index]
    rival = next(row for row in states if row is not own)
    losses = {rival.driver.id: 12.} if committed else {}
    if committed:
        rival.dry_pit_proposal = (5, TireCompound.SOFT)
    before = deepcopy((states, track, weather, simulator.rng.bit_generator.state))
    context = simulator._standard_safety_car_snapshot(own, states, states, track, weather,
                                                     5, queue_delay, losses)
    assert context is not None
    assert before == (states, track, weather, simulator.rng.bit_generator.state)
    physics = LapSimulator()
    for stopped in (False, True):
        rows = deepcopy(states)
        candidate = rows[index]
        pitting = []
        for row in rows:
            if row.driver.id in losses:
                row.total_time += losses[row.driver.id]
                row.current_tire = TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(deep=True)
                row.tire_laps = 0
                pitting.append(row)
        if stopped:
            candidate.total_time += 8 * .55 + expected_stationary_time(candidate.car) + queue_delay
            candidate.current_tire = TIRE_COMPOUNDS[TireCompound.HARD].model_copy(deep=True)
            candidate.tire_laps = 0
            pitting.append(candidate)
        simulator._handle_pit_batch_position_changes(pitting, rows)
        free = {}
        for row in rows:
            row.driver.current_tire_laps = row.tire_laps
            free[row.driver.id] = physics.calculate_lap_time(
                row.driver, row.car, track, row.current_tire, weather, 5, 6,
                active_aero_enabled=False, sample_variation=False,
                gap_to_car_ahead=simulator._get_gap_to_car_ahead(row, rows),
            )
        branch = context.stopped if stopped else context.retained
        assert branch.traffic_gap == simulator._get_gap_to_car_ahead(candidate, rows)
        expected = simulator._safety_car_lap_times(free, rows, 1.4)[own.driver.id]
        assert branch.running_time(free[own.driver.id], 1.4) == pytest.approx(expected)


def test_unresolved_committed_replacement_does_not_guess_queue_pace():
    simulator = RaceSimulator()
    simulator.event_manager.safety_car_active = True
    track = Track(id="s", name="SC", country="test", total_laps=6, base_lap_time=90.)
    states = [DriverRaceState(Driver(id=key, name=key, team_id=key),
                              Car(team_id=key, team_name=key), position)
              for position, key in enumerate("AB", 1)]
    assert simulator._standard_safety_car_snapshot(states[1], states, states, track, Weather(),
                                                  3, 0., {"A": 8.}) is None


def test_failed_inventory_preparation_removes_retired_rival_from_queue():
    simulator = RaceSimulator()
    simulator.event_manager.safety_car_active = True
    track = Track(id="s", name="SC", country="test", total_laps=6, base_lap_time=90.)
    frozen = [DriverRaceState(Driver(id=key, name=key, team_id=key),
                              Car(team_id=key, team_name=key), position)
              for position, key in enumerate("ABC", 1)]
    states = deepcopy(frozen)
    states[1].status = DriverStatus.DNF
    context = simulator._standard_safety_car_snapshot(states[2], frozen, states, track, Weather(),
                                                      3, 0., {"B": 8.})
    assert context is not None
    assert context.retained.traffic_gap == 0.
    assert [state.position for state in frozen] == [1, 2, 3]


@pytest.mark.parametrize("reason", ["vsc", "red", "crossing", "service", "missing_pace"])
def test_chronological_unresolved_queue_uses_uniform_fallback(reason):
    own, track, weather, _, _, _ = setup("chronological", 0., {})
    # Use a minimal new engine snapshot; no actual future service sample is
    # needed to decide that this queue cannot be held through expected entry.
    from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap

    simulator = RaceSimulator()
    simulator.event_manager.safety_car_active = True
    engine = ChronologicalRace(simulator)
    engine.track, engine.weather = track, weather
    ahead = deepcopy(own)
    ahead.driver.id = "A"
    ahead.total_time = 0.
    engine.states, engine.order = {"A": ahead, "B": own}, list("AB")
    engine.running_paces = {key: 90. for key in "AB"}
    pending = _PendingLap(5, 0., 126., weather, True, ahead.current_tire, 10, 90.,
                          running_start=0.)
    engine.pending = {"A": pending}
    if reason == "vsc":
        simulator.event_manager.safety_car_active = False
        simulator.event_manager.vsc_active = True
    elif reason == "red":
        simulator.event_manager.red_flag_active = True
    elif reason == "crossing":
        pending.ready = own.total_time + 1.
    elif reason == "service":
        pending.on_track = False
    else:
        engine.running_paces.pop("B")
    before = deepcopy((engine.states, engine.pending, simulator.rng.bit_generator.state))
    assert engine._safety_car_strategy_snapshot(own, own.total_time, 0.) is None
    assert before == (engine.states, engine.pending, simulator.rng.bit_generator.state)


@pytest.mark.parametrize("values", [dict(queue_pace=0.), dict(queue_pace=float("nan")),
                                    dict(queue_pace=126., traffic_gap=-1.),
                                    dict(queue_pace=126., ahead_progress=1.1),
                                    dict(queue_pace=126., ahead_crossing=90., ahead_progress=.1)])
def test_invalid_queue_observations_are_rejected(values):
    with pytest.raises(ValueError):
        SafetyCarBranch(**values)


def test_paid_commitment_retains_stop_queue_and_frozen_fields():
    context = StrategySafetyCarSnapshot(SafetyCarBranch(126., ahead_crossing=125.),
                                       SafetyCarBranch(126., ahead_crossing=120.))
    committed = context.for_paid_fit()
    assert committed.retained is context.stopped and committed.stopped is context.stopped
    assert context.retained.ahead_crossing == 125.


def test_standard_mean_forecast_does_not_call_the_live_execution_sampler(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(31))
    simulator.event_manager.safety_car_active = True
    track = Track(id="s", name="SC", country="test", total_laps=6, base_lap_time=90.)
    states = [DriverRaceState(Driver(id=key, name=key, team_id=key),
                              Car(team_id=key, team_name=key), position)
              for position, key in enumerate("AB", 1)]
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *a, **kw: pytest.fail("live execution sampler was called"))
    before = deepcopy((states, simulator.rng.bit_generator.state))
    assert simulator._standard_safety_car_snapshot(states[1], states, states, track, Weather(),
                                                  3, 0., {}) is not None
    assert before == (states, simulator.rng.bit_generator.state)
