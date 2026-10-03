"""Engine finish comparisons use frozen, observable rival commitments."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from test_custom_pit_replacements import snapshot
from test_leading_finish_strategy import BoundPhysics

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import chronological_race as chronological_module
from f1sim.simulation import race as race_module
from f1sim.simulation.chronological_race import (
    ChronologicalRace,
    _PendingLap,
    _PitServiceRecord,
)
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_service import expected_remaining_service
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.race_timing import RaceFinishClock, RaceFinishTimeline
from f1sim.simulation.tire_inventory import TireInventory


def inputs():
    simulator = RaceSimulator(np.random.default_rng(29),
                              tire_warmup={"hard": 3., "soft": 9.})
    track = Track(id="timed", name="timed", country="test", total_laps=90,
                  base_lap_time=100., pit_lane_delta=10.)
    states = [DriverRaceState(Driver(id=key, name=key, team_id=key),
                              Car(team_id=key, team_name=key), index,
                              current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM],
                              laps_completed=71, total_time=time, tire_laps=8,
                              tire_compound_history=["hard", "medium"])
              for index, (key, time) in enumerate((("A", 7100.), ("B", 7099.),
                                                   ("C", 7088.)), 1)]
    return simulator, states, track


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("pending_fit", [False, True])
@pytest.mark.parametrize("committed_loss", [0., 12.])
def test_standard_rival_commitment_replaces_old_fitting_cost_without_mutation(
    monkeypatch, finite, pending_fit, committed_loss,
):
    simulator, states, track = inputs()
    rival = states[1]
    rival.current_tire = TIRE_COMPOUNDS[TireCompound.HARD]
    if finite:
        inventory = TireInventory.from_sets([
            dict(id="H", compound="hard", age=1), dict(id="S", compound="soft", age=7),
        ])
        simulator._initialize_inventory(rival, inventory, inventory.sets["H"])
        rival.inventory_pit_proposal = (72, "S")
    else:
        rival.dry_pit_proposal = (72, TireCompound.SOFT)
    rival.fit_lap_pending = pending_fit
    context = simulator._standard_leading_finish_context(
        states, states[0], {key: 100. for key in "ABC"}, RaceFinishClock(90),
    )
    assert context.lockstep and not context.announced
    assert context.rivals[0].next_crossing_time == 7199. + (3. if pending_fit else 0.)
    states[2].status = DriverStatus.DNF
    before = [snapshot(state, simulator) for state in states], deepcopy(context)
    observed = []

    def compare(*args, **kwargs):
        observed.append(kwargs["leading_finish_context"])
        return SimpleNamespace(veto=False)

    monkeypatch.setattr(race_module, "evaluate_finish_protection", compare)
    assert not simulator._protect_leading_finish_distance(
        states[0], track, Weather(), 72, context,
        committed_losses={"B": committed_loss}, active_states=states,
    )
    projected = observed[0].rivals
    assert len(projected) == 1 and projected[0].identifier == "B"
    assert projected[0].next_crossing_time == 7199. + committed_loss + 9.
    assert projected[0].fitting_cost == 9.
    assert before == ([snapshot(state, simulator) for state in states], context)


@pytest.mark.parametrize("pace", [None, 0., float("inf"), float("nan")])
def test_standard_missing_or_invalid_observed_pace_does_not_create_a_clock(pace):
    simulator, states, _ = inputs()
    for identifier in "AB":
        paces = {key: 100. for key in "ABC"}
        paces[identifier] = pace
        assert simulator._standard_leading_finish_context(
            states, states[0], paces, RaceFinishClock(90),
        ) is None


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("uniform_cost", [False, True])
def test_standard_unresolved_rival_fit_needs_an_unambiguous_fitting_cost(
    monkeypatch, finite, uniform_cost,
):
    simulator, states, track = inputs()
    if finite:
        inventory = TireInventory.from_sets([
            dict(id="H", compound="hard", age=1), dict(id="S", compound="soft", age=7),
            dict(id="M", compound="medium", age=0),
        ])
        simulator._initialize_inventory(states[1], inventory, inventory.sets["H"])
    if uniform_cost:
        simulator.tire_warmup = {compound.value: 1. for compound in TireCompound}
    context = simulator._standard_leading_finish_context(
        states, states[0], {key: 100. for key in "ABC"}, RaceFinishClock(90),
    )
    calls = []

    def compare(*args, **kwargs):
        calls.append(kwargs["leading_finish_context"])
        return SimpleNamespace(veto=True)

    monkeypatch.setattr(race_module, "evaluate_finish_protection", compare)
    result = simulator._protect_leading_finish_distance(
        states[0], track, Weather(), 72, context,
        committed_losses={"B": 12.}, active_states=states,
    )
    assert result is uniform_cost
    assert len(calls) == int(uniform_cost)
    if calls:
        assert calls[0].rivals[0].next_crossing_time == 7212.


@pytest.mark.parametrize("sampled_service_end", [7102., 7300.])
@pytest.mark.parametrize("restart", [False, True])
def test_chronological_rival_clock_uses_expected_service_and_pending_fit(
    sampled_service_end, restart,
):
    simulator, states, track = inputs()
    engine = ChronologicalRace(simulator)
    engine.states = {state.driver.id: state for state in states[:2]}
    engine.track, engine.weather = track, Weather()
    engine.timeline = RaceFinishTimeline(90, engine.states)
    engine.running_paces = {"A": 100., "B": 110.}
    engine.order = ["A"]
    rival = states[1]
    rival.current_tire = TIRE_COMPOUNDS[TireCompound.HARD]
    rival.fit_lap_pending = True
    engine.pending = {"B": _PendingLap(72, 7099., sampled_service_end + 10., Weather(),
                                       False, rival.current_tire, 8, 0., on_track=False,
                                       expected_exit=7109.)}
    engine.pit_service_records = [_PitServiceRecord("B", "B", 72, 7099., 7099.,
                                                    sampled_service_end, rival.car, 10.)]
    before = (snapshot(rival, simulator), deepcopy(engine.pending),
              deepcopy(engine.pit_service_records), deepcopy(engine.weather))
    context = engine._leading_finish_context(states[0], 7100., restart=restart)
    expected_exit = 7100. if restart else 7110. + expected_remaining_service(rival.car, 1.)
    assert context.rivals[0].next_crossing_time == pytest.approx(expected_exit + 110. + 3.)
    assert not context.lockstep and not context.announced
    assert before == (snapshot(rival, simulator), engine.pending,
                      engine.pit_service_records, engine.weather)


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
def test_leading_stop_remains_allowed_when_rival_preserves_finish_distance(
    monkeypatch, engine_name,
):
    simulator, states, track = inputs()
    states = states[:2]
    simulator.tire_warmup = {}
    states[0].dry_pit_proposal = (72, TireCompound.SOFT)
    engine = ChronologicalRace(simulator)
    engine.states = {state.driver.id: state for state in states}
    engine.control_intervals = 71
    engine.track, engine.weather = track, Weather()
    engine.timeline = RaceFinishTimeline(90, engine.states)
    engine.running_paces = {"A": 99., "B": 100.}
    engine.order = ["A", "B"]
    # The rival's committed crossing precedes expiry even if A stops now.
    engine.pending = {"B": _PendingLap(72, 7098., 7198., Weather(), False,
                                       states[1].current_tire, 8, 100.,
                                       running_start=7098.)}
    physics = BoundPhysics()
    monkeypatch.setattr(LapSimulator, "calculate_lap_time",
                        lambda self, *args, **kwargs: physics.calculate_lap_time(*args, **kwargs))
    if engine_name == "standard":
        states[1].total_time = 7098.
        context = simulator._standard_leading_finish_context(
            states, states[0], engine.running_paces, RaceFinishClock(90),
        )
        assert not simulator._protect_leading_finish_distance(
            states[0], track, Weather(), 72, context,
        )
    else:
        assert not engine._protect_elective_finish_distance(states[0], track, 7100., 0., None)
    # With no rival, the same retained and paid paths have unequal distances.
    states[1].status = DriverStatus.DNF
    if engine_name == "standard":
        context = simulator._standard_leading_finish_context(
            states, states[0], engine.running_paces, RaceFinishClock(90),
        )
        assert simulator._protect_leading_finish_distance(
            states[0], track, Weather(), 72, context,
        )
    else:
        assert engine._protect_elective_finish_distance(states[0], track, 7100., 0., None)


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
@pytest.mark.parametrize("control_mode", ["vsc", "safety_car", "red_flag"])
@pytest.mark.parametrize("field_size", [1, 3])
def test_neutralized_finish_guard_uses_control_only_with_a_supported_field(
    monkeypatch, engine_name, control_mode, field_size,
):
    simulator, states, track = inputs()
    states = states[:field_size]
    setattr(simulator.event_manager, f"{control_mode}_active", True)
    calls = []

    def compare(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(veto=True)

    engine = ChronologicalRace(simulator)
    engine.states = {state.driver.id: state for state in states}
    engine.track, engine.weather = track, Weather()
    engine.timeline = RaceFinishTimeline(90, engine.states)
    engine.control_intervals = 71
    engine.running_paces = {state.driver.id: 100. for state in states}
    engine.order, engine.pending = list(engine.states), {}
    before = [snapshot(state, simulator) for state in states]
    if engine_name == "standard":
        monkeypatch.setattr(race_module, "evaluate_finish_protection", compare)
        context = simulator._standard_leading_finish_context(
            states, states[0], engine.running_paces, RaceFinishClock(90),
        )
        result = simulator._protect_leading_finish_distance(
            states[0], track, Weather(), 72, context, active_states=states,
        )
    else:
        monkeypatch.setattr(chronological_module, "evaluate_finish_protection", compare)
        result = engine._protect_elective_finish_distance(states[0], track, 7100., 0., None)
    supported = control_mode != "red_flag" and (
        field_size == 1 or engine_name == "standard" and control_mode == "vsc")
    assert result is supported
    assert len(calls) == int(supported)
    if calls:
        assert calls[0]["current_lap_time_modifier"] == (
            1.2 if control_mode == "vsc" else 1.4)
        assert calls[0]["active_aero_enabled"] is False
        assert calls[0]["expected_lane_loss"] == track.pit_lane_delta * (
            .75 if control_mode == "vsc" else .55)
    assert before == [snapshot(state, simulator) for state in states]
