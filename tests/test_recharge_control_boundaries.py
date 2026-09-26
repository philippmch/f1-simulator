"""Overtake Mode recharge follows the completed lap's control snapshot."""

from dataclasses import replace

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


def run_control_case(monkeypatch, engine, control, starting_energy=0.5):
    simulator = RaceSimulator(np.random.default_rng(19))
    manager = simulator.event_manager
    if control == "sc":
        manager.set_forced_safety_car(1)
    elif control == "red":
        manager.set_forced_red_flag(1)

    original_process_lap = manager.process_lap
    control_states = []

    def process_lap(lap, *args, **kwargs):
        events = original_process_lap(lap, *args, **kwargs)
        if lap == 1 and control == "sc":
            # Make lap two the last SC lap so its countdown expires after the
            # neutralized running has completed.
            manager.safety_car_laps_remaining = 1
        elif lap == 1 and control == "vsc":
            manager.vsc_active = True
            manager.vsc_laps_remaining = 1
            event = RaceEvent(EventType.VIRTUAL_SAFETY_CAR, lap, duration_laps=1)
            manager.events.append(event)
            events.append(event)
        control_states.append((lap, manager.safety_car_active, manager.vsc_active))
        return events

    monkeypatch.setattr(manager, "process_lap", process_lap)
    monkeypatch.setattr(manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_deploy_safety_measure", lambda *a, **kw: None)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *a, **kw: 0)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))

    recharges = []
    first_lap_states = []
    original_recharge = simulator._recharge_overtake_mode_energy
    seeded = False

    def recharge(states, neutralized=False):
        nonlocal seeded
        for state in states:
            if state.laps_completed == 1 and not seeded:
                # Energy is internal race state, so seed it at the first real
                # completed-lap boundary for this controlled regression.
                state.overtake_mode_energy = starting_energy
                seeded = True
        original_recharge(states, neutralized=neutralized)
        for state in states:
            recharges.append((state.laps_completed, neutralized, state.overtake_mode_energy))
            if state.laps_completed == 1:
                first_lap_states.append(replace(state))

    monkeypatch.setattr(simulator, "_recharge_overtake_mode_energy", recharge)

    driver = Driver(id="D", name="Driver", team_id="T")
    car = Car(team_id="T", team_name="Team")
    track = Track(id="T", name="Track", country="Test", total_laps=3,
                  base_lap_time=90)
    weather = Weather(change_probability=0)

    if engine == "standard":
        results = simulator.simulate_race(
            [driver], {"T": car}, track, weather, ["D"],
            starting_tires={"D": TireCompound.MEDIUM},
        )
        suspensions = simulator.suspensions
    else:
        chronological = ChronologicalRace(simulator)
        results = chronological.run(
            [driver], {"T": car}, track, weather, ["D"],
            starting_tires={"D": TireCompound.MEDIUM},
        )
        suspensions = chronological.suspensions

    return simulator, track, recharges, first_lap_states, control_states, results, suspensions


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("starting_energy", [0.5, 0.28])
def test_deployment_after_green_crossing_recharges_by_lap_start(
    monkeypatch, engine, control, starting_energy,
):
    simulator, track, recharges, first_lap_states, control_states, results, _ = (
        run_control_case(monkeypatch, engine, control, starting_energy)
    )

    assert [lap for lap, _, _ in recharges] == [1, 2, 3]
    assert [neutralized for _, neutralized, _ in recharges] == [False, True, False]
    assert [energy for _, _, energy in recharges] == pytest.approx([
        starting_energy + 0.04,
        starting_energy + 0.16,
        starting_energy + 0.20,
    ])
    assert control_states == [(1, control == "sc", control == "vsc"), (2, False, False),
                              (3, False, False)]
    assert results[0].laps_completed == 3

    if starting_energy == 0.28:
        # The green recharge leaves the store below a deployment. The old
        # post-event flag read incorrectly raised it above this threshold.
        state_after_lap_one = first_lap_states[0]
        assert state_after_lap_one.overtake_mode_energy < simulator.OVERTAKE_MODE_DEPLOYMENT_COST
        assert not simulator._strategy_overtake_mode_active(
            state_after_lap_one, track, 2, Weather(), 1, mode_allowed=True,
        )
        old_post_event_energy = replace(
            state_after_lap_one,
            overtake_mode_energy=state_after_lap_one.overtake_mode_energy + 0.08,
        )
        assert simulator._strategy_overtake_mode_active(
            old_post_event_energy, track, 2, Weather(), 1, mode_allowed=True,
        )


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_red_flag_pause_and_free_refit_do_not_add_recharge(monkeypatch, engine):
    _, _, recharges, _, _, results, suspensions = run_control_case(
        monkeypatch, engine, "red", starting_energy=0.5,
    )

    assert results[0].laps_completed == 3
    assert results[0].race_suspension_seconds > 0
    assert suspensions
    assert [lap for lap, _, _ in recharges] == [1, 2, 3]
    assert [neutralized for _, neutralized, _ in recharges] == [False, False, False]
    assert [energy for _, _, energy in recharges] == pytest.approx([0.54, 0.58, 0.62])
