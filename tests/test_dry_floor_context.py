"""Dry decisions retain physical fuel and current control across entry points."""

from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import race
from f1sim.simulation.pit_strategy import DryPitDecision
from f1sim.simulation.race import DriverRaceState, RaceSimulator


@pytest.mark.parametrize("action", ["elective", "committed", "restart"])
def test_dry_entry_points_preserve_fuel_and_control(monkeypatch, action):
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator.event_manager.vsc_active = True
    state = DriverRaceState(
        driver=Driver(id="d", name="Driver", team_id="t"),
        car=Car(team_id="t", team_name="Team"), position=1,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM], tire_laps=20,
        tire_compound_history=["soft", "medium"],
    )
    track = Track(id="t", name="Track", country="Test", total_laps=30, base_lap_time=90)
    calls = []

    def plan(*args, **kwargs):
        calls.append((args, kwargs))
        return DryPitDecision(inf, 0, None)

    monkeypatch.setattr(race, "plan_dry_stop", plan)
    if action == "elective":
        assert not simulator._should_pit(
            state, [state], track, 25, Weather(), physical_total_laps=60,
        )
    elif action == "committed":
        simulator._execute_pit_stop(
            state, track, Weather(), current_lap=25,
            physical_total_laps=60, sample_service=False,
        )
    else:
        simulator._fit_red_flag_tires(
            [state], Weather(), track, 25, physical_total_laps=60,
        )
    assert len(calls) == (1 if action == "elective" else 3)
    for args, kwargs in calls:
        assert kwargs["physical_total_laps"] == 60
        assert args[5] == (5 if action == "restart" else 6)
        assert kwargs.get("active_aero_enabled", True) is (action == "restart")
        assert kwargs.get("current_lap_time_modifier", 1) == (
            1 if action == "restart" else 1.2
        )
