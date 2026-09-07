"""Fitting a set is not using it; compliance requires running on that set."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.pit_strategy import plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def fixture(laps=10):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90)
    return driver, car, track


@pytest.mark.parametrize("start", [TireCompound.MEDIUM, TireCompound.WET])
def test_two_lap_race_runs_two_dry_sets_even_after_prestart_wet_refit(monkeypatch, start):
    driver, car, track = fixture(2)
    simulator = RaceSimulator(np.random.default_rng(42))
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    ran = []

    def lap_time(**kwargs):
        ran.append(kwargs["tire"].compound)
        return 90

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap_time)
    result, = simulator.simulate_race([driver], {"A": car}, track, Weather(change_probability=0),
                                      ["A"], starting_tires={"A": start})
    assert len(set(ran)) == 2
    assert all(compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
               for compound in ran)
    assert result.strategy == [compound.value for compound in ran]
    assert result.pit_laps == ([1, 2] if start == TireCompound.WET else [2])


def test_unrun_refit_replaces_tail_but_preserves_prior_actual_occurrence():
    driver, car, _ = fixture()
    state = DriverRaceState(driver, car, 1, current_tire=TIRE_COMPOUNDS[TireCompound.WET],
                            tire_compound_history=["soft", "wet"])
    assert not RaceSimulator._has_used_wet_compound(state)
    RaceSimulator._fit_tire(state, TireCompound.MEDIUM)
    assert state.tire_compound_history == ["soft", "medium"]
    assert RaceSimulator._used_slick_compounds(state) == {TireCompound.SOFT}
    state.current_tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    state.tire_compound_history = ["soft", "medium", "soft"]
    assert RaceSimulator._used_slick_compounds(state) == {TireCompound.SOFT, TireCompound.MEDIUM}


def test_paid_refit_replaces_unrun_red_flag_set_without_wet_exemption(monkeypatch):
    driver, car, track = fixture()
    state = DriverRaceState(driver, car, 1, current_tire=TIRE_COMPOUNDS[TireCompound.SOFT],
                            tire_laps=5)
    simulator = RaceSimulator()
    monkeypatch.setattr(simulator, "_choose_red_flag_tire", lambda *args: TireCompound.WET)
    simulator._handle_red_flag_stop([state], Weather(), track, 5)
    assert state.tire_compound_history == ["soft", "wet"]
    assert not simulator._has_used_wet_compound(state)
    simulator._execute_pit_stop(state, track, Weather(), 6)
    assert "wet" not in state.tire_compound_history
    assert simulator._used_slick_compounds(state) == {TireCompound.SOFT}
    assert state.current_tire.compound in {TireCompound.MEDIUM, TireCompound.HARD}


@pytest.mark.parametrize("remaining", [1, 2])
@pytest.mark.parametrize("paid", [0, 3])
def test_free_unrun_distinct_set_satisfies_stay_without_extra_paid_stop(remaining, paid):
    driver, car, track = fixture()
    state = DriverRaceState(driver, car, 1, current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM],
                            pit_stops=paid, tire_compound_history=["soft", "medium"])
    simulator = RaceSimulator()
    assert simulator._used_slick_compounds(state) == {TireCompound.SOFT}
    assert not simulator._should_pit(state, [state], track, 11 - remaining, False, Weather())


@pytest.mark.parametrize("modifier", [1, 1.2, 1.4])
def test_planner_distinguishes_immediate_replacement_from_running_current_set(modifier):
    driver, car, track = fixture()
    decision = plan_dry_stop(driver, car, track, TIRE_COMPOUNDS[TireCompound.HARD],
                             0, 1, 1, {TireCompound.SOFT}, current_lap_time_modifier=modifier)
    assert decision.compound in (TireCompound.MEDIUM, TireCompound.HARD)
    assert np.isfinite(decision.wait_cost)  # Waiting runs the hard set and meets the rule.
    assert np.isfinite(decision.pit_now_cost)
    # With no actual prior set, a one-lap immediate replacement cannot meet the rule.
    missing = plan_dry_stop(driver, car, track, TIRE_COMPOUNDS[TireCompound.HARD],
                            0, 1, 1, set(), current_lap_time_modifier=modifier)
    assert np.isinf(missing.pit_now_cost) and np.isinf(missing.wait_cost)
