"""Compound-changing proposals remain executable in both race engines."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from examples.check_weather_transitions import CASES, compare_schedules, run_race
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot


def fixture():
    simulator = RaceSimulator(np.random.default_rng(4))
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], tire_laps=10,
                            tire_compound_history=["intermediate"])
    track = Track(id="T", name="T", country="T", total_laps=20, base_lap_time=90)
    return simulator, state, track, Weather(track_wetness=.19, rain_intensity=0)


def test_both_engines_match_bounded_executed_transition_schedules():
    rows = compare_schedules()
    assert len(rows) == 8
    assert {row["engine"] for row in rows} == {"standard", "chronological"}
    for row in rows:
        assert row["schedules_checked"] > 20
        assert row["gap_seconds"] == pytest.approx(0, abs=1e-8)
        assert row["selected"]["laps_completed"] == row["case"]["laps"]
        if row["case"]["name"] == "drying_intermediates":
            assert row["selected"]["pit_laps"] == [3]
        if row["case"]["name"] == "wet_to_intermediate_to_slick":
            assert row["selected"]["compounds"] == ["wet", "intermediate", "soft"]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_drying_transition_avoids_waiting_until_critical(engine):
    selected = run_race(CASES[0], engine)
    late = run_race(CASES[0], engine, ((8, TireCompound.SOFT),))
    assert late["total_seconds"] - selected["total_seconds"] > 24
    assert selected["paid_stops"] == late["paid_stops"] == 1


@pytest.mark.parametrize("neutralized", [False, True])
def test_transition_receives_observed_costs_and_preserves_rng(monkeypatch, neutralized):
    simulator, state, track, weather = fixture()
    if neutralized:
        simulator.event_manager.vsc_active = True
    calls = []

    def plan(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(should_pit=lambda: True, compound=TireCompound.HARD)

    monkeypatch.setattr("f1sim.simulation.race.plan_rain_transition", plan)
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    assert simulator._should_pit(
        state, [state], track, 10, False, weather, 3, physical_total_laps=60,
        traffic_snapshot=StrategyTrafficSnapshot(None, None, 2),
    )
    args, kwargs = calls[0]
    assert args[-3:] == (10, 10, 4)
    factor = simulator.lap_simulator.weather_pace_multiplier(state.driver, state.car, weather)
    assert kwargs["additional_current_stop_cost"] == pytest.approx(
        3 + (0 if neutralized else 2 * factor)
    )
    assert kwargs["physical_total_laps"] == 60
    assert kwargs["remaining_dry_stops"] == 3
    assert kwargs["remaining_damp_stops"] == 1
    assert kwargs["active_aero_enabled"] is not neutralized
    assert kwargs["pit_lane_factor"] == (.75 if neutralized else 1)
    assert kwargs["current_lap_time_modifier"] == (1.2 if neutralized else 1)
    assert simulator.rng.bit_generator.state == before
    # Damp-surface execution must consume the selected slick, not its fallback choice.
    simulator._execute_pit_stop(state, track, weather, 10, sample_service=False)
    assert state.current_tire.compound == TireCompound.HARD
    assert state.weather_pit_proposal is None


def test_stale_or_weather_ineligible_proposal_is_discarded():
    simulator, state, track, weather = fixture()
    state.weather_pit_proposal = (9, TireCompound.HARD)
    simulator._execute_pit_stop(state, track, weather, 10, sample_service=False)
    assert state.weather_pit_proposal is None
    state.weather_pit_proposal = (11, TireCompound.HARD)
    weather.track_wetness = .8
    simulator._execute_pit_stop(state, track, weather, 11, sample_service=False)
    assert state.current_tire.compound == TireCompound.WET
    assert state.weather_pit_proposal is None


def test_unused_rain_set_cannot_bypass_actual_compound_use(monkeypatch):
    simulator, state, track, weather = fixture()
    state.tire_laps = 0
    state.tire_compound_history = ["medium", "intermediate"]
    monkeypatch.setattr("f1sim.simulation.race.plan_rain_transition",
                        lambda *a, **k: pytest.fail("Unused set does not earn wet exemption"))
    simulator._should_pit(state, [state], track, 10, False, weather)
    state.weather_pit_proposal = (20, TireCompound.MEDIUM)
    simulator._execute_pit_stop(state, track, weather, 20, sample_service=False)
    assert state.current_tire.compound != TireCompound.MEDIUM


def test_critical_mismatch_and_exhausted_budget_retain_priority(monkeypatch):
    simulator, state, track, weather = fixture()
    state.pit_stops = 4
    monkeypatch.setattr("f1sim.simulation.race.plan_rain_transition",
                        lambda *a, **k: pytest.fail("Exhausted budget bypasses elective planner"))
    assert not simulator._should_pit(state, [state], track, 10, False, weather)
    weather.track_wetness = .01
    assert simulator._should_pit(state, [state], track, 10, False, weather, 10000)


def test_transition_does_not_rely_on_a_forbidden_later_slick_stop():
    simulator, state, track, weather = fixture()
    state.tire_laps = 1
    state.car.tire_degradation_factor = 1.5
    track.total_laps, track.base_lap_time = 25, 200
    track.pit_lane_delta, track.tire_stress = 1, 1
    assert not simulator._should_pit(state, [state], track, 2, False, weather)
    assert state.weather_pit_proposal is None
