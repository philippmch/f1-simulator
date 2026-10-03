"""Native execution counts accepted race laps against each physical allowance."""

from copy import deepcopy

import pytest
from test_inventory_race import conserved, state_with_pool
from test_inventory_race import race as race

from f1sim.models import Weather
from f1sim.simulation.race import DriverStatus


def check_allowances(result, records):
    conserved(result, records)
    credit = {item["id"]: item["remaining_laps"] for item in records}
    for stint in result.tire_set_history:
        assert stint["remaining_laps_at_fit"] == credit[stint["set_id"]]
        credit[stint["set_id"]] -= stint["laps_used"]
        assert stint["remaining_laps_at_end"] == credit[stint["set_id"]] >= 0
    assert {item["id"]: item["remaining_laps"] for item in result.tire_inventory} == credit


@pytest.mark.parametrize("control", [None, "sc", "vsc"])
@pytest.mark.parametrize("custom", [False, True])
def test_compulsory_expiry_survives_zero_elective_budget_and_empty_plan(
    race, monkeypatch, control, custom,
):
    _, sim, _, run, services, running = race
    monkeypatch.setattr(sim, "_ordinary_stop_budget", lambda *a: 0)
    monkeypatch.setattr(sim, "_dry_stop_budget", lambda *a: 0)
    reset = sim.event_manager.reset

    def reset_control():
        reset()
        sim.event_manager.safety_car_active = control == "sc"
        sim.event_manager.vsc_active = control == "vsc"
        sim.event_manager.safety_car_laps_remaining = 20 if control == "sc" else 0
        sim.event_manager.vsc_laps_remaining = 20 if control == "vsc" else 0

    monkeypatch.setattr(sim.event_manager, "reset", reset_control)
    records = [dict(id="I1", compound="intermediate", age=3, remaining_laps=3),
               dict(id="I2", compound="intermediate", age=3, remaining_laps=3),
               dict(id="I3", compound="intermediate", age=3, remaining_laps=2)]
    original = deepcopy(records)
    result = run(records, "intermediate", 3,
                 Weather(track_wetness=.5, rain_intensity=.5, change_probability=0),
                 **({"pit_plans": {"A": []}} if custom else {}))
    assert result.status == DriverStatus.FINISHED
    assert result.laps_completed == len(running) == 8
    assert result.pit_laps == [4, 4 + result.tire_set_history[1]["remaining_laps_at_fit"]]
    assert len(services) == result.pit_stops == 2
    assert all(item["decision_reason"] == "tyre_usage_limit" for item in result.pit_stop_details)
    if control is not None:
        assert all(item["control"] != "green" for item in result.pit_stop_details)
    assert all(not item["unavailable"] and not item["available"] for item in result.tire_inventory)
    check_allowances(result, records)
    assert records == original


def test_exhausted_pool_retires_before_service_or_running(race, monkeypatch):
    _, sim, _, run, services, running = race
    monkeypatch.setattr(sim, "_ordinary_stop_budget", lambda *a: 0)
    records = [dict(id="I1", compound="intermediate", remaining_laps=3),
               dict(id="I2", compound="intermediate", remaining_laps=2)]
    result = run(records, "intermediate", 0, Weather(track_wetness=.5, rain_intensity=.5))
    assert result.status == DriverStatus.DNF
    assert result.dnf_reason == "No suitable replacement tyre set available"
    assert result.laps_completed == len(running) == 5
    assert result.pit_laps == [4] and len(services) == 1
    check_allowances(result, records)


def test_refitting_a_removed_set_keeps_its_credit(race, monkeypatch):
    _, sim, _, run, _, _ = race
    records = [dict(id="S", compound="soft", age=5, remaining_laps=6),
               dict(id="H", compound="hard", remaining_laps=2)]

    def policy(state, states, track, lap, *a, **kw):
        selected = {2: "H", 4: "S"}.get(lap)
        state.inventory_pit_proposal = (lap, selected)
        return selected is not None

    monkeypatch.setattr(sim, "_should_pit", policy)
    result = run(records, age=5)
    assert result.status == DriverStatus.FINISHED
    assert result.pit_laps == [2, 4]
    assert [item["remaining_laps_at_fit"] for item in result.tire_set_history] == [6, 2, 5]
    check_allowances(result, records)


def test_free_restart_retains_allowance_and_replaces_an_expired_set():
    records = [dict(id="I", compound="intermediate", age=4, remaining_laps=3),
               dict(id="spare", compound="intermediate", remaining_laps=8)]
    sim, state, track = state_with_pool(records, "I", 5)
    wet = Weather(track_wetness=.5, rain_intensity=.5)
    sim._fit_inventory_tire(state, "I", 2, "red_flag")
    assert state.tire_inventory.current_remaining_laps(state.tire_laps) == 2
    assert len(state.tire_set_history) == 1
    state.tire_laps = state.driver.current_tire_laps = 7
    assert sim._refit_inventory_free(state, track, wet, 3)
    assert state.tire_inventory.current_set_id == "spare"
    assert state.tire_inventory.sets["I"].remaining_laps == 0
    assert state.pit_stops == 0
    assert state.tire_set_history[-1]["kind"] == "red_flag"


def test_automatic_opening_keeps_different_allowances_distinct(monkeypatch):
    import numpy as np

    from f1sim.models import Car, Driver, Track
    from f1sim.simulation import opening_strategy
    from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype

    sim = RaceSimulator(np.random.default_rng(1))
    records = [dict(id="expired", compound="intermediate", remaining_laps=0),
               dict(id="short", compound="intermediate", remaining_laps=1),
               dict(id="long", compound="intermediate", remaining_laps=4),
               dict(id="duplicate", compound="intermediate", remaining_laps=4)]
    original = opening_strategy._policy_path_outcome
    observed = []

    def outcome(*args, **kwargs):
        observed.append(kwargs["opening_set_id"])
        return original(*args, **kwargs)

    opening_strategy._cached_inventory_policy_costs.cache_clear()
    monkeypatch.setattr(opening_strategy, "_policy_path_outcome", outcome)
    _, selected = sim._inventory_opening_set(
        Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
        Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90),
        Weather(track_wetness=.5, rain_intensity=.5), TeamStrategyArchetype.BALANCED, records)
    assert selected.id == "long"
    assert observed == ["short", "long"]


def test_expiry_at_the_flag_does_not_trigger_an_extra_stop(race, monkeypatch):
    _, sim, _, run, services, running = race
    monkeypatch.setattr(sim, "_ordinary_stop_budget", lambda *a: 0)
    records = [dict(id="I", compound="intermediate", remaining_laps=8)]
    result = run(records, "intermediate", 0, Weather(track_wetness=.5, rain_intensity=.5))
    assert result.status == DriverStatus.FINISHED and result.laps_completed == len(running) == 8
    assert result.pit_stops == 0 and services == []
    check_allowances(result, records)


def test_leading_finish_guard_cannot_retain_beyond_actual_scheduled_allowance():
    records = [dict(id="I", compound="intermediate", remaining_laps=20)]
    sim, state, planning = state_with_pool(records, "I", 1)
    assert not sim._protect_leading_finish_distance(
        state, planning, Weather(track_wetness=.5, rain_intensity=.5), 2,
        object(), physical_total_laps=58)


def test_chronological_finish_guard_remains_enabled_for_nonbinding_limits(monkeypatch):
    from f1sim.simulation.chronological_race import ChronologicalRace

    records = [dict(id="I", compound="intermediate", remaining_laps=20)]
    sim, state, track = state_with_pool(records, "I", 1)
    engine = ChronologicalRace(sim)
    engine.track, engine.states = track, {"A": state}
    engine.weather = Weather(track_wetness=.5, rain_intensity=.5)
    monkeypatch.setattr(engine, "_weather_projection_clock", lambda *a, **kw: object())
    assert engine._finish_protection_skip_reason(state, now=90.) is None
    engine.track = track.model_copy(update={"total_laps": 58})
    assert engine._finish_protection_skip_reason(state, now=90.) == (
        "current inventory set has usage limit")
