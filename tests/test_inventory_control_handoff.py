"""A green suffix cannot erase accepted crossings from a controlled prefix."""

from copy import deepcopy
from dataclasses import replace
from math import inf

import numpy as np
import pytest
from test_controlled_dry_strategy import single_car_context
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models._native import native_physics
from f1sim.simulation.chronological_finish import ChronologicalFinishContext
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.tire_inventory import TireInventory


def observed_context(track, car, engine, control, intervals):
    context = single_car_context(track, car, control, intervals)
    if engine == "chronological":
        field = ChronologicalFinishContext(
            "A", RaceFinishTimeline(20, ["A"]), ("A",), (), track.base_lap_time,
            1.4 if control == "sc" else 1.2, control == "sc", control_intervals=intervals)
        context = StrategyControlContext(field, 50., context.current_stop_delay)
    return context


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("intervals", [4, 5, 6])
def test_spent_pool_keeps_all_five_crossings_when_control_ends(engine, control, intervals):
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="T", name="Synthetic", country="Test", total_laps=6,
                  base_lap_time=90., pit_lane_delta=20.)
    weather = Weather(track_wetness=.1, rain_intensity=.15, change_probability=0.)
    stock = TireInventory.from_sets([
        dict(id="M", compound="medium", age=35, remaining_laps=2),
        dict(id="I", compound="intermediate", age=34, remaining_laps=2),
        dict(id="old-I", compound="intermediate", age=54, remaining_laps=1)])
    stock.fit("M")
    options = dict(tire_age=35, remaining_stops=0, force_stop=True,
                   remaining_dry_stops=2, remaining_damp_stops=3,
                   used_compounds=("soft", "hard"), physical_total_laps=20,
                   current_fit_pending=True, tire_warmup={"medium": 7., "intermediate": 6.},
                   control_context=observed_context(track, car, engine, control, intervals))
    models = driver, car, track, weather, stock
    assert native_physics(driver, car, track, weather)
    before = deepcopy((models[:-1], stock.__dict__, options))
    expected, first_choices = independent_schedules(models, options)
    result = plan_inventory_strategy(*models, 1, **options)
    assert expected[True][:2] == (False, 5)
    assert_continuation(result.continuation(True), expected[True])
    assert_continuation(result.continuation(False), expected[False])
    assert first_choices[result.set_id] == pytest.approx(expected[True], abs=1.e-8)
    assert result.pit_now_cost == result.wait_cost == inf
    assert result.should_pit()
    assert (models[:-1], stock.__dict__, options) == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("reason", ["exhausted", "critical", "missing_credit"])
def test_retained_last_crossing_survives_an_unavailable_green_action(engine, control, reason):
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="T", name="Synthetic", country="Test", total_laps=3,
                  base_lap_time=90., pit_lane_delta=20.)
    weather = Weather(track_wetness=.09 if reason == "critical" else .1,
                      rain_intensity=0. if reason == "critical" else .1,
                      change_probability=0.)
    compound = "intermediate" if reason == "critical" else "hard"
    stock = TireInventory.from_sets([
        dict(id="only", compound=compound, age=20,
             remaining_laps=None if reason == "critical" else 1)])
    stock.fit("only")
    options = dict(tire_age=20, remaining_stops=0, physical_total_laps=20,
                   used_compounds=() if reason == "missing_credit" else ("soft", "hard"),
                   current_fit_pending=True, tire_warmup={compound: 4.},
                   control_context=observed_context(track, car, engine, control, 1))
    models = driver, car, track, weather, stock
    expected, _ = independent_schedules(models, options)
    result = plan_inventory_strategy(*models, 1, **options)
    assert expected[False][:2] == (False, 1)
    assert_continuation(result.continuation(False), expected[False])
    assert_continuation(result.continuation(True), expected[True])
    assert result.set_id is None and not result.should_pit()
    assert result.wait_cost == inf


def completed_handoff_race(engine, control, forced_set=None):
    """Complete actual crossings and stops with mean physics and no incidents."""
    driver = Driver(id="A", name="Synthetic", team_id="A")
    car = Car(team_id="A", team_name="Synthetic")
    track = Track(id="T", name="Synthetic", country="Test", total_laps=8,
                  base_lap_time=90., pit_lane_delta=20., tire_stress=.8)
    weather = Weather(track_wetness=.1, rain_intensity=.15, change_probability=0.)
    simulator = RaceSimulator(np.random.default_rng(7),
                              tire_warmup={"medium": 7., "intermediate": 6.})
    simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
    events = simulator.event_manager

    def update(lap, *args, **kwargs):
        active = 1 <= lap < 6
        events.safety_car_active = active and control == "sc"
        events.vsc_active = active and control == "vsc"
        events.safety_car_laps_remaining = max(0, 6 - lap) if events.safety_car_active else 0
        events.vsc_laps_remaining = max(0, 6 - lap) if events.vsc_active else 0
        return []

    events.process_lap = update
    calculate = simulator.lap_simulator.calculate_lap_time
    first_running = []

    def mean(*args, **kwargs):
        assert kwargs.get("total_laps", args[6] if args else None) == 8
        value = calculate(*args, **dict(kwargs, sample_variation=False))
        if not first_running:
            first_running.append(value)
        return value

    simulator.lap_simulator.calculate_lap_time = mean
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    plan = simulator._plan_inventory
    first_decisions = []

    def observed(state, track, weather, lap, **options):
        assert native_physics(state.driver, state.car, track, weather)
        decision = plan(state, track, weather, lap, **options)
        if lap == 2:
            first_decisions.append(decision)
            if forced_set is not None:
                return replace(decision, set_id=forced_set,
                               compound=state.tire_inventory.sets[forced_set].compound)
        return decision

    simulator._plan_inventory = observed
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result, = execute([driver], {"A": car}, track, weather, ["A"],
                      starting_tires={"A": TireCompound.HARD},
                      tire_inventory={"A": [
                          dict(id="opening", compound="hard", remaining_laps=1),
                          dict(id="M", compound="medium", age=35, remaining_laps=2),
                          dict(id="I", compound="intermediate", age=34, remaining_laps=2),
                          dict(id="old-I", compound="intermediate", age=54, remaining_laps=1)]})
    return result, first_decisions[0], first_running[0]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
def test_forecast_matches_complete_native_races_and_separately_committed_fits(engine, control):
    selected, forecast, first_lap = completed_handoff_race(engine, control)
    alternatives = [completed_handoff_race(engine, control, key)[0]
                    for key in ("M", "I", "old-I")]
    best = min(alternatives, key=lambda row: (-row.laps_completed, row.total_time))
    assert selected.status.value == best.status.value == "dnf"
    assert selected.laps_completed == best.laps_completed == 6
    assert forecast.pit_now_laps == selected.laps_completed - 1
    assert forecast.pit_now_cost == inf
    assert forecast.pit_now_partial_time == pytest.approx(
        selected.total_time - first_lap, abs=1.e-8)
    assert selected.total_time == pytest.approx(best.total_time, abs=1.e-8)
    assert sum(row["laps_used"] for row in selected.tire_set_history) == 6
    assert all(row["remaining_laps"] == 0 for row in selected.tire_inventory)
