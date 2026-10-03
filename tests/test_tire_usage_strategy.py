"""Usage-limited policies agree with exhaustive, identity-preserving schedules."""

from copy import deepcopy
from dataclasses import replace
from math import inf

import pytest
from test_controlled_inventory_strategy import enumerate_physical_schedules, pool_fixture
from test_controlled_weather_strategy import all_schedules, context, inputs
from test_custom_pit_replacements import execution_costs
from test_custom_pit_replacements import inputs as custom_inputs
from test_inventory_strategy import exhaustive, fixture
from test_timed_inventory_weather_oracle import enumerate_sets

from f1sim.models import TireCompound
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory


@pytest.mark.parametrize("free,budget", [(False, 0), (False, 2), (True, 0)])
@pytest.mark.parametrize("surface", [(0., 0.), (.12, 0.), (.5, .5)])
def test_usage_limits_in_green_physical_schedule_oracle(free, budget, surface):
    models = fixture(*surface)
    stock = TireInventory.from_sets([
        {"id": "S", "compound": "soft", "age": 8, "remaining_laps": 2},
        {"id": "H", "compound": "hard", "age": 2, "remaining_laps": 3},
        {"id": "I", "compound": "intermediate", "age": 3, "remaining_laps": 2},
        {"id": "W", "compound": "wet", "remaining_laps": 3},
    ])
    stock.fit("S")
    options = dict(tire_age=9, remaining_stops=budget, remaining_dry_stops=budget,
                   remaining_damp_stops=budget, free_fit=free,
                   tire_warmup={"soft": 2., "hard": .5, "wet": 3.},
                   current_fit_pending=True, current_lap_time_modifier=1.4)
    before = deepcopy(stock.__dict__)
    expected, _ = exhaustive(models, stock, **options)
    actual = plan_inventory_strategy(*models, stock, 1, **options)
    assert actual.pit_now_cost == pytest.approx(expected[True], abs=1.e-8)
    if not free:
        assert actual.wait_cost == pytest.approx(expected[False], abs=1.e-8)
    assert stock.__dict__ == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("budget", [0, 2])
def test_usage_expiry_during_control_and_green_handoff(engine, control, budget):
    models = pool_fixture(6)
    driver, car, track, weather, stock = models
    credits = {"H": 3, "S": 2, "S2": 4, "M": 3, "W": 0}
    stock.sets = {key: replace(item, remaining_laps=credits[key])
                  for key, item in stock.sets.items()}
    observed = context(track, car, engine, control, 2)
    used = {TireCompound.HARD, TireCompound.MEDIUM}
    result = plan_inventory_strategy(
        driver, car, track, weather, stock, 1, tire_age=14, remaining_stops=budget,
        remaining_dry_stops=budget, remaining_damp_stops=0, used_compounds=used,
        physical_total_laps=12, control_context=observed,
        tire_warmup={"soft": 2., "medium": .4, "hard": 1.1}, current_fit_pending=True)
    expected, selected = enumerate_physical_schedules(
        models, control, 2, budget, budget, used, True)
    assert result.wait_cost == pytest.approx(expected[False], abs=1.e-8)
    assert result.pit_now_cost == pytest.approx(expected[True], abs=1.e-8)
    assert result.set_id == selected


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("surface", [(.5, .5), (.45, 0.), (.12, 0.)])
def test_usage_limits_in_changing_controlled_weather(engine, control, surface):
    models = inputs(*surface)
    driver, car, track, weather, stock = models
    credits = {"I": 4, "I2": 2, "W": 2, "S": 2, "H": 3}
    stock.sets = {key: replace(item, remaining_laps=credits[key])
                  for key, item in stock.sets.items()}
    result = plan_inventory_strategy(
        driver, car, track, weather, stock, 1, tire_age=11, remaining_stops=0,
        remaining_dry_stops=0, remaining_damp_stops=0, physical_total_laps=12,
        control_context=context(track, car, engine, control, 2), current_fit_pending=True,
        tire_warmup={"intermediate": 1., "wet": 2., "soft": .6, "hard": .3})
    expected, costs = all_schedules(models, control, 2, 0, finite=True, pending=True)
    assert result.wait_cost == pytest.approx(expected[False], abs=1.e-8)
    assert result.pit_now_cost == pytest.approx(expected[True], abs=1.e-8)
    if expected[True] == inf:
        assert result.set_id is None
    else:
        assert costs[result.set_id] == pytest.approx(expected[True], abs=1.e-8)


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("surface", [(.1, .75), (.25, 0.), (.44, .6)])
def test_usage_compulsory_stops_advance_external_weather(free, surface):
    models = fixture(*surface)
    stock = TireInventory.from_sets([
        {"compound": c, "age": age, "remaining_laps": cap}
        for c, age, cap in [("soft", 8, 2), ("hard", 2, 3),
                           ("intermediate", 3, 3), ("wet", 0, 2)]
    ])
    stock.fit("set-3")
    service = expected_stationary_time(models[1])
    clock = StrategyWeatherClock((0., 108., 198., 288.), 12., 30., 12,
                                 service + 3 * .75 + 23., service + 3)
    options = dict(tire_age=4, free_fit=free, remaining_stops=0,
                   remaining_dry_stops=0, remaining_damp_stops=0,
                   used_compounds={TireCompound.INTERMEDIATE}, physical_total_laps=10,
                   current_traffic_gaps=(.2, 1.2), current_lap_time_modifier=1.2,
                   active_aero_enabled=False, pit_lane_factor=.75,
                   additional_current_stop_cost=23., current_fit_pending=True,
                   tire_warmup={"intermediate": 4., "wet": 5., "hard": 2.})
    wait, pit, selected = enumerate_sets(models, stock, clock, options)
    actual = plan_inventory_strategy(*models, stock, 1, weather_clock=clock, **options)
    assert actual.wait_cost == pytest.approx(wait, abs=1.e-8)
    assert actual.pit_now_cost == pytest.approx(pit, abs=1.e-8)
    assert actual.set_id == selected


def test_identical_wear_with_distinct_allowances_is_not_interchangeable():
    models = fixture()
    stock = TireInventory.from_sets([
        {"id": "old", "compound": "hard", "remaining_laps": 1},
        {"id": "short", "compound": "soft", "remaining_laps": 1},
        {"id": "long", "compound": "soft", "remaining_laps": 4},
    ])
    stock.fit("old")
    result = plan_inventory_strategy(*models, stock, 1, force_stop=True,
                                     remaining_stops=0, used_compounds={TireCompound.HARD})
    assert result.set_id == "long" and result.pit_now_cost < inf


def test_expired_active_set_and_insufficient_stock_cannot_run_phantom_laps():
    models = fixture()
    stock = TireInventory.from_sets([
        {"compound": "hard", "remaining_laps": 1},
        {"compound": "soft", "remaining_laps": 3},
    ])
    stock.fit("set-1")
    result = plan_inventory_strategy(*models, stock, 1, tire_age=1,
                                     remaining_stops=0, require_compound_rule=False)
    assert result.wait_cost == result.pit_now_cost == inf
    assert result.set_id is None


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("plan", [[], [{"lap": 4, "compound": "hard"},
                                     {"lap": 7, "compound": "soft"}]])
def test_custom_replacement_forecast_preserves_future_compulsory_expiry(free, plan):
    sim, state, track = custom_inputs(True, plan)
    stock = state.tire_inventory
    credits = {"M": 7, "S": 2, "H-used": 2, "H-fresh": 3, "I": 0, "W": 0}
    stock.sets = {key: replace(item, remaining_laps=credits[key])
                  for key, item in stock.sets.items()}
    from f1sim.models import Weather

    weather = Weather()
    expected = execution_costs(sim, deepcopy(state), track, weather, 3,
                               free_fit=free, physical_total_laps=40)
    choice = sim._custom_plan_replacement_choice(
        state, track, weather, 3, free_fit=free, physical_total_laps=40)
    best = min(expected.values())
    assert choice.cost == pytest.approx(best[1], abs=1.e-8)
    assert choice.instructions == -best[0]
    assert expected[choice.set_id] == pytest.approx(best, abs=1.e-8)
