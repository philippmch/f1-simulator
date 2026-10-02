"""Completion pruning preserves independently enumerated physical schedules."""

from copy import deepcopy
from math import inf, ulp
from random import Random

import pytest
from test_inventory_strategy import exhaustive, fixture
from test_timed_inventory_weather_oracle import enumerate_sets

from f1sim.models import TireCompound
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory

USED_COMPOUNDS = (
    (), (TireCompound.SOFT,), (TireCompound.SOFT, TireCompound.HARD),
    (TireCompound.SOFT, TireCompound.MEDIUM), (TireCompound.INTERMEDIATE,),
    (TireCompound.SOFT, TireCompound.HARD, TireCompound.INTERMEDIATE),
)


@pytest.mark.parametrize("used", USED_COMPOUNDS)
@pytest.mark.parametrize("weather", [(0., 0.), (.3, 0.), (.8, .8)])
@pytest.mark.parametrize("budgets", [(12, None, None), (3, 2, 1), (0, 12, None)])
def test_own_lap_paths_match_physical_schedule_oracle(used, weather, budgets):
    args = fixture(*weather)
    records = [dict(id="S", compound="soft", age=18), dict(id="H", compound="hard", age=44),
               dict(id="I", compound="intermediate", age=34),
               dict(id="I-twin", compound="intermediate", age=34)]
    pool = TireInventory.from_sets(records)
    pool.fit("I")
    options = dict(tire_age=35, used_compounds=used, remaining_stops=budgets[0],
                   remaining_dry_stops=budgets[1], remaining_damp_stops=budgets[2],
                   weather_intervals=(0, 1, 4, 9), physical_total_laps=10,
                   current_lap_time_modifier=1.4, active_aero_enabled=False,
                   current_traffic_gaps=(.2, 1.2))
    before = deepcopy((args, pool.__dict__))
    expected, _ = exhaustive(args, pool, **options)
    decision = plan_inventory_strategy(*args, pool, 1, **options)
    assert decision.wait_cost == pytest.approx(expected[False], rel=0, abs=1.e-9)
    assert decision.pit_now_cost == pytest.approx(expected[True], rel=0, abs=1.e-9)
    assert (args, pool.__dict__) == before


@pytest.mark.parametrize("free,require", [(False, True), (True, True), (True, False)])
def test_currently_critical_stock_is_retained_for_later_safe_surfaces(free, require):
    args = fixture(.9, 0.)
    args[2].total_laps = 6
    pool = TireInventory.from_sets([dict(id="S", compound="soft", age=5),
                                    dict(id="H", compound="hard", age=0),
                                    dict(id="I", compound="intermediate", age=0)])
    pool.fit("I")
    options = dict(tire_age=2, free_fit=free, require_compound_rule=require,
                   remaining_stops=0, remaining_dry_stops=0, remaining_damp_stops=0,
                   weather_intervals=(0, 12, 14, 20, 26, 30))
    before = deepcopy((args, pool.__dict__))
    expected, _ = exhaustive(args, pool, **options)
    decision = plan_inventory_strategy(*args, pool, 1, **options)
    assert decision.wait_cost < inf
    assert decision.wait_cost == pytest.approx(expected[False] if not free else expected[True])
    assert decision.pit_now_cost == pytest.approx(expected[True])
    assert (args, pool.__dict__) == before


@pytest.mark.parametrize("used", USED_COMPOUNDS)
@pytest.mark.parametrize("free,budgets", [(False, (12, 12, 12)), (True, (3, 2, 1)),
                                       (False, (0, 4, 2))])
def test_delayed_weather_paths_match_external_event_oracle(used, free, budgets):
    args = fixture(.3, 0.)
    pool = TireInventory.from_sets([dict(id="S", compound="soft", age=5),
                                    dict(id="H", compound="hard", age=44),
                                    dict(id="I", compound="intermediate", age=34)])
    pool.fit("I")
    clock = StrategyWeatherClock((0., 80., 160., 240.), 10., 20., 18, 45., 15.)
    options = dict(tire_age=35, used_compounds=used, free_fit=free,
                   remaining_stops=budgets[0], remaining_dry_stops=budgets[1],
                   remaining_damp_stops=budgets[2], physical_total_laps=10,
                   current_lap_time_modifier=1.4, active_aero_enabled=False,
                   current_traffic_gaps=(.2, 1.2), pit_lane_factor=.5,
                   additional_current_stop_cost=2.)
    before = deepcopy((args, pool.__dict__, clock))
    wait, pit, _ = enumerate_sets(args, pool, clock, options)
    decision = plan_inventory_strategy(*args, pool, 1, weather_clock=clock, **options)
    assert decision.wait_cost == pytest.approx(wait, rel=0, abs=1.e-9)
    assert decision.pit_now_cost == pytest.approx(pit, rel=0, abs=1.e-9)
    assert (args, pool.__dict__, clock) == before


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("scale", [1.e-9, 1., 1.e12])
def test_pruning_keeps_near_tied_and_nonmonotone_physical_optima(monkeypatch, seed, scale):
    args = fixture(*[(0., 0.), (.18, .35), (.3, 0.)][seed % 3])
    args[2].total_laps = 5
    records = [dict(id="S", compound="soft", age=18), dict(id="H", compound="hard", age=44),
               dict(id="I", compound="intermediate", age=34)]
    pool = TireInventory.from_sets(records)
    pool.fit("S")
    rng = Random(seed)
    # Tiny differences exercise downward rounding; arbitrary age curves exercise
    # the first-service bound without assuming that a fresh set is always faster.
    costs = {(lap, row["compound"], row["age"] + elapsed):
             (90. * scale + rng.randrange(16) * ulp(90. * scale)
              if seed < 6 else scale * max(60., rng.uniform(40., 130.)))
             for row in records for elapsed in range(5) for lap in range(1, 6)}

    def running(self, driver, car, track, tire, weather, lap, *args, **kwargs):
        return costs[lap, tire.compound.value, driver.current_tire_laps]

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", running)
    options = dict(tire_age=18, used_compounds=USED_COMPOUNDS[seed % len(USED_COMPOUNDS)],
                   remaining_stops=seed % 3, remaining_dry_stops=seed % 2,
                   remaining_damp_stops=2, free_fit=seed % 2 == 0,
                   force_stop=seed % 4 == 0, require_compound_rule=seed % 3 != 0,
                   weather_intervals=(0, 1, 4, 6, 8))
    before = deepcopy((args, pool.__dict__))
    expected, _ = exhaustive(args, pool, **options)
    retained, _ = exhaustive(args, pool, **{**options, "free_fit": False, "force_stop": False})
    decision = plan_inventory_strategy(*args, pool, 1, **options)
    for actual, reference in ((decision.pit_now_cost, expected[True]),
                              (decision.wait_cost, retained[False] if options["free_fit"]
                               else expected[False])):
        if reference == inf:
            assert actual == inf
        else:
            assert actual == pytest.approx(reference, rel=0., abs=32 * ulp(reference))
    assert (args, pool.__dict__) == before


