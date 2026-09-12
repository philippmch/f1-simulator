"""Finite-stock plans agree with independently enumerated physical-set schedules."""

from copy import deepcopy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.surface_projection import projected_surfaces
from f1sim.simulation.tire_inventory import TireInventory


def fixture(wetness=0, rain=0):
    return (Driver(id="a", name="A", team_id="t"), Car(team_id="t", team_name="T"),
            Track(id="t", name="T", country="T", total_laps=4, base_lap_time=90,
                  pit_lane_delta=3), Weather(track_wetness=wetness, rain_intensity=rain))


def exhaustive(args, inventory, **options):
    driver, car, track, weather = deepcopy(args)
    surfaces = tuple(projected_surfaces(weather, track.total_laps,
                                       options.get("weather_intervals")))
    physics = LapSimulator(np.random.default_rng(10))
    sets = inventory.sets
    ids = [key for key in sets if key not in inventory.unavailable_ids]
    initial = inventory.current_set_id
    ages = {key: item.age for key, item in sets.items()}
    if initial:
        ages[initial] = options.get("tire_age", 0)
    best = {False: inf, True: inf}
    selected = None
    require = options.get("require_compound_rule", True)
    free = options.get("free_fit", False)
    used = set(options.get("used_compounds", ()))

    def legal(compounds):
        return (not require or bool(compounds & {TireCompound.WET, TireCompound.INTERMEDIATE})
                or len(compounds) >= 2)

    def visit(offset, current, wear, compounds, left, dry, damp, total, first_id, first_stop):
        nonlocal selected
        if offset == track.total_laps:
            if legal(compounds) and total < best[first_stop]:
                best[first_stop] = total
                if first_stop:
                    selected = first_id
            return
        surface = surfaces[offset]
        for target in ids:
            changing = target != current
            root_free = offset == 0 and free
            if surface.tire_mismatch(sets[target].compound) == "critical":
                continue
            if offset == 0 and options.get("force_stop", False) and not changing and not free:
                continue
            if changing and not root_free and current in ids:
                old = sets[current].compound
                limit = dry if (surface.track_wetness < .08
                                and surface.rain_intensity < .15) else damp
                elective = left > 0 and (old in (TireCompound.WET, TireCompound.INTERMEDIATE)
                            or surface.track_wetness > .3 or limit is None or limit > 0)
                correction = not legal(compounds) and sets[target].compound not in compounds
                forced = surface.tire_mismatch(old) == "critical" or (
                    offset == 0 and options.get("force_stop", False))
                if not (elective or correction or forced):
                    continue
            paid = changing and not root_free
            next_wear = wear.copy()
            age = next_wear[target]
            driver.current_tire_laps = age
            gaps = options.get("current_traffic_gaps")
            lap_cost = physics.calculate_lap_time(
                driver, car, track, TIRE_COMPOUNDS[sets[target].compound], surface,
                offset + 1, options.get("physical_total_laps", track.total_laps),
                sample_variation=False,
                active_aero_enabled=(options.get("active_aero_enabled", True)
                                     if offset == 0 else True),
                gap_to_car_ahead=gaps[int(changing and not free)] if offset == 0 and gaps else None,
            )
            if offset == 0:
                lap_cost *= options.get("current_lap_time_modifier", 1.)
            if paid:
                lap_cost += expected_stationary_time(car) + track.pit_lane_delta * (
                    options.get("pit_lane_factor", 1.) if offset == 0 else 1.)
                if offset == 0:
                    lap_cost += options.get("additional_current_stop_cost", 0.)
            next_wear[target] += 1
            visit(offset + 1, target, next_wear, compounds | {sets[target].compound},
                  max(0, left - int(paid)),
                  None if dry is None else max(0, dry - int(paid)),
                  None if damp is None else max(0, damp - int(paid)), total + lap_cost,
                  target if offset == 0 else first_id,
                  (changing or free) if offset == 0 else first_stop)

    visit(0, initial, ages, used, options.get("remaining_stops", 3),
          options.get("remaining_dry_stops"), options.get("remaining_damp_stops"), 0., None, False)
    return best, selected


@pytest.mark.parametrize("wetness,rain", [(0, 0), (.1, .1), (.3, 0), (.7, .7)])
@pytest.mark.parametrize("free,budget", [(False, 0), (False, 2), (True, 1)])
def test_matches_exhaustive_reusable_set_schedules(wetness, rain, free, budget):
    args = fixture(wetness, rain)
    pool = TireInventory.from_sets([{"compound": c, "age": age} for c, age in
                                   [("soft", 8), ("hard", 2), ("intermediate", 3), ("wet", 0)]])
    pool.fit("set-1")
    options = dict(tire_age=9, free_fit=free, remaining_stops=budget,
                   remaining_dry_stops=1, remaining_damp_stops=1,
                   weather_intervals=(0, 1, 3, 5), physical_total_laps=10,
                   current_traffic_gaps=(.2, 1.2), current_lap_time_modifier=1.4,
                   active_aero_enabled=False)
    before = deepcopy((args, pool.__dict__))
    expected, _ = exhaustive(args, pool, **options)
    result = plan_inventory_strategy(*args, pool, 1, **options)
    assert result.pit_now_cost == pytest.approx(expected[True])
    if not free:
        assert result.wait_cost == pytest.approx(expected[False])
    assert (args, pool.__dict__) == before


def test_no_phantom_rule_credit_or_freshening_and_free_current_tie():
    args = fixture()
    pool = TireInventory.from_sets([{"compound": "soft", "age": 20}])
    pool.fit("set-1")
    result = plan_inventory_strategy(*args, pool, 1, tire_age=20)
    assert result.wait_cost == result.pit_now_cost == inf
    assert result.set_id is None
    free = plan_inventory_strategy(*args, pool, 1, tire_age=20, free_fit=True,
                                   require_compound_rule=False)
    assert free.set_id == "set-1" and free.wait_cost == free.pit_now_cost
    assert not free.should_pit()


def test_damaged_current_never_returns_and_forced_fit_ignores_budget():
    args = fixture()
    pool = TireInventory.from_sets([{"compound": "soft"}, {"compound": "hard"}])
    pool.fit("set-1")
    pool.mark_current_unavailable(7)
    result = plan_inventory_strategy(*args, pool, 1, tire_age=7, force_stop=True,
                                     remaining_stops=0, used_compounds=(TireCompound.SOFT,))
    assert result.wait_cost == inf
    assert result.set_id == "set-2"
    assert result.should_pit()


def test_identical_sets_keep_first_input_identity_and_opening_is_free():
    args = fixture()
    pool = TireInventory.from_sets([{"compound": "hard"}, {"compound": "hard"}])
    result = plan_inventory_strategy(*args, pool, 1, free_fit=True, require_compound_rule=False)
    assert result.set_id == "set-1"
    assert result.pit_now_cost < inf


def test_optimal_return_to_removed_set_keeps_accumulated_wear(monkeypatch):
    args = fixture()
    args[2].total_laps = 3
    pool = TireInventory.from_sets([{"compound": "soft", "age": 5}, {"compound": "hard"}])
    pool.fit("set-1")

    def running(self, driver, car, track, tire, weather, lap, *args, **kwargs):
        preferred = TireCompound.HARD if lap == 2 else TireCompound.SOFT
        return 100. + driver.current_tire_laps if tire.compound == preferred else 1000.

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", running)
    expected, _ = exhaustive(args, pool, tire_age=5, remaining_stops=2)
    result = plan_inventory_strategy(*args, pool, 1, tire_age=5, remaining_stops=2)
    assert result.wait_cost == pytest.approx(expected[False])
    assert result.wait_cost == pytest.approx(
        105 + 100 + 106 + 2 * (3 + expected_stationary_time(args[1])))


def test_exhausted_budget_skips_relaxation_table(monkeypatch):
    args = fixture()
    args[2].total_laps = 30
    pool = TireInventory.from_sets([{"compound": "soft"}, {"compound": "hard"}])
    pool.fit("set-1")
    calls = []
    original = LapSimulator.calculate_lap_time

    def count(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", count)
    result = plan_inventory_strategy(*args, pool, 1, remaining_stops=0,
                                     used_compounds=(TireCompound.SOFT, TireCompound.HARD))
    assert result.wait_cost < inf
    assert result.pit_now_cost == inf
    # Exactly the retained thirty laps; no hypothetical age/compound table.
    assert len(calls) == 30


@pytest.mark.parametrize("wetness,rain", [(0., 0.), (.18, .35), (.3, 0.)])
def test_exhausted_worn_set_tails_match_physical_schedule_oracle(wetness, rain):
    args = fixture(wetness, rain)
    args[2].total_laps = 6
    pool = TireInventory.from_sets([
        {"compound": "soft", "age": 18},
        {"compound": "hard", "age": 44},
        {"compound": "intermediate", "age": 34},
    ])
    pool.fit("set-1")
    options = dict(tire_age=18, remaining_stops=2,
                   remaining_dry_stops=2, remaining_damp_stops=2,
                   used_compounds=(TireCompound.SOFT, TireCompound.HARD))
    # Different fitted/removed-set histories exhaust the budget at different
    # physical ages around each cliff; a shared final-stint cost must retain
    # that age and the projected surface at every remaining lap.
    expected, _ = exhaustive(args, pool, **options)
    result = plan_inventory_strategy(*args, pool, 1, **options)
    assert result.wait_cost == pytest.approx(expected[False], abs=1e-9)
    assert result.pit_now_cost == pytest.approx(expected[True], abs=1e-9)
