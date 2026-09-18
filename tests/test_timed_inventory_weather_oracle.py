"""Enumerate physical set schedules against explicit external weather events."""

from copy import deepcopy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory


def enumerate_sets(models, inventory, clock, options):
    driver, car, track, weather = deepcopy(models)
    physics = LapSimulator(np.random.default_rng(7))
    sets = inventory.sets
    ids = [key for key in sets if key not in inventory.unavailable_ids]
    initial = inventory.current_set_id
    ages = {key: item.age for key, item in sets.items()}
    ages[initial] = options["tire_age"]
    free = options["free_fit"]
    best = {key: inf for key in ids}
    events = [clock.first_update_after + index * clock.update_interval
              for index in range(clock.max_updates)]

    def surface_at(time):
        value = weather
        for event in events:
            if event <= time:
                value = value.project_surface()
        return value

    def legal(used):
        return bool(used & {TireCompound.INTERMEDIATE, TireCompound.WET}) or len(used) >= 2

    def visit(offset, current, wear, used, left, dry, damp, delay, total, first):
        if offset == track.total_laps:
            if legal(used):
                best[first] = min(best[first], total)
            return
        before = surface_at(clock.lap_start_offsets[offset] + delay)
        for target in ids:
            changing = target != current
            root_free = offset == 0 and free
            old = sets[current].compound
            compound = sets[target].compound
            if before.tire_mismatch(compound) == "critical":
                continue
            if changing and not root_free:
                limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
                elective = left > 0 and (
                    old in (TireCompound.INTERMEDIATE, TireCompound.WET)
                    or before.track_wetness > .3 or limit > 0
                )
                mandatory = before.tire_mismatch(old) == "critical"
                correction = not legal(used) and compound not in used
                if not (elective or mandatory or correction):
                    continue
            paid = changing and not root_free
            after_delay = delay
            if paid:
                after_delay += clock.current_stop_delay if offset == 0 else clock.future_stop_delay
            running_surface = surface_at(clock.lap_start_offsets[offset] + after_delay)
            driver.current_tire_laps = wear[target]
            gap = options["current_traffic_gaps"][int(paid)] if offset == 0 else None
            value = physics.calculate_lap_time(
                driver, car, track, TIRE_COMPOUNDS[compound], running_surface,
                offset + 1, options["physical_total_laps"], sample_variation=False,
                gap_to_car_ahead=gap,
                active_aero_enabled=options["active_aero_enabled"] if offset == 0 else True,
            )
            if offset == 0:
                value *= options["current_lap_time_modifier"]
            if paid:
                value += expected_stationary_time(car) + track.pit_lane_delta * (
                    options["pit_lane_factor"] if offset == 0 else 1
                )
                if offset == 0:
                    value += options["additional_current_stop_cost"]
            next_wear = wear.copy()
            next_wear[target] += 1
            visit(offset + 1, target, next_wear, used | {compound},
                  max(0, left - paid), max(0, dry - paid), max(0, damp - paid),
                  after_delay, total + value, target if offset == 0 else first)

    visit(0, initial, ages, set(options["used_compounds"]), options["remaining_stops"],
          options["remaining_dry_stops"], options["remaining_damp_stops"], 0, 0, None)
    choices = ([initial] if free else []) + [key for key in ids if key != initial]
    selected = min(choices, key=best.__getitem__)
    if best[selected] == inf:
        selected = None
    return best[initial], min(best[key] for key in choices), selected


@pytest.mark.parametrize("wetness,rain,current", [
    (.1, .75, "set-1"), (.25, 0, "set-4"),
    (.12, 0, "set-3"), (.44, .6, "set-3"),
])
@pytest.mark.parametrize("free,budget", [(False, 0), (False, 2), (True, 1)])
@pytest.mark.parametrize("update_cap,queue_delay,period", [
    (4, 25, 90), (1, 0, 90), (2, 220, 90), (12, 25, 30),
])
def test_physical_schedules_match_timed_inventory_costs(
    wetness, rain, current, free, budget, update_cap, queue_delay, period,
):
    models = (
        Driver(id="d", name="D", team_id="t"), Car(team_id="t", team_name="T"),
        Track(id="t", name="T", country="T", total_laps=4,
              base_lap_time=90, pit_lane_delta=3),
        Weather(track_wetness=wetness, rain_intensity=rain),
    )
    inventory = TireInventory.from_sets([
        {"compound": compound, "age": age}
        for compound, age in [("soft", 8), ("hard", 2), ("intermediate", 3), ("wet", 0)]
    ])
    inventory.fit(current)
    service = expected_stationary_time(models[1])
    clock = StrategyWeatherClock(
        (0, 108, 198, 288), 12, period, update_cap,
        service + 3 * .75 + queue_delay, service + 3,
    )
    options = dict(
        tire_age=19, free_fit=free, remaining_stops=budget,
        remaining_dry_stops=1, remaining_damp_stops=1,
        used_compounds=(inventory.sets[current].compound,),
        physical_total_laps=10, current_traffic_gaps=(.2, 1.2),
        current_lap_time_modifier=1.2, active_aero_enabled=False,
        pit_lane_factor=.75, additional_current_stop_cost=23,
    )
    before = deepcopy((models, inventory.__dict__))
    wait, pit, selected = enumerate_sets(models, inventory, clock, options)
    actual = plan_inventory_strategy(*models, inventory, 1, weather_clock=clock, **options)
    assert actual.wait_cost == pytest.approx(wait)
    assert actual.pit_now_cost == pytest.approx(pit)
    assert actual.set_id == selected
    assert (models, inventory.__dict__) == before
