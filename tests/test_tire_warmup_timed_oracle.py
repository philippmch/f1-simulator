"""Enumerate small physical-set schedules with post-fit timing costs."""

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


def _enumerate_warmup_schedules(models, inventory, clock, options):
    """Exhaust the physical set choices without using the strategy planner."""
    driver, car, track, weather = deepcopy(models)
    physics = LapSimulator(np.random.default_rng(7))
    sets = inventory.sets
    ids = [key for key in sets if key not in inventory.unavailable_ids]
    initial = inventory.current_set_id
    current_lap = options.get("current_lap", 1)
    horizon = track.total_laps - current_lap + 1
    ages = {key: item.age for key, item in sets.items()}
    ages[initial] = options["tire_age"]
    profile = options["tire_warmup"]
    free = options.get("free_fit", False)
    current_fit_pending = options.get("current_fit_pending", False)
    best = {key: (inf, inf) for key in ids}
    events = [clock.first_update_after + index * clock.update_interval
              for index in range(clock.max_updates)]

    def surface_at(elapsed):
        value = weather
        for event in events:
            if event <= elapsed:
                value = value.project_surface()
        return value

    def legal(used):
        return bool(used & {TireCompound.INTERMEDIATE, TireCompound.WET}) or len(used) >= 2

    def visit(offset, current, wear, used, left, dry, damp, stop_delay,
              fit_delay, total, first, paid_count):
        nonlocal best
        if offset == horizon:
            if legal(used):
                candidate = (total, paid_count)
                if candidate[0] < best[first][0]:
                    best[first] = candidate
            return

        nominal_start = clock.lap_start_offsets[offset]
        # A fitting penalty delays later entries. It never moves the surface
        # seen on the lap that exits the pit.
        entry_time = nominal_start + stop_delay
        if offset:
            entry_time += fit_delay
        before = surface_at(entry_time)
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
            after_stop_delay = stop_delay
            if paid:
                after_stop_delay += (
                    clock.current_stop_delay if offset == 0 else clock.future_stop_delay
                )
            running_time = nominal_start + after_stop_delay
            if offset:
                running_time += fit_delay
            running_surface = surface_at(running_time)
            driver.current_tire_laps = wear[target]
            gaps = options.get("current_traffic_gaps")
            gap = gaps[int(paid)] if offset == 0 and gaps is not None else None
            value = physics.calculate_lap_time(
                driver, car, track, TIRE_COMPOUNDS[compound], running_surface,
                current_lap + offset, options["physical_total_laps"], sample_variation=False,
                gap_to_car_ahead=gap,
                active_aero_enabled=options.get("active_aero_enabled", True)
                if offset == 0 else True,
            )
            if offset == 0:
                value *= options["current_lap_time_modifier"]

            # Cost is an absolute, post-modifier amount added once per actual
            # fitting. The current stop does not shift its pit-exit surface.
            fit_cost = profile.get(compound.value, 0.0) if changing else 0.0
            if offset == 0 and current_fit_pending and not changing:
                fit_cost = profile.get(compound.value, 0.0)
            value += fit_cost
            if paid:
                value += expected_stationary_time(car) + track.pit_lane_delta * (
                    options["pit_lane_factor"] if offset == 0 else 1
                )
                if offset == 0:
                    value += options["additional_current_stop_cost"]

            next_wear = wear.copy()
            next_wear[target] += 1
            next_fit_delay = fit_delay + fit_cost
            visit(
                offset + 1, target, next_wear, used | {compound},
                max(0, left - paid), max(0, dry - paid), max(0, damp - paid),
                after_stop_delay, next_fit_delay, total + value,
                target if offset == 0 else first, paid_count + paid,
            )

    visit(
        0, initial, ages, set(options["used_compounds"]),
        options["remaining_stops"], options["remaining_dry_stops"],
        options["remaining_damp_stops"], 0.0, 0.0, 0.0, None, 0,
    )
    choices = ([initial] if free else []) + [key for key in ids if key != initial]
    selected = min(choices, key=lambda key: best[key][0])
    if best[selected][0] == inf:
        selected = None
    return best[initial][0], min(best[key][0] for key in choices), selected, best


def _models(total_laps=3):
    driver = Driver(id="d", name="D", team_id="t")
    car = Car(team_id="t", team_name="T", pit_stop_avg=1.5, pit_stop_std=.1)
    track = Track(
        id="t", name="T", country="T", total_laps=total_laps,
        base_lap_time=90, pit_lane_delta=3,
    )
    weather = Weather(track_wetness=.12, rain_intensity=0)
    return driver, car, track, weather


def _inventory(records, current="wet"):
    inventory = TireInventory.from_sets(records)
    inventory.fit(current)
    return inventory


def _clock(car, track):
    service = expected_stationary_time(car)
    return StrategyWeatherClock(
        (0, 90, 180), 50, 80, 2, 20, service + track.pit_lane_delta,
    )


def _options(inventory, profile, *, budget=1):
    return dict(
        tire_age=0, remaining_stops=budget, remaining_dry_stops=1,
        remaining_damp_stops=budget, used_compounds=(inventory.sets["wet"].compound,),
        physical_total_laps=10, current_traffic_gaps=(None, None),
        current_lap_time_modifier=1.2, active_aero_enabled=False,
        pit_lane_factor=1., additional_current_stop_cost=0.,
        tire_warmup=profile,
    )


def test_fit_fee_shifts_future_surface_but_not_current_pit_exit():
    driver, car, track, weather = _models()
    inventory = _inventory([
        {"id": "wet", "compound": "wet", "age": 0},
        {"id": "soft", "compound": "soft", "age": 60},
    ])
    clock = _clock(car, track)
    profile = {"soft": 30.0}
    options = _options(inventory, profile)

    # The root fitting fee is paid after the pit-exit surface is selected.
    assert clock.updates(0, 1, True, fit_delay=30) == 0
    assert clock.updates(1, 1, True) == 1
    assert clock.updates(1, 1, True, fit_delay=30) == 2
    assert weather.project_surface().track_wetness == pytest.approx(.09)
    assert weather.project_surface().project_surface().track_wetness == pytest.approx(.06)

    expected_wait, expected_pit, expected_id, _ = _enumerate_warmup_schedules(
        (driver, car, track, weather), inventory, clock, options,
    )
    actual = plan_inventory_strategy(
        driver, car, track, weather, inventory, 1, weather_clock=clock, **options,
    )
    assert actual.wait_cost == pytest.approx(expected_wait)
    assert actual.pit_now_cost == pytest.approx(expected_pit)
    assert actual.set_id == expected_id == "soft"


def test_fee_shifted_intermediate_boundary_matches_multistop_inventory_oracle():
    driver, car, track, weather = _models()
    inventory = _inventory([
        {"id": "wet", "compound": "wet", "age": 0},
        {"id": "inter", "compound": "intermediate", "age": 0},
        {"id": "soft", "compound": "soft", "age": 1000},
        {"id": "hard", "compound": "hard", "age": 1000},
    ])
    clock = _clock(car, track)
    profile = {"intermediate": 30.0}
    options = _options(inventory, profile, budget=2)

    # Without the fee the intermediate remains safe at .09. The fee delays
    # the next entry to .06, where a second physical fit is required.
    no_fee_surface = weather.project_surface()
    with_fee_surface = no_fee_surface.project_surface()
    assert no_fee_surface.track_wetness == pytest.approx(.09)
    assert with_fee_surface.track_wetness == pytest.approx(.06)
    assert no_fee_surface.tire_mismatch(TireCompound.INTERMEDIATE) != "critical"
    assert with_fee_surface.tire_mismatch(TireCompound.INTERMEDIATE) == "critical"

    expected_wait, expected_pit, expected_id, candidates = _enumerate_warmup_schedules(
        (driver, car, track, weather), inventory, clock, options,
    )
    actual = plan_inventory_strategy(
        driver, car, track, weather, inventory, 1, weather_clock=clock, **options,
    )
    assert actual.wait_cost == pytest.approx(expected_wait)
    assert actual.pit_now_cost == pytest.approx(expected_pit)
    assert actual.set_id == expected_id == "inter"
    assert candidates["inter"][1] == 2
