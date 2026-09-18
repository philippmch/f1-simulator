"""Independent small-schedule checks for externally timed strategy weather."""

from copy import deepcopy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_strategy import weather_stop_costs


def models(laps=4, lane=10):
    return (
        Driver(id="D", name="D", team_id="T"),
        Car(team_id="T", team_name="T"),
        Track(id="T", name="T", country="T", total_laps=laps,
              base_lap_time=90, pit_lane_delta=lane),
        Weather(track_wetness=.44, rain_intensity=.6),
    )


def clock_for(track, horizon=None, *, current=None, future=None):
    horizon = track.total_laps if horizon is None else horizon
    physical = track.pit_lane_delta + expected_stationary_time(Car(team_id="T", team_name="T"))
    return StrategyWeatherClock(
        tuple(170.0 * index for index in range(horizon)),
        10.0,
        90.0,
        6,
        physical if current is None else current,
        physical if future is None else future,
    )


def event_surface(weather, clock, offset, paid, stopped_first):
    elapsed = clock.lap_start_offsets[offset] + paid * clock.future_stop_delay
    if stopped_first:
        elapsed += clock.current_stop_delay - clock.future_stop_delay
    count = min(
        clock.max_updates,
        sum(
            clock.first_update_after + index * clock.update_interval
            <= elapsed + 1.0e-12
            for index in range(clock.max_updates)
        ),
    )
    surface = weather
    for _ in range(count):
        surface = surface.project_surface()
    return surface


def exhaustive_same(driver, car, track, weather, tire, age, current_lap, budget, clock,
                    *, queue=0.0, modifier=1.0, aero=True):
    simulator = LapSimulator(np.random.default_rng(7))
    horizon = track.total_laps - current_lap + 1
    values = [inf, inf]

    def visit(offset, fitted, tire_age, paid, total, first_stop):
        if offset == horizon:
            values[int(first_stop)] = min(values[int(first_stop)], total)
            return
        before = event_surface(weather, clock, offset, paid, first_stop)
        driver.current_tire_laps = tire_age
        running = simulator.calculate_lap_time(
            driver, car, track, fitted, before, current_lap + offset, track.total_laps,
            active_aero_enabled=aero if offset == 0 else True,
            sample_variation=False,
        ) * (modifier if offset == 0 else 1.0)
        visit(offset + 1, fitted, tire_age + 1, paid, total + running, first_stop)
        if paid >= budget:
            return
        after = event_surface(weather, clock, offset, paid + 1,
                              first_stop or offset == 0)
        fresh = TIRE_COMPOUNDS[tire.compound]
        # The stop runs the replacement on the post-delay surface.
        driver.current_tire_laps = 0
        fresh_running = simulator.calculate_lap_time(
            driver, car, track, fresh, after, current_lap + offset, track.total_laps,
            active_aero_enabled=aero if offset == 0 else True,
            sample_variation=False,
        ) * (modifier if offset == 0 else 1.0)
        stop = (track.pit_lane_delta + expected_stationary_time(car)
                + (queue if offset == 0 else 0.0) + fresh_running)
        visit(offset + 1, fresh, 1, paid + 1, total + stop,
              first_stop or offset == 0)

    visit(0, tire, age, 0, 0.0, False)
    return values[1], values[0]


def exhaustive_transition(driver, car, track, weather, tire, age, current_lap,
                          budget, clock, *, lane=1.0, queue=0.0, modifier=1.0,
                          aero=True, used=()):
    simulator = LapSimulator(np.random.default_rng(8))
    horizon = track.total_laps - current_lap + 1
    bits = {compound: (1 << index if index < 3 else 8)
            for index, compound in enumerate(TireCompound)}
    used_mask = 0
    for compound in used:
        used_mask |= bits[compound]
    minima = [inf, inf]
    selected = None

    def legal(mask):
        return bool(mask & 8) or (mask & 7).bit_count() >= 2

    def visit(offset, fitted, tire_age, left, dry, damp, mask, paid, stopped,
              total, first_compound):
        nonlocal selected
        if offset == horizon:
            if not legal(mask):
                return
            index = int(stopped)
            if total < minima[index]:
                minima[index] = total
                if stopped:
                    selected = first_compound
            return
        before = event_surface(weather, clock, offset, paid, stopped)
        current = fitted.compound
        critical = before.tire_mismatch(current) == "critical"
        if not critical:
            driver.current_tire_laps = tire_age
            running = simulator.calculate_lap_time(
                driver, car, track, fitted, before, current_lap + offset,
                track.total_laps, active_aero_enabled=aero if offset == 0 else True,
                sample_variation=False,
            ) * (modifier if offset == 0 else 1.0)
            visit(offset + 1, fitted, tire_age + 1, left, dry, damp,
                  mask | bits[current], paid, stopped, total + running,
                  first_compound)
        limit = dry if before.track_wetness < .08 and before.rain_intensity < .15 else damp
        allowed = critical or (left > 0 and (
            current in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or before.track_wetness > .3 or limit is None or limit > 0
        ))
        candidates = ((before.fresh_rain_compound(),) if before.fresh_rain_compound()
                      else (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD))
        if not (allowed or not legal(mask)):
            return
        for candidate in candidates:
            if before.tire_mismatch(candidate) == "critical":
                continue
            if not allowed and (legal(mask) or mask & bits[candidate]):
                continue
            after = event_surface(weather, clock, offset, paid + 1,
                                  stopped or offset == 0)
            driver.current_tire_laps = 0
            running = simulator.calculate_lap_time(
                driver, car, track, TIRE_COMPOUNDS[candidate], after,
                current_lap + offset, track.total_laps,
                active_aero_enabled=aero if offset == 0 else True,
                sample_variation=False,
            ) * (modifier if offset == 0 else 1.0)
            stop = track.pit_lane_delta * (lane if offset == 0 else 1.0) \
                + expected_stationary_time(car) + (queue if offset == 0 else 0)
            visit(offset + 1, TIRE_COMPOUNDS[candidate], 1,
                  max(0, left - 1),
                  None if dry is None else max(0, dry - 1),
                  None if damp is None else max(0, damp - 1),
                  mask | bits[candidate], paid + 1, stopped or offset == 0,
                  total + stop + running,
                  candidate if offset == 0 else first_compound)

    visit(0, tire, age, budget, None, None, used_mask, 0, False, 0.0, None)
    return minima[1], minima[0], selected


def exhaustive_weather_bound(driver, car, track, weather, tire, age, current_lap,
                             clock, *, modifier=1.0, aero=True,
                             traffic_possible=True, lane=1.0, queue=0.0):
    simulator = LapSimulator(np.random.default_rng(9))
    horizon = track.total_laps - current_lap + 1
    traffic_gap = 0.0 if traffic_possible else None

    def run(offset, fitted, tire_age, surface, *, retained=False, first=False):
        driver.current_tire_laps = tire_age
        value = simulator.calculate_lap_time(
            driver, car, track, fitted, surface, current_lap + offset,
            track.total_laps, gap_to_car_ahead=traffic_gap if retained else None,
            active_aero_enabled=aero if offset == 0 else True,
            sample_variation=False,
        )
        return value * modifier if offset == 0 else value

    def future(offset, fitted, tire_age, paid, stopped):
        if offset == horizon:
            return 0.0
        before = event_surface(weather, clock, offset, paid, stopped)
        best = inf
        if before.tire_mismatch(fitted.compound) != "critical":
            best = run(offset, fitted, tire_age, before) + future(
                offset + 1, fitted, tire_age + 1, paid, stopped,
            )
        for candidate in TireCompound:
            if before.tire_mismatch(candidate) == "critical":
                continue
            after = event_surface(weather, clock, offset, paid + 1, stopped)
            cost = track.pit_lane_delta + expected_stationary_time(car)
            cost += run(offset, TIRE_COMPOUNDS[candidate], 0, after)
            cost += future(offset + 1, TIRE_COMPOUNDS[candidate], 1,
                           paid + 1, stopped)
            best = min(best, cost)
        return best

    def retained(offset, fitted, tire_age, paid, stopped):
        if offset == horizon:
            return 0.0
        before = event_surface(weather, clock, offset, paid, stopped)
        if before.tire_mismatch(fitted.compound) != "critical":
            return run(offset, fitted, tire_age, before, retained=True) + retained(
                offset + 1, fitted, tire_age + 1, paid, stopped,
            )
        required = before.fresh_rain_compound()
        candidates = ((required,) if required is not None else tuple(
            compound for compound in TireCompound
            if compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
            and before.tire_mismatch(compound) != "critical"
        ))
        return min((
            track.pit_lane_delta + expected_stationary_time(car)
            + run(offset, TIRE_COMPOUNDS[candidate], 0,
                  event_surface(weather, clock, offset, paid + 1, stopped))
            + retained(offset + 1, TIRE_COMPOUNDS[candidate], 1,
                       paid + 1, stopped)
            for candidate in candidates
        ), default=inf)

    first = event_surface(weather, clock, 0, 0, False)
    pit_candidates = ((first.fresh_rain_compound(),) if first.fresh_rain_compound()
                      else tuple(compound for compound in (
                          TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
                      ) if first.tire_mismatch(compound) != "critical"))
    pit = min((
        track.pit_lane_delta * lane + expected_stationary_time(car) + queue
        + run(0, TIRE_COMPOUNDS[candidate], 0,
              event_surface(weather, clock, 0, 1, True), first=True)
        + future(1, TIRE_COMPOUNDS[candidate], 1, 1, True)
        for candidate in pit_candidates
    ), default=inf)
    return pit, retained(0, tire, age, 0, False)


def test_clocked_rain_stop_matches_explicit_timeline_and_reverses_choice():
    driver, car, track, weather = models(laps=5, lane=10)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy()
    delay = track.pit_lane_delta + expected_stationary_time(car)
    clock = clock_for(track, horizon=4, current=delay, future=delay)
    expected = exhaustive_same(driver, car, track, weather, tire, 39, 2, 2, clock)

    actual = plan_rain_stop(
        driver, car, track, weather, tire, 39, 2, 2, weather_clock=clock,
    )

    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
    assert not actual.should_pit()

    discounted = plan_rain_stop(
        driver, car, track, weather, tire, 39, 2, 2,
        weather_clock=clock, additional_current_stop_cost=-5,
    )
    assert discounted.pit_now_cost == pytest.approx(actual.pit_now_cost - 5)
    assert discounted.wait_cost == pytest.approx(actual.wait_cost)


def test_clocked_planners_preserve_inputs_and_custom_retained_tire():
    driver, car, track, weather = models(laps=4, lane=5)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
        update={"degradation_rate": .08}
    )
    clock = clock_for(track)
    before = deepcopy((driver, car, track, weather, tire))
    oracle_models = deepcopy((driver, car, track, weather, tire))
    expected = exhaustive_same(*oracle_models[:4], oracle_models[4], 3, 1, 2, clock)
    first = plan_rain_stop(driver, car, track, weather, tire, 3, 1, 2,
                           weather_clock=clock)
    second = plan_rain_stop(driver, car, track, weather, tire, 3, 1, 2,
                            weather_clock=clock)
    assert first == second
    assert first.pit_now_cost == pytest.approx(expected[0])
    assert first.wait_cost == pytest.approx(expected[1])
    assert (driver, car, track, weather, tire) == before


def test_clocked_transition_and_weather_bound_are_finite():
    driver, car, track, weather = models(laps=4, lane=5)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy()
    clock = clock_for(track)
    expected = exhaustive_transition(
        driver, car, track, weather, tire, 2, 1, 2, clock,
        lane=.8, queue=-2, modifier=1.2, aero=False,
    )
    transition = plan_rain_transition(
        driver, car, track, weather, tire, 2, 1, 2, weather_clock=clock,
        pit_lane_factor=.8, additional_current_stop_cost=-2,
        current_lap_time_modifier=1.2, active_aero_enabled=False,
    )
    expected_weather = exhaustive_weather_bound(
        driver, car, track, weather, tire, 2, 1, clock,
    )
    costs = weather_stop_costs(
        driver, car, track, weather, tire, 2, 1, weather_clock=clock,
    )
    assert transition.pit_now_cost == pytest.approx(expected[0])
    assert transition.wait_cost == pytest.approx(expected[1])
    assert transition.compound == expected[2]
    assert costs.pit_now_cost == pytest.approx(expected_weather[0])
    assert costs.stay_cost == pytest.approx(expected_weather[1])


def test_clocked_inventory_free_fit_has_no_physical_weather_shift():
    driver, car, track, weather = models(laps=3, lane=8)
    inventory = TireInventory.from_sets([
        {"id": "current", "compound": "intermediate", "age": 4},
        {"id": "soft", "compound": "soft", "age": 0},
        {"id": "medium", "compound": "medium", "age": 2},
    ])
    inventory.current_set_id = "current"
    clock = clock_for(track, current=70, future=70)
    free = plan_inventory_strategy(
        driver, car, track, weather, inventory, 1, tire_age=4,
        free_fit=True, weather_clock=clock,
    )
    paid = plan_inventory_strategy(
        driver, car, track, weather, inventory, 1, tire_age=4,
        free_fit=False, weather_clock=clock,
    )
    assert free.wait_cost < inf
    assert free.pit_now_cost < inf
    assert paid.pit_now_cost < inf
    assert paid.pit_now_cost != free.pit_now_cost


def test_clocked_weather_stop_uses_slick_fallback_before_rain_threshold():
    driver, car, track, _ = models(laps=4, lane=5)
    weather = Weather(track_wetness=.19, rain_intensity=.39)
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy()
    delay = track.pit_lane_delta + expected_stationary_time(car)
    clock = clock_for(track, current=delay, future=delay)

    expected = exhaustive_weather_bound(
        driver, car, track, weather, tire, 3, 1, clock,
    )
    actual = weather_stop_costs(
        driver, car, track, weather, tire, 3, 1, weather_clock=clock,
    )

    # At the decision surface neither rain compound is required.  A paid
    # stop must therefore begin with a slick; the delayed post-stop surface
    # may cross the rain threshold, but cannot retroactively change selection.
    assert weather.fresh_rain_compound() is None
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.stay_cost == pytest.approx(expected[1])
