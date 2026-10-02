"""Small complete stop enumerations on known future rainfall paths."""

from copy import deepcopy
from math import inf

import numpy as np
import pytest
import test_strategy_pit_weather as oracle

from f1sim.models import TireCompound, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.opening_strategy import _policy_path_cost, opening_policy_costs
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import normalize_weather_intervals, projected_surfaces
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext
from f1sim.simulation.weather_strategy import weather_stop_costs


def explicit_surface(weather, entries, count, start=1):
    """Independent prescribed-change dispatch on enumerated external events."""
    surface = weather.model_copy(deep=True)
    by_lap = {entry["lap"]: entry for entry in entries}
    for lap in range(start + 1, start + count + 1):
        if lap in by_lap:
            entry = by_lap[lap]
            surface.rain_intensity = entry["rain_intensity"]
        surface = surface.project_surface()
    return surface


def events_before(clock, offset, paid, first, fit_delay=0.):
    time = clock.lap_start_offsets[offset] + paid * clock.future_stop_delay + fit_delay
    if first:
        time += clock.current_stop_delay - clock.future_stop_delay
    return sum(clock.first_update_after + i * clock.update_interval <= time + 1e-12
               for i in range(clock.max_updates))


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("entries", [
    [{"lap": 2, "rain_intensity": .9}, {"lap": 4, "rain_intensity": 0}],
    [{"lap": 2, "rain_intensity": 0}, {"lap": 5, "rain_intensity": 1}],
])
@pytest.mark.parametrize("budget", [0, 2])
def test_unlimited_transition_matches_complete_enumeration(monkeypatch, external, entries, budget):
    driver, car, track, _ = oracle.models(laps=4, lane=3)
    weather = Weather(track_wetness=.12, rain_intensity=.2)
    compound = TireCompound.MEDIUM
    tire = TIRE_COMPOUNDS[compound]
    context = WeatherForecastContext.from_schedule(entries)
    clock = (oracle.clock_for(track, current=27, future=13) if external else
             StrategyWeatherClock((0., 1., 2., 3.), 1., 1., 3, 0., 0.))

    def event_surface(weather, clock, offset, paid, first):
        return explicit_surface(weather, entries, events_before(clock, offset, paid, first))

    monkeypatch.setattr(oracle, "event_surface", event_surface)
    from test_paid_weather_compounds import exhaustive_safe_actions

    expected = exhaustive_safe_actions(
        driver, car, track, weather, tire, 19, 1, budget, context,
        clock=clock if external else None, used=(compound,),
    )
    actual = plan_rain_transition(
        driver, car, track, weather, tire, 19, 1, budget, used_compounds=(compound,),
        forecast_context=context, **({"weather_clock": clock} if external else {}),
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
    assert actual.compound == expected[2]


def test_weather_bound_matches_scheduled_external_event_oracle(monkeypatch):
    driver, car, track, weather = oracle.models(laps=4, lane=3)
    entries = [{"lap": 2, "rain_intensity": 0}, {"lap": 5, "rain_intensity": .9}]
    clock = oracle.clock_for(track, current=110, future=13)

    def event_surface(weather, clock, offset, paid, first):
        return explicit_surface(weather, entries, events_before(clock, offset, paid, first))

    monkeypatch.setattr(oracle, "event_surface", event_surface)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    expected = oracle.exhaustive_weather_bound(
        deepcopy(driver), car, track, weather, tire, 19, 1, clock,
    )
    actual = weather_stop_costs(
        driver, car, track, weather, tire, 19, 1, weather_clock=clock,
        forecast_context=WeatherForecastContext.from_schedule(entries),
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.stay_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("warmup", [0., 35.])
@pytest.mark.parametrize("free", [False, True])
def test_finite_schedule_matches_physical_set_oracle(external, warmup, free):
    driver, car, track, weather = oracle.models(laps=4, lane=3)
    weather = Weather(track_wetness=.12, rain_intensity=.2)
    entries = [{"lap": 2, "rain_intensity": .9}, {"lap": 4, "rain_intensity": 0}]
    pool = TireInventory.from_sets([
        {"compound": "medium", "age": 4}, {"compound": "hard", "age": 0},
        {"compound": "intermediate", "age": 0},
    ])
    pool.fit("set-1")
    clock = (oracle.clock_for(track, current=95, future=13) if external else
             StrategyWeatherClock((0., 1., 2., 3.), 1., 1., 3, 0., 0.))
    physics = LapSimulator(np.random.default_rng(7))
    best = {key: inf for key in pool.sets}

    def legal(used):
        return TireCompound.INTERMEDIATE in used or len(used) >= 2

    def visit(offset, current, ages, used, left, paid, first_paid, delay, cost, choice):
        if offset == 4:
            if legal(used):
                best[choice] = min(best[choice], cost)
            return
        before = explicit_surface(weather, entries, events_before(
            clock, offset, paid, first_paid, delay if external else 0,
        ))
        old = pool.sets[current].compound
        for key, item in pool.sets.items():
            changed = key != current
            fitting = changed
            stop = changed and not (offset == 0 and free)
            mandatory = before.tire_mismatch(old) == "critical"
            correction = not legal(used) and item.compound not in used
            if stop and not (left > 0 or mandatory or correction):
                continue
            if before.tire_mismatch(item.compound) == "critical":
                continue
            after_paid = paid + stop
            after_first = first_paid or stop and offset == 0
            after = explicit_surface(weather, entries, events_before(
                clock, offset, after_paid, after_first, delay if external else 0,
            ))
            driver.current_tire_laps = ages[key]
            running = physics.calculate_lap_time(
                driver, car, track, TIRE_COMPOUNDS[item.compound], after,
                offset + 1, track.total_laps, sample_variation=False,
            )
            fee = warmup if fitting else 0.
            next_ages = ages.copy()
            next_ages[key] += 1
            visit(offset + 1, key, next_ages, used | {item.compound}, max(0, left - stop),
                  after_paid, after_first, delay + fee,
                  cost + running + fee + stop * (
                      track.pit_lane_delta + expected_stationary_time(car)),
                  key if offset == 0 else choice)

    visit(0, "set-1", {key: item.age for key, item in pool.sets.items()},
          {TireCompound.MEDIUM}, 2, 0, False, 0., 0., None)
    actual = plan_inventory_strategy(
        driver, car, track, weather, pool, 1, tire_age=4, remaining_stops=2,
        used_compounds=(TireCompound.MEDIUM,), free_fit=free,
        forecast_context=WeatherForecastContext.from_schedule(entries),
        tire_warmup={compound.value: warmup for compound in TireCompound},
        **({"weather_clock": clock} if external else {}),
    )
    choices = list(pool.sets) if free else [key for key in pool.sets if key != "set-1"]
    assert actual.wait_cost == pytest.approx(best["set-1"])
    assert actual.pit_now_cost == pytest.approx(min(best[key] for key in choices))
    if actual.set_id is not None:
        assert best[actual.set_id] == pytest.approx(actual.pit_now_cost)


def test_automatic_opening_responds_to_known_future_rain():
    driver, car, track, _ = oracle.models(laps=12, lane=20)
    simulator = RaceSimulator(np.random.default_rng(7))
    strategy = simulator._infer_team_strategy(car, track)
    dry = simulator._choose_starting_compound(strategy, track, Weather(), driver, car)
    simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
        [{"lap": 2, "rain_intensity": 1}],
    )
    wet = simulator._choose_starting_compound(strategy, track, Weather(), driver, car)
    assert dry != wet


def test_finite_automatic_opening_responds_to_known_future_rain():
    driver, car, track, _ = oracle.models(laps=35, lane=30)
    simulator = RaceSimulator(np.random.default_rng(7))
    strategy = simulator._infer_team_strategy(car, track)
    records = [{"id": str(index), "compound": compound.value, "age": 0}
               for index, compound in enumerate(TireCompound)]
    dry = simulator._inventory_opening_set(driver, car, track, Weather(), strategy, records)[1]
    simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
        [{"lap": 2, "rain_intensity": 1}],
    )
    wet = simulator._inventory_opening_set(driver, car, track, Weather(), strategy, records)[1]
    assert dry.compound == TireCompound.MEDIUM
    assert wet.compound == TireCompound.SOFT


def test_paid_stint_rebases_shared_schedule_after_service_and_fitting_delay():
    _, _, track, _ = oracle.models(laps=4)
    entries = [{"lap": 2, "rain_intensity": .9}, {"lap": 4, "rain_intensity": 0}]
    simulator = RaceSimulator(np.random.default_rng(7))
    simulator.weather_forecast_context = WeatherForecastContext.from_schedule(entries)
    clock = oracle.clock_for(track, current=95, future=13)
    weather = Weather(track_wetness=.12, rain_intensity=.2)
    surface, cadence = simulator._projected_stint_weather(
        weather, clock, None, 4, fit_delay=35.,
    )
    projected = projected_surfaces(surface, 4, cadence)
    for offset, actual in enumerate(projected):
        count = events_before(clock, offset, 1, True, 35.) if offset else events_before(
            clock, offset, 1, True,
        )
        assert actual == explicit_surface(weather, entries, count)
    assert cadence.context.leading_lap == 1 + events_before(clock, 0, 1, True)


def test_known_rain_intensification_ranks_safe_wet_opening_without_changing_default():
    driver, car, track, _ = oracle.models(laps=12, lane=20)
    weather = Weather(track_wetness=.69, rain_intensity=.7)
    simulator = RaceSimulator(np.random.default_rng(7))
    strategy = simulator._infer_team_strategy(car, track)
    before = deepcopy((driver, car, track, weather, simulator.rng.bit_generator.state))
    assert simulator._choose_starting_compound(
        strategy, track, weather, driver, car,
    ) == TireCompound.INTERMEDIATE
    constant_costs = {compound: _policy_path_cost(
        driver, car, track, weather, strategy, simulator.strategy_tuning,
        simulator.strategy_profiles, compound, 0,
    ) for compound in (TireCompound.INTERMEDIATE, TireCompound.WET)}
    assert constant_costs[TireCompound.INTERMEDIATE] < constant_costs[TireCompound.WET]
    context = WeatherForecastContext.from_schedule([{"lap": 2, "rain_intensity": 1}])
    scores = dict(opening_policy_costs(
        driver, car, track, weather, strategy, simulator.strategy_tuning,
        simulator.strategy_profiles, forecast_context=context,
    ))
    assert set(scores) == {TireCompound.INTERMEDIATE, TireCompound.WET}
    assert scores[TireCompound.WET] < scores[TireCompound.INTERMEDIATE]
    simulator.weather_forecast_context = context
    assert simulator._choose_starting_compound(
        strategy, track, weather, driver, car,
    ) == TireCompound.WET
    assert context.leading_lap == 1
    assert (driver, car, track, weather, simulator.rng.bit_generator.state) == before


@pytest.mark.parametrize("planner", ["same", "transition", "weather", "inventory"])
@pytest.mark.parametrize("warmup,pending", [(0., False), (35., False), (35., True)])
def test_clock_planners_resolve_interval_carried_context(planner, warmup, pending):
    driver, car, track, _ = oracle.models(laps=4, lane=3)
    weather = Weather(track_wetness=.12, rain_intensity=.2)
    context = WeatherForecastContext.from_schedule([
        {"lap": 2, "rain_intensity": .9}, {"lap": 4, "rain_intensity": 0},
    ])
    clock = StrategyWeatherClock((0., 170., 340., 510.), 10., 90., 6, 27., 13.)
    intervals = normalize_weather_intervals(4, forecast_context=context)
    options = {"weather_clock": clock}
    if warmup:
        options.update(tire_warmup={compound.value: warmup for compound in TireCompound},
                       current_fit_pending=pending)
    if planner == "inventory":
        pool = TireInventory.from_sets([
            {"compound": compound.value, "age": 19 if compound == TireCompound.MEDIUM else 0}
            for compound in TireCompound
        ])
        pool.fit("set-2")

        def evaluate(extra):
            return plan_inventory_strategy(
                driver, car, track, weather, pool, 1, tire_age=19, remaining_stops=2,
                used_compounds=(TireCompound.MEDIUM,), **options, **extra,
            )
    else:
        compound = TireCompound.INTERMEDIATE if planner == "same" else TireCompound.MEDIUM
        tire = TIRE_COMPOUNDS[compound]

        def evaluate(extra):
            if planner == "same":
                return plan_rain_stop(driver, car, track, weather, tire, 19, 1, 2,
                                      **options, **extra)
            if planner == "transition":
                return plan_rain_transition(
                    driver, car, track, weather, tire, 19, 1, 2,
                    used_compounds=(compound,), **options, **extra,
                )
            return weather_stop_costs(driver, car, track, weather, tire, 19, 1,
                                      **options, **extra)

    explicit = evaluate({"forecast_context": context})
    carried = evaluate({"weather_intervals": intervals})
    assert carried == explicit
    assert context.leading_lap == 1
    with pytest.raises(ValueError, match="conflicting forecast contexts"):
        evaluate({"weather_intervals": intervals, "forecast_context": context.advanced()})
