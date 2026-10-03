"""Independent physical schedules for known control on changing surfaces."""

from copy import deepcopy
from dataclasses import replace
from math import inf

import pytest
from pydantic import PrivateAttr
from test_chronological_field_finish import FieldPhysics, field, native_path
from test_controlled_dry_strategy import single_car_context
from test_controlled_strategy_field import signature

from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Track, Weather, _native
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import inventory_strategy
from f1sim.simulation.chronological_finish import (
    ChronologicalFinishContext,
    ObservedChronologicalField,
)
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.controlled_weather_strategy import green_weather_forecast
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("relative_laps", [-1, 1])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("horizon", [2, 18])
@pytest.mark.parametrize("native_clock", [False, True])
def test_green_weather_clock_rebases_native_entries_and_excludes_the_flag(
    monkeypatch, control, relative_laps, off_track, horizon, native_clock,
):
    models = field(control=control, intervals=2, relative_laps=relative_laps,
                   off_track=off_track, remaining=120., neutralized=True,
                   fee=7. if off_track else 0., now=1000.)
    *_, observed, now = models
    entries, updates = [], []
    begin, after = ChronologicalRace._begin_running, ChronologicalRace._after_leader_crossing

    def entry(engine, state, pending, time):
        if state.driver.id == "A":
            entries.append((time, len(updates)))
        return begin(engine, state, pending, time)

    def leading(engine, time, red):
        if engine.timeline.chequered_time is None:
            updates.append(time)
        return after(engine, time, red)

    monkeypatch.setattr(ChronologicalRace, "_begin_running", entry)
    monkeypatch.setattr(ChronologicalRace, "_after_leader_crossing", leading)
    native_path(models, stopped=False)
    projected = ObservedChronologicalField(observed, now)
    prefix = 0
    while projected.projection_required:
        projected.enter()
        projected.cross(99.)
        prefix += 1
    assert not projected.finished
    before = signature(projected)
    intervals, clock = green_weather_forecast(projected, horizon, 200., native=native_clock)
    assert signature(projected) == before
    native_entries = entries[prefix:prefix + horizon]
    origin = native_entries[0][0]
    for index, (time, count) in enumerate(native_entries):
        expected = count - projected.updates
        if clock is None:
            assert intervals[index] == expected
        else:
            assert clock.lap_start_offsets[index] == pytest.approx(time - origin, abs=1.e-8)
            assert clock.updates(index, 0) == expected
    if clock is not None:
        expected_updates = [time - origin for time in updates[projected.updates:]]
        assert clock.update_offsets == pytest.approx(expected_updates, abs=1.e-8)
        # A short own horizon still exposes later external updates to paid
        # service. A long horizon freezes weather once the native field flags.
        assert clock.updates(horizon - 1, 100) == len(expected_updates)
        assert clock.max_updates == len(expected_updates)
    if horizon > len(native_entries):
        frozen = len(updates) - projected.updates
        counts = intervals if clock is None else tuple(
            clock.updates(index, 0) for index in range(horizon))
        assert all(count == frozen for count in counts[len(native_entries):])


@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("stopped", [False, True])
def test_changing_surface_cost_and_distance_match_native_service_and_flag(
    monkeypatch, control, off_track, stopped,
):
    models = field(control=control, intervals=6, relative_laps=0,
                   off_track=off_track, remaining=120., neutralized=True,
                   fee=7. if off_track else 0., now=6850.)
    driver, car, track, observed, now = models
    weather = Weather(track_wetness=.12, rain_intensity=0., change_probability=0.)
    calculator = LapSimulator()

    def physics(self, driver, car, track, tire, surface, lap, total, **options):
        if driver.id != "A":
            return self.paces[driver.id]
        return calculator.calculate_lap_time(
            driver, car, track, tire, surface, lap, total,
            sample_variation=False, **options)

    monkeypatch.setattr(FieldPhysics, "calculate_lap_time", physics)
    warmup = {"medium": 3., "soft": 5., "hard": 7.}
    distance, crossing, _ = native_path(
        models, stopped=stopped, delay=200., weather=weather,
        warmup=warmup, pending_fit=True)
    pool = TireInventory.from_sets([
        {"id": "M", "compound": "medium", "age": 8},
        {"id": "S", "compound": "soft", "age": 0},
    ])
    pool.fit("M")
    result = plan_inventory_strategy(
        driver, car, track, weather, pool, 72, tire_age=8,
        remaining_stops=0, force_stop=stopped, current_fit_pending=True,
        used_compounds={TireCompound.HARD, TireCompound.MEDIUM},
        tire_warmup=warmup, control_context=StrategyControlContext(observed, now, 200.))
    assert (result.pit_now_laps if stopped else result.wait_laps) == distance
    assert (result.pit_now_cost if stopped else result.wait_cost) == pytest.approx(
        crossing - now, abs=1.e-8)


@pytest.mark.parametrize("planner", ["inventory", "transition", "rain_stop", "weather_bound"])
def test_custom_weather_clock_preserves_dispatch_with_a_control_context(planner):
    calls = []

    class CustomClock(StrategyWeatherClock):
        def updates(self, *args, **kwargs):
            calls.append(args)
            return super().updates(*args, **kwargs)

    driver, car, track, weather, pool = inputs(.5, .5, 3)
    clock = CustomClock((0., 100., 200.), 10., 100., 3, 10., 10.)
    observed = context(track, car, "standard", "sc", 2)
    options = dict(weather_clock=clock, physical_total_laps=12)
    current = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    def execute(**extra):
        if planner == "inventory":
            return plan_inventory_strategy(
                driver, car, track, weather, pool, 1, tire_age=11, remaining_stops=1,
                **options, **extra)
        if planner == "transition":
            return plan_rain_transition(
                driver, car, track, weather, current, 11, 1, 1, **options, **extra)
        if planner == "rain_stop":
            return plan_rain_stop(
                driver, car, track, weather, current, 11, 1, 1, **options, **extra)
        return weather_stop_costs(
            driver, car, track, weather, current, 11, 1, **options, **extra)
    expected = execute()
    calls.clear()
    assert execute(control_context=observed) == expected
    assert calls


def test_green_rebase_stops_at_flag_with_a_large_custom_scheduled_distance(monkeypatch):
    *_, observed, now = field(control="vsc", intervals=2, relative_laps=1,
                              remaining=120., neutralized=True, now=1000., scheduled=1_000_000)
    projection = ObservedChronologicalField(observed, now)
    while projection.projection_required:
        projection.enter()
        projection.cross(99.)
    entries = []
    enter = ObservedChronologicalField.enter

    def counted(self, *args):
        entries.append(self.now)
        return enter(self, *args)

    monkeypatch.setattr(ObservedChronologicalField, "enter", counted)
    _, clock = green_weather_forecast(projection, 2, 200.)
    assert clock is not None
    assert clock.max_updates < 100
    assert 2 < len(entries) < 100


def inputs(wetness, rain, horizon=5):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A", pit_stop_avg=1.7, pit_stop_std=.1,
              tire_degradation_factor=1.4)
    track = Track(id="T", name="T", country="Test", total_laps=horizon,
                  base_lap_time=100., pit_lane_delta=.5, tire_stress=1.)
    stock = TireInventory.from_sets([
        {"id": "I", "compound": "intermediate", "age": 8},
        {"id": "I2", "compound": "intermediate", "age": 3},
        {"id": "W", "compound": "wet", "age": 2},
        {"id": "S", "compound": "soft", "age": 3},
        {"id": "H", "compound": "hard", "age": 9},
    ])
    stock.fit("I")
    return driver, car, track, Weather(track_wetness=wetness, rain_intensity=rain), stock


def context(track, car, engine, control, intervals):
    value = single_car_context(track, car, control, intervals)
    if engine == "chronological":
        field = ChronologicalFinishContext(
            "A", RaceFinishTimeline(12, ["A"]), ("A",), (), 100.,
            1.4 if control == "sc" else 1.2, control == "sc", control_intervals=intervals)
        value = StrategyControlContext(field, 50., value.current_stop_delay)
    return value


def all_schedules(models, control, intervals, budget, *, finite, pending=False,
                  force=False, damaged=False, forecast=None, current_age=11):
    """Enumerate real set IDs, their accumulated wear and actual rule credit."""
    driver, car, track, weather, stock = deepcopy(models)
    surfaces = [weather]
    for _ in range(track.total_laps - 1):
        surfaces.append(surfaces[-1].project_surface() if forecast is None else
                        forecast.advanced(len(surfaces) - 1).project_next(surfaces[-1]))
    sets = stock.sets if finite else TIRE_COMPOUNDS
    available = tuple(key for key in sets if not finite or key not in stock.unavailable_ids)
    current = "I" if finite else TireCompound.INTERMEDIATE
    ages = {key: value.age if finite else 0 for key, value in sets.items()}
    ages[current] = current_age
    warmup = {"intermediate": 1., "wet": 2., "soft": .6, "hard": .3}
    best, root_costs = {False: inf, True: inf}, {}
    simulator = LapSimulator()

    def legal(used):
        return bool(used & {TireCompound.WET, TireCompound.INTERMEDIATE}) or len(used) >= 2

    def visit(offset, current, wear, used, left, dry, damp, total, first, stopped):
        if offset == track.total_laps:
            if legal(used):
                best[stopped] = min(best[stopped], total)
                if stopped:
                    root_costs[first] = min(root_costs.get(first, inf), total)
            return
        surface = surfaces[offset]
        old = sets[current].compound
        critical = surface.tire_mismatch(old) == "critical" or offset == 0 and damaged
        for target in available:
            compound = sets[target].compound
            fitting = target != current if finite else True
            # Unlimited retention is a distinct action from fitting a fresh
            # set of the same compound. Enumerate both without merging wear.
            actions = (False, True) if not finite and target == current else (fitting,)
            for fitting in actions:
                if not fitting and (critical or offset == 0 and force):
                    continue
                if fitting and surface.tire_mismatch(compound) == "critical":
                    continue
                limit = (dry if surface.track_wetness < .08 and surface.rain_intensity < .15
                         else damp)
                allowed = (critical or left > 0 and (old in (
                    TireCompound.INTERMEDIATE, TireCompound.WET)
                    or surface.track_wetness > .3 or limit > 0)
                    or not legal(used) and compound not in used)
                if fitting and not allowed and not (offset == 0 and force):
                    continue
                age = wear[target] if finite or not fitting else 0
                driver.current_tire_laps = age
                running = simulator.calculate_lap_time(
                    driver, car, track, TIRE_COMPOUNDS[compound], surface, offset + 1, 12,
                    sample_variation=False, active_aero_enabled=offset >= intervals)
                value = running * ((1.4 if control == "sc" else 1.2) if offset < intervals else 1.)
                if fitting:
                    factor = (.55 if control == "sc" else .75) if offset < intervals else 1.
                    value += track.pit_lane_delta * factor + expected_stationary_time(car)
                if fitting or offset == 0 and pending:
                    value += warmup.get(compound.value, 0.)
                updated = dict(wear)
                updated[target] = age + 1
                visit(offset + 1, target, updated, used | {compound},
                      max(0, left - int(fitting)), max(0, dry - int(fitting)),
                      max(0, damp - int(fitting)), total + value,
                      target if offset == 0 else first, fitting if offset == 0 else stopped)

    visit(0, current, ages, set(), budget, budget, budget, 0., None, False)
    return best, root_costs


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("wetness,rain,prescribed", [
    (.5, .5, False), (.45, 0., False), (.05, .7, False), (.12, 0., False), (.12, .1, True),
])
@pytest.mark.parametrize("finite,budget,pending,force,damaged", [
    (False, 2, False, False, False),
    (True, 2, True, False, False),
    (True, 0, False, True, False),
    (True, 0, False, False, True),
    (False, 9, False, False, False),
    (True, 9, False, False, False),
])
def test_known_weather_control_matches_complete_schedules(
    engine, control, wetness, rain, prescribed, finite, budget, pending, force, damaged,
):
    models = inputs(wetness, rain)
    driver, car, track, weather, stock = models
    if damaged:
        stock.unavailable_ids.add("I")
    observed = context(track, car, engine, control, 2)
    forecast = (WeatherForecastContext.from_schedule([
        {"lap": 2, "rain_intensity": .8}, {"lap": 4, "rain_intensity": 0.},
    ]) if prescribed else None)
    before = deepcopy(models), deepcopy(observed)
    options = dict(control_context=observed, physical_total_laps=12, used_compounds=set(),
                   tire_warmup={"intermediate": 1., "wet": 2., "soft": .6, "hard": .3},
                   remaining_dry_stops=budget, remaining_damp_stops=budget,
                   current_fit_pending=pending, forecast_context=forecast)
    if finite:
        actual = plan_inventory_strategy(
            driver, car, track, weather, stock, 1, tire_age=11, remaining_stops=budget,
            force_stop=force, **options)
    else:
        actual = plan_rain_transition(
            driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
            11, 1, budget, **options)
    expected, costs = all_schedules(models, control, 2, budget, finite=finite,
                                    pending=pending, force=force, damaged=damaged,
                                    forecast=forecast)
    assert actual.pit_now_cost == pytest.approx(expected[True], abs=1.e-8)
    assert actual.wait_cost == pytest.approx(expected[False], abs=1.e-8)
    selected = actual.set_id if finite else actual.compound
    assert selected is None if expected[True] == inf else costs[selected] == pytest.approx(
        expected[True], abs=1.e-8)
    assert actual.pit_now_laps == (-1 if expected[True] == inf else track.total_laps)
    assert actual.wait_laps == (-1 if expected[False] == inf else track.total_laps)
    assert models[:-1] == before[0][:-1]
    assert stock.__dict__ == before[0][-1].__dict__
    assert observed == before[1]


@pytest.mark.parametrize("control", ["sc", "vsc"])
def test_same_rain_compound_prices_all_known_intervals(control):
    driver, car, track, weather, _ = inputs(.5, .5, 4)
    current = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    observed = context(track, car, "standard", control, 4)
    result = plan_rain_stop(driver, car, track, weather, current, 11, 1, 0,
                            control_context=observed, physical_total_laps=12)
    expected = 0.
    for offset in range(4):
        driver.current_tire_laps = 11 + offset
        expected += LapSimulator().calculate_lap_time(
            driver, car, track, current, weather, offset + 1, 12,
            sample_variation=False, active_aero_enabled=False) * (1.4 if control == "sc" else 1.2)
    assert result.wait_cost == pytest.approx(expected, abs=1.e-8)
    assert result.pit_now_cost == inf


@pytest.mark.parametrize("extreme", ["aero_floor", "old_sets"])
def test_weather_bounds_preserve_complete_schedules_for_extreme_native_inputs(extreme):
    driver, car, track, weather, stock = models = inputs(.12, 0.)
    if extreme == "aero_floor":
        track.base_lap_time = .08
        track.active_aero_zones = [ActiveAeroZone(zone_id=index + 1, sector=1, time_gain=1.)
                                  for index in range(3)]
    else:
        stock.sets = {key: replace(item, age=1000) for key, item in stock.sets.items()}
    age = 1000 if extreme == "old_sets" else 11
    observed = context(track, car, "standard", "sc", 2)
    result = plan_inventory_strategy(
        driver, car, track, weather, stock, 1, tire_age=age, remaining_stops=2,
        remaining_dry_stops=2, remaining_damp_stops=2,
        physical_total_laps=12, control_context=observed,
        tire_warmup={"intermediate": 1., "wet": 2., "soft": .6, "hard": .3})
    expected, _ = all_schedules(models, "sc", 2, 2, finite=True, current_age=age)
    assert result.pit_now_cost == pytest.approx(expected[True], abs=1.e-8)
    assert result.wait_cost == pytest.approx(expected[False], abs=1.e-8)


@pytest.mark.parametrize("extreme", ["ordinary", "aero_floor", "old_sets"])
def test_common_stock_bound_matches_independent_suffix_bounds(monkeypatch, extreme):
    driver, car, track, weather, stock = inputs(.12, 0.)
    *_, observed, now = field(control="sc", intervals=2, relative_laps=1,
                              remaining=120., neutralized=True, now=1000.)
    track.total_laps = 76
    if extreme == "aero_floor":
        track.base_lap_time = .08
        track.active_aero_zones = [ActiveAeroZone(zone_id=index + 1, sector=1, time_gain=1.)
                                  for index in range(3)]
    if extreme == "old_sets":
        stock.sets = {key: replace(item, age=1000) for key, item in stock.sets.items()}
    options = dict(tire_age=1000 if extreme == "old_sets" else 11, remaining_stops=2,
                   remaining_dry_stops=2, remaining_damp_stops=2,
                   used_compounds={TireCompound.INTERMEDIATE}, physical_total_laps=90,
                   control_context=StrategyControlContext(observed, now, 10.),
                   tire_warmup={"intermediate": 1., "wet": 2., "soft": .6, "hard": .3})
    bounded = plan_inventory_strategy(driver, car, track, weather, stock, 72, **options)
    original = inventory_strategy.control_wear_bound
    calls = []

    def separate(*args):
        assert original(*args) is not None
        calls.append(args)
        return None

    monkeypatch.setattr(inventory_strategy, "control_wear_bound", separate)
    # This audit changes only which proven lower bound is used. Preserve
    # native dispatch so the reference computes each suffix's original bound.
    monkeypatch.setattr(_native, "_HELPERS", [
        (namespace, name, separate if namespace is inventory_strategy.__dict__
         and name == "control_wear_bound" else value)
        for namespace, name, value in _native._HELPERS])
    reference = plan_inventory_strategy(driver, car, track, weather, stock, 72, **options)
    assert calls
    assert bounded == reference


@pytest.mark.parametrize("planner", ["inventory", "transition", "rain_stop", "weather_bound"])
def test_actual_model_identity_and_snapshot_state_survive_the_green_suffix(planner):
    calls = []

    class CustomCar(Car):
        _ledger: list = PrivateAttr(default_factory=list)

        def pace_delta_seconds(self, *args, **kwargs):
            self._ledger.append(self.team_id)
            calls.append((self.team_id, len(self._ledger)))
            return (super().pace_delta_seconds(*args, **kwargs)
                    + (5. if self.team_id == "A" else 40.) + len(self._ledger))

    driver, car, track, weather, stock = inputs(.5, .5)
    car = CustomCar.model_validate(car.model_dump())
    observed = context(track, car, "standard", "sc", 2)
    options = dict(control_context=observed, physical_total_laps=12)
    if planner == "inventory":
        result = plan_inventory_strategy(driver, car, track, weather, stock, 1,
                                         tire_age=11, remaining_stops=0,
                                         used_compounds={TireCompound.INTERMEDIATE}, **options)
    elif planner == "transition":
        result = plan_rain_transition(driver, car, track, weather,
                                      TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                                      11, 1, 0, used_compounds={TireCompound.INTERMEDIATE},
                                      **options)
    elif planner == "rain_stop":
        result = plan_rain_stop(driver, car, track, weather,
                               TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 11, 1, 0, **options)
    else:
        result = weather_stop_costs(driver, car, track, weather,
                                    TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 11, 1, **options)
    expected = 0.
    for offset in range(5):
        clean = driver.model_copy(deep=True)
        clean.current_tire_laps = 11 + offset
        running = LapSimulator().calculate_lap_time(
            clean, Car.model_validate(car.model_dump()), track,
            TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], weather, offset + 1, 12,
            sample_variation=False, active_aero_enabled=offset >= 2)
        scale = LapSimulator.weather_pace_multiplier(driver, car, weather)
        expected += (running + 6. * scale) * (1.4 if offset < 2 else 1.)
    cost = result.stay_cost if planner == "weather_bound" else result.wait_cost
    assert cost == pytest.approx(expected, abs=1.e-8)
    assert calls and set(calls) == {("A", 1)}
    assert car._ledger == []
