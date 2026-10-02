"""Reuse deterministic weather and stint physics without changing the forecast."""

from copy import deepcopy
from functools import lru_cache
from sys import _getframe

import pytest
from test_strategy_pit_weather import clock_for, exhaustive_same, models

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.models import Car, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.weather import WeatherCondition
from f1sim.simulation import rain_strategy
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.rain_strategy import _running_row, plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("planner", ["transition", "weather", "inventory"])
def test_timed_surface_cache_advances_each_update_once(monkeypatch, planner):
    driver, car, track, _ = models(laps=5)
    weather = Weather(track_wetness=.19, rain_intensity=.39)
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    clock = clock_for(track)
    calls = 0
    original = Weather.project_surface

    def counted(self):
        nonlocal calls
        calls += 1
        return original(self)

    monkeypatch.setattr(Weather, "project_surface", counted)
    if planner == "inventory":
        inventory = TireInventory.from_sets([
            {"compound": "medium", "age": 3}, {"compound": "intermediate"},
            {"compound": "hard"},
        ])
        inventory.fit("set-1")
        plan_inventory_strategy(driver, car, track, weather, inventory, 1,
                                remaining_stops=2, weather_clock=clock)
    elif planner == "transition":
        plan_rain_transition(driver, car, track, weather, tire, 3, 1, 2,
                             used_compounds=("medium",), weather_clock=clock)
    else:
        weather_stop_costs(driver, car, track, weather, tire, 3, 1, weather_clock=clock)
    assert 0 < calls <= clock.max_updates


def test_clocked_stints_reuse_physics_but_include_changed_car_performance(monkeypatch, request):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = clock_for(track)
    rain_strategy._green_laps.clear()
    first = plan_rain_stop(driver, car, track, weather, tire, 39, 1, 2,
                           weather_clock=clock)
    initial_cache = dict(rain_strategy._green_laps)
    renamed_driver = driver.model_copy(update={"id": "other", "team_id": "other"})
    renamed_car = car.model_copy(update={"team_id": "other", "team_name": "Other"})
    same = plan_rain_stop(renamed_driver, renamed_car, track, weather, tire, 39, 1, 2,
                          weather_clock=clock)
    assert same == first
    assert dict(rain_strategy._green_laps) == initial_cache

    changed_car = renamed_car.model_copy(update={"tire_degradation_factor": 1.5})
    before = len(rain_strategy._green_laps)
    changed = plan_rain_stop(renamed_driver, changed_car, track, weather, tire, 39, 1, 2,
                             weather_clock=clock)
    assert len(rain_strategy._green_laps) > before
    expected = exhaustive_same(deepcopy(driver), changed_car, track, weather,
                               tire, 39, 1, 2, clock)
    assert changed.pit_now_cost == pytest.approx(expected[0])
    assert changed.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("budget", [0, 2])
def test_clocked_stints_match_control_traffic_and_physical_fuel(budget):
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
        update={"degradation_rate": .08},
    )
    clock = clock_for(track, current=27, future=13)
    expected = exhaustive_same(
        deepcopy(driver), car, track, weather, tire, 12, 1, budget, clock,
        queue=-2, modifier=1.2, aero=False, lane=.75,
        current_gaps=(.2, 1.1), physical_total_laps=10,
    )
    actual = plan_rain_stop(
        driver, car, track, weather, tire, 12, 1, budget, weather_clock=clock,
        additional_current_stop_cost=-2, current_lap_time_modifier=1.2,
        active_aero_enabled=False, pit_lane_factor=.75,
        current_traffic_gaps=(.2, 1.1), physical_total_laps=10,
    )
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])


@pytest.mark.parametrize("budget", [0, 1, 3])
@pytest.mark.parametrize("wetness", [.07999999999999999, .08, .08000000000000002,
                                   .29999999999999993, .3, .30000000000000004, .44731])
@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("cadence", [(23.25, 83.125, 9, 103.5, 17.125),
                                    (81.25, 47.5, 4, 0., 128.75),
                                    (0., 90., 0, 25., 25.)])
def test_native_clock_costs_are_exactly_generic_with_fitting_and_cadence(
    monkeypatch, request, budget, wetness, pending, cadence,
):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, _ = models(laps=7)
    weather = Weather(track_wetness=wetness, rain_intensity=.39127)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
        update={"degradation_rate": .08123, "initial_grip": .9317,
                "cliff_threshold": 14, "cliff_multiplier": 4.17},
    )
    clock = StrategyWeatherClock((0., 81.25, 176.125, 233., 310.5, 475., 521.),
                                 *cadence)
    inputs = tuple(item.model_dump_json() for item in (driver, car, track, weather, tire))
    options = dict(weather_clock=clock, physical_total_laps=17,
                   current_lap_time_modifier=1.4, active_aero_enabled=False,
                   current_traffic_gaps=(.3125, 1.375), pit_lane_factor=.6,
                   additional_current_stop_cost=-3.125,
                   tire_warmup={"intermediate": 11.75}, current_fit_pending=pending)
    native = plan_rain_stop(driver, car, track, weather, tire, 13, 1, budget, **options)
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time",
                        lambda *args: None)
    generic = plan_rain_stop(driver, car, track, weather, tire, 13, 1, budget, **options)
    assert native == generic
    assert tuple(item.model_dump_json() for item in (driver, car, track, weather, tire)) == inputs


def test_native_clock_shares_surface_path_and_never_reconstructs_rows(monkeypatch, request):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models(laps=8)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = StrategyWeatherClock(tuple(index * 83. for index in range(8)),
                                 20., 90., 10, 95., 13.)
    row_calls = surface_calls = 0
    original_row = rain_strategy._running_row
    original_surface = Weather.project_surface

    def row(*args):
        nonlocal row_calls
        row_calls += 1
        return original_row(*args)

    def surface(self):
        nonlocal surface_calls
        surface_calls += 1
        return original_surface(self)

    monkeypatch.setattr(rain_strategy, "_running_row", row)
    monkeypatch.setattr(Weather, "project_surface", surface)
    native = plan_rain_stop(driver, car, track, weather, tire, 18, 1, 3,
                           weather_clock=clock)
    assert row_calls == 0
    native_surfaces = surface_calls
    assert 0 < native_surfaces <= clock.max_updates
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time",
                        lambda *args: None)
    generic = plan_rain_stop(driver, car, track, weather, tire, 18, 1, 3,
                            weather_clock=clock)
    assert generic == native
    assert row_calls > 1
    assert surface_calls - native_surfaces > native_surfaces


@pytest.mark.parametrize("custom", ["physics", "clock"])
def test_native_shortcut_keeps_custom_dispatch(monkeypatch, request, custom, green_cache):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = clock_for(track)
    plan_rain_stop(driver, car, track, weather, tire, 12, 1, 2, weather_clock=clock)
    shared_before = dict(green_cache)
    calls = 0
    original = rain_strategy._running_row

    def row(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(rain_strategy, "_running_row", row)
    if custom == "physics":
        physics = LapSimulator.calculate_lap_time
        monkeypatch.setattr(LapSimulator, "calculate_lap_time",
                            lambda self, *args, **kwargs: physics(self, *args, **kwargs) + .125)
    else:
        class CustomClock(StrategyWeatherClock):
            def updates(self, offset, paid_stops=0, stopped_first=False, *, fit_delay=0.):
                return min(self.max_updates, super().updates(offset, paid_stops, stopped_first) + 1)
        clock = CustomClock(**{name: getattr(clock, name) for name in clock.__slots__})
    actual = plan_rain_stop(driver, car, track, weather, tire, 12, 1, 2,
                           weather_clock=clock)
    assert calls > 1
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert actual == plan_rain_stop(driver, car, track, weather, tire, 12, 1, 2,
                                   weather_clock=clock)
    assert dict(green_cache) == shared_before


def test_native_clock_planning_does_not_sample_randomness(monkeypatch):
    class NoRandomDraws:
        def normal(self, *args, **kwargs):
            pytest.fail("Deterministic planning sampled driver variation")

    monkeypatch.setattr(rain_strategy.np.random, "default_rng", lambda *args: NoRandomDraws())
    driver, car, track, weather = models()
    plan_rain_stop(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                   12, 1, 2, weather_clock=clock_for(track))


@pytest.mark.parametrize("compound", list(TireCompound))
@pytest.mark.parametrize("age", [0, 19, 43])
@pytest.mark.parametrize("intervals", [None, (0, 0, 1, 1, 3), (0, 2, 4, 6, 8)])
def test_cached_stint_preparation_is_bit_exact(monkeypatch, request, compound, age, intervals):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, _ = models(laps=9)
    driver.skill_rating = .87123
    car.tire_degradation_factor = 1.2317
    weather = Weather(track_wetness=.30000000000000004, rain_intensity=.32719)
    tire = TIRE_COMPOUNDS[compound].model_copy(
        update={"initial_grip": .95171, "degradation_rate": .07923,
                "cliff_threshold": 21, "cliff_multiplier": 4.173},
    )
    args = (tuple(item.model_dump_json() for item in (driver, car, track)),
            weather.model_dump_json(), tire.model_dump_json(), age, 5, 17, intervals)
    native = _running_row(*args)
    _running_row.cache_clear()
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert native == _running_row(*args)


def test_cached_stint_preparation_preserves_custom_physics(monkeypatch, request):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    args = (tuple(item.model_dump_json() for item in (driver, car, track)),
            weather.model_dump_json(), tire.model_dump_json(), 9, 1, 10)
    native = _running_row(*args)
    _running_row.cache_clear()
    original = LapSimulator.calculate_lap_time
    calls = 0

    def physics(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(self, *args, **kwargs) + .125

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", physics)
    assert _running_row(*args) == tuple(value + .125 for value in native)
    assert calls == track.total_laps


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("cancel_after", [1, 2])
def test_zero_stop_clock_row_cancels_immediately_or_mid_row(
    monkeypatch, request, native, cancel_after,
):
    _running_row.cache_clear()
    request.addfinalizer(_running_row.cache_clear)
    driver, car, track, weather = models(laps=40)
    clock = clock_for(track)
    if not native:
        monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    polls = 0

    def cancel():
        nonlocal polls
        polls += 1
        return polls >= cancel_after

    with cancellation_scope(cancel), pytest.raises(SimulationCancelled):
        plan_rain_stop(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                       12, 1, 0, weather_clock=clock)
    assert polls == cancel_after


def test_native_row_cost_cache_hits_still_poll_cancellation(monkeypatch):
    driver, car, track, weather = models(laps=40)
    clock = StrategyWeatherClock(tuple(index * 90. for index in range(40)),
                                 0., 90., 0, 25., 25.)
    hits = 0

    def observed_cache(*args, **kwargs):
        decorate = lru_cache(*args, **kwargs)

        def apply(function):
            cached = decorate(function)
            if function.__name__ != "running":
                return cached

            def running(*key):
                nonlocal hits
                before = cached.cache_info().hits
                value = cached(*key)
                hits += cached.cache_info().hits - before
                return value

            return running

        return apply

    original_checkpoint = rain_strategy.cancellation_checkpoint

    def row_checkpoint():
        # Isolate row cancellation: the outer search also polls, and would
        # otherwise mask missing checks when all scalar costs are cache hits.
        if _getframe(1).f_code.co_name == "row":
            original_checkpoint()

    monkeypatch.setattr(rain_strategy, "lru_cache", observed_cache)
    monkeypatch.setattr(rain_strategy, "cancellation_checkpoint", row_checkpoint)
    with cancellation_scope(lambda: hits > 0), pytest.raises(SimulationCancelled):
        plan_rain_stop(driver, car, track, weather, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                       12, 1, 3, weather_clock=clock)
    assert hits > 0


@pytest.fixture
def green_cache():
    rain_strategy._reset_green_cache_after_fork()
    yield rain_strategy._green_laps
    rain_strategy._reset_green_cache_after_fork()


def _green_decision(*, tire=None, weather=None, **options):
    driver, car, track, default_weather = models(laps=options.pop("laps", 7))
    return plan_rain_stop(
        driver, car, track, weather or default_weather,
        tire or TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 13, 1,
        options.pop("budget", 3), weather_clock=clock_for(track), **options,
    )


def test_shared_green_costs_reuse_only_immutable_values(monkeypatch, green_cache):
    calls = 0
    original = rain_strategy._shared_green_lap

    def observed(key, evaluate, *args):
        def counted(*values):
            nonlocal calls
            calls += 1
            return evaluate(*values)
        return original(key, counted, *args)

    monkeypatch.setattr(rain_strategy, "_shared_green_lap", observed)
    first = _green_decision()
    initial = calls
    assert initial > 0
    assert _green_decision() == first
    assert calls == initial
    assert green_cache and all(type(cost) is float for cost in green_cache.values())

    def immutable(value):
        assert isinstance(value, (tuple, str, int, float, bool, type(None)))
        if isinstance(value, tuple):
            for item in value:
                immutable(item)
    for key in green_cache:
        immutable(key)


@pytest.mark.parametrize("target,field,value", [
    (target, field, value)
    for target in ("retained", "fresh")
    for field, value in (("initial_grip", .913), ("degradation_rate", .081),
                         ("cliff_threshold", 12), ("cliff_multiplier", 4.2),
                         ("optimal_temp_range", (79., 99.)))
] + [("weather", field, value) for field, value in (
    ("condition", WeatherCondition.LIGHT_RAIN), ("track_temperature", 34.),
    ("air_temperature", 24.), ("humidity", .6), ("rain_intensity", .421),
    ("track_wetness", .30000000000000004), ("change_probability", .2),
)])
def test_shared_green_keys_include_complete_tire_and_weather(
    monkeypatch, green_cache, target, field, value,
):
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(deep=True)
    _, _, _, weather = models(laps=7)
    _green_decision(tire=tire, weather=weather)
    before = set(green_cache)
    if target == "fresh":
        monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.INTERMEDIATE,
                            tire.model_copy(update={field: value}))
    elif target == "retained":
        tire = tire.model_copy(update={field: value})
    else:
        weather = weather.model_copy(update={field: value})
    actual = _green_decision(tire=tire, weather=weather)
    assert set(green_cache) - before
    monkeypatch.setattr(rain_strategy, "_native_green_cache_available", lambda _: False)
    assert _green_decision(tire=tire, weather=weather) == actual


def test_shared_green_eviction_preserves_exact_decision_and_tie(monkeypatch, green_cache):
    expected = _green_decision()
    monkeypatch.setattr(rain_strategy, "_GREEN_LAP_LIMIT", 2)
    green_cache.clear()
    actual = _green_decision()
    assert actual == expected
    assert actual.should_pit() == expected.should_pit()
    assert len(green_cache) == 2
    assert _green_decision() == expected
    assert len(green_cache) == 2


@pytest.mark.parametrize("target", ["driver", "car", "track", "physical", "lap", "age"])
def test_shared_green_package_lap_and_age_sensitivity(monkeypatch, green_cache, target):
    driver, car, track, weather = models(laps=7)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    age, lap, physical = 13, 1, 17
    def decide():
        clock_track = track.model_copy(update={"total_laps": track.total_laps - lap + 1})
        return plan_rain_stop(driver, car, track, weather, tire, age, lap, 3,
                              weather_clock=clock_for(clock_track), physical_total_laps=physical)
    decide()
    before = set(green_cache)
    if target == "driver":
        driver.skill_rating = .9123
    elif target == "car":
        car.tire_degradation_factor = 1.237
    elif target == "track":
        track.base_lap_time += .137
    elif target == "physical":
        physical += 3
    elif target == "lap":
        lap += 1
    else:
        age += 1
    actual = decide()
    assert set(green_cache) - before
    monkeypatch.setattr(rain_strategy, "_native_green_cache_available", lambda _: False)
    assert decide() == actual


def test_shared_green_hits_still_poll_row_cancellation(monkeypatch, green_cache):
    _green_decision(budget=0, laps=40)
    assert green_cache
    original = rain_strategy.cancellation_checkpoint
    polls = 0

    def row_checkpoint():
        if _getframe(1).f_code.co_name == "row":
            original()

    def cancel():
        nonlocal polls
        polls += 1
        return polls == 3

    monkeypatch.setattr(rain_strategy, "cancellation_checkpoint", row_checkpoint)
    with cancellation_scope(cancel), pytest.raises(SimulationCancelled):
        _green_decision(budget=0, laps=40)
    assert polls == 3


@pytest.mark.parametrize("owner,name", [
    (Tire, "time_penalty_per_lap"), (Tire, "wear_loss_at_lap"),
    (Weather, "lap_time_multiplier"), (Weather, "wet_severity"),
    (Car, "pace_delta_seconds"), (Track, "total_active_aero_gain"),
])
def test_shared_green_bypasses_stateful_model_hooks(monkeypatch, green_cache, owner, name):
    _green_decision()
    before = dict(green_cache)
    hook = getattr(owner, name)
    delta = .125
    calls = 0

    def changed(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return (hook.fget(self) if isinstance(hook, property)
                else hook(self, *args, **kwargs)) + delta

    monkeypatch.setattr(owner, name, property(changed) if isinstance(hook, property) else changed)
    first = _green_decision()
    initial = calls
    delta = .25
    second = _green_decision()
    assert first != second
    assert calls > initial > 0
    assert dict(green_cache) == before


def test_shared_green_bypasses_custom_prepared_evaluator(monkeypatch, green_cache):
    _green_decision()
    before = dict(green_cache)
    prepare = LapSimulator.prepare_deterministic_lap_time
    calls = 0

    def custom(self, *args):
        evaluate = prepare(self, *args)
        def stateful(*values):
            nonlocal calls
            calls += 1
            return evaluate(*values) + calls * .001
        return stateful

    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", custom)
    assert _green_decision() != _green_decision()
    assert calls > 0
    assert dict(green_cache) == before


def test_shared_green_compound_factor_mutations_remain_visible(monkeypatch, green_cache):
    first = _green_decision()
    before = dict(green_cache)
    monkeypatch.setitem(LapSimulator._COMPOUND_PACE_FACTORS, TireCompound.INTERMEDIATE, .017)
    changed = _green_decision()
    assert changed != first
    assert dict(green_cache) == before
    monkeypatch.setattr(rain_strategy, "_native_green_cache_available", lambda _: False)
    assert _green_decision() == changed


@pytest.mark.parametrize("name", ["minimum_lap_time", "_track_car_delta_from_values",
                                  "_weather_pace_multiplier_from_values"])
def test_shared_green_bypasses_scalar_physics_hooks_and_resumes_native_hits(
    monkeypatch, green_cache, name,
):
    first = _green_decision()
    before = dict(green_cache)
    physics = rain_strategy.lap_physics
    hook = getattr(physics, name)
    calls = 0

    def changed(*args, **kwargs):
        nonlocal calls
        calls += 1
        return hook(*args, **kwargs) + .125

    with monkeypatch.context() as patch:
        patch.setattr(physics, name, changed)
        _green_decision()
        initial = calls
        _green_decision()
        assert calls > initial > 0
        assert dict(green_cache) == before
    # Restoring native dispatch can safely reuse the original entries.
    def no_miss(*args):
        pytest.fail("Restored native physics should hit the existing shared cache")
    original = rain_strategy._shared_green_lap
    monkeypatch.setattr(rain_strategy, "_shared_green_lap",
                        lambda key, evaluate, *args: original(key, no_miss, *args))
    assert _green_decision() == first


def test_shared_green_after_fork_replaces_storage_and_lock(green_cache):
    _green_decision()
    old_lock = rain_strategy._green_lap_lock
    assert green_cache
    rain_strategy._reset_green_cache_after_fork()
    assert rain_strategy._green_laps is not green_cache
    assert not rain_strategy._green_laps
    assert rain_strategy._green_lap_lock is not old_lock


@pytest.mark.parametrize("target,name,budget", [
    (target, name, budget)
    for target in ("retained", "fresh")
    for name in ("time_penalty_per_lap", "wear_loss_at_lap")
    for budget in ([3] if target == "fresh" else [0, 3])
] + [(target, name, budget)
     for target in ("weather", "projected")
     for name in ("lap_time_multiplier", "wet_severity")
     for budget in (0, 3)
] + [("car", "pace_delta_seconds", budget) for budget in (0, 3)])
def test_shared_green_bypasses_nonserialized_instance_hooks(
    monkeypatch, green_cache, target, name, budget,
):
    driver, car, track, weather = models(laps=7)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(deep=True)
    clock = clock_for(track)

    def decide():
        return plan_rain_stop(driver, car, track, weather, tire, 13, 1, budget,
                              weather_clock=clock)

    native = decide()
    before = dict(green_cache)
    delta = .125
    calls = 0

    def attach(model):
        original = getattr(model, name)

        def changed(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs) + delta

        changed_model = model.model_copy(update={name: changed})
        assert changed_model.model_dump_json() == model.model_dump_json()
        return changed_model

    if target == "retained":
        tire = attach(tire)
    elif target == "fresh":
        monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.INTERMEDIATE, attach(tire))
    elif target == "weather":
        weather = attach(weather)
    elif target == "car":
        car = attach(car)
    else:
        project = Weather.project_surface
        monkeypatch.setattr(Weather, "project_surface", lambda self: attach(project(self)))

    first = decide()
    initial = calls
    delta = .25
    second = decide()
    assert first != native
    assert second != first
    assert calls > initial > 0
    assert dict(green_cache) == before
    monkeypatch.setattr(rain_strategy, "_native_green_cache_available", lambda _: False)
    assert decide() == second


def test_shared_green_bypasses_instance_shadow_of_track_property(monkeypatch, green_cache):
    driver, car, track, weather = models(laps=7)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    clock = clock_for(track)
    track = track.model_copy(update={"total_active_aero_gain": lambda: .125})
    assert not rain_strategy._native_green_model_available(track)
    actual = plan_rain_stop(driver, car, track, weather, tire, 13, 1, 0, weather_clock=clock)
    assert not green_cache
    monkeypatch.setattr(rain_strategy, "_native_green_cache_available", lambda _: False)
    assert plan_rain_stop(driver, car, track, weather, tire, 13, 1, 0,
                          weather_clock=clock) == actual
