"""Behavioral extensions cannot share forecasts keyed only by schema values."""

import os
import subprocess
import sys
from math import inf
from textwrap import dedent

import numpy as np
import pytest

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.models import Car, Driver, Tire, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation import opening_strategy, pit_strategy, rain_strategy, weather_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import projected_surfaces


def models(laps=3):
    return (Driver(id="d", name="D", team_id="t"), Car(team_id="t", team_name="T"),
            Track(id="t", name="T", country="T", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=1))


def clock(laps):
    return StrategyWeatherClock(tuple(90. * lap for lap in range(laps)), 1000., 1., 0, 1., 1.)


def public_wait(driver, car, track, weather, tire, age=0):
    driver = driver.model_copy(deep=True)
    simulator = LapSimulator(np.random.default_rng(0))
    result = 0.
    for lap in range(1, track.total_laps + 1):
        driver.current_tire_laps = age + lap - 1
        result += simulator.calculate_lap_time(
            driver, car, track, tire, weather, lap, track.total_laps, sample_variation=False,
        )
        weather = weather.project_surface()
    return result


@pytest.mark.parametrize("case", ["field", "helper", "clock", "rain", "transition", "dry"])
def test_hooks_installed_before_consumer_import(case):
    script = dedent('''
        import sys
        from f1sim.models import Car, Driver, Tire, Track, Weather
        from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
        state = {"extra": 0.}
        case = sys.argv[1]
        calls = []
        if case == "field":
            def skill(self):
                calls.append(1)
                return self.__dict__["skill_rating"] + state["extra"]
            Driver.skill_rating = property(skill)
        elif case == "clock":
            original = Weather.lap_time_multiplier
            Weather.lap_time_multiplier = lambda self: original(self) + state["extra"]
        elif case == "rain":
            original = Car.pace_delta_seconds
            Car.pace_delta_seconds = lambda self, *args: original(self, *args) + state["extra"]
        elif case == "transition":
            original = Weather.tire_mismatch
            Weather.tire_mismatch = lambda self, compound: (
                "critical" if state["extra"] else original(self, compound))
        elif case == "dry":
            original = Tire.time_penalty_per_lap
            Tire.time_penalty_per_lap = lambda self, *args: original(self, *args) + state["extra"]
        from f1sim.simulation.lap import LapSimulator
        if case == "helper":
            from f1sim.simulation import lap
            original = lap._track_car_delta_from_values
            lap._track_car_delta_from_values = lambda *args: original(*args) + state["extra"]
        from f1sim.simulation import rain_strategy as rain, pit_strategy as pit
        driver = Driver(id="d", name="D", team_id="t")
        car = Car(team_id="t", team_name="T")
        track = Track(id="t", name="T", country="T", total_laps=3,
                      base_lap_time=90, pit_lane_delta=1)
        if case in ("field", "helper"):
            assert LapSimulator().prepare_deterministic_lap_time(driver, car, track, 3) is None
            assert not calls
            sys.exit(0)
        weather = Weather(track_wetness=.5, rain_intensity=.5)
        tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
        options = {}
        if case == "clock":
            from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
            options["weather_clock"] = StrategyWeatherClock((0., 90., 180.), 1000., 1., 0, 1., 1.)
        if case == "transition":
            def decide():
                return rain.plan_rain_transition(driver, car, track, weather, tire, 2, 1, 1,
                                                used_compounds={"intermediate"})
        elif case == "dry":
            def decide():
                return pit.plan_dry_stop(driver, car, track,
                    TIRE_COMPOUNDS[TireCompound.MEDIUM], 2, 3, 1, {TireCompound.SOFT})
        else:
            def decide():
                return rain.plan_rain_stop(driver, car, track, weather, tire, 2, 1, 1, **options)
        first = decide()
        state["extra"] = 1.
        changed = decide()
        assert changed != first
        if case == "transition":
            assert changed.wait_cost == changed.pit_now_cost == float("inf")
        for module in (rain, pit):
            for value in vars(module).values():
                if hasattr(value, "cache_clear"):
                    value.cache_clear()
        rain._green_laps.clear()
        rain._transition_suffixes.clear()
        assert decide() == changed
    ''')
    result = subprocess.run(
        [sys.executable, "-c", script, case], capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("clocked", [False, True])
@pytest.mark.parametrize("planner", ["stop", "transition", "weather"])
def test_warm_forecasts_observe_later_stateful_car_hook(monkeypatch, clocked, planner):
    driver, car, track = models()
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    options = {"weather_clock": clock(3)} if clocked else {}

    def decide():
        if planner == "stop":
            return rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 0,
                                                **options).wait_cost
        if planner == "transition":
            return rain_strategy.plan_rain_transition(
                driver, car, track, weather, tire, 0, 1, 0,
                used_compounds={"intermediate"}, **options,
            ).wait_cost
        return weather_strategy.weather_stop_costs(
            driver, car, track, weather, tire, 0, 1, traffic_possible=False, **options,
        ).stay_cost

    native = decide()
    extra = [1.]
    original = Car.pace_delta_seconds
    monkeypatch.setattr(Car, "pace_delta_seconds",
                        lambda self, *args: original(self, *args) + extra[0])
    assert decide() == pytest.approx(public_wait(driver, car, track, weather, tire))
    assert decide() > native
    extra[0] = 2.
    assert decide() == pytest.approx(public_wait(driver, car, track, weather, tire))
    assert driver.current_tire_laps == 0


@pytest.mark.parametrize("kind", ["subclass", "instance", "extra"])
def test_equal_schema_retained_tire_keeps_distinct_extension_dispatch(kind):
    driver, car, track = models()
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    fresh = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]

    class CustomTire(Tire):
        def time_penalty_per_lap(self, *args):
            return super().time_penalty_per_lap(*args) + 2.

    retained = (CustomTire(**fresh.model_dump()) if kind == "subclass"
                else fresh.model_copy(deep=True))
    if kind != "subclass":
        def hook(*args):
            return fresh.time_penalty_per_lap(*args) + 2.
        if kind == "instance":
            object.__setattr__(retained, "time_penalty_per_lap", hook)
        else:
            object.__setattr__(retained, "__pydantic_extra__", {"time_penalty_per_lap": hook})
    native = rain_strategy.plan_rain_stop(driver, car, track, weather, fresh, 0, 1, 1)
    actual = rain_strategy.plan_rain_stop(driver, car, track, weather, retained, 0, 1, 1)
    if kind == "extra":
        # Pydantic extra cannot shadow an existing class method, but it must
        # never require serializing a callable to reach public dispatch.
        assert actual == native
    else:
        assert actual.wait_cost > native.wait_cost
        assert actual.pit_now_cost == native.pit_now_cost
        no_stop = rain_strategy.plan_rain_stop(driver, car, track, weather, retained, 0, 1, 0)
        expected = public_wait(driver, car, track, weather, retained)
        assert no_stop.wait_cost == pytest.approx(expected)


def test_custom_projection_at_equilibrium_is_not_canonicalized_or_shared(monkeypatch):
    weather = Weather(track_wetness=.4, rain_intensity=.4)
    assert projected_surfaces(weather, 3, (0, 2, 4)) == projected_surfaces(weather, 3)
    step = [.01]
    monkeypatch.setattr(Weather, "project_surface", lambda self: self.model_copy(
        update={"track_wetness": self.track_wetness + step[0]},
    ))
    first = projected_surfaces(weather, 3, (0, 2, 4))
    assert [value.track_wetness for value in first] == pytest.approx([.4, .42, .44])
    step[0] = .02
    second = projected_surfaces(weather, 3, (0, 2, 4))
    assert [value.track_wetness for value in second] == pytest.approx([.4, .44, .48])
    assert weather.track_wetness == .4


@pytest.mark.parametrize("planner", ["stop", "transition", "weather"])
def test_clocked_forecast_keeps_actual_weather_subclass(planner):
    class CustomWeather(Weather):
        def lap_time_multiplier(self):
            return super().lap_time_multiplier() + .1

    driver, car, track = models()
    weather = CustomWeather(track_wetness=.5, rain_intensity=.5)
    tire = TIRE_COMPOUNDS[TireCompound.WET if planner == "transition"
                               else TireCompound.INTERMEDIATE]
    options = {"weather_clock": clock(3)}
    if planner == "stop":
        actual = rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 0,
                                              **options).wait_cost
    elif planner == "transition":
        actual = rain_strategy.plan_rain_transition(
            driver, car, track, weather, tire, 0, 1, 0,
            used_compounds={"intermediate"}, **options,
        ).wait_cost
    else:
        actual = weather_strategy.weather_stop_costs(
            driver, car, track, weather, tire, 0, 1, traffic_possible=False, **options,
        ).stay_cost
    assert actual == pytest.approx(public_wait(driver, car, track, weather, tire))


def test_dry_floor_observes_stateful_tire_hook_after_warm(monkeypatch):
    driver, car, track = models()
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    used = {TireCompound.SOFT, TireCompound.HARD}
    pit_strategy.plan_dry_stop(driver, car, track, tire, 0, 3, 0, used)
    original = Tire.time_penalty_per_lap
    extra = [1.]
    monkeypatch.setattr(Tire, "time_penalty_per_lap",
                        lambda self, *args: original(self, *args) + extra[0])
    for value in (1., 2.):
        extra[0] = value
        result = pit_strategy.plan_dry_stop(driver, car, track, tire, 0, 3, 0, used)
        assert result.wait_cost == public_wait(driver, car, track, Weather(), tire)


@pytest.mark.parametrize("entry", ["dry", "wet", "inventory"])
def test_opening_cache_bypasses_stateful_model_extension(monkeypatch, entry):
    from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype

    driver, car, track = models(4)
    weather = Weather()
    simulator = RaceSimulator(np.random.default_rng(0))
    args = (driver, car, track, weather, TeamStrategyArchetype.BALANCED,
            simulator.strategy_tuning, simulator.strategy_profiles)
    extra = [0.]
    original = Car.pace_delta_seconds
    monkeypatch.setattr(Car, "pace_delta_seconds",
                        lambda self, *args: original(self, *args) + extra[0])
    entries = {"dry": opening_strategy.dry_opening_policy_costs,
               "wet": opening_strategy.opening_policy_costs,
               "inventory": opening_strategy.inventory_opening_policy_costs}
    records = [{"id": "s", "compound": "soft", "age": 0},
               {"id": "m", "compound": "medium", "age": 0}]

    def decide():
        return entries[entry](*args, *([records] if entry == "inventory" else []))

    first = decide()
    extra[0] = 1.
    changed = decide()
    assert changed != first
    extra[0] = 2.
    assert decide() != changed


def test_native_repeat_decisions_keep_shared_hits():
    driver, car, track = models()
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    rain_strategy._plan.cache_clear()
    first = rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 1)
    assert rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 1) == first
    assert rain_strategy._plan.cache_info().hits == 1
    assert first.pit_now_cost < inf


@pytest.mark.parametrize("planner", ["rain", "dry_floor"])
def test_warm_forecast_observes_transitive_service_hook(monkeypatch, planner):
    from f1sim.models import ActiveAeroZone

    driver, car, track = models(2)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    if planner == "dry_floor":
        track.active_aero_zones = [ActiveAeroZone(zone_id=index + 1, sector=1, time_gain=1.)
                                  for index in range(16)]
        tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]

    def decide():
        if planner == "rain":
            return rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 1)
        return pit_strategy.plan_dry_stop(
            driver, car, track, tire, 0, 2, 1, {TireCompound.SOFT, TireCompound.HARD},
        )

    baseline = decide()
    assert decide() == baseline
    original = pit_strategy._expected_service
    extra = [0.]
    monkeypatch.setattr(pit_strategy, "_expected_service",
                        lambda *args: original(*args) + extra[0])
    assert decide() == baseline
    for value in (3., 5.):
        extra[0] = value
        assert decide().pit_now_cost == pytest.approx(baseline.pit_now_cost + value)


def test_projection_cache_observes_changed_calibration_and_resumes_native_hits(monkeypatch):
    from f1sim.models import weather as weather_model
    from f1sim.simulation import surface_projection

    weather = Weather(track_wetness=.1, rain_intensity=.8)
    surface_projection._surface_snapshots.cache_clear()
    baseline = projected_surfaces(weather, 3)
    assert [value.track_wetness for value in baseline] == pytest.approx([.1, .24, .352])
    monkeypatch.setattr(weather_model, "WETNESS_RESPONSE_PER_LAP", .5)
    changed = projected_surfaces(weather, 3)
    assert [value.track_wetness for value in changed] == pytest.approx([.1, .45, .625])
    assert surface_projection._surface_snapshots.cache_info().hits == 0
    monkeypatch.setattr(weather_model, "WETNESS_RESPONSE_PER_LAP", .2)
    assert projected_surfaces(weather, 3) == baseline
    assert surface_projection._surface_snapshots.cache_info().hits == 1


@pytest.mark.parametrize("cancelled", [False, True])
def test_extension_scope_is_released_after_exception_and_preserves_caller(cancelled):
    class MutatingCar(Car):
        def pace_delta_seconds(self, base):
            self.base_pace = 0.
            raise RuntimeError("extension failed")

    driver, car, track = models()
    extension = MutatingCar(**car.model_dump())
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    if cancelled:
        with cancellation_scope(lambda: True), pytest.raises(SimulationCancelled):
            rain_strategy.plan_rain_stop(driver, extension, track, weather, tire, 0, 1, 0)
    else:
        with pytest.raises(RuntimeError, match="extension failed"):
            rain_strategy.plan_rain_stop(driver, extension, track, weather, tire, 0, 1, 0)
    assert extension.base_pace == car.base_pace
    rain_strategy._plan.cache_clear()
    first = rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 0)
    assert rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 0) == first
    assert rain_strategy._plan.cache_info().hits == 1
