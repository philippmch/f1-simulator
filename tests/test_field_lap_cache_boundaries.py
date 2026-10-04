"""Native field reuse preserves own weather clocks and custom surface dispatch."""

import runpy
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import PrivateAttr
from test_controlled_weather_strategy import inputs
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.models import Weather, _native
from f1sim.simulation import inventory_strategy
from f1sim.simulation.chronological_finish import ChronologicalFinishCar, ChronologicalFinishContext
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.strategy_lap import control_lap_scope
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.weather_schedule import WeatherForecastContext


class OpaqueWeather(Weather):
    _scale: float = PrivateAttr(default=1.)

    def lap_time_multiplier(self):
        return super().lap_time_multiplier() * self._scale


class EvolvingOpaqueWeather(OpaqueWeather):
    def project_surface(self):
        projected = super().project_surface()
        projected._scale += .1
        return projected


def disable_field_lap_reuse(monkeypatch):
    def replacement(*args):
        return None
    monkeypatch.setattr(inventory_strategy, "control_lap_memo", replacement)
    # Preserve the native solver on both sides; only sharing is disabled.
    monkeypatch.setattr(_native, "_HELPERS", [
        (namespace, name, replacement if namespace is inventory_strategy.__dict__
         and name == "control_lap_memo" else value)
        for namespace, name, value in _native._HELPERS])


@pytest.mark.parametrize("extension", ["subclass", "instance_hook"])
def test_custom_weather_with_equal_serialized_fields_keeps_its_actual_lap_physics(extension):
    driver, car, track, weather, stock = inputs(.2, .2, 4)
    clock = StrategyWeatherClock((0., 90., 180., 270.), 35., 90., 5, 20., 20.)
    options = dict(tire_age=11, remaining_stops=2, weather_clock=clock,
                   physical_total_laps=12, used_compounds=("intermediate",))
    surfaces = []
    for scale in (1., 1.2, 1.):
        if extension == "subclass":
            surface = OpaqueWeather(**weather.model_dump())
            surface._scale = scale
        else:
            surface = weather.model_copy(deep=True)
            surface.__dict__["lap_time_multiplier"] = lambda scale=scale: 1.04 * scale
        assert not _native.native_physics(driver, car, track, surface)
        surfaces.append(surface)

    @control_lap_scope
    def evaluate():
        decisions = []
        for surface in surfaces:
            models = driver, car, track, surface, stock
            expected, _ = independent_schedules(models, options)
            actual = plan_inventory_strategy(*models, 1, **options)
            for stopped in (False, True):
                assert_continuation(actual.continuation(stopped), expected[stopped])
            decisions.append(actual)
        assert decisions[0] == decisions[2]
        assert decisions[0].wait_cost < decisions[1].wait_cost

    before = deepcopy((driver, car, track, surfaces, stock.__dict__))
    evaluate()
    assert (driver, car, track, surfaces, stock.__dict__) == before


@pytest.mark.parametrize("control", ["sc", "vsc"])
def test_paid_weather_updates_in_the_public_controlled_planner_preserve_custom_dispatch(
    monkeypatch, control,
):
    driver, car, track, weather, stock = inputs(.2, .2, 4)
    track.pit_lane_delta = 200.
    weather = EvolvingOpaqueWeather(**weather.model_dump())
    ledger = RaceFinishTimeline(12, ["A", "B"])
    observed = StrategyControlContext(ChronologicalFinishContext(
        "A", ledger, ("A", "B"),
        (ChronologicalFinishCar("B", 0, 80., 80., 0., False, 1),), 100.,
        1.4 if control == "sc" else 1.2, control == "sc", control_intervals=3,
    ), 50., 200.)
    options = dict(tire_age=11, remaining_stops=2, control_context=observed,
                   physical_total_laps=12, used_compounds=("intermediate",))
    before = deepcopy((driver, car, track, weather, stock.__dict__, observed))
    actual = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    disable_field_lap_reuse(monkeypatch)
    independent = plan_inventory_strategy(driver, car, track, weather, stock, 1, **options)
    assert actual == independent
    assert (driver, car, track, weather, stock.__dict__, observed) == before


@pytest.mark.parametrize("control", ["sc", "vsc"])
def test_leading_native_candidate_reuses_laps_and_fresh_service_without_changing_decisions(
    monkeypatch, control,
):
    benchmark = runpy.run_path(str(
        Path(__file__).parents[1] / "examples/benchmark_controlled_strategy.py"))["benchmark"]
    options = dict(control=control, laps=30, drivers=4, intervals=3, profile=True, leader=True)
    shared = benchmark(**options)
    disable_field_lap_reuse(monkeypatch)
    independent = benchmark(**options)
    assert shared["native"] and independent["native"]
    assert shared["decision"] == independent["decision"]
    assert shared["outcome_sha256"] == independent["outcome_sha256"]
    assert shared["green_suffix_evaluations"] == independent["green_suffix_evaluations"]
    for count in ("fresh_service_frame_visits", "native_lap_evaluations"):
        assert 0 < shared[count] < independent[count]


@pytest.mark.parametrize("changed", ["cadence", "forecast", "fuel", "weather", "lane", "stock"])
def test_own_clock_shared_costs_keep_changed_forecasts_and_physical_limits_independent(changed):
    models = inputs(.2, .2, 4)
    forecast = WeatherForecastContext.from_schedule([
        dict(lap=2, rain_intensity=.6), dict(lap=4, rain_intensity=0.),
    ])
    options = dict(tire_age=11, remaining_stops=2, physical_total_laps=12,
                   used_compounds=("intermediate",), forecast_context=forecast)

    @control_lap_scope
    def compare():
        for alternate in (False, True, False):
            active_models = deepcopy(models)
            active_options = dict(options)
            if alternate:
                if changed == "cadence":
                    active_options["weather_intervals"] = (0, 0, 2, 5)
                elif changed == "forecast":
                    active_options["forecast_context"] = forecast.advanced(2)
                elif changed == "fuel":
                    active_options["physical_total_laps"] = 30
                elif changed == "weather":
                    active_models[3].track_temperature = 50.
                    active_models[3].track_wetness = .4
                elif changed == "lane":
                    active_models[2].pit_lane_delta = 25.
                elif changed == "stock":
                    active_models[-1].sets = {
                        key: replace(item, remaining_laps=
                                     5 if key == active_models[-1].current_set_id else 2)
                        for key, item in active_models[-1].sets.items()}
            expected, _ = independent_schedules(active_models, active_options)
            actual = plan_inventory_strategy(*active_models, 1, **active_options)
            for stopped in (False, True):
                assert_continuation(actual.continuation(stopped), expected[stopped])

    compare()
