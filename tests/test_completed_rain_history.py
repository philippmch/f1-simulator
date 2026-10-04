"""Reuse completed compound credit without changing executable rain strategies."""

from copy import deepcopy
from inspect import currentframe
from math import inf

import pytest
from test_paid_weather_compounds import exhaustive_safe_actions, models

from f1sim.cancellation import SimulationCancelled
from f1sim.models import TireCompound, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import rain_strategy
from f1sim.simulation.rain_strategy import plan_rain_transition
from f1sim.simulation.weather_schedule import WeatherForecastContext

USED = [
    (), (TireCompound.SOFT,), (TireCompound.SOFT, TireCompound.MEDIUM),
    (TireCompound.SOFT, TireCompound.HARD),
    (TireCompound.MEDIUM, TireCompound.HARD),
    (TireCompound.INTERMEDIATE,), (TireCompound.SOFT, TireCompound.WET),
]


def clear_transition_work():
    rain_strategy._transition_plan.cache_clear()
    rain_strategy._running_row.cache_clear()
    rain_strategy._reset_transition_cache_after_fork()


@pytest.mark.parametrize("used", USED)
@pytest.mark.parametrize("budget,limit", [(0, 0), (2, 1)])
def test_completed_credit_matches_independent_safe_schedules(used, budget, limit):
    driver, car, track = models(laps=5)
    weather = Weather(track_wetness=.1, rain_intensity=.1)
    context = WeatherForecastContext.from_schedule(
        [{"lap": 3, "rain_intensity": .8}, {"lap": 5, "rain_intensity": 0.}],
    )
    tire = TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(deep=True)
    expected = exhaustive_safe_actions(
        driver, car, track, weather, tire, 7, 1, budget, context,
        used=used, dry=limit, damp=limit, warmup=7., pending=True,
        physical=9, lane=.75, queue=3., modifier=1.2, aero=False, gaps=(.5, 1.7),
    )
    options = dict(
        used_compounds=used, remaining_dry_stops=limit, remaining_damp_stops=limit,
        tire_warmup={compound.value: 7. for compound in TireCompound}, current_fit_pending=True,
        physical_total_laps=9, pit_lane_factor=.75, additional_current_stop_cost=3.,
        current_lap_time_modifier=1.2, active_aero_enabled=False, current_traffic_gaps=(.5, 1.7),
        forecast_context=context,
    )
    before = deepcopy((driver, car, track, weather, tire, options))
    actual = plan_rain_transition(driver, car, track, weather, tire, 7, 1, budget, **options)
    assert actual.pit_now_cost == pytest.approx(expected[0])
    assert actual.wait_cost == pytest.approx(expected[1])
    assert actual.compound == expected[2]
    assert (driver, car, track, weather, tire, options) == before


def test_completed_histories_reduce_physics_rows_with_identical_exact_costs(monkeypatch):
    driver, car, track = models(laps=20)
    args = (driver, car, track, Weather(track_wetness=.1, rain_intensity=.1),
            TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(deep=True), 5, 1, 3)
    clear_transition_work()
    native = plan_rain_transition(*args, used_compounds=())
    info = rain_strategy._running_row.cache_info()
    native_rows = info.hits + info.misses
    native_completed = len(rain_strategy._transition_suffixes)

    # Disable suffix sharing and the new history quotient while retaining the
    # identical native physics. The unmerged graph keeps every compound mask.
    monkeypatch.setattr(rain_strategy, "shared_forecast_available", lambda: False)
    clear_transition_work()
    unmerged = plan_rain_transition(*args, used_compounds=())
    info = rain_strategy._running_row.cache_info()
    original_rows = info.hits + info.misses

    assert native == unmerged
    assert native_completed > 50
    assert native_rows < original_rows


def test_one_slick_cannot_borrow_completed_credit_after_stop_budget_is_exhausted():
    driver, car, track = models(laps=2)
    args = (driver, car, track, Weather(), TIRE_COMPOUNDS[TireCompound.SOFT], 0, 1, 0)
    completed = plan_rain_transition(
        *args, used_compounds=(TireCompound.SOFT, TireCompound.HARD),
    )
    incomplete = plan_rain_transition(*args, used_compounds=(TireCompound.SOFT,))
    assert completed.pit_now_cost == inf
    assert incomplete.pit_now_cost < inf
    assert incomplete.wait_cost > completed.wait_cost


def test_unrun_rain_set_does_not_complete_credit_when_it_is_unsafe():
    driver, car, track = models(laps=1)
    actual = plan_rain_transition(
        driver, car, track, Weather(), TIRE_COMPOUNDS[TireCompound.WET], 0, 1, 0,
        used_compounds=(),
    )
    assert actual.wait_cost == actual.pit_now_cost == inf


def test_behavioral_weather_extensions_retain_unmerged_work(monkeypatch):
    class CustomWeather(Weather):
        def lap_time_multiplier(self):
            return super().lap_time_multiplier() * 1.01

    driver, car, track = models(laps=8)
    args = (driver, car, track, CustomWeather(track_wetness=.1, rain_intensity=.1),
            TIRE_COMPOUNDS[TireCompound.SOFT], 5, 1, 2)
    clear_transition_work()
    actual = plan_rain_transition(*args, used_compounds=(TireCompound.SOFT, TireCompound.HARD))
    assert not rain_strategy._transition_suffixes
    monkeypatch.setattr(rain_strategy, "shared_forecast_available", lambda: False)
    clear_transition_work()
    expected = plan_rain_transition(*args, used_compounds=(TireCompound.SOFT, TireCompound.HARD))
    assert actual == expected


def test_hot_native_stint_can_cancel_without_yielding_a_dependency(monkeypatch):
    driver, car, track = models(laps=20)
    args = (driver, car, track, Weather(track_wetness=.1, rain_intensity=.1),
            TIRE_COMPOUNDS[TireCompound.SOFT], 5, 1, 0)
    used = (TireCompound.SOFT, TireCompound.HARD)
    clear_transition_work()
    expected = plan_rain_transition(*args, used_compounds=used)
    rain_strategy._transition_plan.cache_clear()
    before = deepcopy(args)
    original = rain_strategy.cancellation_checkpoint
    cancelled = []

    def cancel_stint():
        original()
        # Completed credit and zero stop allowance leave no dependency to
        # yield. Warm running rows must still permit cancellation mid-scan.
        if currentframe().f_back.f_code.co_name == "stint":
            cancelled.append(True)
            raise SimulationCancelled("cancelled hot native scan")

    monkeypatch.setattr(rain_strategy, "cancellation_checkpoint", cancel_stint)
    with pytest.raises(SimulationCancelled, match="hot native scan"):
        plan_rain_transition(*args, used_compounds=used)
    assert cancelled
    assert rain_strategy._transition_plan.cache_info().currsize == 0
    assert args == before
    monkeypatch.setattr(rain_strategy, "cancellation_checkpoint", original)
    assert plan_rain_transition(*args, used_compounds=used) == expected
