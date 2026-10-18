"""Exact native row reuse without changing the shared scalar row cache."""

import gc
import sys
import weakref

import pytest
from test_paid_weather_compounds import models

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.models import Driver, TireCompound, Weather
from f1sim.models._native import forecast_decision, native_forecast_memo
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import rain_strategy as strategy
from f1sim.simulation.weather_schedule import WeatherForecastContext


def clear_work():
    strategy._transition_plan.cache_clear()
    strategy._running_row.cache_clear()
    strategy._reset_transition_cache_after_fork()


def bits(row):
    return tuple(value.hex() for value in row)


def collect_rows(args, options, *, reuse):
    rows = []
    reused = []
    code = strategy._running_row.__wrapped__.__code__

    def profile(frame, event, result):
        if frame.f_code is not code:
            return
        if event == 'call' and not reuse:
            native_forecast_memo(strategy._PREPARED_ROWS).clear()
        if event == 'return' and result is not None:
            values = frame.f_locals
            inputs = tuple(values[name] for name in (
                'models', 'weather_json', 'tire_json', 'age', 'lap', 'physical',
                'intervals', 'observed_surface'))
            rows.append((inputs, bits(result)))
            reused.append(values['reuse'])

    clear_work()
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        result = strategy.plan_rain_transition(*args, **options)
    finally:
        sys.setprofile(previous)
    return result, rows, reused, strategy._running_row.cache_info()


@pytest.mark.parametrize('cadence', [None, (0, 1, 2, 3, 4, 5),
                                    (0, 0, 2, 2, 7, 8), (0, 2, 4, 8, 8, 13)])
@pytest.mark.parametrize('wet,rain,scheduled', [(.1, .1, False), (.19, 0., False),
                                             (.3, .1, True)])
def test_every_native_cell_and_lru_access_matches_original_suffix(cadence, wet, rain, scheduled):
    driver, car, track = models(laps=7)
    context = (WeatherForecastContext.from_schedule(
        [{'lap': 5, 'rain_intensity': .85}, {'lap': 9, 'rain_intensity': 0.}],
        leading_lap=3) if scheduled else None)
    args = (driver, car, track, Weather(track_wetness=wet, rain_intensity=rain),
            TIRE_COMPOUNDS[TireCompound.SOFT], 4, 2, 2)
    options = dict(weather_intervals=cadence, forecast_context=context,
                   physical_total_laps=13, used_compounds=(TireCompound.SOFT,),
                   remaining_dry_stops=1, remaining_damp_stops=1,
                   tire_warmup={compound.value: 7.3 for compound in TireCompound},
                   current_fit_pending=True)
    actual, rows, reused, info = collect_rows(args, options, reuse=True)
    assert rows and all(reused)
    # Outside a decision there is no native workspace: this is the original
    # independent preparation/projection path, including every suffix cell.
    for inputs, expected_bits in rows:
        assert bits(strategy._running_row.__wrapped__(*inputs)) == expected_bits
    expected, ordinary_rows, ordinary_reuse, ordinary_info = collect_rows(
        args, options, reuse=False,
    )
    assert actual == expected
    assert rows == ordinary_rows
    assert not any(ordinary_reuse)
    assert info == ordinary_info


def test_workspace_lifetime_nested_replacement_and_custom_scope():
    class CustomDriver(Driver):
        pass

    driver, car, track = models(laps=5)
    args = (driver, car, track, Weather(track_wetness=.1, rain_intensity=.1),
            TIRE_COMPOUNDS[TireCompound.SOFT], 2, 1, 1)
    refs = []

    @forecast_decision
    def custom(model):
        assert native_forecast_memo(strategy._PREPARED_ROWS) is None

    @forecast_decision
    def nested():
        assert native_forecast_memo(strategy._PREPARED_ROWS) is memo

    @forecast_decision
    def outer():
        nonlocal memo
        memo = native_forecast_memo(strategy._PREPARED_ROWS)
        nested()
        custom(CustomDriver(**driver.model_dump()))
        assert native_forecast_memo(strategy._PREPARED_ROWS) is memo
        for age in (2, 3):
            clear_work()
            strategy.plan_rain_transition(*args[:5], age, *args[6:], used_compounds=())
            assert list(memo) == ['current']
            refs.append(weakref.ref(memo['current'][6]))
        gc.collect()
        assert refs[0]() is None and refs[1]() is not None

    memo = None
    assert native_forecast_memo(strategy._PREPARED_ROWS) is None
    outer()
    # Release the explicit test reference to the workspace as real callers do.
    memo = None
    gc.collect()
    assert all(ref() is None for ref in refs)
    assert native_forecast_memo(strategy._PREPARED_ROWS) is None


def test_mismatched_packages_and_observed_surface_keep_independent_path():
    driver, car, track = models(laps=7)
    rows = []
    code = strategy._running_row.__wrapped__.__code__

    def profile(frame, event, result):
        if frame.f_code is code and event == 'return':
            assert not frame.f_locals['reuse']

    @forecast_decision
    def run():
        clear_work()
        strategy.plan_rain_transition(
            driver, car, track, Weather(track_wetness=.23, rain_intensity=0.),
            TIRE_COMPOUNDS[TireCompound.SOFT], 4, 2, 1, used_compounds=())
        record = native_forecast_memo(strategy._PREPARED_ROWS)['current']
        base = [record[0], record[3][0], TIRE_COMPOUNDS[TireCompound.HARD].model_dump_json(),
                6, record[2], record[1], record[4][0], False]
        variants = []
        changed_driver = driver.model_copy(update={'skill_rating': .99})
        variants.append((0, (changed_driver.model_dump_json(), *record[0][1:])))
        variants.extend([(1, Weather(track_wetness=.24).model_dump_json()),
                         (4, 1), (5, record[1] + 5),
                         (6, (0, 0, 2, 2, 4, 5)), (7, True)])
        previous = sys.getprofile()
        sys.setprofile(profile)
        try:
            for index, value in variants:
                inputs = base.copy()
                inputs[index] = value
                result = strategy._running_row.__wrapped__(*inputs)
                rows.append((inputs, bits(result)))
        finally:
            sys.setprofile(previous)

    run()
    for inputs, expected in rows:
        assert bits(strategy._running_row.__wrapped__(*inputs)) == expected


@pytest.mark.parametrize('cancel', [False, True])
def test_reused_row_keeps_cancellation_and_exception_cleanup(cancel):
    driver, car, track = models(laps=30)
    refs = []

    @forecast_decision
    def run():
        clear_work()
        strategy.plan_rain_transition(
            driver, car, track, Weather(track_wetness=.1, rain_intensity=.1),
            TIRE_COMPOUNDS[TireCompound.SOFT], 2, 1, 1, used_compounds=())
        record = native_forecast_memo(strategy._PREPARED_ROWS)['current']
        refs.append(weakref.ref(record[6]))
        if not cancel:
            raise RuntimeError('after completed native decision')
        polls = 0

        def request_cancel():
            nonlocal polls
            polls += 1
            return polls == 2

        with cancellation_scope(request_cancel):
            strategy._running_row.__wrapped__(
                record[0], record[3][0], TIRE_COMPOUNDS[TireCompound.HARD].model_dump_json(),
                3, record[2], record[1], record[4][0])

    with pytest.raises(SimulationCancelled if cancel else RuntimeError):
        run()
    gc.collect()
    assert refs[0]() is None
    assert native_forecast_memo(strategy._PREPARED_ROWS) is None
