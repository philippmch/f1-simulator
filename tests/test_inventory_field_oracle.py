"""Independent tyre enumeration observes actual service exits and field gaps."""

from copy import deepcopy

import pytest
from test_chronological_field_finish import FieldPhysics, field, native_path
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.models import TireCompound, Weather
from f1sim.models._native import native_physics
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext


@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("relative_laps", [-1, 0, 1])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("stopped", [False, True])
def test_enumerated_weather_and_gaps_match_complete_native_field_crossings(
    monkeypatch, control, relative_laps, off_track, stopped,
):
    inputs = field(control=control, intervals=6, relative_laps=relative_laps,
                   off_track=off_track, remaining=120., neutralized=True,
                   fee=7. if off_track else 0., now=6850.)
    driver, car, physical_track, observed, now = inputs
    weather = Weather(track_wetness=.12, change_probability=0.)
    calculator = LapSimulator()

    def mean(self, driver, car, track, tire, surface, lap, total, **options):
        if driver.id != "A":
            return self.paces[driver.id]
        return calculator.calculate_lap_time(
            driver, car, track, tire, surface, lap, total, sample_variation=False, **options)

    monkeypatch.setattr(FieldPhysics, "calculate_lap_time", mean)
    warmup = {"medium": 3., "soft": 5., "hard": 7.}
    distance, crossing, _ = native_path(inputs, stopped=stopped, delay=200., weather=weather,
                                       warmup=warmup, pending_fit=True)
    # This planning horizon covers the timed finish; fuel still uses all 90 laps.
    track = physical_track.model_copy(update={"total_laps": 75})
    stock = TireInventory.from_sets([
        dict(id="M", compound="medium", age=8), dict(id="S", compound="soft")])
    stock.fit("M")
    options = dict(current_lap=72, tire_age=8, remaining_stops=0, force_stop=stopped,
                   used_compounds=(TireCompound.HARD, TireCompound.MEDIUM),
                   physical_total_laps=90, tire_warmup=warmup, current_fit_pending=True,
                   control_context=StrategyControlContext(observed, now, 200.))
    assert native_physics(driver, car, track, weather)
    expected, _ = independent_schedules((driver, car, track, weather, stock), options)
    assert expected[stopped] == pytest.approx((True, distance, now - crossing), abs=1.e-8)


@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("relative_laps", [-1, 0, 1])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("scenario", ["drying", "rising_rain", "compound_credit"])
@pytest.mark.parametrize("budget", [0, 2])
def test_physical_choices_through_a_timed_field_match_all_crossing_schedules(
    control, relative_laps, off_track, scenario, budget,
):
    driver, car, track, observed, now = field(
        control=control, intervals=6, relative_laps=relative_laps,
        off_track=off_track, remaining=120., neutralized=True,
        fee=7. if off_track else 0., now=6850.)
    track = track.model_copy(update={"total_laps": 75})
    water, rain, records, history, forecast = {
        "drying": (.24, 0., [dict(id="I", compound="intermediate", age=8, remaining_laps=2),
                             dict(id="W", compound="wet", age=4, remaining_laps=1),
                             dict(id="M", compound="medium", age=35, remaining_laps=2)], (), None),
        "rising_rain": (.4, .8, [dict(id="I", compound="intermediate", age=8, remaining_laps=1),
                                  dict(id="W", compound="wet", age=4, remaining_laps=2),
                                  dict(id="old-W", compound="wet", age=40, remaining_laps=1)], (),
                        WeatherForecastContext.from_schedule([
                            dict(lap=2, rain_intensity=.85), dict(lap=4, rain_intensity=0.)])),
        "compound_credit": (.1, .15, [dict(id="H", compound="hard", age=8, remaining_laps=2),
                                      dict(id="H2", compound="hard", age=35, remaining_laps=1),
                                      dict(id="S", compound="soft", age=8, remaining_laps=1)],
                            (TireCompound.HARD,), None),
    }[scenario]
    weather = Weather(track_wetness=water, rain_intensity=rain, change_probability=0.)
    stock = TireInventory.from_sets(records)
    stock.fit(records[0]["id"])
    options = dict(tire_age=8, remaining_stops=budget,
                   remaining_dry_stops=budget, remaining_damp_stops=budget,
                   used_compounds=history, physical_total_laps=90,
                   tire_warmup={"medium": 3., "soft": 5., "hard": 7.,
                                "intermediate": 2., "wet": 4.},
                   current_fit_pending=True, forecast_context=forecast,
                   control_context=StrategyControlContext(observed, now, 200.))
    models = driver, car, track, weather, stock
    before = deepcopy((models[:-1], stock.__dict__, options))
    assert native_physics(driver, car, track, weather)
    expected, choices = independent_schedules(models, dict(options, current_lap=72))
    actual = plan_inventory_strategy(*models, 72, **options)
    for stopped in (False, True):
        assert_continuation(actual.continuation(stopped), expected[stopped])
    if actual.set_id is not None:
        assert choices[actual.set_id] == pytest.approx(expected[True], abs=1.e-8)
    assert (models[:-1], stock.__dict__, options) == before
