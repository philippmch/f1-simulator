"""Pure deterministic finish-distance protection forecasts."""

from copy import deepcopy

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.finish_strategy import ReplacementOption, evaluate_finish_protection
from f1sim.simulation.lap import LapSimulator, minimum_lap_time


class ConstantPhysics:
    def __init__(self, pace=100.0):
        self.pace = pace
        self.calls = []

    def calculate_lap_time(self, driver, car, track, tire, weather, lap, total, **kwargs):
        self.calls.append((lap, driver.current_tire_laps, tire.compound, weather.track_wetness,
                           kwargs.get("gap_to_car_ahead"), kwargs.get("sample_variation")))
        return self.pace


def models(total_laps=10):
    return (
        Driver(id="A", name="A", team_id="A"),
        Car(team_id="A", team_name="A"),
        Track(id="T", name="T", country="T", total_laps=total_laps, base_lap_time=100,
              pit_lane_delta=20),
        Weather(),
    )


def forecast(physics, *, flag=250, max_lap=10, lane=80, service=0, queue=0,
             current_lap=1, tire_age=0, replacements=None, surface_at=None,
             observed_gap=None):
    driver, car, track, weather = models(max_lap)
    return evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], tire_age,
        current_lap, weather, 0, flag, max_lap, lap_simulator=physics,
        expected_lane_loss=lane, expected_service_time=service,
        expected_queue_delay=queue, replacements=replacements,
        projected_surface_at=surface_at, observed_gap=observed_gap,
    )


def test_veto_requires_more_retained_laps_than_optimistic_stop():
    result = forecast(ConstantPhysics(), flag=250, lane=80)

    assert result.retained_laps == 3
    assert result.stop_laps == 2
    assert result.veto


def test_exact_flag_distance_tie_preserves_native_decision():
    result = forecast(ConstantPhysics(), flag=400, lane=100)

    assert result.retained_laps == result.stop_laps == 4
    assert not result.veto


def test_stop_future_laps_use_absolute_green_floor_without_physics_rollouts():
    physics = ConstantPhysics(pace=200)
    result = forecast(physics, flag=550, lane=80)

    # The outlap uses mean physics at 200 seconds; later optimistic laps use
    # the shared 95%-reference-lap floor (95 seconds), not the 200-second
    # mocked physics value.
    assert result.stop_laps == 4
    assert result.stop_crossing_time == 565
    assert [call[0] for call in physics.calls if call[2] == TireCompound.SOFT] == [1]


def test_finite_replacement_age_is_used_without_inventory_mutation():
    physics = ConstantPhysics()
    options = (ReplacementOption(TireCompound.SOFT, age=7, identifier="fresh-ish"),)
    result = forecast(physics, flag=250, lane=80, replacements=options)

    assert result.stop_compound == TireCompound.SOFT
    assert result.veto
    assert [call[1] for call in physics.calls if call[2] == TireCompound.SOFT] == [7]
    assert options[0].age == 7


def test_absolute_surface_callback_includes_expected_stop_exit_and_future_entries():
    physics = ConstantPhysics()
    observed = []

    def surface_at(absolute_time):
        observed.append(absolute_time)
        return Weather()

    result = forecast(physics, flag=250, lane=80, surface_at=surface_at)

    assert result.stop_laps == 2
    assert 0 in observed
    assert 80 in observed
    assert any(time > 80 for time in observed)


def test_forecast_does_not_mutate_models_or_rng():
    driver, car, track, weather = models()
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    rng = np.random.default_rng(7)
    physics = LapSimulator(rng)
    before_driver = driver.model_dump_json()
    before_weather = weather.model_dump_json()
    before_rng = deepcopy(rng.bit_generator.state)
    result = evaluate_finish_protection(
        driver, car, track, tire, 4, 3, weather, 100, 450, track.total_laps,
        lap_simulator=physics, expected_lane_loss=20,
        replacements=(ReplacementOption(TireCompound.SOFT, age=2),),
    )

    assert result.retained_feasible and result.stop_feasible
    assert driver.model_dump_json() == before_driver
    assert weather.model_dump_json() == before_weather
    assert rng.bit_generator.state == before_rng
    assert minimum_lap_time(track) == 95


def test_critical_retained_surface_is_inconclusive():
    physics = ConstantPhysics()
    driver, car, track, weather = models()
    wet = Weather(track_wetness=.6)
    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 0, 1, weather,
        0, 250, track.total_laps, lap_simulator=physics,
        projected_surface_at=lambda _: wet,
    )

    assert not result.retained_feasible
    assert not result.veto
