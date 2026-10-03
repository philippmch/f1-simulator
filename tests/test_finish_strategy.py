"""Pure deterministic finish-distance protection forecasts."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Track, Weather
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


@pytest.mark.parametrize("modifier", [1.2, 1.4])
@pytest.mark.parametrize("pending_fit", [False, True])
def test_neutralized_first_lap_prices_native_aero_and_fitting_costs(modifier, pending_fit):
    driver, car, track, weather = models()
    track.active_aero_zones = [ActiveAeroZone(zone_id=1, sector=1, time_gain=.7)]
    driver.current_tire_laps = 4
    physics = LapSimulator(np.random.default_rng(13))
    before_rng = deepcopy(physics.rng.bit_generator.state)
    current = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    options = dict(lap_simulator=physics, expected_lane_loss=7., expected_service_time=4.,
                   expected_queue_delay=3., current_lap_time_modifier=modifier,
                   replacements=(ReplacementOption(TireCompound.SOFT, age=7),),
                   tire_warmup={"soft": 5., "medium": 3.}, current_fit_pending=pending_fit)
    result = evaluate_finish_protection(
        driver, car, track, current, 4, 1, weather, 0., 10., 10,
        active_aero_enabled=False, **options,
    )
    retained = physics.calculate_lap_time(driver, car, track, current, weather, 1, 10,
                                         sample_variation=False, active_aero_enabled=False)
    replacement_driver = driver.model_copy(update={"current_tire_laps": 7}, deep=True)
    outlap = physics.calculate_lap_time(replacement_driver, car, track,
                                       TIRE_COMPOUNDS[TireCompound.SOFT], weather, 1, 10,
                                       sample_variation=False, active_aero_enabled=False)
    assert result.retained_crossing_time == pytest.approx(
        retained * modifier + (3. if pending_fit else 0.))
    assert result.stop_crossing_time == pytest.approx(14. + outlap * modifier + 5.)
    assert result.retained_laps == result.stop_laps == 1 and not result.veto
    aero = evaluate_finish_protection(
        driver, car, track, current, 4, 1, weather, 0., 10., 10,
        active_aero_enabled=True, **options,
    )
    assert aero.retained_crossing_time < result.retained_crossing_time
    assert aero.stop_crossing_time < result.stop_crossing_time
    assert physics.rng.bit_generator.state == before_rng
    assert driver.current_tire_laps == 4


@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_unavailable_aero_state_cannot_veto(invalid):
    driver, car, track, weather = models()
    physics = ConstantPhysics()
    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 0, 1, weather,
        0., 250., 10, lap_simulator=physics, active_aero_enabled=invalid,
    )
    assert not result.veto and result.reason == "invalid active aero state"
    assert not physics.calls
