"""Actual qualifying laps must realize the fixed candidate without changing race pace."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.qualifying_calibration import calibrate_qualifying_drivers
from f1sim.analysis.qualifying_history import recent_team_q1_predictions
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.saved_validation import validate_saved_model
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.output import Exporter
from f1sim.simulation.lap import LapSimulator


def inputs():
    drivers = [Driver(id=code, name=code, team_id=team, skill_rating=skill)
               for code, team, skill in [("A1", "a", .9), ("A2", "a", .96),
                                        ("B1", "b", .86), ("B2", "b", .92)]]
    cars = {team: Car(team_id=team, team_name=team, base_pace=pace)
            for team, pace in [("a", .8), ("b", .96)]}
    track = Track(id="track", name="Track", country="Test", total_laps=8, base_lap_time=90.)
    history = [{"round": n, "status": "scored", "teams": {
        "a": {"q1_count": 2, "residual": -.005 - n * .001},
        "b": {"q1_count": 2, "residual": .005 + n * .001},
    }} for n in range(1, 7)]
    return drivers, cars, track, history


def predict(drivers, cars, track):
    simulator = LapSimulator(np.random.default_rng(0))
    return [{"driver_id": driver.id, "team_id": driver.team_id,
             "predicted_seconds": min(simulator.calculate_qualifying_lap(
                 driver, cars[driver.team_id], track, tire, Weather(), sample_variation=False,
             ) for tire in TIRE_COMPOUNDS.values())} for driver in drivers]


def test_actual_qualifying_laps_match_fixed_candidate_without_changing_race_laps():
    drivers, cars, track, history = inputs()
    before = deepcopy((drivers, cars, track, history))
    reference, evidence = recent_team_q1_predictions(predict(drivers, cars, track), history, 5)
    calibrated, metadata = calibrate_qualifying_drivers(drivers, cars, track, history, 5)
    expected = {row["driver_id"]: row["predicted_seconds"] for row in reference}
    actual = {row["driver_id"]: row["predicted_seconds"]
              for row in predict(calibrated, cars, track)}
    assert actual == pytest.approx(expected, abs=1e-12)
    assert metadata["training_rounds"] == evidence["training_rounds"] == [2, 3, 4]
    assert any(driver.qualifying_pace_adjustment != 0. for driver in calibrated)
    simulator = LapSimulator(np.random.default_rng(0))
    for native, fitted in zip(drivers, calibrated):
        for tire in TIRE_COMPOUNDS.values():
            assert simulator.calculate_lap_time(
                native, cars[native.team_id], track, tire, Weather(), 3, 8,
                sample_variation=False,
            ) == simulator.calculate_lap_time(
                fitted, cars[fitted.team_id], track, tire, Weather(), 3, 8,
                sample_variation=False,
            )
    assert (drivers, cars, track, history) == before


def test_future_history_changes_and_repeated_calibration_do_not_compound_inputs():
    drivers, cars, track, history = inputs()
    first, _ = calibrate_qualifying_drivers(drivers, cars, track, history, 5)
    history[-1]["teams"]["a"]["residual"] = 50.
    history[-2]["teams"]["b"]["residual"] = -50.
    again, _ = calibrate_qualifying_drivers(first, cars, track, history, 5)
    assert again == first


def test_cold_start_and_out_of_bounds_evidence_retain_native_inputs():
    drivers, cars, track, history = inputs()
    cold, metadata = calibrate_qualifying_drivers(drivers, cars, track, history, 1)
    assert cold == drivers
    assert metadata["training_rounds"] == []
    for event in history:
        event["teams"]["a"]["residual"] = .5
    retained, metadata = calibrate_qualifying_drivers(drivers, cars, track, history, 5)
    assert retained == drivers
    assert metadata["candidate_fallback"] == "adjustment_out_of_bounds"


def test_calibration_respects_behavioral_extensions_and_strict_saved_inputs(monkeypatch):
    drivers, cars, track, history = inputs()
    fitted, _ = calibrate_qualifying_drivers(drivers, cars, track, history, 5)
    for driver in fitted:
        assert validate_saved_model(Driver, driver.model_dump(mode="json")) == driver
    original = LapSimulator.calculate_qualifying_lap
    monkeypatch.setattr(LapSimulator, "calculate_qualifying_lap",
                        lambda *a, **k: original(*a, **k) + 1.)
    retained, metadata = calibrate_qualifying_drivers(drivers, cars, track, history, 5)
    assert retained == drivers
    assert metadata["candidate_fallback"] == "custom_physics"


@pytest.mark.parametrize("value", [True, "0.01", float("inf"), float("nan"), .11, -.11])
def test_saved_qualifying_adjustments_reject_invalid_values(value):
    driver = inputs()[0][0].model_dump(mode="json")
    driver["qualifying_pace_adjustment"] = value
    with pytest.raises(ValueError):
        validate_saved_model(Driver, driver)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_calibrated_grids_survive_real_workers_and_second_trial_replay(tmp_path, engine):
    drivers, cars, track, history = inputs()
    fitted, _ = calibrate_qualifying_drivers(drivers, cars, track, history, 5)
    configured = MonteCarloRunner(
        fitted, cars, track, Weather(change_probability=0.), seed=42, race_engine=engine,
        qualifying_weather={"Q1": {"rain_intensity": .3, "track_wetness": .3}},
    )
    sequential = configured.run(2, parallel=False)
    parallel = configured.run(2, parallel=True, max_workers=2)
    assert parallel.race_results == sequential.race_results
    assert parallel.qualifying_results == sequential.qualifying_results
    assert parallel.event_stats == sequential.event_stats
    assert parallel.input_snapshot == sequential.input_snapshot
    path = Exporter(tmp_path).export_statistics_json(sequential)
    restored, count = _load_saved_runner(path)
    assert count == 2
    assert [driver.qualifying_pace_adjustment for driver in restored.drivers] == [
        driver.qualifying_pace_adjustment for driver in fitted]
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [sequential.race_results[1]]
    assert replay.qualifying_results == [sequential.qualifying_results[1]]


def test_legacy_driver_snapshot_has_zero_qualifying_adjustment():
    legacy = inputs()[0][0].model_dump(mode="json")
    legacy.pop("qualifying_pace_adjustment")
    assert validate_saved_model(Driver, legacy).qualifying_pace_adjustment == 0.
