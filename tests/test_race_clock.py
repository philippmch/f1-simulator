"""Dry clock scope, native physics parity, historical inputs and replay."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.race_clock import (
    DRY_RACE_CLOCK_ADJUSTMENT,
    dry_race_pace_adjustment_for_event,
)
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.data.current import CurrentSeasonDataLoader
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.output import Exporter
from f1sim.simulation.lap import LapSimulator


def models(adjustment=0.0):
    return (Driver(id="A", name="A", team_id="a", skill_rating=.9),
            Car(team_id="a", team_name="A", base_pace=.85),
            Track(id="test", name="Test", country="Test", total_laps=5,
                  base_lap_time=90., dry_race_pace_adjustment=adjustment))


@pytest.mark.parametrize("value", [True, "0.01", float("nan"), float("inf"), -.051, .051])
def test_clock_rejects_non_numeric_non_finite_or_unbounded_values(value):
    with pytest.raises(ValidationError):
        models(value)


@pytest.mark.parametrize("adjustment", [0., -.003, DRY_RACE_CLOCK_ADJUSTMENT])
@pytest.mark.parametrize("compound", [TireCompound.SOFT, TireCompound.HARD, TireCompound.WET])
@pytest.mark.parametrize("rain,wetness", [(0., 0.), (0., .001), (.001, 0.), (.8, .9)])
def test_actual_and_prepared_laps_match_with_dry_only_clock(adjustment, compound, rain, wetness):
    driver, car, track = models(adjustment)
    weather = Weather(rain_intensity=rain, track_wetness=wetness, change_probability=0.)
    tire = TIRE_COMPOUNDS[compound]
    driver.current_tire_laps = 4
    simulator = LapSimulator()
    prepared = simulator.prepare_deterministic_lap_time(driver, car, track, track.total_laps)
    assert prepared is not None
    actual = simulator.calculate_lap_time(driver, car, track, tire, weather, 3,
                                          track.total_laps, sample_variation=False)
    assert prepared(tire, weather, 3, 4) == actual
    native = track.model_copy(update={"dry_race_pace_adjustment": 0.})
    before = simulator.calculate_lap_time(driver, car, native, tire, weather, 3,
                                          track.total_laps, sample_variation=False)
    if rain or wetness:
        assert actual == before
    else:
        assert actual - before == pytest.approx(track.base_lap_time * adjustment, abs=1e-12)
    # Identical seed preserves sampled qualifying times, including mistakes.
    first = LapSimulator(np.random.default_rng(42)).calculate_qualifying_lap(
        driver, car, track, tire, weather)
    second = LapSimulator(np.random.default_rng(42)).calculate_qualifying_lap(
        driver, car, native, tire, weather)
    assert first == second


@pytest.mark.parametrize("year,round_number,enabled", [
    (2026, 5, False), (2026, 6, True), (2026, 17, True), (2027, 17, False),
    (2026, None, False), (2026, True, False), (2026, "6", False),
])
def test_calibration_is_season_specific_and_strictly_after_training(year, round_number, enabled):
    assert dry_race_pace_adjustment_for_event(year, round_number) == (
        DRY_RACE_CLOCK_ADJUSTMENT if enabled else 0.)


def test_current_loader_propagates_fixed_clock_without_target_performance():
    loader = CurrentSeasonDataLoader()
    year = loader.current_year
    for round_number in (5, 6):
        event = {"round": round_number, "circuit_id": "test", "race": "Test"}
        stats = loader._track_stats_from_event(year, event)
        track = loader.create_track_from_stats(stats)
        assert track.dry_race_pace_adjustment == dry_race_pace_adjustment_for_event(
            year, round_number)
        # Changing a venue reference does not refit the frozen relative coefficient.
        changed = loader._track_stats_from_event(year, event, fastest_lap=110.)
        assert changed.dry_race_pace_adjustment == stats.dry_race_pace_adjustment


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_process_parity_and_saved_second_trial_preserve_clock(tmp_path, engine):
    driver, car, track = models(DRY_RACE_CLOCK_ADJUSTMENT)
    second = driver.model_copy(update={"id": "B", "name": "B"})
    runner = MonteCarloRunner([driver, second], {"a": car}, track,
                             Weather(change_probability=0.), seed=42, race_engine=engine)
    sequential = runner.run(2, parallel=False)
    workers = runner.run(2, parallel=True, max_workers=2)
    assert workers.race_results == sequential.race_results
    assert workers.qualifying_results == sequential.qualifying_results
    path = Exporter(tmp_path).export_statistics_json(sequential)
    restored, _ = _load_saved_runner(path)
    assert restored.track.dry_race_pace_adjustment == DRY_RACE_CLOCK_ADJUSTMENT
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [sequential.race_results[1]]
    assert replay.qualifying_results == [sequential.qualifying_results[1]]


def test_old_saved_track_uses_zero_clock_and_replays(tmp_path):
    driver, car, track = models()
    original = MonteCarloRunner([driver], {"a": car}, track,
                               Weather(change_probability=0.), seed=42).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"]["track"].pop("dry_race_pace_adjustment")
    path.write_text(json.dumps(saved))
    restored, _ = _load_saved_runner(path)
    assert restored.track.dry_race_pace_adjustment == 0.
    assert replay_saved_simulation(path).race_results == original.race_results
