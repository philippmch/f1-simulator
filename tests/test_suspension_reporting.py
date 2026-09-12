"""Global red-flag suspension reporting across engines and API boundaries."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import (
    DriverStatus,
    RaceResult,
    RaceSimulator,
    get_race_suspension_seconds,
)
from f1sim.web.server import _serialize_race_result, _summarize_scenario_results


def _controlled_run(monkeypatch, engine: str, *, laps=3, red=(1,), pause=600.0,
                    paces=None, retiree=None):
    paces = paces or {"A": 90.0, "B": 110.0}
    drivers = [Driver(id=driver_id, name=driver_id, team_id=driver_id)
               for driver_id in paces]
    cars = {driver_id: Car(team_id=driver_id, team_name=driver_id, reliability=1.0)
            for driver_id in paces}
    track = Track(id="t", name="Test", country="Test", total_laps=laps,
                  base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(7), red_flag_pause_seconds=pause)
    control = simulator.event_manager
    for lap in red:
        control.set_forced_red_flag(lap)
    monkeypatch.setattr(simulator, "_should_pit", lambda *args, **kwargs: False)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *args, **kwargs: 0)
    def lap_time(*args, **kwargs):
        driver = args[0] if args else kwargs["driver"]
        return paces[driver.id]

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lap_time)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *args, **kwargs: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *args, **kwargs: None)
    if retiree is not None:
        def retire_on_lap(driver, _car, _track, lap, _weather=None):
            if driver.id != retiree[0] or lap != retiree[1]:
                return None
            driver.dnf = True
            driver.dnf_reason = "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])

        monkeypatch.setattr(control, "_check_mechanical_failure", retire_on_lap)

    starting_tires = {driver_id: TireCompound.MEDIUM for driver_id in paces}
    if engine == "chronological":
        runner = ChronologicalRace(simulator, red_flag_pause_seconds=pause)
        results = runner.run(
            drivers, cars, track, Weather(change_probability=0), list(paces),
            starting_tires=starting_tires,
        )
    else:
        runner = simulator
        results = runner.simulate_race(
            drivers, cars, track, Weather(change_probability=0), list(paces),
            starting_tires=starting_tires,
        )
    return results


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_engine_rows_report_shared_collection_and_pause_duration(monkeypatch, engine):
    results = _controlled_run(monkeypatch, engine)

    assert [row.race_suspension_seconds for row in results] == [620.0, 620.0]
    assert get_race_suspension_seconds(results) == 620.0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_engine_reports_zero_without_suspension_and_final_lap_flag(monkeypatch, engine):
    no_flag = _controlled_run(monkeypatch, engine, red=())
    final_flag = _controlled_run(monkeypatch, engine, laps=2, red=(2,))

    assert [row.race_suspension_seconds for row in no_flag] == [0.0, 0.0]
    assert [row.race_suspension_seconds for row in final_flag] == [0.0, 0.0]
    assert get_race_suspension_seconds(no_flag) == 0.0
    assert get_race_suspension_seconds(final_flag) == 0.0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_engine_accumulates_repeated_suspensions(monkeypatch, engine):
    results = _controlled_run(monkeypatch, engine, laps=4, red=(1, 3))

    assert get_race_suspension_seconds(results) == 1260.0
    assert all(row.race_suspension_seconds == 1260.0 for row in results)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_engine_reports_full_duration_above_finish_extension_cap(monkeypatch, engine):
    results = _controlled_run(monkeypatch, engine, pause=4000.0)

    # The finish-deadline extension is capped at one hour, while reporting
    # retains the full collection plus pause duration.
    assert get_race_suspension_seconds(results) == 4020.0
    assert all(row.race_suspension_seconds == 4020.0 for row in results)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_retired_row_keeps_global_suspension_duration(monkeypatch, engine):
    results = _controlled_run(monkeypatch, engine, retiree=("B", 1))

    expected = 600.0 if engine == "standard" else 620.0
    assert get_race_suspension_seconds(results) == expected
    assert all(row.race_suspension_seconds == expected for row in results)
    assert any(row.status == DriverStatus.DNF for row in results)


@pytest.mark.parametrize("value", [None, True, "600", -1, math.inf, math.nan,
                                    10**1000])
def test_race_suspension_helper_rejects_unknown_or_invalid_values(value):
    row = SimpleNamespace(race_suspension_seconds=value)
    assert get_race_suspension_seconds([row]) is None


def test_race_suspension_helper_requires_every_row_to_match():
    rows = [SimpleNamespace(race_suspension_seconds=value) for value in (0, 0.0)]
    assert get_race_suspension_seconds(rows) == 0.0
    assert get_race_suspension_seconds([
        SimpleNamespace(race_suspension_seconds=10),
        SimpleNamespace(),
    ]) is None
    assert get_race_suspension_seconds([
        SimpleNamespace(race_suspension_seconds=10),
        SimpleNamespace(race_suspension_seconds=11),
    ]) is None
    large = 2**53
    assert get_race_suspension_seconds([
        SimpleNamespace(race_suspension_seconds=large),
        SimpleNamespace(race_suspension_seconds=large + 1),
    ]) is None
    assert get_race_suspension_seconds([
        SimpleNamespace(race_suspension_seconds=large),
        SimpleNamespace(race_suspension_seconds=float(large)),
    ]) == float(large)


def _result(driver_id: str, suspension=None, *, position=1) -> RaceResult:
    return RaceResult(
        driver_id, driver_id, "T", position, 100.0, 0.0, 0, 90.0,
        DriverStatus.FINISHED, race_suspension_seconds=suspension,
    )


def test_suspension_statistics_use_one_race_denominator_and_include_known_zero():
    results = SimulationResults(
        5,
        "Test",
        {"A": DriverStatistics("A", "A", "T")},
        [[_result("A", 0.0)], [_result("A", 10.0), _result("B", 10.0)],
         [_result("A", 10.0), _result("B", 11.0)], [_result("A")], []],
        [],
    )

    assert results.get_suspension_statistics() == {
        "recorded_races": 2,
        "races_with_recorded_suspension": 1,
        "mean_completed_suspension_seconds": 5.0,
    }


def test_suspension_statistics_mean_does_not_overflow_on_large_finite_values():
    value = float.fromhex("0x1.fffffffffffffp+1023")
    results = SimulationResults(
        2,
        "Test",
        {},
        [[_result("A", value)], [_result("A", value)]],
        [],
    )

    statistics = results.get_suspension_statistics()
    assert statistics["races_with_recorded_suspension"] == 2
    assert statistics["mean_completed_suspension_seconds"] == value


def test_api_sanitizes_row_and_representative_race_suspension_fields():
    valid = _result("A", 12.0)
    invalid = _result("B", math.inf)
    assert _serialize_race_result(valid)["race_suspension_seconds"] == 12.0
    assert _serialize_race_result(invalid)["race_suspension_seconds"] is None

    results = SimulationResults(
        2,
        "Test",
        {"A": DriverStatistics("A", "A", "T", avg_position=2.0, positions=[2])},
        [[_result("A", 12.0), _result("B", 13.0)],
         [_result("A", 4.0, position=2), _result("B", 4.0, position=2)]],
        [],
    )
    summary = _summarize_scenario_results({"test": results})["scenarios"]["test"]
    assert summary["suspension_statistics"] == {
        "recorded_races": 1,
        "races_with_recorded_suspension": 1,
        "mean_completed_suspension_seconds": 4.0,
    }
    assert summary["sample_race_suspension_seconds"] == 4.0


def test_api_sanitizes_nonfinite_aggregate_for_strict_json():
    results = SimulationResults(1, "Test", {}, [], [])
    results.get_suspension_statistics = lambda: {
        "recorded_races": 1,
        "races_with_recorded_suspension": 1,
        "mean_completed_suspension_seconds": math.nan,
    }

    summary = _summarize_scenario_results({"test": results})
    assert summary["scenarios"]["test"]["suspension_statistics"] == {
        "recorded_races": 1,
        "races_with_recorded_suspension": 1,
        "mean_completed_suspension_seconds": None,
    }
    import json

    json.dumps(summary, allow_nan=False)
