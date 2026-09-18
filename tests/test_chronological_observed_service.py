"""Observed pit-service phases stay separate from future sampled durations."""

from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.models import Car
from f1sim.simulation.chronological_race import (
    ChronologicalRace,
    _PitServiceRecord,
)
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator
from tests.test_chronological_planning_information import setup as planning_setup


def lifecycle_fixture():
    simulator = RaceSimulator(np.random.default_rng(3))
    engine = ChronologicalRace(simulator)
    car = Car(team_id="T", team_name="T")
    engine.expected_box_releases = {}
    engine.pit_service_records = []
    return engine, car


def record(car, *, driver="A", lap=1, arrival=90, start=90, end=190, lane=22):
    return _PitServiceRecord(
        driver, car.team_id, lap, arrival, start, end,
        car.model_copy(deep=True), lane,
    )


def test_active_service_uses_conditional_remaining_and_ignores_future_end(monkeypatch):
    engine, car = lifecycle_fixture()
    engine.pit_service_records = [record(car, end=190)]
    monkeypatch.setattr(engine, "_remaining_service", lambda item, elapsed: 2.26)

    assert engine._team_release_forecast("T", 96) == pytest.approx(98.26)

    # The unobserved end timestamp is private: changing it cannot change the
    # forecast while the same service is still active at the observation time.
    engine.pit_service_records = [record(car, end=900)]
    assert engine._team_release_forecast("T", 96) == pytest.approx(98.26)


def test_overdue_service_and_completed_service_while_in_lane_are_observable():
    engine, car = lifecycle_fixture()
    engine.pit_service_records = [record(car, start=90, end=95, lane=22)]
    pending = SimpleNamespace(lap=1, expected_exit=999.0)

    assert engine._team_release_forecast("T", 100) == pytest.approx(100)
    assert engine._pending_service_exit("A", pending, 100) == pytest.approx(117)


def test_queued_jobs_use_expected_service_after_active_tail_without_future_leakage(
    monkeypatch,
):
    engine, car = lifecycle_fixture()
    queued = record(car, driver="B", lap=2, arrival=91, start=190, end=193)
    engine.pit_service_records = [record(car, end=190), queued]
    monkeypatch.setattr(engine, "_remaining_service", lambda item, elapsed: 2.26)
    expected_tail = 96 + 2.26 + expected_stationary_time(car)

    assert engine._team_release_forecast("T", 96) == pytest.approx(expected_tail)

    queued.service_start = 900
    queued.service_end = 1200
    assert engine._team_release_forecast("T", 96) == pytest.approx(expected_tail)


def test_repeated_stops_retain_observed_completed_tail(monkeypatch):
    engine, car = lifecycle_fixture()
    engine.pit_service_records = [
        record(car, driver="A", lap=1, arrival=10, start=10, end=13),
        record(car, driver="A", lap=9, arrival=90, start=90, end=190),
    ]
    monkeypatch.setattr(engine, "_remaining_service", lambda item, elapsed: 2.5)

    # The completed first stop is not charged again; only the active second
    # stop contributes its conditional remaining service.
    assert engine._team_release_forecast("T", 96) == pytest.approx(98.5)


def test_red_flag_reset_drops_stale_box_ledger_but_keeps_observed_records():
    engine, car = lifecycle_fixture()
    engine.expected_box_releases["T"] = 300
    engine.pit_service_records = [record(car, start=90, end=95)]
    engine.expected_box_releases.clear()

    assert engine._team_release_forecast("T", 200) == pytest.approx(200)
    assert len(engine.pit_service_records) == 1


def test_live_queue_forecast_hides_unfinished_service_duration(monkeypatch):
    """The planner sees one positive conditional wait for both future samples."""
    forecasts = []
    snapshots = []
    for service, expected_queue in ((8, 2), (100, 94)):
        with monkeypatch.context() as patch:
            engine, run, decisions, _ = planning_setup(
                patch, service=service, first_b=96,
            )
            results = run()
            forecasts.append(decisions["B", 2][1])
            snapshots.append(decisions["B", 2][2])
            b = next(result for result in results if result.driver_id == "B")
            assert b.pit_stop_details[0]["queue_time"] == pytest.approx(expected_queue)
            assert b.pit_stop_details[0]["service_time"] == pytest.approx(service)

    assert forecasts[0] == pytest.approx(2.2600000078614575)
    assert forecasts[0] == pytest.approx(forecasts[1])
    assert snapshots[0] == snapshots[1]


def test_same_time_queued_exits_follow_record_order_without_future_timestamp_leakage(
    monkeypatch,
):
    engine, car = lifecycle_fixture()
    engine.pit_service_records = [
        record(car, driver="A", lap=1, arrival=90, start=90, end=190),
        record(car, driver="B", lap=2, arrival=96, start=190, end=193, lane=10),
        record(car, driver="C", lap=2, arrival=96, start=193, end=196, lane=12),
    ]
    monkeypatch.setattr(engine, "_remaining_service", lambda item, elapsed: 2.26)
    pending_b = SimpleNamespace(lap=2, expected_exit=999.0)
    pending_c = SimpleNamespace(lap=2, expected_exit=999.0)

    exit_b = engine._pending_service_exit("B", pending_b, 96)
    exit_c = engine._pending_service_exit("C", pending_c, 96)
    assert exit_c > exit_b

    engine.pit_service_records[2].service_start = 900
    engine.pit_service_records[2].service_end = 1200
    assert engine._pending_service_exit("C", pending_c, 96) == pytest.approx(exit_c)


def test_collected_restart_clears_observed_records_after_real_stop(monkeypatch):
    engine, run, _, _ = planning_setup(
        monkeypatch, service=100, laps=5, both=False,
    )
    control = engine.simulator.event_manager
    control.set_forced_red_flag(2)
    forced_process = control.process_lap
    process = type(control).process_lap

    def process_with_restart(*args, **kwargs):
        forced_process(*args, **kwargs)
        return process(control, *args, **kwargs)

    monkeypatch.setattr(control, "process_lap", process_with_restart)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *args: None)

    results = run()

    assert sum(result.pit_stops for result in results) == 1
    assert engine.pit_service_records == []
