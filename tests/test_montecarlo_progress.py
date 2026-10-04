"""Progress reports belong to the caller and preserve seeded race outcomes."""

import os
from dataclasses import asdict
from multiprocessing import active_children
from threading import Event

import pytest
from test_simulation_cancellation import _runner

from f1sim.cancellation import SimulationCancelled


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("cancellable", [False, True])
def test_progress_preserves_every_trial_and_runs_in_caller(engine, parallel, cancellable):
    baseline = _runner(engine, laps=3).run(4, parallel=False)
    notifications = []

    def report(completed, total):
        notifications.append((completed, total, os.getpid()))

    options = {"cancel_requested": lambda: False} if cancellable else {}
    observed = _runner(engine, laps=3).run(
        4, parallel=parallel, max_workers=2, progress_callback=report, **options,
    )
    assert notifications == [(count, 4, os.getpid()) for count in range(5)]
    assert [[asdict(row) for row in race] for race in observed.race_results] == [
        [asdict(row) for row in race] for race in baseline.race_results
    ]
    assert [[asdict(row) for row in grid] for grid in observed.qualifying_results] == [
        [asdict(row) for row in grid] for grid in baseline.qualifying_results
    ]
    assert observed.driver_stats == baseline.driver_stats
    assert observed.event_stats == baseline.event_stats


@pytest.mark.parametrize("parallel", [False, True])
def test_progress_callback_error_closes_workers_and_returns_no_result(parallel):
    existing_children = {child.pid for child in active_children()}
    completed_counts = []

    def fail(completed, total):
        completed_counts.append(completed)
        if completed == 1:
            raise RuntimeError("progress receiver failed")

    with pytest.raises(RuntimeError, match="progress receiver failed"):
        _runner(laps=3).run(6, parallel=parallel, max_workers=2, progress_callback=fail)
    assert completed_counts == [0, 1]
    assert {child.pid for child in active_children()} <= existing_children


@pytest.mark.parametrize("parallel", [False, True])
def test_progress_can_request_cooperative_cancellation(parallel):
    cancelled = Event()
    seen = []

    def stop(completed, total):
        seen.append(completed)
        if completed == 1:
            cancelled.set()

    with pytest.raises(SimulationCancelled):
        _runner(laps=3).run(6, parallel=parallel, max_workers=2,
                            progress_callback=stop, cancel_requested=cancelled.is_set)
    assert seen[0] == 0 and 1 in seen and max(seen) < 6


def test_progress_callback_is_validated_before_running_a_trial():
    with pytest.raises(TypeError, match="progress_callback must be callable"):
        _runner().run(1, parallel=False, progress_callback=True)
