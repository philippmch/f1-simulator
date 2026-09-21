"""Parent-process cancellation for bounded Monte Carlo runs."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from threading import Event, Lock, Thread

import pytest

import f1sim.analysis.montecarlo as montecarlo
from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather


def _runner(engine="standard", laps=5):
    drivers = [Driver(id=str(index), name=str(index), team_id=str(index))
               for index in range(2)]
    cars = {driver.team_id: Car(team_id=driver.team_id, team_name=driver.name)
            for driver in drivers}
    track = Track(id="cancel", name="Cancel", country="Test", total_laps=laps,
                  base_lap_time=90)
    return MonteCarloRunner(
        drivers, cars, track, Weather(change_probability=0), seed=42,
        race_engine=engine,
    )


def test_precancelled_run_does_no_preparation_or_execution(monkeypatch):
    runner = _runner()
    monkeypatch.setattr(
        montecarlo.Driver,
        "model_dump",
        lambda *args, **kwargs: pytest.fail("cancelled run prepared worker data"),
    )
    monkeypatch.setattr(
        montecarlo,
        "_run_single_simulation",
        lambda *args, **kwargs: pytest.fail("cancelled run started a trial"),
    )

    with pytest.raises(SimulationCancelled, match="cancelled"):
        runner.run(5, parallel=True, cancel_requested=lambda: True)


def test_serial_cancellation_discards_partial_results(monkeypatch):
    runner = _runner()
    started = []
    aggregate_calls = []

    def fake_run(args):
        started.append(args[4])
        return [], [], {"weather_history": []}

    def cancel_after_first():
        return bool(started)

    monkeypatch.setattr(montecarlo, "_run_single_simulation", fake_run)
    monkeypatch.setattr(
        runner,
        "_aggregate_statistics",
        lambda *args: aggregate_calls.append(args) or {},
    )

    with pytest.raises(SimulationCancelled, match="cancelled"):
        runner.run(5, parallel=False, cancel_requested=cancel_after_first)

    assert started == [42]
    assert aggregate_calls == []


def _fake_executor_patch(monkeypatch):
    state = {}

    class FakeExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers
            self.futures = []
            self.submitted = []
            self.waited = False
            self.exited = False
            self.drained = False
            state["executor"] = self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            self.exited = True
            self.drained = all(future.done() or future.cancelled()
                               for future in self.futures)
            return False

        def submit(self, function, args):
            self.submitted.append(args[4])
            future = Future()
            self.futures.append(future)
            if len(self.futures) == 1:
                future.set_result(([], [], {"weather_history": []}))
            return future

    def fake_wait(futures, *, timeout, return_when):
        executor = state["executor"]
        executor.waited = True
        first = executor.futures[0]
        return {first}, set(futures) - {first}

    monkeypatch.setattr(montecarlo, "ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr(montecarlo, "wait", fake_wait)
    return state


@pytest.mark.parametrize("callback_error", [False, True])
def test_parallel_cancellation_bounds_submissions_and_drains_workers(
    monkeypatch, callback_error
):
    runner = _runner(laps=8)
    state = _fake_executor_patch(monkeypatch)

    def callback():
        executor = state.get("executor")
        if executor is not None and executor.waited:
            if callback_error:
                raise RuntimeError("callback failed")
            return True
        return False

    expected = RuntimeError if callback_error else SimulationCancelled
    with pytest.raises(expected):
        runner.run(6, parallel=True, max_workers=2, cancel_requested=callback)

    executor = state["executor"]
    assert executor.max_workers == 2
    assert executor.submitted == [42, 43]
    assert executor.exited
    assert executor.drained
    assert all(future.done() or future.cancelled() for future in executor.futures)


@pytest.mark.parametrize("callback_error", [False, True])
def test_parallel_cancellation_waits_for_running_workers(monkeypatch, callback_error):
    runner = _runner(laps=8)
    started = Event()
    release = Event()
    finished = Event()
    lock = Lock()
    state = {"started": 0, "submissions": [], "exception": None}

    def fake_run(args):
        with lock:
            state["started"] += 1
            if state["started"] == 2:
                started.set()
        release.wait(timeout=10)
        return [], [], {"weather_history": []}

    class TrackingExecutor(ThreadPoolExecutor):
        def submit(self, function, *args, **kwargs):
            state["submissions"].append(args[0][4])
            return super().submit(function, *args, **kwargs)

    monkeypatch.setattr(montecarlo, "_run_single_simulation", fake_run)
    monkeypatch.setattr(montecarlo, "ProcessPoolExecutor", TrackingExecutor)
    cancel_requested = Event()

    def callback():
        if cancel_requested.is_set():
            if callback_error:
                raise RuntimeError("callback failed")
            return True
        return False

    def run():
        try:
            runner.run(
                6,
                parallel=True,
                max_workers=2,
                cancel_requested=callback,
            )
        except BaseException as exc:  # noqa: BLE001 - assert worker cleanup below
            state["exception"] = exc
        finally:
            finished.set()

    thread = Thread(target=run)
    thread.start()
    try:
        assert started.wait(timeout=5)
        cancel_requested.set()
        assert not finished.wait(timeout=0.5)
        assert state["submissions"] == [42, 43]
    finally:
        release.set()
        thread.join(timeout=10)

    assert not thread.is_alive()
    expected = RuntimeError if callback_error else SimulationCancelled
    assert isinstance(state["exception"], expected)
    assert state["submissions"] == [42, 43]


@pytest.mark.parametrize("use_callback", [False, True])
@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_uncancelled_serial_and_parallel_results_match(engine, use_callback):
    run_options = {"cancel_requested": lambda: False} if use_callback else {}
    serial = _runner(engine).run(2, parallel=False, **run_options)
    parallel = _runner(engine).run(
        2, parallel=True, max_workers=2, **run_options,
    )

    assert serial.race_results == parallel.race_results
    assert serial.qualifying_results == parallel.qualifying_results
    assert serial.event_stats == parallel.event_stats
    assert serial.weather_histories == parallel.weather_histories
    parallel_data = asdict(parallel)
    parallel_data.update(parallel=False, max_workers=None)
    assert asdict(serial) == parallel_data


def test_cancel_requested_must_be_callable():
    with pytest.raises(TypeError, match="cancel_requested must be callable"):
        _runner().run(1, parallel=False, cancel_requested=True)
