"""Cooperative cancellation for simulations and strategy searches."""

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict
from multiprocessing import get_context
from threading import Event, Lock, Thread

import numpy as np
import pytest

import f1sim.analysis.montecarlo as montecarlo
from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.cancellation import (
    cancellation_scope,
    current_cancellation_callback,
    raise_if_cancelled,
)
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation import chronological_race as chronological_module
from f1sim.simulation import opening_strategy as opening
from f1sim.simulation import race as race_module
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype
from f1sim.simulation.tire_inventory import TireInventory


def _runner(engine="standard", laps=5, **options):
    drivers = [Driver(id=str(index), name=str(index), team_id=str(index))
               for index in range(2)]
    cars = {driver.team_id: Car(team_id=driver.team_id, team_name=driver.name)
            for driver in drivers}
    track = Track(id="cancel", name="Cancel", country="Test", total_laps=laps,
                  base_lap_time=90)
    return MonteCarloRunner(
        drivers, cars, track, Weather(change_probability=0), seed=42,
        race_engine=engine,
        **options,
    )


def _opening_args(laps=12):
    simulator = RaceSimulator(np.random.default_rng(42))
    return (
        Driver(id="a", name="A", team_id="a"),
        Car(team_id="a", team_name="A"),
        Track(id="t", name="T", country="Test", total_laps=laps,
              base_lap_time=90, pit_lane_delta=22),
        Weather(),
        TeamStrategyArchetype.BALANCED,
        simulator.strategy_tuning,
        simulator.strategy_profiles,
    )


def _spawn_checkpoint_worker(started):
    """Keep a real spawn worker in a cancellable checkpoint until signalled."""
    started.set()
    while True:
        raise_if_cancelled()


def test_cancellation_scope_restores_nested_thread_and_error_state():
    def outer():
        return False

    with cancellation_scope(outer):
        assert current_cancellation_callback() is outer
        with pytest.raises(SimulationCancelled):
            with cancellation_scope(lambda: True):
                raise_if_cancelled()
        assert current_cancellation_callback() is outer

        seen = []
        thread = Thread(target=lambda: seen.append(current_cancellation_callback()))
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert seen == [None]

        def callback_error():
            raise ValueError("callback failed")

        with pytest.raises(ValueError, match="callback failed"):
            with cancellation_scope(callback_error):
                raise_if_cancelled()
        assert current_cancellation_callback() is outer

    assert current_cancellation_callback() is None


@pytest.mark.parametrize(
    ("engine", "race_module_under_test"),
    [("standard", race_module), ("chronological", chronological_module)],
)
def test_in_trial_cancellation_aborts_actual_engine(monkeypatch, engine,
                                                    race_module_under_test):
    runner = _runner(
        engine, laps=53,
        starting_tires={"0": "soft", "1": "soft"},
    )
    race_started = Event()

    original_checkpoint = race_module_under_test.raise_if_cancelled

    def mark_race_checkpoint():
        race_started.set()
        return original_checkpoint()

    monkeypatch.setattr(race_module_under_test, "raise_if_cancelled", mark_race_checkpoint)

    def cancel_after_race_starts():
        return race_started.is_set()

    with pytest.raises(SimulationCancelled, match="cancelled"):
        runner.run(1, parallel=False, cancel_requested=cancel_after_race_starts)
    assert race_started.is_set()


def test_opening_search_cancellation_does_not_poison_cache():
    args = _opening_args()
    opening._cached_dry_policy_costs.cache_clear()
    calls = 0

    def cancel_after_search_work():
        nonlocal calls
        calls += 1
        return calls >= 5

    try:
        with pytest.raises(SimulationCancelled, match="cancelled"):
            with cancellation_scope(cancel_after_search_work):
                opening.dry_opening_policy_costs(*args)
        assert calls >= 5

        baseline = opening.dry_opening_policy_costs(*args)
        opening._cached_dry_policy_costs.cache_clear()
        assert opening.dry_opening_policy_costs(*args) == baseline
    finally:
        opening._cached_dry_policy_costs.cache_clear()


def test_finite_inventory_search_cancellation_does_not_poison_call():
    driver = Driver(id="a", name="A", team_id="a")
    car = Car(team_id="a", team_name="A")
    track = Track(id="t", name="T", country="Test", total_laps=8,
                  base_lap_time=90, pit_lane_delta=3)
    inventory = TireInventory.from_sets([
        {"compound": "soft"},
        {"compound": "hard"},
    ])
    inventory.fit("set-1")
    calls = 0

    def cancel_after_search_work():
        nonlocal calls
        calls += 1
        return calls >= 4

    with pytest.raises(SimulationCancelled, match="cancelled"):
        with cancellation_scope(cancel_after_search_work):
            plan_inventory_strategy(
                driver, car, track, Weather(), inventory, 1,
                remaining_stops=1, require_compound_rule=False,
            )
    assert calls >= 4

    baseline = plan_inventory_strategy(
        driver, car, track, Weather(), inventory, 1,
        remaining_stops=1, require_compound_rule=False,
    )
    assert baseline.pit_now_cost < float("inf")
    assert plan_inventory_strategy(
        driver, car, track, Weather(), inventory, 1,
        remaining_stops=1, require_compound_rule=False,
    ) == baseline


def test_spawn_worker_cancels_after_start_and_joins():
    multiprocessing_context = get_context("spawn")
    cancel_event = multiprocessing_context.Event()
    with multiprocessing_context.Manager() as manager:
        started = manager.Event()
        with ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing_context,
            initializer=montecarlo._initialize_cancellation_worker,
            initargs=(cancel_event,),
        ) as executor:
            try:
                future = executor.submit(_spawn_checkpoint_worker, started)
                assert started.wait(timeout=15)
                cancel_event.set()
                with pytest.raises(SimulationCancelled, match="cancelled"):
                    future.result(timeout=15)
            finally:
                cancel_event.set()


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
        def __init__(self, max_workers, *, mp_context, initializer, initargs):
            self.max_workers = max_workers
            self.mp_context = mp_context
            self.initializer = initializer
            self.initargs = initargs
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
        def __init__(self, max_workers=None, *, mp_context=None, initializer=None,
                     initargs=(), **kwargs):
            self.worker_initializer = initializer
            self.worker_initargs = initargs
            super().__init__(max_workers=max_workers, **kwargs)

        def submit(self, function, *args, **kwargs):
            state["submissions"].append(args[0][4])
            if self.worker_initializer is None:
                return super().submit(function, *args, **kwargs)

            def initialized(*inner_args, **inner_kwargs):
                self.worker_initializer(*self.worker_initargs)
                return function(*inner_args, **inner_kwargs)

            return super().submit(initialized, *args, **kwargs)

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


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_finite_pool_false_callback_matches_uncancelled_run(engine):
    inventory = {
        driver_id: [
            {"id": "soft-set", "compound": "soft", "age": 0},
            {"id": "hard-set", "compound": "hard", "age": 0},
        ]
        for driver_id in ("0", "1")
    }
    options = {
        "starting_tires": {"0": "soft", "1": "soft"},
        "tire_inventory": inventory,
    }
    baseline = _runner(engine, laps=8, **options).run(2, parallel=False)
    uncancelled = _runner(engine, laps=8, **options).run(
        2, parallel=False, cancel_requested=lambda: False,
    )

    assert asdict(uncancelled) == asdict(baseline)


def test_cancel_requested_must_be_callable():
    with pytest.raises(TypeError, match="cancel_requested must be callable"):
        _runner().run(1, parallel=False, cancel_requested=True)
