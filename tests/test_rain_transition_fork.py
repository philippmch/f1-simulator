"""Parallel runs cannot inherit a lock held by another request's thread."""

import multiprocessing
import threading

import pytest

from f1sim.simulation import rain_strategy


def test_child_cache_reset_replaces_lock_and_storage():
    original_lock = rain_strategy._transition_suffix_lock
    original_cache = rain_strategy._transition_suffixes
    rain_strategy._reset_transition_cache_after_fork()
    assert rain_strategy._transition_suffix_lock is not original_lock
    assert rain_strategy._transition_suffixes is not original_cache
    assert not rain_strategy._transition_suffixes
    assert rain_strategy._transition_suffix_lock.acquire(timeout=.1)
    rain_strategy._transition_suffix_lock.release()


def child_probe(connection):
    acquired = rain_strategy._transition_suffix_lock.acquire(timeout=1)
    connection.send((acquired, len(rain_strategy._transition_suffixes)))
    if acquired:
        rain_strategy._transition_suffix_lock.release()
    connection.close()


@pytest.mark.skipif("fork" not in multiprocessing.get_all_start_methods(),
                    reason="Requires POSIX fork")
def test_forked_worker_replaces_lock_owned_by_another_thread():
    context = multiprocessing.get_context("fork")
    received, sending = context.Pipe(duplex=False)
    held, release = threading.Event(), threading.Event()
    original_lock = rain_strategy._transition_suffix_lock

    def hold_parent_lock():
        with original_lock:
            held.set()
            release.wait(10)

    thread = threading.Thread(target=hold_parent_lock, daemon=True)
    child = context.Process(target=child_probe, args=(sending,))
    thread.start()
    try:
        assert held.wait(2)
        child.start()
        sending.close()
        child.join(5)
        assert not child.is_alive(), "Worker hung on the inherited cache lock"
        assert child.exitcode == 0
        assert received.poll(1)
        assert received.recv() == (True, 0)
        assert rain_strategy._transition_suffix_lock is original_lock
    finally:
        if child.pid is not None and child.is_alive():
            child.terminate()
            child.join(2)
        release.set()
        thread.join(2)
        received.close()
        sending.close()
