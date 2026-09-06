"""Admission control across threads, independent apps, and server processes."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

import pytest

from f1sim.web.capacity import RunCapacity


@pytest.mark.parametrize("value", ["0", "5", "-1", "many", "1.5", ""])
def test_invalid_environment_limit_fails_early(monkeypatch, tmp_path, value):
    monkeypatch.setenv("F1SIM_MAX_CONCURRENT_RUNS", value)
    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="between 1 and 4"):
        RunCapacity.from_environment()


def test_default_and_configured_capacity(monkeypatch, tmp_path):
    monkeypatch.delenv("F1SIM_MAX_CONCURRENT_RUNS", raising=False)
    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))
    assert RunCapacity.from_environment().limit == 1
    monkeypatch.setenv("F1SIM_MAX_CONCURRENT_RUNS", "4")
    capacity = RunCapacity.from_environment()
    assert capacity.limit == 4
    assert capacity.directory == tmp_path.resolve()


def test_independent_limiters_share_slots_across_threads(tmp_path):
    ready = Barrier(7)
    release = Event()
    admitted = []

    def attempt():
        with RunCapacity(2, tmp_path).acquire() as available:
            admitted.append(available)
            ready.wait(timeout=10)
            assert release.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(attempt) for _ in range(6)]
        try:
            ready.wait(timeout=10)
            assert sum(admitted) == 2
        finally:
            release.set()
        for future in futures:
            future.result()
    with RunCapacity(2, tmp_path).acquire() as available:
        assert available
    assert all(path.stat().st_size == 0 for path in tmp_path.iterdir())


def test_process_lock_is_shared_and_released_after_crash(tmp_path):
    code = """
import os, sys
from pathlib import Path
from f1sim.web.capacity import RunCapacity
with RunCapacity(1, Path(sys.argv[1])).acquire() as available:
    print('acquired' if available else 'busy', flush=True)
    sys.stdin.readline()
    os._exit(17)
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    process = subprocess.Popen(
        [sys.executable, "-c", code, str(tmp_path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )
    try:
        # Bound startup waits even if the child fails before reporting acquisition.
        with ThreadPoolExecutor(max_workers=1) as executor:
            line = executor.submit(process.stdout.readline)
            try:
                assert line.result(timeout=15).strip() == "acquired"
            except BaseException:
                process.kill()
                raise
        with RunCapacity(1, tmp_path).acquire() as available:
            assert not available
        process.communicate("exit\n", timeout=10)
        assert process.returncode == 17
        with RunCapacity(1, tmp_path).acquire() as available:
            assert available
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="POSIX descriptor inheritance")
def test_release_unlocks_slot_while_unrelated_forked_worker_stays_alive(tmp_path):
    read_fd, write_fd = os.pipe()
    child_pid = None
    try:
        with RunCapacity(1, tmp_path).acquire() as available:
            assert available
            child_pid = os.fork()
            if child_pid == 0:
                # Model another request's pool worker inheriting the active slot.
                os.close(write_fd)
                os.read(read_fd, 1)
                os._exit(0)
            os.close(read_fd)
            read_fd = None
        # Child still holds the inherited descriptor, but normal release must
        # explicitly unlock the shared open-file description before closing it.
        with RunCapacity(1, tmp_path).acquire() as available:
            assert available
        assert os.waitpid(child_pid, os.WNOHANG) == (0, 0)
    finally:
        os.close(write_fd)
        if read_fd is not None:
            os.close(read_fd)
        if child_pid:
            os.waitpid(child_pid, 0)


@pytest.mark.parametrize("failure", [None, ValueError("invalid live race"), RuntimeError("oops")])
def test_apps_reject_busy_runs_and_release_on_success_or_failure(monkeypatch, tmp_path, failure):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from f1sim.web import server

    monkeypatch.setenv("F1SIM_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))
    entered = Event()
    release = Event()
    calls = []

    def simulate(payload):
        calls.append(payload)
        entered.set()
        assert release.wait(timeout=10)
        if failure is not None:
            raise failure
        return {"ok": True}

    monkeypatch.setattr(server, "run_dashboard_simulation", simulate)
    payload = {"year": server._current_season(), "simulations": 10, "scenarios": "dry"}
    with TestClient(server.build_fastapi_app()) as first:
        with TestClient(server.build_fastapi_app()) as second:
            with ThreadPoolExecutor(max_workers=1) as executor:
                ongoing = executor.submit(first.post, "/api/run", json=payload)
                try:
                    assert entered.wait(timeout=10)
                    busy = second.post("/api/run", json=payload)
                    assert busy.status_code == 429
                    assert busy.headers["Retry-After"] == "5"
                    assert "retry" in busy.json()["detail"].lower()
                    invalid = second.post("/api/run", json={**payload, "simulations": 1})
                    assert invalid.status_code == 400
                    assert second.get("/api/health").status_code == 200
                    assert len(calls) == 1
                finally:
                    release.set()
                expected = (
                    200 if failure is None else (400 if isinstance(failure, ValueError) else 500)
                )
                assert ongoing.result(timeout=10).status_code == expected
            monkeypatch.setattr(server, "run_dashboard_simulation", lambda request: {"ok": True})
            assert second.post("/api/run", json=payload).status_code == 200
