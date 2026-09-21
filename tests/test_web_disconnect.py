"""Real ASGI disconnects must drain dashboard workers before capacity release."""

from __future__ import annotations

import asyncio
import json
from threading import Event
from typing import Any

import pytest

from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.web import server


def _scope() -> dict[str, Any]:
    return {
        "type": "http",
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/api/run",
        "raw_path": b"/api/run",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
        "client": ("test", 1234),
        "server": ("test", 80),
        "root_path": "",
    }


async def _serve(
    app: Any,
    payload: dict[str, Any],
    receive: Any,
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(_scope(), receive, send)
    return sent


def _static_receive(payload: dict[str, Any]):
    body = json.dumps(payload).encode("utf-8")
    first = True

    async def receive() -> dict[str, Any]:
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.request", "body": b"", "more_body": False}

    return receive


def _disconnecting_receive(payload: dict[str, Any], disconnect: asyncio.Event):
    body = json.dumps(payload).encode("utf-8")
    first = True

    async def receive() -> dict[str, Any]:
        nonlocal first
        if first:
            first = False
            return {"type": "http.request", "body": body, "more_body": False}
        if disconnect.is_set():
            return {"type": "http.disconnect"}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    return receive


def _status(sent: list[dict[str, Any]]) -> int:
    return next(message["status"] for message in sent if message["type"] == "http.response.start")


async def _wait_thread_event(event: Event) -> None:
    assert await asyncio.to_thread(event.wait, 10)


def _payload() -> dict[str, Any]:
    return {
        "year": server._current_season(),
        "simulations": 10,
        "scenarios": "dry",
    }


@pytest.mark.anyio
async def test_asgi_disconnect_stops_following_work_and_drains_capacity(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("F1SIM_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))

    entered = Event()
    cancellation_seen = Event()
    release = Event()
    calls: list[tuple[Any, Any]] = []

    def simulate(payload, cancel_requested=None):
        calls.append((payload, cancel_requested))
        entered.set()
        while not release.wait(0.01):
            if cancel_requested is not None and cancel_requested():
                cancellation_seen.set()
        return {"ok": True}

    monkeypatch.setattr(server, "run_dashboard_simulation", simulate)
    first_app = server.build_fastapi_app()
    second_app = server.build_fastapi_app()
    disconnect = asyncio.Event()
    first_task = asyncio.create_task(
        _serve(first_app, _payload(), _disconnecting_receive(_payload(), disconnect)),
    )

    finished = None
    try:
        await _wait_thread_event(entered)
        disconnect.set()
        await _wait_thread_event(cancellation_seen)
        # Exercise transport cancellation after the disconnect watcher has
        # already requested cancellation and the route is draining.
        first_task.cancel()
        done, _ = await asyncio.wait({first_task}, timeout=0.2)
        assert not done

        busy = await _serve(second_app, _payload(), _static_receive(_payload()))
        assert _status(busy) == 429
        assert len(calls) == 1
    finally:
        release.set()
        try:
            finished = await asyncio.wait_for(first_task, timeout=10)
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            pytest.fail("dashboard worker did not drain after disconnect", pytrace=False)
    if finished is not None:
        assert _status(finished) == 499

    reacquired = await _serve(second_app, _payload(), _static_receive(_payload()))
    assert _status(reacquired) == 200
    assert len(calls) == 2


@pytest.mark.anyio
async def test_endpoint_task_cancellation_drains_worker_before_capacity_release(
    monkeypatch, tmp_path,
):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("F1SIM_MAX_CONCURRENT_RUNS", "1")
    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))

    entered = Event()
    cancellation_seen = Event()
    release = Event()

    def simulate(payload, cancel_requested=None):
        entered.set()
        while not release.wait(0.01):
            if cancel_requested is not None and cancel_requested():
                cancellation_seen.set()
        return {"ok": True}

    monkeypatch.setattr(server, "run_dashboard_simulation", simulate)
    first_app = server.build_fastapi_app()
    second_app = server.build_fastapi_app()
    first_task = asyncio.create_task(
        _serve(first_app, _payload(), _static_receive(_payload())),
    )

    finished = None
    try:
        await _wait_thread_event(entered)
        first_task.cancel()
        await _wait_thread_event(cancellation_seen)
        first_task.cancel()
        done, _ = await asyncio.wait({first_task}, timeout=0.2)
        assert not done

        busy = await _serve(second_app, _payload(), _static_receive(_payload()))
        assert _status(busy) == 429
    finally:
        release.set()
        try:
            finished = await asyncio.wait_for(first_task, timeout=10)
        except asyncio.CancelledError:
            # Some ASGI servers cancel the outer application task after the
            # route has drained; either outcome must leave the slot available.
            pass
        except asyncio.TimeoutError:
            pytest.fail("dashboard worker did not drain after task cancellation", pytrace=False)
    if finished is not None:
        assert _status(finished) == 499

    reacquired = await _serve(second_app, _payload(), _static_receive(_payload()))
    assert _status(reacquired) == 200


def test_dashboard_cancellation_is_checked_before_live_loading(monkeypatch):
    request = server.DashboardRunRequest(scenarios="dry")
    monkeypatch.setattr(
        server,
        "_get_loader",
        lambda: pytest.fail("cancellation must precede live loading"),
    )

    with pytest.raises(SimulationCancelled, match="disconnected"):
        server.run_dashboard_simulation(request, cancel_requested=lambda: True)
