"""HTTP request bodies keep strict scalar types before resource admission."""

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

import f1sim.web.server as server_module
from f1sim.web.server import DashboardRunRequest, _validate_dashboard_request


def _asgi_post(app, payload: dict):
    httpx = pytest.importorskip("httpx")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post("/api/run", json=payload)

    return asyncio.run(request())


class _NoExecutionCapacity:
    acquire_calls = 0

    @classmethod
    def from_environment(cls):
        return cls()

    def acquire(self):
        type(self).acquire_calls += 1
        pytest.fail("malformed request must be rejected before capacity admission")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("year", True),
        ("year", str(datetime.now(timezone.utc).year)),
        ("year", float(datetime.now(timezone.utc).year)),
        ("simulations", True),
        ("simulations", "10"),
        ("simulations", 10.0),
        ("seed", True),
        ("seed", "42"),
        ("seed", 42.0),
        ("parallel", "false"),
        ("parallel", 0),
        ("parallel", 1),
        ("max_workers", True),
        ("max_workers", "2"),
        ("max_workers", 2.0),
    ],
)
def test_run_rejects_non_strict_scalars_before_execution(monkeypatch, field, value):
    pytest.importorskip("fastapi")
    monkeypatch.setattr(server_module, "RunCapacity", _NoExecutionCapacity)
    monkeypatch.setattr(
        server_module,
        "run_dashboard_simulation",
        lambda _request, cancel_requested=None: pytest.fail(
            "malformed request reached simulation"
        ),
    )

    response = _asgi_post(
        server_module.build_fastapi_app(),
        {field: value},
    )

    assert response.status_code == 422
    assert _NoExecutionCapacity.acquire_calls == 0
    assert any(error["loc"][-1] == field for error in response.json()["detail"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("simulations", 10),
        ("simulations", 1000),
        ("seed", 0),
        ("seed", 2**32 - 1),
        ("max_workers", 1),
        ("max_workers", 16),
    ],
)
def test_run_accepts_strict_integer_bounds(monkeypatch, field, value):
    pytest.importorskip("fastapi")
    calls = []

    class _Capacity:
        @classmethod
        def from_environment(cls):
            return cls()

        @contextmanager
        def acquire(self):
            calls.append("capacity")
            yield True

    captured = []
    monkeypatch.setattr(server_module, "RunCapacity", _Capacity)
    monkeypatch.setattr(
        server_module,
        "run_dashboard_simulation",
        lambda request, cancel_requested=None: captured.append(request) or {"ok": True},
    )

    response = _asgi_post(
        server_module.build_fastapi_app(),
        {field: value},
    )

    assert response.status_code == 200
    assert calls == ["capacity"]
    assert len(captured) == 1
    assert getattr(captured[0], field) == value


def test_run_preserves_zero_seed_boolean_controls_null_workers_and_defaults(monkeypatch):
    pytest.importorskip("fastapi")
    captured = []

    class _Capacity:
        @classmethod
        def from_environment(cls):
            return cls()

        @contextmanager
        def acquire(self):
            yield True

    monkeypatch.setattr(server_module, "RunCapacity", _Capacity)
    monkeypatch.setattr(
        server_module,
        "run_dashboard_simulation",
        lambda request, cancel_requested=None: captured.append(request) or {"ok": True},
    )

    response = _asgi_post(
        server_module.build_fastapi_app(),
        {"simulations": 10, "seed": 0, "parallel": False, "max_workers": None},
    )

    assert response.status_code == 200
    request = captured[0]
    assert request.year == datetime.now(timezone.utc).year
    assert request.simulations == 10
    assert request.seed == 0
    assert request.parallel is False
    assert request.max_workers is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("year", True, "live"),
        ("simulations", "10", "simulations must be an integer"),
        ("seed", 42.0, "seed must be an integer"),
        ("parallel", "false", "parallel must be a boolean"),
        ("max_workers", 2.0, "max_workers must be an integer"),
    ],
)
def test_direct_validation_still_rejects_coercible_scalars(field, value, message):
    request = DashboardRunRequest(**{field: value})

    with pytest.raises(ValueError, match=message):
        _validate_dashboard_request(request)
