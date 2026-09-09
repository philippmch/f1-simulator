import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

import f1sim.web.server as server_module
from f1sim.data import CurrentSeasonDataError
from f1sim.data.current import DriverStats
from f1sim.models import Car, Driver
from f1sim.web.server import (
    DashboardRunRequest,
    _representative_sample_index,
    _serialize_track,
    _summarize_scenario_results,
    _validate_dashboard_request,
    build_dashboard_html,
    run_dashboard_simulation,
)


@pytest.mark.parametrize("source,actual,component,expected", [
    ("model_prior", 0.95, 0.95, "model_prior"),
    ("model_prior", 0.8, 0.95, "provided"),
    ("provided", 0.95, 0.95, "provided"),
    ("model_prior", 0.95, 0.8, "provided"),
])
def test_car_snapshot_reports_its_mechanical_assumption(source, actual, component, expected):
    driver = Driver(id="A", name="A", team_id="team")
    stats = DriverStats(driver_id="A", driver_name="A", team_id="team", team_name="Team",
                        team_reliability=0.95, team_reliability_source=source)
    car = Car(team_id="team", team_name="Team", reliability=actual)
    car.brakes_reliability = component
    snapshot = server_module._serialize_ratings_snapshot([driver], {"team": car}, {"A": stats})
    assert snapshot["cars"][0]["reliability_source"] == expected
    assert snapshot["cars"][0]["reliability"] == actual


def test_track_payload_uses_2026_active_aero_terms() -> None:
    track = SimpleNamespace(
        id="monza",
        name="Monza",
        country="Italy",
        total_laps=53,
        base_lap_time=80.0,
        pit_lane_delta=24.0,
        overtake_difficulty=0.2,
        tire_stress=0.5,
        safety_car_probability=0.25,
        weather_variability=0.2,
        active_aero_zone_count=3,
        total_active_aero_gain=0.75,
        overtake_mode_detection_gap=1.0,
    )

    payload = _serialize_track(track)

    assert payload["active_aero_zones"] == 3
    assert payload["active_aero_time_gain"] == 0.75
    assert payload["overtake_mode_detection_gap"] == 1.0


def _asgi_get(app, path: str, *, params: dict | None = None):
    httpx = pytest.importorskip("httpx")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.get(path, params=params)

    return asyncio.run(request())


def _asgi_post(app, path: str, *, json: dict):
    httpx = pytest.importorskip("httpx")

    async def request():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.post(path, json=json)

    return asyncio.run(request())


class _FakeResults:
    def __init__(self, seed: int) -> None:
        self.num_simulations = 100
        self.seed = seed

    def get_win_probabilities(self):
        return {"VER": 55.0, "NOR": 25.0, "HAM": 20.0}

    def get_event_rates(self):
        return {"safety_car_race_rate": 0.4}

    def get_team_championship_projection(self):
        return {"Red Bull": 35.0, "McLaren": 28.0}


def test_dashboard_summary_shape() -> None:
    summary = _summarize_scenario_results(
        {
            "dry": _FakeResults(seed=42),
            "heavy_rain": _FakeResults(seed=1042),
        }
    )

    assert set(summary["scenarios"].keys()) == {"dry", "heavy_rain"}
    assert summary["scenarios"]["dry"]["seed"] == 42
    assert summary["scenarios"]["dry"]["top3_win_probabilities"][0][0] == "VER"
    assert "win_probabilities" in summary["scenarios"]["dry"]
    assert "runtime_seconds" in summary["scenarios"]["dry"]
    assert "simulations_per_second" in summary["scenarios"]["dry"]
    assert "driver_statistics" in summary["scenarios"]["dry"]
    assert "sample_race" in summary["scenarios"]["dry"]
    assert "sample_qualifying" in summary["scenarios"]["dry"]
    assert summary["scenarios"]["dry"]["sample_index"] == 0
    assert summary["scenarios"]["dry"]["qualifying_mode"] == "simulated"


def test_representative_sample_uses_race_closest_to_aggregate() -> None:
    def row(driver_id: str, position: int) -> SimpleNamespace:
        return SimpleNamespace(driver_id=driver_id, position=position)

    results = SimpleNamespace(
        race_results=[
            [row("A", 1), row("B", 2)],
            [row("A", 2), row("B", 1)],
            [row("A", 2), row("B", 1)],
        ],
        driver_stats={
            "A": SimpleNamespace(avg_position=5 / 3),
            "B": SimpleNamespace(avg_position=4 / 3),
        },
    )

    assert _representative_sample_index(results) == 1


def test_dashboard_html_contains_controls() -> None:
    html = build_dashboard_html()
    assert "Run Simulation" in html
    assert "/api/calendar" in html
    assert "fetchCalendar" in html
    assert "scenarioSetInput" in html
    assert "panel-scenarios" in html
    assert "renderScenarioLab" in html
    assert "renderScenarioCards" in html
    assert "renderScenarioWinChart" in html
    assert "renderScenarioTrends" in html
    assert "renderDriverMatrix" in html
    assert "compareTopN" in html
    assert "compareTrendMetric" in html
    assert "compareTrendScale" in html
    assert "compareMatrixSort" in html
    assert "compareMatrixHighlight" in html
    assert "compareDriverFilter" in html
    assert "compareScenarioFilter" in html
    assert "downloadScenarioJsonBtn" in html
    assert "downloadResultJson" in html
    assert "exportScenarioMatrixCsv" in html
    assert "qualifyingModeSelect" in html
    assert "presetDryBtn" in html
    assert "presetMixedBtn" in html
    assert "presetChaosBtn" in html
    assert "f1sim:uiPrefs" in html
    assert "scenarios" in html
    assert "Live only" in html
    assert "Fresh current-season data" in html
    assert "Historical Grid" not in html
    assert "FALLBACK_EVENTS" not in html
    assert "LAST_PAYLOAD_KEY" not in html
    assert "/api/runs" not in html
    assert "const BACKEND_URL" not in html
    assert "repeat(20" not in html
    assert "drivers.slice(0, 12)" not in html
    assert "seed: parseInt" not in html
    assert 'role="tablist"' in html
    assert 'aria-live="polite"' in html
    assert "escapeHtml(result.driver_name)" in html
    assert 'id="btnRefresh"' in html
    assert "names.flatMap" in html
    assert "readOptionalStorage" in html
    assert "writeOptionalStorage" in html
    assert "window.localStorage.getItem" in html
    assert "window.localStorage.setItem" in html


def test_dashboard_request_defaults_to_current_season_and_simulated_qualifying() -> None:
    request = DashboardRunRequest()

    assert request.year == datetime.now(timezone.utc).year
    assert request.qualifying_mode == "simulated"


def test_api_responses_disable_client_and_proxy_caching() -> None:
    pytest.importorskip("fastapi")

    response = _asgi_get(server_module.build_fastapi_app(), "/api/health")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, max-age=0"
    assert response.headers["pragma"] == "no-cache"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


def test_dashboard_rejects_older_season_before_fetching() -> None:
    with pytest.raises(ValueError, match="Only the live"):
        run_dashboard_simulation(
            DashboardRunRequest(year=datetime.now(timezone.utc).year - 1)
        )


def test_dashboard_rejects_non_simulated_qualifying() -> None:
    with pytest.raises(ValueError, match="Only freshly simulated qualifying"):
        run_dashboard_simulation(DashboardRunRequest(qualifying_mode="historical"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"race_engine": "unknown"}, "race_engine must be"),
        ({"race_engine": None}, "race_engine must be"),
        ({"race": "  "}, "race must be"),
        ({"simulations": 0}, "simulations must be between"),
        ({"simulations": 1001}, "simulations must be between"),
        ({"seed": -1}, "seed must be between"),
        ({"max_workers": 0}, "max_workers must be between"),
        ({"max_workers": 17}, "max_workers must be between"),
        ({"scenarios": "dry,snow"}, "Unknown scenario label"),
    ],
)
def test_dashboard_rejects_invalid_run_before_live_fetch(
    monkeypatch,
    overrides: dict,
    message: str,
) -> None:
    monkeypatch.setattr(
        server_module,
        "_get_loader",
        lambda: pytest.fail("invalid input must be rejected before live fetch"),
    )

    with pytest.raises(ValueError, match=message):
        run_dashboard_simulation(DashboardRunRequest(**overrides))


def test_dashboard_validation_preserves_zero_seed() -> None:
    request = DashboardRunRequest(seed=0, scenarios="dry,cloudy")

    assert _validate_dashboard_request(request) == ["dry", "cloudy"]
    assert request.seed == 0


def test_run_endpoint_rejects_unbounded_simulations_as_bad_request(monkeypatch) -> None:
    pytest.importorskip("fastapi")
    monkeypatch.setattr(
        server_module,
        "_get_loader",
        lambda: pytest.fail("invalid input must be rejected before live fetch"),
    )

    response = _asgi_post(
        server_module.build_fastapi_app(),
        "/api/run",
        json={"simulations": 100000000},
    )

    assert response.status_code == 400
    assert "simulations must be between" in response.json()["detail"]


def test_ratings_endpoint_reports_old_season_as_bad_request() -> None:
    pytest.importorskip("fastapi")
    response = _asgi_get(
        server_module.build_fastapi_app(),
        "/api/ratings",
        params={
            "year": datetime.now(timezone.utc).year - 1,
            "race": "1",
        },
    )

    assert response.status_code == 400
    assert "Only the current UTC season" in response.json()["detail"]


def test_calendar_endpoint_reports_live_source_failure_as_unavailable(monkeypatch) -> None:
    pytest.importorskip("fastapi")

    class _UnavailableLoader:
        def list_available_events(self, year: int):
            raise CurrentSeasonDataError(f"fresh {year} calendar unavailable")

    monkeypatch.setattr(server_module, "_get_loader", lambda: _UnavailableLoader())
    response = _asgi_get(server_module.build_fastapi_app(), "/api/calendar")

    assert response.status_code == 503
    assert "fresh" in response.json()["detail"]


def test_calendar_endpoint_does_not_leak_unexpected_exception_details(monkeypatch) -> None:
    pytest.importorskip("fastapi")

    class _BrokenLoader:
        def list_available_events(self, year: int):
            raise RuntimeError(f"private provider detail for {year}")

    monkeypatch.setattr(server_module, "_get_loader", lambda: _BrokenLoader())
    response = _asgi_get(server_module.build_fastapi_app(), "/api/calendar")

    assert response.status_code == 500
    assert response.json()["detail"] == "Unexpected server error"
