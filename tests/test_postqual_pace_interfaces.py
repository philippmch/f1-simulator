"""Selected race-pace policies must reach every public forecast entry point."""
from datetime import datetime, timezone

import pytest
from test_dashboard_plan_comparison import MultiDriverLoader
from test_official_grid_interfaces import OfficialLoader, cli_module
from test_post_qualifying_forecast import PostQualifyingLoader
from test_server_summary import _asgi_get, _asgi_post

from f1sim.analysis.recorded_forecast import (
    _digest,
    record_race_forecast,
    validate_recorded_forecast,
)
from f1sim.data.current import DriverStats
from f1sim.web import server


class PolicyLoader(MultiDriverLoader):
    def get_weighted_driver_stats(self, **kwargs):
        self.calls.append("driver_stats")
        self.selected_policy = kwargs["postqual_pace_policy"]
        return {key: DriverStats(driver_id=key, driver_name=key,
                                 team_id=key, team_name=key, sample_size=3)
                for key in ("A", "B")}

    def create_cars_from_stats(self, *args):
        cars = super().create_cars_from_stats(*args)
        for car in cars.values():
            car.base_pace = .95 if self.selected_policy == "qualifying_led" else .85
        return cars

    def get_provenance(self):
        return {"source": "Offline test", "postqual_pace_policy": self.selected_policy}


@pytest.mark.parametrize("policy", ["qualifying_led", "season_points"])
def test_dashboard_policy_reaches_loaded_cars_metadata_and_automatic_reference(monkeypatch, policy):
    adapter = PolicyLoader()
    monkeypatch.setattr(server, "_get_loader", lambda: adapter)
    payload = server.run_dashboard_simulation(server.DashboardRunRequest(
        simulations=10, scenarios="dry", parallel=False, race_grid_mode="simulated",
        postqual_pace_policy=policy, pit_plans={"A": []}, compare_automatic=True,
    ))
    assert adapter.selected_policy == policy
    assert adapter.calls.count("driver_stats") == 1
    for variant in (payload, payload["automatic_reference"]):
        assert variant["request"]["postqual_pace_policy"] == policy
        assert variant["provenance"]["postqual_pace_policy"] == policy
        cars = variant["scenarios"]["dry"]["simulation_inputs"]["cars"]
        assert {row["base_pace"] for row in cars.values()} == {
            .95 if policy == "qualifying_led" else .85}


@pytest.mark.parametrize("interface", ["direct", "http"])
def test_invalid_dashboard_policy_rejected_before_loader_creation(monkeypatch, interface):
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("Live I/O started"))
    if interface == "direct":
        with pytest.raises(ValueError, match="postqual_pace_policy"):
            server.run_dashboard_simulation(server.DashboardRunRequest(
                postqual_pace_policy="unknown"))
    else:
        pytest.importorskip("fastapi")
        response = _asgi_post(server.build_fastapi_app(), "/api/run",
                             json={"postqual_pace_policy": "unknown"})
        assert response.status_code == 400
        assert "postqual_pace_policy" in response.json()["detail"]


@pytest.mark.parametrize("policy", ["qualifying_led", "season_points"])
def test_ratings_query_passes_policy_to_loader_and_returns_matching_ratings(monkeypatch, policy):
    pytest.importorskip("fastapi")
    adapter = PolicyLoader()
    monkeypatch.setattr(server, "_get_loader", lambda: adapter)
    response = _asgi_get(server.build_fastapi_app(), "/api/ratings", params={
        "year": datetime.now(timezone.utc).year, "race": "1", "postqual_pace_policy": policy,
    })
    assert response.status_code == 200
    assert adapter.selected_policy == policy
    assert response.json()["provenance"]["postqual_pace_policy"] == policy
    assert set(response.json()["car_pace"].values()) == {
        .95 if policy == "qualifying_led" else .85}


def test_invalid_ratings_policy_rejected_before_live_io(monkeypatch):
    pytest.importorskip("fastapi")
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("Live I/O started"))
    response = _asgi_get(server.build_fastapi_app(), "/api/ratings", params={
        "race": "1", "postqual_pace_policy": "unknown",
    })
    assert response.status_code == 400
    assert "postqual_pace_policy" in response.json()["detail"]


def test_postqual_record_can_explicitly_reproduce_legacy_pace_policy():
    adapter = PostQualifyingLoader()
    record = record_race_forecast(
        adapter, adapter.instant.year, 3, trials=1, now=lambda: adapter.instant,
        stage="post_qualifying", postqual_pace_policy="season_points",
    )
    assert adapter.requested_pace_policy == "season_points"
    assert record["postqual_pace_policy"] == "season_points"
    assert record["simulation_inputs"]["starting_grid"] == ["BB", "AA"]
    validate_recorded_forecast(record)


def test_invalid_record_policy_rejected_before_fetch_or_trial_setup(monkeypatch):
    adapter = PostQualifyingLoader()
    for method in ("_event_for_race", "_season_data", "get_weighted_driver_stats"):
        monkeypatch.setattr(adapter, method, lambda *args, **kwargs: pytest.fail(
            "Live I/O started"))
    monkeypatch.setattr("f1sim.analysis.recorded_forecast.MonteCarloRunner",
                        lambda *args, **kwargs: pytest.fail("Trials started"))
    with pytest.raises(ValueError, match="postqual_pace_policy"):
        record_race_forecast(adapter, adapter.instant.year, 3, trials=1,
                             stage="post_qualifying", postqual_pace_policy="unknown")


@pytest.mark.parametrize("policy", [None, "season_points"])
def test_simulation_cli_passes_default_or_explicit_pace_policy(monkeypatch, capsys, policy):
    adapter = OfficialLoader()
    supplied = []
    original = adapter.get_weighted_driver_stats

    def capture(**kwargs):
        supplied.append(kwargs["postqual_pace_policy"])
        return original(**kwargs)

    monkeypatch.setattr(adapter, "get_weighted_driver_stats", capture)
    arguments = [] if policy is None else ["--postqual-pace-policy", policy]
    module = cli_module(monkeypatch, adapter, "--race-grid", "simulated", *arguments)
    assert module.main() == 0
    selected = policy or "qualifying_led"
    assert supplied == [selected]
    assert f"Race pace policy: {selected}" in capsys.readouterr().out


def test_old_schema_four_record_without_policy_remains_valid():
    adapter = PostQualifyingLoader()
    record = record_race_forecast(
        adapter, adapter.instant.year, 3, trials=1, now=lambda: adapter.instant,
        stage="post_qualifying", postqual_pace_policy="season_points",
    )
    del record["postqual_pace_policy"]
    record["content_sha256"] = _digest({k: v for k, v in record.items() if k != "content_sha256"})
    validate_recorded_forecast(record)
    record["postqual_pace_policy"] = "unrecognized"
    record["content_sha256"] = _digest({k: v for k, v in record.items() if k != "content_sha256"})
    with pytest.raises(ValueError, match="postqual_pace_policy"):
        validate_recorded_forecast(record)
