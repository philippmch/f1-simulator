"""Dashboard held-out pit-plan request preflight stays ahead of trial work."""

from types import SimpleNamespace

import pytest

from f1sim.models import Car, Driver, Track
from f1sim.web import server


def _selection(**overrides):
    value = {
        "plans": {"reference": None, "alternative": [{"lap": 2, "compound": "hard"}]},
        "reference_label": "reference",
        "driver_id": "A",
        "training_simulations": 1,
        "validation_simulations": 1,
    }
    value.update(overrides)
    return value


class TinyLoader:
    def resolve_race_identifier(self, *args):
        return 1

    def list_available_events(self, *args):
        return [{"round": 1, "race": "Synthetic"}]

    def get_weighted_driver_stats(self, **kwargs):
        return {}

    def get_track_stats(self, *args):
        return SimpleNamespace(
            track_name="Synthetic", country="Test", total_laps=3, avg_lap_time=90,
        )

    def create_drivers_from_stats(self, *args):
        return [Driver(id="A", name="A", team_id="A")]

    def create_cars_from_stats(self, *args):
        return {"A": Car(team_id="A", team_name="A")}

    def create_track_from_stats(self, *args):
        return Track(id="synthetic", name="Synthetic", country="Test", total_laps=3,
                     base_lap_time=90)

    def get_provenance(self):
        return {"source": "test"}


def _install_tiny_loader(monkeypatch):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", TinyLoader)


def _request(**overrides):
    values = {
        "year": 2026,
        "simulations": 10,
        "scenarios": "dry",
        "parallel": False,
        "pit_plan_selection": _selection(),
    }
    values.update(overrides)
    return server.DashboardRunRequest(**values)


@pytest.mark.parametrize("objective", ["win", "podium"])
def test_dashboard_selection_carries_objective_into_response_and_html(monkeypatch, objective):
    _install_tiny_loader(monkeypatch)
    response = server.run_dashboard_simulation(_request(pit_plan_selection=_selection(
        objective=objective,
    )))
    entry = response["strategy_selections"]["dry"]
    assert entry["selection"]["objective"] == objective
    assert entry["selection"]["score_unit"] == "probability"
    assert "Held-out objective probabilities" in entry["validation_report_html"]
    assert response["request"]["pit_plan_selection"]["objective"] == objective


@pytest.mark.parametrize("objective", ["finish", True, None, {}, []])
def test_http_invalid_objective_fails_before_capacity_or_live_loading(monkeypatch, objective):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class NoCapacity:
        @classmethod
        def from_environment(cls):
            return cls()

        def acquire(self, *args, **kwargs):
            pytest.fail("invalid objective must fail before capacity admission")

    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "RunCapacity", NoCapacity)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live load"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={
            "year": 2026, "simulations": 10, "scenarios": "dry",
            "pit_plan_selection": _selection(objective=objective),
        })
    assert response.status_code == 422 and "objective" in response.text


def test_selection_budget_uses_all_candidates_and_two_validation_variants(monkeypatch):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live load"))
    identity = {"plans": {"reference": None, "same": None}}

    with pytest.raises(ValueError, match="worst-case work.*at most 1000"):
        server.run_dashboard_simulation(_request(pit_plan_selection=_selection(
            **identity,
            training_simulations=490,
            validation_simulations=6,
        )))


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (_selection(training_simulations=True), "training_simulations"),
        (_selection(validation_simulations=0), "positive integer"),
        (_selection(validation_simulations=1001), "at most 1000"),
        (_selection(driver_id="A", constructor_id="A"), "exactly one"),
    ],
)
def test_strict_selection_shape_and_counts_are_rejected_before_loading(
    monkeypatch, selection, message,
):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live load"))
    with pytest.raises(ValueError, match=message):
        server.run_dashboard_simulation(_request(pit_plan_selection=selection))


def test_selection_rejects_automatic_comparison_before_loading(monkeypatch):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live load"))
    with pytest.raises(ValueError, match="cannot be combined with compare_automatic"):
        server.run_dashboard_simulation(_request(
            pit_plans={"A": []},
            compare_automatic=True,
        ))


def test_http_selection_counts_are_strict_before_capacity_admission(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    class NoCapacity:
        @classmethod
        def from_environment(cls):
            return cls()

        def acquire(self):
            pytest.fail("invalid nested selection must fail before capacity admission")

    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "RunCapacity", NoCapacity)
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={
            "year": 2026,
            "simulations": 10,
            "scenarios": "dry",
            "pit_plan_selection": _selection(training_simulations=True),
        })
    assert response.status_code == 422
    assert "training_simulations" in response.text


def test_invalid_candidate_plan_fails_after_load_but_before_source_trials(monkeypatch):
    _install_tiny_loader(monkeypatch)
    calls = []
    monkeypatch.setattr(
        server.MonteCarloRunner,
        "run",
        lambda self, *args, **kwargs: calls.append(self.base_seed),
    )
    selection = _selection(plans={
        "reference": None,
        "invalid": [{"lap": 8, "compound": "hard"}],
    })

    with pytest.raises(ValueError, match="must not exceed total_laps"):
        server.run_dashboard_simulation(_request(pit_plan_selection=selection))
    assert calls == []


def test_late_weather_seed_overflow_fails_before_any_weather_trials(monkeypatch):
    _install_tiny_loader(monkeypatch)
    calls = []
    monkeypatch.setattr(
        server.MonteCarloRunner,
        "run",
        lambda self, *args, **kwargs: calls.append(self.base_seed),
    )
    maximum_seed = 2**32 - 1

    with pytest.raises(ValueError, match="must not exceed"):
        server.run_dashboard_simulation(_request(
            seed=maximum_seed - 1000,
            scenarios="dry,light_rain",
        ))
    assert calls == []
