"""Dashboard custom-plan versus automatic-reference comparisons."""

import copy
import json
from types import SimpleNamespace

import pytest

from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track
from f1sim.web import server


class CountingLoader:
    """Small offline provider snapshot used by dashboard comparison tests."""

    instances = 0

    def __init__(self):
        type(self).instances += 1
        self.calls = []

    def resolve_race_identifier(self, *args):
        self.calls.append("resolve")
        return 1

    def list_available_events(self, *args):
        self.calls.append("events")
        return [{"round": 1, "race": "Synthetic"}]

    def get_weighted_driver_stats(self, **kwargs):
        self.calls.append("driver_stats")
        return {}

    def get_track_stats(self, *args):
        self.calls.append("track_stats")
        return SimpleNamespace(track_name="Synthetic", country="Test", total_laps=3,
                               avg_lap_time=90)

    def create_drivers_from_stats(self, *args):
        self.calls.append("drivers")
        return [Driver(id="A", name="A", team_id="A")]

    def create_cars_from_stats(self, *args):
        self.calls.append("cars")
        return {"A": Car(team_id="A", team_name="A")}

    def create_track_from_stats(self, *args):
        self.calls.append("track")
        return Track(id="synthetic", name="Synthetic", country="Test", total_laps=3,
                     base_lap_time=90)

    def get_provenance(self):
        self.calls.append("provenance")
        return {"source": "Synthetic test"}


class MultiDriverLoader(CountingLoader):
    def create_drivers_from_stats(self, *args):
        self.calls.append("drivers")
        return [
            Driver(id="A", name="A", team_id="A"),
            Driver(id="B", name="B", team_id="B"),
        ]

    def create_cars_from_stats(self, *args):
        self.calls.append("cars")
        return {
            "A": Car(team_id="A", team_name="A"),
            "B": Car(team_id="B", team_name="B"),
        }


def _request(**overrides):
    values = {
        "simulations": 10,
        "scenarios": "dry,light_rain",
        "seed": 7,
        "parallel": False,
        "pit_plans": {"A": []},
        "compare_automatic": True,
    }
    values.update(overrides)
    return server.DashboardRunRequest(**values)


@pytest.fixture
def offline_loader(monkeypatch):
    CountingLoader.instances = 0
    loader = CountingLoader()
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: loader)
    return loader


def test_comparison_runs_one_provider_snapshot_and_keeps_replayable_variants(
    offline_loader, tmp_path,
):
    source_plans = {"A": []}
    request = _request(pit_plans=source_plans)
    payload = server.run_dashboard_simulation(request)

    assert CountingLoader.instances == 1
    assert offline_loader.calls.count("driver_stats") == 1
    assert offline_loader.calls.count("track_stats") == 1
    assert offline_loader.calls.count("provenance") == 1
    assert source_plans == {"A": []}
    assert payload["request"]["compare_automatic"] is True
    assert payload["request"]["pit_plans"] == {"A": []}
    assert payload["automatic_reference"]["request"]["compare_automatic"] is False
    assert payload["automatic_reference"]["request"]["pit_plans"] == {}
    assert "comparison_report_html" not in payload["automatic_reference"]
    assert "automatic_reference" not in payload["automatic_reference"]
    assert set(payload["strategy_comparisons"]) == {"dry", "light_rain"}

    for name, custom in payload["scenarios"].items():
        reference = payload["automatic_reference"]["scenarios"][name]
        assert custom["seed"] == reference["seed"]
        assert custom["sample_qualifying"] == reference["sample_qualifying"]
        assert custom["simulation_inputs"]["track"] == reference["simulation_inputs"]["track"]
        assert custom["simulation_inputs"]["weather"] == reference["simulation_inputs"]["weather"]
        assert custom["simulation_inputs"]["starting_tires"] == reference[
            "simulation_inputs"
        ]["starting_tires"]
        assert custom["simulation_inputs"]["pit_plans"] == {"A": []}
        assert "pit_plans" not in reference["simulation_inputs"]
        assert payload["strategy_comparisons"][name]["reference_scenario"] == "automatic"
        assert payload["strategy_comparisons"][name]["variants"]["custom"]["status"] == "paired"

    main_path = tmp_path / "custom.json"
    reference_path = tmp_path / "automatic.json"
    main_path.write_text(json.dumps(payload), encoding="utf-8")
    reference_path.write_text(json.dumps(payload["automatic_reference"]), encoding="utf-8")
    custom_replay = replay_saved_simulation(
        main_path,
        payload["scenarios"]["dry"]["sample_index"] + 1,
        "dry",
    )
    reference_replay = replay_saved_simulation(
        reference_path,
        payload["automatic_reference"]["scenarios"]["dry"]["sample_index"] + 1,
        "dry",
    )
    assert server._serialize_sample_race(custom_replay) == payload["scenarios"]["dry"][
        "sample_race"
    ]
    assert server._serialize_sample_race(reference_replay) == payload["automatic_reference"][
        "scenarios"
    ]["dry"]["sample_race"]
    assert server._serialize_sample_qualifying(custom_replay) == payload["scenarios"]["dry"][
        "sample_qualifying"
    ]
    assert server._serialize_sample_qualifying(reference_replay) == payload[
        "automatic_reference"
    ]["scenarios"]["dry"]["sample_qualifying"]


def test_compare_automatic_requires_nonempty_plan_and_caps_work_before_loading(monkeypatch):
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading"))

    with pytest.raises(ValueError, match="nonempty pit_plans"):
        server.run_dashboard_simulation(_request(pit_plans={}))
    with pytest.raises(ValueError, match="at most 500"):
        server.run_dashboard_simulation(_request(simulations=501))
    with pytest.raises(ValueError, match="compare_automatic must be a boolean"):
        server.run_dashboard_simulation(_request(compare_automatic=1))


def test_false_comparison_mode_runs_only_submitted_plan(offline_loader):
    request = _request(compare_automatic=False)
    payload = server.run_dashboard_simulation(request)

    assert "automatic_reference" not in payload
    assert "strategy_comparisons" not in payload
    assert payload["request"]["compare_automatic"] is False


def test_false_comparison_mode_constructs_one_runner_per_scenario(monkeypatch, offline_loader):
    original_runner = server.MonteCarloRunner
    constructed = []

    class RecordingRunner(original_runner):
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs["seed"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(server, "MonteCarloRunner", RecordingRunner)
    server.run_dashboard_simulation(_request(compare_automatic=False))

    assert constructed == [7, 1007]


def test_multi_driver_reference_clears_every_override_and_keeps_finite_settings(monkeypatch):
    loader = MultiDriverLoader()
    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: loader)
    plans = {"A": [], "B": [{"lap": 2, "compound": "hard"}]}
    inventory = {
        "A": [{"compound": "soft", "age": 1}, {"compound": "hard"}],
        "B": [{"compound": "medium", "age": 2}, {"compound": "hard"}],
    }
    request = _request(
        scenarios="dry",
        pit_plans=plans,
        starting_tires={"A": "soft", "B": "medium"},
        starting_tire_ages={"A": 1, "B": 2},
        tire_inventory=inventory,
    )
    payload = server.run_dashboard_simulation(request)
    custom = payload["scenarios"]["dry"]["simulation_inputs"]
    reference = payload["automatic_reference"]["scenarios"]["dry"]["simulation_inputs"]

    assert custom["pit_plans"] == plans
    assert "pit_plans" not in reference
    assert custom["tire_inventory"] == reference["tire_inventory"]
    assert custom["starting_tires"] == reference["starting_tires"] == {
        "A": "soft", "B": "medium",
    }
    assert custom["starting_tire_ages"] == reference["starting_tire_ages"] == {
        "A": 1, "B": 2,
    }


def test_cancellation_between_variants_forwards_callback(monkeypatch, offline_loader):
    calls = []
    callback_calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            calls.append((self.kwargs, kwargs))
            callback = kwargs.get("cancel_requested")
            callback_calls.append(callback)
            return object()

    def cancel_requested():
        return bool(calls)

    monkeypatch.setattr(server, "MonteCarloRunner", FakeRunner)
    with pytest.raises(SimulationCancelled):
        server.run_dashboard_simulation(_request(scenarios="dry"),
                                        cancel_requested=cancel_requested)

    assert len(calls) == 1
    assert callback_calls == [cancel_requested]


def test_cancellation_during_second_variant_reaches_it_and_returns_no_payload(
    monkeypatch, offline_loader,
):
    calls = []
    callback_calls = []

    class FakeRunner:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def run(self, **kwargs):
            calls.append(self.kwargs)
            callback = kwargs["cancel_requested"]
            callback_calls.append(callback)
            callback()
            return object()

    def cancel_requested():
        return len(calls) >= 2

    monkeypatch.setattr(server, "MonteCarloRunner", FakeRunner)
    with pytest.raises(SimulationCancelled):
        server.run_dashboard_simulation(_request(scenarios="dry"),
                                        cancel_requested=cancel_requested)

    assert len(calls) == 2
    assert callback_calls == [cancel_requested, cancel_requested]


def test_http_compare_automatic_is_strict_boolean(monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "_current_season", lambda: 2026)
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={
            "simulations": 10,
            "scenarios": "dry",
            "pit_plans": {"A": []},
            "compare_automatic": 1,
        })
    assert response.status_code == 422
    assert "compare_automatic" in response.text


def test_comparison_does_not_mutate_nested_input_configuration(offline_loader):
    plan = {"A": [{"lap": 2, "compound": "hard"}]}
    inventory = {"A": [{"id": "hard-1", "compound": "hard", "age": 0}]}
    original_plan = copy.deepcopy(plan)
    original_inventory = copy.deepcopy(inventory)
    request = _request(pit_plans=plan, tire_inventory=inventory)

    server.run_dashboard_simulation(request)

    assert plan == original_plan
    assert inventory == original_inventory
