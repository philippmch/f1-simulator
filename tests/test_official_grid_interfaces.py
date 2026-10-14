"""Explicit official-grid requests must not become simulated-grid forecasts."""

import importlib.util
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from test_dashboard_plan_comparison import MultiDriverLoader

from f1sim.web import server


class OfficialLoader(MultiDriverLoader):
    def __init__(self):
        super().__init__()
        self.evidence = None
        self.change = None
        self.qualifying = datetime.now(timezone.utc) - timedelta(hours=3)
        self.race_start = datetime.now(timezone.utc) + timedelta(days=1)

    def list_available_events(self, year):
        return [{"round": 1, "race": "Synthetic", "date": self.race_start.date().isoformat(),
                 "time": self.race_start.time().isoformat() + "Z",
                 "sessions": {"Qualifying": {"date": self.qualifying.date().isoformat(),
                                              "time": self.qualifying.time().isoformat() + "Z"}}}]

    def get_starting_grid(self, year, race, drivers):
        self.calls.append("grid")
        if self.change == "unavailable":
            self.evidence = {"mode": "simulated", "reason": "published_grid_unavailable"}
            return None
        self.evidence = {
            "mode": "published", "year": year, "round": 1,
            "starting_grid": ["B", "A"], "pit_lane_starters": ["A"],
            "source_url": f"https://www.formula1.com/en/results/{year}/races/1234/test/starting-grid",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "qualifying_started_at": self.qualifying.isoformat(),
        }
        if self.change == "stale":
            self.evidence["fetched_at"] = (datetime.now(timezone.utc)
                                            - timedelta(minutes=5)).isoformat()
        elif self.change == "wrong_event":
            self.evidence["round"] = 2
        elif self.change == "mismatched":
            self.evidence["starting_grid"] = ["A", "B"]
        elif self.change == "wrong_source":
            self.evidence["source_url"] = "https://example.com/grid"
        elif self.change == "pit_mismatch":
            self.evidence["pit_lane_starters"] = []
        elif self.change == "future":
            self.evidence["fetched_at"] = (datetime.now(timezone.utc)
                                             + timedelta(minutes=5)).isoformat()
        elif self.change == "wrong_year":
            self.evidence["year"] = year - 1
        elif self.change == "wrong_qualifying":
            self.evidence["qualifying_started_at"] = (self.qualifying
                                                        - timedelta(hours=1)).isoformat()
        elif self.change == "untimed":
            self.evidence["fetched_at"] = datetime.now().isoformat()
        return ["B"] if self.change == "incomplete" else ["B", "A"]

    def get_pit_lane_starters(self):
        return ["A"]

    def get_provenance(self):
        return {"source": "Synthetic test", "race_grid": deepcopy(self.evidence)}


def configure(monkeypatch, adapter):
    monkeypatch.setattr(server, "_get_loader", lambda: adapter)
    monkeypatch.setattr(server, "_current_season", lambda: datetime.now(timezone.utc).year)


def request(**kwargs):
    return server.DashboardRunRequest(simulations=10, scenarios="dry", parallel=False,
                                      race_grid_mode="official", **kwargs)


def test_official_grid_and_pit_queue_reach_real_runners_and_reference(monkeypatch):
    adapter = OfficialLoader()
    configure(monkeypatch, adapter)
    contexts = []

    def prefer_native(year, race, drivers, weather, **context):
        contexts.append(context)
        return context["starting_grid"] == ["B", "A"]

    adapter.prefer_native_winner_forecast = prefer_native
    adapter.get_winner_allocation = lambda *a: pytest.fail("Native chances reallocated")
    submitted = request(pit_plans={"A": []}, compare_automatic=True)
    before = deepcopy(submitted)
    payload = server.run_dashboard_simulation(submitted)
    assert submitted == before
    assert adapter.calls.count("grid") == 1
    assert len(contexts) == 1
    for variant in (payload, payload["automatic_reference"]):
        assert variant["request"]["race_grid_mode"] == "official"
        inputs = variant["scenarios"]["dry"]["simulation_inputs"]
        assert inputs["starting_grid"] == ["B", "A"]
        assert inputs["pit_lane_starters"] == ["A"]
        assert variant["provenance"]["race_grid"]["mode"] == "published"


@pytest.mark.parametrize("change", ["unavailable", "incomplete", "stale", "wrong_event",
                                   "mismatched", "wrong_source", "pit_mismatch", "future",
                                   "wrong_year", "wrong_qualifying", "untimed"])
def test_unverified_official_grid_stops_before_monte_carlo(monkeypatch, change):
    adapter = OfficialLoader()
    adapter.change = change
    configure(monkeypatch, adapter)
    monkeypatch.setattr(server, "MonteCarloRunner", lambda **k: pytest.fail("Trial setup started"))
    with pytest.raises(ValueError, match="Official race grid"):
        server.run_dashboard_simulation(request())


@pytest.mark.parametrize("overrides", [
    {"starting_grid": ["B", "A"]}, {"pit_lane_starters": []},
    {"qualifying_weather": {"Q1": {"rain_intensity": .2}}},
])
def test_official_custom_input_conflicts_precede_live_io(monkeypatch, overrides):
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("Live I/O started"))
    with pytest.raises(ValueError, match="[Oo]fficial"):
        server.run_dashboard_simulation(request(**overrides))


def test_official_request_before_qualifying_stops_before_trials(monkeypatch):
    adapter = OfficialLoader()
    adapter.qualifying = datetime.now(timezone.utc) - timedelta(minutes=30)
    configure(monkeypatch, adapter)
    monkeypatch.setattr(server, "MonteCarloRunner", lambda **k: pytest.fail("Trial setup started"))
    with pytest.raises(ValueError, match="Official race grid"):
        server.run_dashboard_simulation(request())


def test_completed_current_season_race_can_use_official_grid_for_simulation(monkeypatch):
    adapter = OfficialLoader()
    adapter.race_start = datetime.now(timezone.utc) - timedelta(minutes=1)
    configure(monkeypatch, adapter)
    payload = server.run_dashboard_simulation(request())
    assert payload["request"]["race_grid_mode"] == "official"
    assert payload["scenarios"]["dry"]["simulation_inputs"]["starting_grid"] == ["B", "A"]
    assert "grid_references" not in payload


def test_official_failure_is_a_clear_http_validation_error(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    adapter = OfficialLoader()
    adapter.change = "unavailable"
    configure(monkeypatch, adapter)
    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(server, "MonteCarloRunner", lambda **k: pytest.fail("Trial setup started"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={"race_grid_mode": "official",
                              "simulations": 10, "scenarios": "dry", "parallel": False})
    assert response.status_code == 400
    assert "Official race grid" in response.text


def test_simulated_grid_does_not_fetch_published_grid(monkeypatch):
    adapter = OfficialLoader()
    adapter.get_starting_grid = lambda *a: pytest.fail("Published grid fetched")
    configure(monkeypatch, adapter)
    payload = server.run_dashboard_simulation(server.DashboardRunRequest(
        simulations=10, scenarios="dry", parallel=False, race_grid_mode="simulated",
    ))
    assert payload["request"]["race_grid_mode"] == "simulated"
    assert "starting_grid" not in payload["scenarios"]["dry"]["simulation_inputs"]


def cli_module(monkeypatch, adapter, *arguments):
    path = Path(__file__).resolve().parents[1] / "examples" / "simulate_race.py"
    spec = importlib.util.spec_from_file_location("official_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CurrentSeasonDataLoader", lambda **kwargs: adapter)
    monkeypatch.setattr(sys, "argv", [str(path), "-n", "1", "--no-parallel", "--scenarios", "dry",
                                      *arguments])
    return module


def test_cli_official_unavailable_stops_before_runner(monkeypatch, capsys):
    adapter = OfficialLoader()
    adapter.change = "unavailable"
    module = cli_module(monkeypatch, adapter, "--race-grid", "official")
    monkeypatch.setattr(module, "MonteCarloRunner", lambda **k: pytest.fail("Trial setup started"))
    assert module.main() == 1
    assert "Official race grid" in capsys.readouterr().out


@pytest.mark.parametrize("mode,completed", [("official", False), ("official", True),
                                           ("simulated", False)])
def test_cli_grid_selection_reaches_real_runner(monkeypatch, mode, completed):
    adapter = OfficialLoader()
    if completed:
        adapter.race_start = datetime.now(timezone.utc) - timedelta(minutes=1)
    if mode == "simulated":
        adapter.get_starting_grid = lambda *a: pytest.fail("Published grid fetched")
    module = cli_module(monkeypatch, adapter, "--race-grid", mode)
    captured = []
    runner = module.MonteCarloRunner

    def capture(**kwargs):
        captured.append(kwargs)
        return runner(**kwargs)

    monkeypatch.setattr(module, "MonteCarloRunner", capture)
    assert module.main() == 0
    assert len(captured) == 1
    assert captured[0].get("starting_grid") == (["B", "A"] if mode == "official" else None)
    assert captured[0].get("pit_lane_starters", []) == (["A"] if mode == "official" else [])
