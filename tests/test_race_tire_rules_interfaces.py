"""Race-wide tyre constraints survive public boundaries without changing old runs."""

import csv
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest
from test_engine_entrypoints import SyntheticLoader

import f1sim.data.current as current_module
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.data import CurrentSeasonDataLoader
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.tire_rules import RACE_TIRE_RULES_POLICY
from f1sim.web import server

ROOT = Path(__file__).resolve().parents[1]
RULES = {"max_set_laps": 2, "minimum_sets": 2}


def runner(engine="standard", **options):
    values = dict(
        drivers=[Driver(id=key, name=key, team_id=key) for key in "AB"],
        cars={key: Car(team_id=key, team_name=key) for key in "AB"},
        track=Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90),
        weather=Weather(change_probability=0),
        seed=19,
        race_engine=engine,
        starting_tires={"A": "medium", "B": "hard"},
    )
    return MonteCarloRunner(**(values | options))


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("supplied", [False, True])
def test_rules_reach_real_process_workers_and_replay_with_both_grid_sources(
    monkeypatch, tmp_path, engine, supplied,
):
    grid = {"starting_grid": ["B", "A"], "pit_lane_starters": ["A"]} if supplied else {}
    inputs = RULES.copy()
    model = runner(engine, race_tire_rules=inputs, **grid)
    inputs["max_set_laps"] = 99
    engine_class = RaceSimulator if engine == "standard" else ChronologicalRace
    method = "simulate_race" if engine == "standard" else "run"
    actual = getattr(engine_class, method)
    observed = []

    def capture(self, *args, **kwargs):
        observed.append((
            kwargs["starting_grid"].copy(), kwargs.get("pit_lane_starters", []).copy(),
        ))
        return actual(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(engine_class, method, capture)
        serial = model.run(2, parallel=False)
    process = model.run(2, parallel=True, max_workers=1)
    assert serial.race_results == process.race_results
    assert serial.qualifying_results == process.qualifying_results
    assert serial.weather_histories == process.weather_histories
    assert serial.event_stats == process.event_stats
    assert serial.input_snapshot == process.input_snapshot
    saved = serial.input_snapshot
    assert saved["schema_version"] == 16
    assert saved["race_tire_rules"] == RULES
    assert saved["race_tire_rules_policy"] == RACE_TIRE_RULES_POLICY
    assert saved.get("starting_grid") == (["B", "A"] if supplied else None)
    assert saved.get("pit_lane_starters", []) == (["A"] if supplied else [])
    assert all({row.driver_id for row in race} == {"A", "B"}
               for race in serial.qualifying_results)
    for (race_grid, pits), qualifying in zip(observed, serial.qualifying_results, strict=True):
        assert race_grid == (["B", "A"] if supplied else [row.driver_id for row in qualifying])
        assert pits == (["A"] if supplied else [])
    for race in serial.race_results:
        for result in race:
            assert result.race_tire_usage["rules"] == RULES
            assert sum(result.race_tire_usage["completed_race_laps_by_set"].values()) == (
                result.laps_completed
            )
    path = Exporter(tmp_path).export_statistics_json(serial)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [serial.race_results[1]]
    assert replay.qualifying_results == [serial.qualifying_results[1]]
    assert replay.input_snapshot["race_tire_rules"] == RULES


def test_schema16_keeps_all_optional_overlays_and_offline_replays_together(tmp_path):
    pool = {key: [
        {"id": "opening", "compound": compound, "age": 3, "remaining_laps": 3},
        {"id": "replacement", "compound": "soft", "remaining_laps": 3},
    ] for key, compound in (("A", "medium"), ("B", "hard"))}
    options = dict(
        race_tire_rules={"max_set_laps": 3, "minimum_sets": 2},
        control_schedule=[],
        qualifying_weather={"Q1": {"condition": "cloudy", "rain_intensity": 0}},
        weather_schedule=[{"lap": 2, "rain_intensity": 0, "condition": "dry"}],
        tire_warmup={"soft": 1.25},
        tire_inventory=pool,
        starting_tire_ages={"A": 3, "B": 3},
        pit_plans={key: [{"lap": 3, "compound": "soft"}] for key in "AB"},
    )
    result = runner("chronological", **options).run(1, parallel=False)
    snapshot = result.input_snapshot
    assert snapshot["schema_version"] == 16
    for key, value in options.items():
        if key == "tire_inventory":
            assert {driver: [row["id"] for row in records]
                    for driver, records in snapshot[key].items()} == {
                        "A": ["opening", "replacement"], "B": ["opening", "replacement"],
                    }
        elif key == "qualifying_weather":
            assert snapshot[key]["Q1"]["condition"] == "cloudy"
            assert snapshot[key]["Q1"]["rain_intensity"] == 0
        else:
            assert snapshot[key] == value
    assert snapshot["tire_usage_policy"] and snapshot["control_schedule_policy"]
    path = Exporter(tmp_path).export_statistics_json(result)
    replay = replay_saved_simulation(path)
    assert replay.input_snapshot == snapshot
    assert replay.race_results == result.race_results
    assert replay.qualifying_results == result.qualifying_results


def test_exports_include_usage_only_when_enabled_and_detach_public_rows(tmp_path):
    enabled = runner(race_tire_rules=RULES).run(1, parallel=False)
    disabled = runner().run(1, parallel=False)
    default = runner(race_tire_rules={"minimum_sets": 1}).run(1, parallel=False)
    assert default.input_snapshot == disabled.input_snapshot
    assert default.race_results == disabled.race_results
    assert default.qualifying_results == disabled.qualifying_results
    assert "race_tire_rules" not in disabled.input_snapshot
    export = Exporter(tmp_path)
    enabled_json = json.loads(export.export_statistics_json(enabled, "enabled.json").read_text())
    disabled_json = json.loads(export.export_statistics_json(disabled, "disabled.json").read_text())
    assert "race_tire_usage" not in disabled_json
    enabled_html = export.export_report_html(enabled, "enabled.html").read_text(encoding="utf-8")
    disabled_html = export.export_report_html(disabled, "disabled.html").read_text(encoding="utf-8")
    assert "Race tyre constraints:" in enabled_html
    assert "at most 2 completed race laps per physical set" in enabled_html
    assert "at least 2 distinct sets actually run" in enabled_html
    assert "started but incomplete laps are not reconstructed" in enabled_html
    assert "Race tyre constraints:" not in disabled_html
    assert len(enabled_json["race_tire_usage"]) == 2
    for entry in enabled_json["race_tire_usage"]:
        actual = next(row for row in enabled.race_results[0] if row.driver_id == entry["driver_id"])
        assert entry == {"simulation": 1, "driver_id": actual.driver_id,
                         "usage": actual.race_tire_usage}
    for name, result in (("on", enabled), ("off", disabled)):
        with export.export_race_results_csv(result, name + ".csv").open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            assert ("race_tire_usage" in reader.fieldnames) == (name == "on")
        for actual, row in zip(result.race_results[0], rows, strict=True):
            if name == "on":
                assert json.loads(row["race_tire_usage"]) == actual.race_tire_usage
            serialized = server._serialize_race_result(actual)
            assert ("race_tire_usage" in serialized) == (name == "on")
            if name == "on":
                serialized["race_tire_usage"]["rules"]["max_set_laps"] = 99
                assert actual.race_tire_usage["rules"] == RULES


def test_legacy_saved_inputs_cannot_enable_new_rules_during_replay(tmp_path):
    original = runner().run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    body = json.loads(path.read_text())
    assert body["simulation_inputs"]["schema_version"] < 16
    body["simulation_inputs"].update(
        race_tire_rules=RULES,
        race_tire_rules_policy=RACE_TIRE_RULES_POLICY,
    )
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="require schema 16"):
        replay_saved_simulation(path)


@pytest.mark.parametrize("rules", [
    {"max_set_laps": True}, {"minimum_sets": 0}, {"max_set_laps": "25"},
])
def test_invalid_http_rules_fail_before_loading(monkeypatch, tmp_path, rules):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_get_loader", lambda: pytest.fail("live loading attempted"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={"race_tire_rules": rules})
    assert response.status_code == 400
    assert "race_tire_rules" in response.text


def test_dashboard_runner_and_metadata_keep_detached_canonical_rules():
    base = runner()
    request = server.DashboardRunRequest(race_tire_rules=RULES.copy())
    arguments = dict(request=request, tire_inventory=None, starting_tires={},
                     starting_tire_ages={}, pit_plans={})
    configured = server._dashboard_runner(
        **arguments, drivers=base.drivers, cars=base.cars, track=base.track,
        weather=base.weather, seed=19, copy_inputs=True,
    )
    metadata = server._dashboard_request_metadata(
        **arguments, canonical_race="1", effective_max_workers=1, compare_automatic=False,
    )
    request.race_tire_rules["max_set_laps"] = 99
    assert configured.race_tire_rules == metadata["race_tire_rules"] == RULES
    metadata["race_tire_rules"]["minimum_sets"] = 4
    assert configured.race_tire_rules == RULES


def test_dashboard_known_rules_and_manual_constraints_combine_before_trials(monkeypatch):
    loader = SyntheticLoader()
    source = {"max_set_laps": 25, "minimum_sets": 3}
    monkeypatch.setattr(loader, "get_race_tire_rules", lambda *args: source, raising=False)
    monkeypatch.setattr(server, "_get_loader", lambda: loader)
    request = server.DashboardRunRequest(
        simulations=10, scenarios="dry", parallel=False, seed=7,
        race_tire_rules={"max_set_laps": 2, "minimum_sets": 1},
    )
    payload = server.run_dashboard_simulation(request)
    expected = {"max_set_laps": 2, "minimum_sets": 3}
    assert payload["request"]["race_tire_rules"] == expected
    assert payload["scenarios"]["dry"]["simulation_inputs"]["race_tire_rules"] == expected
    assert source == {"max_set_laps": 25, "minimum_sets": 3}
    assert request.race_tire_rules == {"max_set_laps": 2, "minimum_sets": 1}


def test_loader_uses_source_cutoff_and_clears_detached_provenance_without_http(monkeypatch):
    clock = {"now": datetime(2025, 11, 16, tzinfo=timezone.utc)}
    monkeypatch.setattr(current_module, "_utc_now", lambda: clock["now"])
    loader = CurrentSeasonDataLoader(
        current_year=2025, http_getter=lambda *a, **kw: pytest.fail("HTTP attempted"),
    )
    event = {"round": 23, "circuit_id": "losail", "date": "2025-11-30"}
    monkeypatch.setattr(loader, "_event_for_race", lambda *args: event)
    assert loader.get_race_tire_rules(2025, "23") == {}
    assert "race_tire_rules" not in loader.get_provenance()
    clock["now"] = datetime(2025, 11, 29, tzinfo=timezone.utc)
    rules = loader.get_race_tire_rules(2025, "23")
    assert rules == {"max_set_laps": 25}
    evidence = loader.get_provenance()["race_tire_rules"]
    assert evidence["year"] == 2025 and evidence["round"] == 23
    assert evidence["sources"]["max_set_laps"]["published_at"].startswith("2025-11-17")
    rules["max_set_laps"] = 99
    evidence["rules"]["max_set_laps"] = 99
    evidence["sources"]["max_set_laps"]["limitations"].clear()
    assert loader.get_provenance()["race_tire_rules"]["rules"] == {"max_set_laps": 25}
    assert loader.get_provenance()["race_tire_rules"]["sources"]["max_set_laps"]["limitations"]
    assert loader.get_race_tire_rules(2025, "23", cutoff="2025-11-16T00:00:00Z") == {}
    assert "race_tire_rules" not in loader.get_provenance()
    assert loader.get_race_tire_rules(2025, "23") == {"max_set_laps": 25}
    loader.refresh()
    assert "race_tire_rules" not in loader.get_provenance()


@pytest.mark.parametrize("flag,value", [
    ("--max-tyre-set-laps", "0"), ("--minimum-tyre-sets", "-1"),
    ("--max-tire-set-laps", "1.5"), ("--minimum-tire-sets", "true"),
])
def test_cli_invalid_rules_fail_before_live_io(flag, value):
    result = subprocess.run(
        [sys.executable, str(ROOT / "examples/simulate_race.py"), flag, value],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert result.returncode == 2
    assert "Fetching the current calendar" not in result.stdout
    assert "positive integer" in result.stderr or "invalid _positive_int value" in result.stderr


def test_cli_forwards_stricter_merged_rules_to_real_export(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "tyre_rules_cli", ROOT / "examples/simulate_race.py",
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    source = {"max_set_laps": 25, "minimum_sets": 3}
    loader = SyntheticLoader()
    monkeypatch.setattr(
        loader, "get_race_tire_rules", lambda *args: deepcopy(source), raising=False,
    )
    monkeypatch.setattr(cli, "CurrentSeasonDataLoader", lambda **kwargs: loader)
    monkeypatch.setattr(sys, "argv", [
        "simulate_race", "--simulations", "1", "--no-parallel", "--scenarios", "dry",
        "--max-tyre-set-laps", "2", "--minimum-tyre-sets", "1",
        "--export", "--output-dir", str(tmp_path),
    ])
    assert cli.main() == 0
    path = next(tmp_path.glob("*statistics*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["simulation_inputs"]["race_tire_rules"] == {
        "max_set_laps": 2, "minimum_sets": 3,
    }
    assert saved["simulation_inputs"]["schema_version"] == 16
    assert saved["race_tire_usage"]
