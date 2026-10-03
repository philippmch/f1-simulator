"""Usage constraints survive saved inputs, workers, replay and public requests."""

from copy import deepcopy

import pytest
from test_inventory_analysis import make_runner, save_result

from f1sim.analysis.paired_comparison import _snapshot
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.strategy_comparison import compare_saved_race_engines
from f1sim.output import Exporter
from f1sim.output.qualifying_context import qualifying_weather_context
from f1sim.output.weather_schedule_context import weather_schedule_context
from f1sim.simulation.tire_inventory import TIRE_USAGE_POLICY
from f1sim.web import server


def pool():
    return {"A": [{"id": "S", "compound": "soft", "age": 5, "remaining_laps": 2},
                  {"id": "H", "compound": "hard", "age": 0, "remaining_laps": 3}]}


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("combined", [False, True])
def test_usage_schema_replays_and_combines_other_optional_features(tmp_path, engine, combined):
    options = (dict(tire_warmup={"hard": 1.}, pit_plans={"A": []},
                    qualifying_weather={"Q1": {"rain_intensity": .1}},
                    weather_schedule=[{"lap": 2, "rain_intensity": .05}]) if combined else {})
    runner = make_runner(engine, tire_inventory=pool(), starting_tires={"A": "soft"},
                         starting_tire_ages={"A": 5}, **options)
    serial = runner.run(2, parallel=False)
    assert serial.input_snapshot["schema_version"] == 9
    assert serial.input_snapshot["tire_usage_policy"] == TIRE_USAGE_POLICY
    assert serial.input_snapshot["tire_inventory"] == pool()
    assert _snapshot(serial) is not None
    summaries = serial.get_pit_decision_statistics()
    assert all(item["missing_reason_stops"] == 0 for item in summaries.values())
    path = save_result(tmp_path, serial)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results[0] == serial.race_results[1]
    assert replay.input_snapshot == serial.input_snapshot
    if combined:
        assert "unrecognized" not in qualifying_weather_context(serial.input_snapshot)
        assert "unrecognized" not in weather_schedule_context(serial.input_snapshot)
    else:
        parallel = runner.run(2, parallel=True, max_workers=2)
        assert parallel.race_results == serial.race_results
        variants = compare_saved_race_engines(path, num_simulations=1)
        assert all(item.input_snapshot["tire_inventory"] == pool() for item in variants.values())
    report = Exporter(tmp_path)._tire_set_ledger_html(serial)
    assert "remaining laps at fit" in report and "remaining laps at end" in report
    assert "remaining laps" in report


@pytest.mark.parametrize("mutation", ["downgrade", "missing_policy", "unknown_policy", "no_limit"])
def test_replay_and_pairing_reject_silent_usage_policy_changes(tmp_path, mutation):
    result = make_runner(tire_inventory=pool()).run(1, parallel=False)
    altered = deepcopy(result)
    snapshot = altered.input_snapshot
    if mutation == "downgrade":
        snapshot["schema_version"] = 4
        snapshot.pop("tire_usage_policy")
    elif mutation == "missing_policy":
        snapshot.pop("tire_usage_policy")
    elif mutation == "unknown_policy":
        snapshot["tire_usage_policy"] = "unknown"
    else:
        for item in snapshot["tire_inventory"]["A"]:
            item.pop("remaining_laps")
    with pytest.raises(ValueError, match="usage"):
        _load_saved_runner(save_result(tmp_path, altered))
    assert _snapshot(altered) is None
    changed = deepcopy(result)
    changed.input_snapshot["tire_inventory"]["A"][0]["remaining_laps"] += 1
    assert _snapshot(changed)[0] != _snapshot(result)[0]


@pytest.mark.parametrize("remaining", [True, "2", 2., -1, 1001])
def test_api_rejects_invalid_allowances_before_live_loading(tmp_path, monkeypatch, remaining):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("F1SIM_RUN_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_get_loader", lambda **kw: pytest.fail("live loading"))
    with TestClient(server.build_fastapi_app()) as client:
        response = client.post("/api/run", json={"tire_inventory": {
            "A": [{"compound": "hard", "remaining_laps": remaining}]}})
    assert response.status_code == 400 and "remaining_laps" in response.text


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_dashboard_keeps_initial_and_consumed_allowances(monkeypatch, engine):
    from test_engine_entrypoints import SyntheticLoader

    monkeypatch.setattr(server, "_get_loader", SyntheticLoader)
    request = server.DashboardRunRequest(
        simulations=10, scenarios="dry", seed=7, parallel=False, race_engine=engine,
        starting_tires={"A": "soft"}, starting_tire_ages={"A": 5}, tire_inventory=pool())
    result = server.run_dashboard_simulation(request)
    assert result["request"]["tire_inventory"] == pool()
    scenario = result["scenarios"]["dry"]
    assert scenario["simulation_inputs"]["schema_version"] == 9
    assert scenario["simulation_inputs"]["tire_usage_policy"] == TIRE_USAGE_POLICY
    first = scenario["sample_race"][0]
    assert first["tire_set_history"][0]["remaining_laps_at_fit"] == 2
    assert all(item["remaining_laps"] >= 0 for item in first["tire_inventory"])
    assert "remaining laps at fit" in result["comparison_report_html"]
