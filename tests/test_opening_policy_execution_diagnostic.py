"""Public forecast/execution comparison includes timed fuel and physical sets."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def script():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / "examples" / "check_opening_policy_execution.py"))


def test_representative_native_paths_match_in_all_modes(script):
    report = script["check_cases"](cases=("drying", "timed_drying"),
                                    compounds=("soft", "intermediate"), seeds=(0,))
    assert report["comparisons"] == 16
    assert report["mismatches"] == 0
    assert {row["engine"] for row in report["rows"]} == {"standard", "chronological"}
    for row in report["rows"]:
        assert row["executed"]["status"] == "finished"
        assert row["gap_seconds"] == pytest.approx(0, abs=1e-7)
        assert row["physical_fuel_distances"] == [row["inputs"]["scheduled_laps"]]
        if row["case"] == "timed_drying":
            assert row["executed"]["time_limited"]
            assert row["executed"]["laps"] < row["inputs"]["scheduled_laps"]
        if row["inventory"] == "finite":
            assert row["wear_conserved"] is True
            assert len(row["executed"]["tire_inventory"]) == 5
            if row["inputs"]["opening_compound"] == "soft":
                assert row["executed"]["tire_set_history"][0]["age_at_fit"] == 5
    json.dumps(report, allow_nan=False)


def test_diagnostic_reports_nonfinite_prediction_without_invalid_json(script, monkeypatch):
    # Explicit infeasibility remains visible rather than becoming Infinity
    # or an accidental successful comparison through NaN arithmetic.
    monkeypatch.setitem(script["run_case"].__globals__, "_policy_path_outcome",
                        lambda *args, **kwargs: (0, float("inf")))
    row = script["run_case"]("dry", "soft", 0, "standard", "finite")
    assert row["predicted"] == dict(laps=0, seconds=None, status="infeasible")
    assert row["gap_seconds"] is None
    assert "policy_cost" in row["errors"]
    assert "completed_distance" in row["errors"]
    json.dumps(row, allow_nan=False)


def test_invalid_cli_mode_is_rejected(script, monkeypatch):
    monkeypatch.setattr("sys.argv", ["check_opening_policy_execution.py", "--inventory", "unknown"])
    with pytest.raises(SystemExit) as error:
        script["main"]()
    assert error.value.code == 2


@pytest.mark.parametrize("value", [float("nan"), float("-inf")])
def test_invalid_forecast_time_is_an_error(script, monkeypatch, value):
    monkeypatch.setitem(script["run_case"].__globals__, "_policy_path_outcome",
                        lambda *args, **kwargs: (6, value))
    simulator = script["RaceSimulator"]
    execute = simulator.simulate_race

    def retire(self, *args, **kwargs):
        results = execute(self, *args, **kwargs)
        results[0].status = type(results[0].status).DNF
        return results

    monkeypatch.setattr(simulator, "simulate_race", retire)
    row = script["run_case"]("dry", "soft", 0, "standard", "finite")
    # Matching distance and non-completion must not disguise invalid numbers
    # as a legitimate positive-infinity infeasible forecast.
    assert row["executed"]["status"] == "dnf"
    assert row["predicted"]["status"] == "invalid"
    assert row["predicted"]["seconds"] is None
    assert "invalid_predicted_time" in row["errors"]
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("value", [float("nan"), float("-inf"), float("inf")])
def test_invalid_execution_time_is_null_and_reported(script, monkeypatch, value):
    simulator = script["RaceSimulator"]
    execute = simulator.simulate_race

    def invalid_time(self, *args, **kwargs):
        results = execute(self, *args, **kwargs)
        results[0].total_time = value
        return results

    monkeypatch.setattr(simulator, "simulate_race", invalid_time)
    row = script["run_case"]("dry", "soft", 0, "standard", "finite")
    assert row["executed"]["seconds"] is None
    assert row["gap_seconds"] is None
    assert "invalid_executed_time" in row["errors"]
    json.dumps(row, allow_nan=False)


@pytest.mark.parametrize("mismatches,status", [(0, 0), (1, 1)])
def test_cli_prints_report_and_returns_outcome_status(script, monkeypatch, capsys,
                                                      mismatches, status):
    report = dict(comparisons=1, mismatches=mismatches, rows=[])
    monkeypatch.setattr("sys.argv", ["check_opening_policy_execution.py"])
    monkeypatch.setitem(script["main"].__globals__, "check_cases", lambda **kwargs: report)
    assert script["main"]() == status
    assert json.loads(capsys.readouterr().out) == report
