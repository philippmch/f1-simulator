"""Actual offline execution, information cutoff and recoverable-output boundaries."""

import importlib.util
import json
import socket
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "examples.backtest_qualifying_led", ROOT / "examples/backtest_qualifying_led.py",
)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def forbidden(*args, **kwargs):
    pytest.fail("This operation must not execute")


def test_list_does_not_construct_or_run_models_or_fetch(monkeypatch, capsys):
    monkeypatch.setattr(RUNNER, "MonteCarloRunner", forbidden)
    monkeypatch.setattr(RUNNER.inputs_verifier, "verify_data", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    RUNNER.main(["--list", "--year", "2026"])
    events = json.loads(capsys.readouterr().out)
    assert events == [{"year": 2026, "round": n} for n in range(1, 17)]


@pytest.mark.parametrize("arguments", [
    ["--trials", "0"], ["--trials", "10001"], ["--workers", "0"],
    ["--workers", "33"], ["--round", "0"], ["--round", "25"], ["--year", "2024"],
])
def test_invalid_bounds_fail_before_loading_or_constructing(arguments, monkeypatch, tmp_path):
    monkeypatch.setattr(RUNNER.inputs_verifier, "read", forbidden)
    monkeypatch.setattr(RUNNER, "MonteCarloRunner", forbidden)
    with pytest.raises(SystemExit) as error:
        RUNNER.main(["--year", "2026", "--output", str(tmp_path / "new.json"), *arguments])
    assert error.value.code == 2


@pytest.mark.parametrize("suffix", ["", ".partial", ".partial.tmp"])
def test_existing_output_or_checkpoint_is_never_overwritten(suffix, monkeypatch, tmp_path):
    output = tmp_path / "new.json"
    existing = output.with_name(output.name + suffix)
    existing.write_text("preserve this", encoding="utf-8")
    monkeypatch.setattr(RUNNER.inputs_verifier, "read", forbidden)
    with pytest.raises(SystemExit, match="2"):
        RUNNER.main(["--year", "2026", "--output", str(output)])
    assert existing.read_text(encoding="utf-8") == "preserve this"


def fake_forecast(event, policy, trials, workers):
    grid = event["race_context"]["starting_grid"]
    return {"trials": trials, "wins": {d: int(i == 0) * trials for i, d in enumerate(grid)},
            "no_classified_winner_count": 0,
            "marginals": {d: {"position_sum": (i + 1) * trials, "podium_count":
                              int(i < 3) * trials, "nonfinish_count": 0}
                          for i, d in enumerate(grid)}, "joint_race_results": []}


def test_outcomes_open_only_after_both_forecasts_are_checkpointed(monkeypatch, tmp_path):
    output = tmp_path / "new.json"
    order = []
    original_read = Path.read_text

    def tracked_read(path, *args, **kwargs):
        if path == RUNNER.labels_verifier.EVIDENCE:
            partial = json.loads(output.with_name(output.name + ".partial").read_bytes())
            assert set(partial["events"][0]["forecasts"]) == {
                "season_points", "qualifying_led",
            }
            assert partial["scoring_performed"] is False
            order.append("labels")
        return original_read(path, *args, **kwargs)

    def tracked_forecast(event, policy, trials, workers):
        order.append(policy)
        return fake_forecast(event, policy, trials, workers)

    monkeypatch.setattr(Path, "read_text", tracked_read)
    monkeypatch.setattr(RUNNER.inputs_verifier, "verify_data", lambda _: None)
    monkeypatch.setattr(RUNNER, "forecast", tracked_forecast)
    RUNNER.main(["--year", "2026", "--round", "16", "--trials", "1",
                 "--output", str(output)])
    assert order == ["season_points", "qualifying_led", "labels"]


def test_failed_second_model_retains_first_unscored_checkpoint(monkeypatch, tmp_path):
    output = tmp_path / "new.json"

    def failing(event, policy, trials, workers):
        if policy == "qualifying_led":
            raise RuntimeError("second model failed")
        return fake_forecast(event, policy, trials, workers)

    monkeypatch.setattr(RUNNER.inputs_verifier, "verify_data", lambda _: None)
    monkeypatch.setattr(RUNNER, "forecast", failing)
    monkeypatch.setattr(RUNNER.labels_verifier, "assess", forbidden)
    with pytest.raises(RuntimeError, match="second model failed"):
        RUNNER.main(["--year", "2026", "--round", "16", "--trials", "1",
                     "--output", str(output)])
    assert not output.exists()
    partial = json.loads(output.with_name(output.name + ".partial").read_bytes())
    assert set(partial["events"][0]["forecasts"]) == {"season_points"}
    assert partial["scoring_performed"] is False


def test_nested_output_directory_is_created_before_checkpointing(monkeypatch, tmp_path):
    output = tmp_path / "new" / "nested" / "backtest.json"
    monkeypatch.setattr(RUNNER.inputs_verifier, "verify_data", lambda _: None)
    monkeypatch.setattr(RUNNER, "forecast", fake_forecast)
    result = RUNNER.main(["--year", "2026", "--round", "16", "--trials", "1",
                          "--output", str(output)])
    assert output.is_file()
    assert result["scoring_performed"] is True
    assert not output.with_name(output.name + ".partial").exists()


def test_changed_scoring_grid_preserves_unscored_checkpoint(monkeypatch, tmp_path):
    output = tmp_path / "new.json"
    original = RUNNER.inputs_verifier.read()
    changed = json.loads(RUNNER.labels_verifier.EVIDENCE.read_text(encoding="utf-8"))
    event = next(e for e in changed["events"] if e["year"] == 2026 and e["round"] == 16)
    event["starting_grid"].reverse()
    source_read = Path.read_text

    def changed_labels(path, *args, **kwargs):
        if path == RUNNER.labels_verifier.EVIDENCE:
            return json.dumps(changed)
        return source_read(path, *args, **kwargs)

    monkeypatch.setattr(RUNNER.inputs_verifier, "read", lambda: original)
    monkeypatch.setattr(RUNNER.inputs_verifier, "verify_data", lambda _: None)
    monkeypatch.setattr(RUNNER.labels_verifier, "assess", lambda _: None)
    monkeypatch.setattr(RUNNER, "forecast", fake_forecast)
    monkeypatch.setattr(Path, "read_text", changed_labels)
    with pytest.raises(ValueError, match="frozen grid"):
        RUNNER.main(["--year", "2026", "--round", "16", "--trials", "1",
                     "--output", str(output)])
    assert not output.exists()
    partial = json.loads(output.with_name(output.name + ".partial").read_bytes())
    assert partial["scoring_performed"] is False
    assert "observed_winner" not in partial["events"][0]


def test_actual_one_trial_pair_rebuilds_and_races_offline(monkeypatch, tmp_path):
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    output = tmp_path / "actual.json"
    result = RUNNER.main(["--year", "2026", "--round", "16", "--trials", "1",
                          "--workers", "1", "--output", str(output)])
    assert output.exists() and not output.with_name(output.name + ".partial").exists()
    assert result["scoring_performed"] is True
    event = result["events"][0]
    first, second = (event["forecasts"][p] for p in ("season_points", "qualifying_led"))
    assert first["seed"] == second["seed"] == 26007
    assert first["input_snapshot"]["drivers"] == second["input_snapshot"]["drivers"]
    assert first["input_snapshot"]["cars"] != second["input_snapshot"]["cars"]
    for forecast in event["forecasts"].values():
        assert sum(forecast["wins"].values()) + forecast["no_classified_winner_count"] == 1
        assert len(forecast["joint_race_results"]) == 1
        rows = forecast["joint_race_results"][0]
        assert sorted(r["position"] for r in rows) == list(range(1, 23))
        assert all({"status", "pit_laps", "pit_stop_details", "tire_set_history"} <= r.keys()
                   for r in rows)
    assert RUNNER.inputs_verifier.digest(
        {k: v for k, v in result.items() if k != "content_sha256"}
    ) == result["content_sha256"]


def test_zero_observed_winner_probability_is_explicit_infinite_loss():
    value = {"trials": 1, "wins": {"A": 1, "B": 0}, "no_classified_winner_count": 0,
             "marginals": {d: {"position_sum": p, "podium_count": 1, "nonfinish_count": 0}
                           for d, p in (("A", 1), ("B", 2))}}
    observed = {"observed_winner": "B", "observed_drivers": {
        d: {"position": p, "podium": True, "dnf": False} for d, p in (("A", 2), ("B", 1))}}
    scores = RUNNER.score(value, observed)
    assert scores["winner_brier"] == 2
    assert scores["winner_log_loss"] is None
    assert scores["winner_log_loss_infinite"] is True
    assert scores["zero_probability_observed_winner"] is True


@pytest.mark.parametrize("field", ["wins", "marginals"])
def test_score_rejects_silent_roster_intersection(field):
    value = {"trials": 1, "wins": {"A": 1, "B": 0}, "no_classified_winner_count": 0,
             "marginals": {d: {"position_sum": p, "podium_count": 1, "nonfinish_count": 0}
                           for d, p in (("A", 1), ("B", 2))}}
    observed = {"observed_winner": "A", "observed_drivers": {
        d: {"position": p, "podium": True, "dnf": False} for d, p in (("A", 1), ("B", 2))}}
    value[field].pop("B")
    with pytest.raises(ValueError, match="same complete field"):
        RUNNER.score(value, observed)
