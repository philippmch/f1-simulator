"""The benchmark compares outcomes independently of process cache warmth."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.fixture
def script():
    return runpy.run_path(str(Path(__file__).resolve().parents[1]
                             / "examples" / "benchmark_strategy_planning.py"))


def test_benchmark_digest_excludes_timings_and_cache_warmth():
    benchmark = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                  / "examples" / "benchmark_strategy_planning.py"))["benchmark"]
    cold = benchmark(drivers=4, laps=8, trials=2)
    warm = benchmark(drivers=4, laps=8, trials=2)
    changed = benchmark(drivers=4, laps=8, trials=2, seed=43)
    assert cold["outcome_sha256"] == warm["outcome_sha256"]
    assert cold["outcome_sha256"] != changed["outcome_sha256"]
    assert len(cold["outcome_sha256"]) == 64
    assert len(cold["trial_seconds"]) == cold["trials"] == 2
    assert all(value >= 0 for value in cold["trial_seconds"])
    assert benchmark(drivers=1, laps=2, trials=1)["later_trial_mean_seconds"] is None


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_finite_automatic_benchmark_is_reproducible(script, engine):
    options = dict(engine=engine, scenario="wetting", inventory="finite", opening="automatic",
                   drivers=2, laps=6, trials=2)
    cold = script["benchmark"](**options)
    warm = script["benchmark"](**options)
    assert cold["outcome_sha256"] == warm["outcome_sha256"]
    assert cold["inventory"] == "finite" and cold["opening"] == "automatic"
    assert cold["starting_tires"] is cold["starting_tire_ages"] is None
    assert cold["tire_inventory"]["D00"] == [
        {"id": "S", "compound": "soft", "age": 5},
        {"id": "H", "compound": "hard", "age": 0},
        {"id": "I", "compound": "intermediate", "age": 4},
    ]


@pytest.mark.parametrize("scenario,compound,age", [("dry", "soft", 5),
                                                  ("rain_transition", "intermediate", 4)])
def test_explicit_finite_opening_matches_a_physical_set(script, scenario, compound, age):
    result = script["benchmark"](scenario=scenario, inventory="finite",
                                 drivers=1, laps=3, trials=1)
    assert result["starting_tires"] == {"D00": compound}
    assert result["starting_tire_ages"] == {"D00": age}


def test_default_opening_keeps_existing_unlimited_ages(script):
    result = script["benchmark"](drivers=3, laps=3, trials=1)
    assert result["benchmark_version"] == 3
    assert result["change_probability"] == 0.0
    assert result["inventory"] == "unlimited"
    assert result["opening"] == "explicit"
    assert result["tire_inventory"] is None
    assert result["starting_tire_ages"] == {"D00": 0, "D01": 6, "D02": 12}


@pytest.mark.parametrize("probability", [0, .35, 1])
def test_change_probability_is_forwarded_to_weather(script, monkeypatch, probability):
    globals_ = script["benchmark"].__globals__
    original_weather = globals_["Weather"]
    observed = []

    def capture_weather(*args, **kwargs):
        weather = original_weather(*args, **kwargs)
        observed.append(weather.change_probability)
        return weather

    monkeypatch.setitem(globals_, "Weather", capture_weather)
    result = script["benchmark"](drivers=1, laps=2, trials=1,
                                 change_probability=probability)
    assert observed == [float(probability)]
    assert result["change_probability"] == float(probability)


def test_evolving_weather_benchmark_is_reproducible(script):
    options = dict(engine="standard", scenario="dry", drivers=2, laps=6,
                   trials=2, seed=17, change_probability=.4)
    first = script["benchmark"](**options)
    second = script["benchmark"](**options)
    assert first["outcome_sha256"] == second["outcome_sha256"]
    assert first["change_probability"] == .4


@pytest.mark.parametrize("probability", [
    True, False, "0.5", None, 1j, float("nan"), float("inf"), float("-inf"),
    -.01, 1.01, 10**1000,
])
def test_python_api_rejects_invalid_change_probability(script, probability):
    with pytest.raises(ValueError, match="change_probability"):
        script["benchmark"](drivers=1, laps=2, trials=1,
                            change_probability=probability)


@pytest.mark.parametrize("field,key", [("tire_set_history", "set_id"), ("tire_inventory", "id")])
def test_digest_includes_physical_ledger_and_final_inventory(script, monkeypatch, field, key):
    options = dict(inventory="finite", opening="automatic", drivers=1, laps=3, trials=1)
    before = script["benchmark"](**options)
    runner = script["MonteCarloRunner"]
    run = runner.run

    def change_identity(self, *args, **kwargs):
        result = run(self, *args, **kwargs)
        # Change only the physical audit identity, keeping race timing and
        # classification untouched. The complete-output digest must notice.
        getattr(result.race_results[0][0], field)[0][key] += "-audit"
        return result

    monkeypatch.setattr(runner, "run", change_identity)
    after = script["benchmark"](**options)
    assert before["outcome_sha256"] != after["outcome_sha256"]


@pytest.mark.parametrize("flag,value", [("--inventory", "invented"),
                                       ("--opening", "invented"), ("--laps", "1")])
def test_cli_rejects_invalid_modes_and_bounds(script, monkeypatch, capsys, flag, value):
    monkeypatch.setattr("sys.argv", ["benchmark_strategy_planning.py", flag, value])
    with pytest.raises(SystemExit) as error:
        script["main"]()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_accepts_and_records_change_probability(script, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["benchmark_strategy_planning.py", "--drivers", "1",
                                     "--laps", "2", "--trials", "1",
                                     "--change-probability", ".35"])
    script["main"]()
    result = json.loads(capsys.readouterr().out)
    assert result["change_probability"] == .35
    assert result["benchmark_version"] == 3


@pytest.mark.parametrize("value", ["nan", "inf", "-0.01", "1.01", "not-a-number"])
def test_cli_rejects_invalid_change_probability(script, monkeypatch, capsys, value):
    monkeypatch.setattr("sys.argv", ["benchmark_strategy_planning.py",
                                     "--change-probability", value])
    with pytest.raises(SystemExit) as error:
        script["main"]()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("options", [{"inventory": "invented"}, {"opening": "invented"}])
def test_python_api_rejects_invalid_modes(script, options):
    with pytest.raises(ValueError):
        script["benchmark"](**options)
