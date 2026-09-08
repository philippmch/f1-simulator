"""The offline comparison command prints uncertainty and exports replayable runs."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def module():
    path = Path(__file__).resolve().parents[1] / "examples" / "compare_starting_tyres.py"
    spec = importlib.util.spec_from_file_location("compare_starting_cli", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def saved_input(tmp_path):
    result = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")], {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Track", country="Test", total_laps=3, base_lap_time=90),
        Weather(change_probability=0), seed=41, starting_tires={"A": "medium"},
    ).run(1, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


def test_comparison_cli_export_is_unique_and_replayable(monkeypatch, tmp_path, capsys):
    path = saved_input(tmp_path)
    before = path.read_bytes()
    target = tmp_path / "comparisons"
    monkeypatch.setattr(sys, "argv", ["compare_starting_tyres.py", str(path), "--driver", "A",
                                     "--compounds", "automatic,hard", "--simulations", "2",
                                     "--export", "--output-dir", str(target)])
    command = module()
    for _ in range(2):
        assert command.main() == 0
    output = capsys.readouterr().out
    assert "seeds 41–42" in output
    assert "Win % [95% range]" in output
    assert "Equal seeds do not freeze later race events" in output
    comparisons = list(target.glob("starting_tyres_*.json"))
    assert len(comparisons) == 2
    assert len(list(target.glob("*statistics.json"))) == 4
    for comparison in comparisons:
        scenarios = json.loads(comparison.read_text(encoding="utf-8"))["scenarios"]
        assert list(scenarios) == ["automatic", "hard"]
        assert scenarios["automatic"]["simulation_inputs"]["starting_tires"] == {}
        assert scenarios["hard"]["simulation_inputs"]["starting_tires"] == {"A": "hard"}
        replay = replay_saved_simulation(comparison, simulation=2, scenario="hard")
        assert replay.seed == 42
        assert replay.race_results[0][0].strategy[0] == "hard"
    assert path.read_bytes() == before


@pytest.mark.parametrize("arguments,expected", [
    (["--simulations", "0"], "between 1 and 1000"),
    (["--simulations", "1001"], "between 1 and 1000"),
    (["--max-workers", "17"], "between 1 and 16"),
])
def test_invalid_resource_options_fail_before_comparison(monkeypatch, capsys, arguments, expected):
    command = module()
    monkeypatch.setattr(command, "compare_saved_starting_tires", lambda *a, **k: pytest.fail("run"))
    monkeypatch.setattr(sys, "argv", ["compare_starting_tyres.py", "missing.json", "--driver", "A",
                                     *arguments])
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2
    assert expected in capsys.readouterr().err


def test_unknown_driver_has_clean_error_and_no_exports(monkeypatch, tmp_path, capsys):
    path = saved_input(tmp_path)
    target = tmp_path / "should_not_exist"
    monkeypatch.setattr(sys, "argv", ["compare_starting_tyres.py", str(path), "--driver", "UNKNOWN",
                                     "--export", "--output-dir", str(target)])
    with pytest.raises(SystemExit) as error:
        module().main()
    assert error.value.code == 2
    assert "UNKNOWN" in capsys.readouterr().err
    assert not target.exists()
