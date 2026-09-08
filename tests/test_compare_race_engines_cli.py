"""Offline engine comparisons export distinct, replayable model variants."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def command_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "compare_race_engines.py"
    spec = importlib.util.spec_from_file_location("compare_engines_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_input(tmp_path):
    result = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")], {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Track", country="Test", total_laps=3, base_lap_time=90),
        Weather(change_probability=0), seed=41, starting_tires={"A": "medium"},
        race_engine="chronological",
    ).run(1, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


def test_engine_cli_exports_unique_comparisons_and_replays(monkeypatch, tmp_path, capsys):
    path = saved_input(tmp_path)
    before = path.read_bytes()
    target = tmp_path / "comparisons"
    monkeypatch.setattr(sys, "argv", ["compare_race_engines.py", str(path),
                                     "--simulations", "2", "--export", "--output-dir", str(target)])
    command = command_module()
    for _ in range(2):
        assert command.main() == 0
    output = capsys.readouterr().out
    assert "seeds 41–42" in output
    assert "Win % [95% range]" in output
    assert "mean winning distance 3.00 laps (2 known winners)" in output
    assert "Equal seeds do not freeze later race events" in output
    assert "experimental" in output
    comparisons = list(target.glob("race_engines_*.json"))
    assert len(comparisons) == 2
    reports = list(target.glob("race_engines_*.html"))
    assert {p.stem for p in reports} == {p.stem for p in comparisons}
    assert len(list(target.glob("*statistics.json"))) == 4
    for comparison in comparisons:
        scenarios = json.loads(comparison.read_text(encoding="utf-8"))["scenarios"]
        assert list(scenarios) == ["standard", "chronological"]
        for engine, scenario in scenarios.items():
            assert scenario["race_engine"] == engine
            assert scenario["simulation_inputs"]["starting_tires"] == {"A": "medium"}
            assert scenario["simulation_inputs"]["weather"]["change_probability"] == 0
            assert scenario["probability_intervals"]["A"]["trials"] == 2
            replay = replay_saved_simulation(comparison, simulation=2, scenario=engine)
            assert replay.race_engine == engine
            assert replay.seed == 42
    assert path.read_bytes() == before


@pytest.mark.parametrize("arguments,expected", [
    (["--simulations", "0"], "between 1 and 1000"),
    (["--simulations", "1001"], "between 1 and 1000"),
    (["--max-workers", "17"], "between 1 and 16"),
])
def test_invalid_limits_precede_comparison(monkeypatch, capsys, arguments, expected):
    command = command_module()
    monkeypatch.setattr(command, "compare_saved_race_engines", lambda *a, **k: pytest.fail("run"))
    monkeypatch.setattr(sys, "argv", ["compare_race_engines.py", "missing.json", *arguments])
    with pytest.raises(SystemExit) as error:
        command.main()
    assert error.value.code == 2
    assert expected in capsys.readouterr().err


def test_invalid_engine_is_clean_error_without_exports(monkeypatch, tmp_path, capsys):
    target = tmp_path / "should_not_exist"
    monkeypatch.setattr(sys, "argv", ["compare_race_engines.py", "missing.json",
                                     "--engines", "standard,unknown", "--export",
                                     "--output-dir", str(target)])
    with pytest.raises(SystemExit) as error:
        command_module().main()
    assert error.value.code == 2
    assert "engine" in capsys.readouterr().err
    assert not target.exists()


def test_comparison_without_export_writes_nothing(monkeypatch, tmp_path):
    path = saved_input(tmp_path)
    target = tmp_path / "should_not_exist"
    monkeypatch.setattr(sys, "argv", ["compare_race_engines.py", str(path), "--simulations", "1",
                                     "--engines", "chronological,standard",
                                     "--output-dir", str(target)])
    assert command_module().main() == 0
    assert not target.exists()
