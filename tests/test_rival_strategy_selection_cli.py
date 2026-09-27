"""The rival-plan CLI exports separate, replayable scenario evidence."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def _load_cli():
    sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(
        "validate_rival_pit_plan_selection",
        EXAMPLES / "validate_rival_pit_plan_selection.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rival_selection_cli = _load_cli()


def _saved(tmp_path):
    drivers = [
        Driver(id="A", name="Target", team_id="T"),
        Driver(id="B", name="Rival", team_id="R"),
    ]
    result = MonteCarloRunner(
        drivers,
        {
            "T": Car(team_id="T", team_name="Target Team"),
            "R": Car(team_id="R", team_name="Rival Team"),
        },
        Track(id="t", name="Saved", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0),
        seed=71,
        starting_tires={"A": "medium", "B": "medium"},
        pit_plans={"B": [{"lap": 2, "compound": "hard"}]},
    ).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    return path


def _write_inputs(tmp_path, scenarios):
    plans_path = tmp_path / "target-plans.json"
    plans_path.write_text(json.dumps({
        "automatic": None,
        "planned": [{"lap": 2, "compound": "hard"}],
    }), encoding="utf-8")
    scenarios_path = tmp_path / "rival-scenarios.json"
    scenarios_path.write_text(scenarios, encoding="utf-8")
    return plans_path, scenarios_path


def _invoke(monkeypatch, source, plans, scenarios, output, *extra):
    monkeypatch.setattr(sys, "argv", [
        "validate_rival_pit_plan_selection.py", str(source), "--driver", "A",
        "--plans", str(plans), "--reference", "automatic",
        "--rival-scenarios", str(scenarios),
        "--training-simulations", "1", "--validation-simulations", "1",
        "--output-dir", str(output), *extra,
    ])
    return rival_selection_cli.main()


def test_cli_exports_weighted_selection_and_each_scenario_for_replay(
    tmp_path, monkeypatch, capsys,
):
    source = _saved(tmp_path)
    original_source = source.read_bytes()
    plans, scenarios = _write_inputs(tmp_path, json.dumps({
        "../conservative <rival>": {
            "weight": 1,
            "pit_plans": {"B": None},
        },
        "no-elective-stops": {
            "weight": 3,
            "pit_plans": {"B": []},
        },
    }))
    output = tmp_path / "rival-selection-output"

    assert _invoke(monkeypatch, source, plans, scenarios, output, "--export") == 0

    printed = capsys.readouterr().out
    assert "Weighted training mean points:" in printed
    assert "supplied 1, normalized 0.250" in printed
    assert "supplied 3, normalized 0.750" in printed
    assert (
        "Weighted held-out validation: no change" in printed
        or "Weighted held-out selected-minus-reference mean:" in printed
    )
    assert "cross-scenario covariance is retained" in printed
    assert "not causal proof" in printed

    manifests = list(output.glob("rival_selection_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    selection = manifest["selection"]
    assert selection["rival_scenarios"] == [
        {
            "name": "../conservative <rival>",
            "weight": 1,
            "normalized_weight": 0.25,
            "rival_pit_plans": {"B": None},
        },
        {
            "name": "no-elective-stops",
            "weight": 3,
            "normalized_weight": 0.75,
            "rival_pit_plans": {"B": []},
        },
    ]
    weights = {row["name"]: row["normalized_weight"] for row in selection["rival_scenarios"]}
    weighted_scenario_delta = sum(
        weights[name] * value["mean_points_difference"]
        for name, value in selection["validation_scenario_metrics"].items()
    )
    assert selection["validation_target_metrics"]["mean_points_difference"] == pytest.approx(
        weighted_scenario_delta,
    )

    for name, files in manifest["rival_scenarios"].items():
        assert files["normalized_weight"] == weights[name]
        for key in (
            "training_comparison_json", "training_comparison_html",
            "validation_comparison_json", "validation_comparison_html",
        ):
            assert Path(files[key]).name == files[key]
            assert (output / files[key]).is_file()

        training_path = output / files["training_comparison_json"]
        validation_path = output / files["validation_comparison_json"]
        training_json = json.loads(training_path.read_text(encoding="utf-8"))
        validation_json = json.loads(validation_path.read_text(encoding="utf-8"))
        assert set(training_json["scenarios"]) == {"automatic", "planned"}
        expected_validation = {"automatic"}
        if selection["selected_label"] != "automatic":
            expected_validation.add(selection["selected_label"])
        assert set(validation_json["scenarios"]) == expected_validation
        training_replay = replay_saved_simulation(training_path, 1, scenario="automatic")
        validation_replay = replay_saved_simulation(validation_path, 1, scenario="automatic")
        assert training_replay.seed == selection["seed_ranges"]["training"]["first_seed"]
        assert validation_replay.seed == selection["seed_ranges"]["validation"]["first_seed"]

    assert source.read_bytes() == original_source
    assert all(path.parent == output for path in output.iterdir())


def test_cli_export_uses_core_normalized_weights_for_large_finite_values(
    tmp_path, monkeypatch,
):
    source = _saved(tmp_path)
    plans, scenarios = _write_inputs(tmp_path, json.dumps({
        "first": {"weight": 1e308, "pit_plans": {"B": None}},
        "second": {"weight": 1e308, "pit_plans": {"B": []}},
    }))
    output = tmp_path / "large-weight-output"

    assert _invoke(monkeypatch, source, plans, scenarios, output, "--export") == 0

    manifest_path = next(output.glob("rival_selection_manifest_*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest["selection"]["rival_scenarios"]:
        assert item["normalized_weight"] == pytest.approx(0.5)
    for item in manifest["rival_scenarios"].values():
        assert item["normalized_weight"] == pytest.approx(0.5)


@pytest.mark.parametrize("bad_scenarios", [
    '{"one": {"weight": 1, "pit_plans": {}}, "one": {"weight": 2, "pit_plans": {}}}',
    '{"one": {"weight": NaN, "pit_plans": {}}}',
    '{"one": {"weight": 0, "pit_plans": {}}}',
    '{"one": {"weight": 1, "plans": {}}}',
    '[]',
])
def test_cli_rejects_invalid_scenario_manifest_before_creating_exports(
    tmp_path, monkeypatch, bad_scenarios,
):
    source = _saved(tmp_path)
    plans, scenarios = _write_inputs(tmp_path, bad_scenarios)
    output = tmp_path / "invalid-rival-selection"

    with pytest.raises(SystemExit) as raised:
        _invoke(monkeypatch, source, plans, scenarios, output, "--export")

    assert raised.value.code == 2
    assert not output.exists()
