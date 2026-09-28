"""Command-line selection reports phases and exports replayable evidence."""

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
        "validate_pit_plan_selection", EXAMPLES / "validate_pit_plan_selection.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selection_cli = _load_cli()


def _saved(tmp_path):
    result = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")],
        {"T": Car(team_id="T", team_name="Team")},
        Track(id="t", name="Saved", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), seed=10,
    ).run(1, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


def _invoke(monkeypatch, source, plans, output, *extra, driver=True):
    target_args = ["--driver", "A"] if driver else []
    monkeypatch.setattr(sys, "argv", [
        "validate_pit_plan_selection.py", str(source), *target_args,
        "--plans", str(plans), "--reference", "automatic",
        "--training-simulations", "1", "--validation-simulations", "1",
        "--output-dir", str(output), *extra,
    ])
    return selection_cli.main()


def test_cli_formats_gain_tie_and_loss_profile(capsys):
    selection_cli._print_points_outcome_profile({
        "paired_races": 4,
        "more_points_races": 2,
        "equal_points_races": 1,
        "fewer_points_races": 1,
        "mean_points_gain_when_ahead": 5,
        "mean_points_loss_when_behind": 2,
    })

    assert capsys.readouterr().out.strip() == (
        "Held-out paired points outcomes (more/equal/fewer): 2/1/1 across 4 seeds; "
        "mean gain when ahead 5.000 points; mean loss when behind 2.000 points."
    )


def test_cli_exports_separate_replayable_phase_evidence_and_manifest(
    tmp_path, monkeypatch, capsys,
):
    source = _saved(tmp_path)
    plans = tmp_path / "plans.json"
    plans.write_text(json.dumps({"automatic": None, "same": None}), encoding="utf-8")
    output = tmp_path / "selection-output"

    assert _invoke(monkeypatch, source, plans, output, "--export") == 0

    printed = capsys.readouterr().out
    assert "Fixed reference: automatic" in printed
    assert "Training seeds 11–11" in printed
    assert "validation seeds 12–12" in printed
    assert "Selected: automatic because an exact training tie preferred the reference." in printed
    assert "no separate standard error is estimated" in printed
    assert "outcome profile: not independently estimated" in printed
    assert "repeating the same request reuses them" in printed
    manifests = list(output.glob("selection_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["selection"]["validation_status"] == "no_change"
    assert manifest["selection"]["validation_target_metrics"][
        "points_outcome_profile"
    ] is None
    for key in (
        "training_comparison_json", "training_comparison_html",
        "validation_comparison_json", "validation_comparison_html",
    ):
        assert (output / manifest[key]).is_file()

    training_replay = replay_saved_simulation(
        output / manifest["training_comparison_json"], 1, scenario="same",
    )
    validation_replay = replay_saved_simulation(
        output / manifest["validation_comparison_json"], 1, scenario="automatic",
    )
    assert training_replay.seed == 11
    assert validation_replay.seed == 12


@pytest.mark.parametrize("failure", ["duplicate", "mutually-exclusive", "constructor-shape"])
def test_cli_errors_do_not_create_exports(tmp_path, monkeypatch, failure):
    source = _saved(tmp_path)
    plans = tmp_path / "plans.json"
    output = tmp_path / f"no-exports-{failure}"
    extra = []
    if failure == "duplicate":
        plans.write_text('{"automatic": null, "automatic": null}', encoding="utf-8")
    elif failure == "constructor-shape":
        plans.write_text(json.dumps({"automatic": None, "partial": {"unknown": []}}),
                         encoding="utf-8")
        extra = ["--constructor", "T"]
    else:
        plans.write_text(json.dumps({"automatic": None, "same": None}), encoding="utf-8")
        extra = ["--constructor", "T"]

    with pytest.raises(SystemExit) as raised:
        _invoke(monkeypatch, source, plans, output, *extra, "--export",
                driver=failure != "constructor-shape")
    assert raised.value.code == 2
    assert not output.exists()
