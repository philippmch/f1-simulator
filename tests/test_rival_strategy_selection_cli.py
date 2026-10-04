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


def test_cli_formats_seed_outcomes_and_identity_profile():
    assert rival_selection_cli._points_outcome_profile_text({
        "paired_races": 4,
        "more_points_races": 2,
        "equal_points_races": 1,
        "fewer_points_races": 1,
        "mean_points_gain_when_ahead": 5,
        "mean_points_loss_when_behind": 2,
    }) == (
        "more/equal/fewer 2/1/1 of 4 seeds; mean gain when ahead 5.000 points; "
        "mean loss when behind 2.000 points"
    )
    assert rival_selection_cli._points_outcome_profile_text(
        None, identity=True,
    ) == "not independently estimated; the selected plan is the reference"


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


@pytest.mark.parametrize("gap,tied,expected", [
    (0, True, "0.000 (exact tie)"),
    (0, False, "below numeric reporting precision"),
    (7e-20, False, "7.000e-20"),
    (None, None, "not recorded"),
    (0, 1, "not recorded"),
    (True, False, "not recorded"),
    (float("nan"), False, "not recorded"),
    (1, True, "not recorded"),
])
def test_cli_shortfall_preserves_exact_tie_evidence(gap, tied, expected):
    assert rival_selection_cli._training_shortfall({
        "label": "reference", "mean_points_behind_selected": gap, "tied_for_best": tied,
    }, "planned") == expected
    assert rival_selection_cli._training_shortfall({
        "label": "planned", "mean_points_behind_selected": 0, "tied_for_best": True,
    }, "planned") == "0.000 (selected)"


def test_cli_keeps_tiny_profile_quantities_visible_and_unknown_reason_unavailable():
    text = rival_selection_cli._points_outcome_profile_text({
        "paired_races": 2, "more_points_races": 1, "equal_points_races": 0,
        "fewer_points_races": 1, "mean_points_gain_when_ahead": 7e-20,
        "mean_points_loss_when_behind": 3e-20,
    })
    assert "7.000e-20 points" in text
    assert "3.000e-20 points" in text
    assert rival_selection_cli._selection_number(0) == "0.000"
    assert rival_selection_cli._selection_number(-1e-20) == "-1.000e-20"
    assert rival_selection_cli._selection_number(float("inf")) == "not recorded"
    assert rival_selection_cli._selection_number(True) == "not recorded"
    assert rival_selection_cli._selection_reason(None) == "selection reason not recorded"
    assert rival_selection_cli._selection_reason("unknown") == "selection reason not recorded"
    assert "exact training tie preferred the reference" in rival_selection_cli._selection_reason(
        "reference_preferred_on_exact_tie",
    )
    assert "exact training tie used the first candidate" in rival_selection_cli._selection_reason(
        "first_plan_order_on_exact_tie",
    )


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


@pytest.mark.parametrize("objective", ["points", "win", "podium"])
def test_cli_minimax_criterion_exports_frozen_evidence(tmp_path, monkeypatch, capsys, objective):
    source = _saved(tmp_path)
    plans, scenarios = _write_inputs(tmp_path, json.dumps({
        "automatic": {"weight": 9, "pit_plans": {"B": None}},
        "no_stop": {"weight": 1, "pit_plans": {"B": []}},
    }))
    output = tmp_path / "output"
    before = source.read_bytes()
    assert _invoke(
        monkeypatch, source, plans, scenarios, output,
        "--objective", objective, "--selection-method", "minimax_regret", "--export",
    ) == 0
    text = capsys.readouterr().out
    assert "Minimax regret training choice" in text
    assert "scenario weights do not affect selection" in text
    assert "maximum shortfall" in text and "best candidate mean" in text
    assert "selected minus candidate mean points" in text
    assert "Weighted means below are context" in text
    manifest = json.loads(next(output.glob("rival_selection_manifest_*.json")).read_text(
        encoding="utf-8",
    ))
    selection = manifest["selection"]
    assert selection["selection_method"] == "minimax_regret"
    assert selection["objective"] == objective
    assert len(selection["training_regret_table"]) == 2
    assert all(set(row["scenarios"]) == {"automatic", "no_stop"}
               for row in selection["training_regret_table"])
    html = (output / manifest["selection_report_html"]).read_text(encoding="utf-8")
    assert "Minimax regret training choice" in html
    assert "Selected minus candidate mean points" in html
    assert source.read_bytes() == before


@pytest.mark.parametrize("objective", ["win", "podium"])
def test_cli_probability_objective_exports_weighted_and_scenario_scores(
    tmp_path, monkeypatch, capsys, objective,
):
    source = _saved(tmp_path)
    plans, scenarios = _write_inputs(tmp_path, json.dumps({
        "Aggregate": {"weight": 1, "pit_plans": {"B": None}},
    }))
    output = tmp_path / "output"
    assert _invoke(monkeypatch, source, plans, scenarios, output,
                   "--objective", objective, "--export") == 0
    text = capsys.readouterr().out
    assert "Training objective probabilities:" in text and "percentage points" in text
    assert text.count("Aggregate: reference") == 2
    manifest = json.loads(next(output.glob("rival_selection_manifest_*.json")).read_text(
        encoding="utf-8",
    ))
    assert manifest["selection"]["objective"] == objective
    assert "Held-out objective probabilities" in (
        output / manifest["selection_report_html"]
    ).read_text(encoding="utf-8")


def test_cli_invalid_objective_is_rejected_before_source_and_exports(tmp_path, monkeypatch):
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        _invoke(monkeypatch, tmp_path / "missing.json", tmp_path / "plans.json",
                tmp_path / "rivals.json", output, "--objective", "finish", "--export")
    assert error.value.code == 2 and not output.exists()


def test_cli_joint_weather_exports_replay_each_case_with_shared_qualifying(
    tmp_path, monkeypatch, capsys,
):
    source = _saved(tmp_path)
    before = source.read_bytes()
    assumptions = {
        "dry": {"weight": 2, "pit_plans": {}, "weather_schedule": []},
        "rain <case>": {"weight": 1, "pit_plans": {"B": []},
                        "weather": {"condition": "light_rain", "rain_intensity": .3,
                                    "track_wetness": .3},
                        "weather_schedule": [{"lap": 3, "rain_intensity": .8,
                                              "condition": "heavy_rain"}]},
    }
    plans, scenarios = _write_inputs(tmp_path, json.dumps(assumptions))
    output = tmp_path / "weather-selection"
    assert _invoke(monkeypatch, source, plans, scenarios, output,
                   "--objective", "win", "--export") == 0
    text = capsys.readouterr().out
    assert "Frozen initial race weather" in text and "Frozen known rainfall steps" in text
    assert "Shared qualifying weather (frozen)" in text
    manifest = json.loads(next(output.glob("rival_selection_manifest_*.json")).read_text(
        encoding="utf-8",
    ))
    selection = manifest["selection"]
    assert "weather_and_rival" in selection["method"]
    assert "Race weather and schedule" in (output / manifest["selection_report_html"]).read_text(
        encoding="utf-8",
    )
    for case in selection["rival_scenarios"]:
        for phase in ("training", "validation"):
            path = output / manifest["rival_scenarios"][case["name"]][f"{phase}_comparison_json"]
            saved = json.loads(path.read_text(encoding="utf-8"))
            for label in saved["scenarios"]:
                replay = replay_saved_simulation(path, 1, scenario=label)
                assert replay.seed == selection["seed_ranges"][phase]["first_seed"]
                assert replay.input_snapshot["weather"] == case["weather"]
                assert replay.input_snapshot.get("weather_schedule", []) == case["weather_schedule"]
                assert replay.input_snapshot["qualifying_weather"] == (
                    selection["frozen_qualifying_weather"]
                )
    assert source.read_bytes() == before


@pytest.mark.parametrize("override", [
    {"weather": {"humidity": "0.5"}},
    {"weather_schedule": [{"lap": 2, "rain_intensity": True}]},
])
def test_cli_weather_preflight_rejects_before_source_loading_or_exports(
    tmp_path, monkeypatch, override,
):
    plans, scenarios = _write_inputs(tmp_path, json.dumps({
        "invalid": {"weight": 1, "pit_plans": {}, **override},
    }))
    output = tmp_path / "output"
    with pytest.raises(SystemExit) as error:
        _invoke(monkeypatch, tmp_path / "missing.json", plans, scenarios, output, "--export")
    assert error.value.code == 2 and not output.exists()


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
    assert "mean points behind selected:" in printed
    assert "supplied 1, normalized 0.250" in printed
    assert "supplied 3, normalized 0.750" in printed
    assert (
        "Weighted held-out validation: no change" in printed
        or "Weighted held-out selected-minus-reference mean:" in printed
    )
    assert "Weighted held-out paired points outcome profile:" in printed
    assert "cross-scenario covariance is retained" in printed
    assert "not causal proof" in printed

    manifests = list(output.glob("rival_selection_manifest_*.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    selection = manifest["selection"]
    aggregate_profile = selection["validation_target_metrics"]["points_outcome_profile"]
    if selection["validation_status"] == "no_change":
        assert aggregate_profile is None
        assert all(
            metrics["points_outcome_profile"] is None
            for metrics in selection["validation_scenario_metrics"].values()
        )
    else:
        assert aggregate_profile["paired_races"] == 1
        assert sum(aggregate_profile[key] for key in (
            "more_points_races", "equal_points_races", "fewer_points_races",
        )) == 1
    assert manifest["selection_report_html"].endswith("_summary.html")
    report_path = output / manifest["selection_report_html"]
    assert report_path.is_file()
    report_html = report_path.read_text(encoding="utf-8")
    assert "Weighted rival strategy selection" in report_html
    assert "Track: Saved" in report_html
    assert "Race engine:" in report_html
    assert "Weighted mean target points" in report_html
    assert "within-seed cross-scenario covariance" in report_html
    assert "not zero uncertainty" in report_html
    assert "../conservative &lt;rival&gt;" in report_html
    assert manifest["target_plans"][selection["reference_label"]] is None
    assert selection["selected_label"] in manifest["target_plans"]
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
    assert str(report_path) in printed


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
