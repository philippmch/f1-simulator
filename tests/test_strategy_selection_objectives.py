"""Frozen win/podium objectives select and validate complete, paired cohorts."""

from fractions import Fraction
from math import sqrt
from statistics import mean, stdev

import pytest
from test_rival_strategy_selection import (
    _classification_runner,
    _controlled_run,
    _set_classification,
)

from f1sim.analysis import rival_strategy_selection as rival_selector
from f1sim.analysis import strategy_selection as selector
from f1sim.output import Exporter
from f1sim.output.comparison import (
    _selection_objective_html,
    render_rival_strategy_selection_report,
)
from f1sim.simulation.race import DriverStatus


def _plans():
    return {"reference": None,
            "win": [{"lap": 2, "compound": "hard"}],
            "podium": [{"lap": 3, "compound": "hard"}]}


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("objective,winner", [
    ("points", "reference"), ("win", "win"), ("podium", "podium"),
])
def test_objective_changes_training_winner_and_heldout_cannot_reselect(
    monkeypatch, weighted, objective, winner,
):
    calls = []

    def outcomes(runner, result):
        plan = (runner.pit_plans or {}).get("A")
        if runner.base_seed == 72:
            awards = [12] * 4 if not plan else (
                [25, 0, 0, 0] if plan[0]["lap"] == 2 else [15, 15, 15, 0]
            )
        else:
            awards = [25] * 4 if not plan else [0] * 4
        for race, points in zip(result.race_results, awards):
            _set_classification(race, {"A": points})

    _controlled_run(monkeypatch, outcomes, calls)
    runner = _classification_runner()
    kwargs = dict(driver_id="A", objective=objective,
                  training_simulations=4, validation_simulations=4)
    if weighted:
        prepared = rival_selector.prepare_rival_pit_plan_selection(
            runner, 1, _plans(), "reference",
            {"one": {"weight": 1, "pit_plans": {}},
             "two": {"weight": 3, "pit_plans": {"B": []}}}, **kwargs,
        )
        outcome = rival_selector.evaluate_prepared_rival_pit_plan_selection(prepared)
        validation = outcome["validation_results"].values()
    else:
        outcome = selector.evaluate_pit_plan_selection(runner, 1, _plans(), "reference", **kwargs)
        validation = [outcome["validation_results"]]
    selection = outcome["selection"]
    assert selection["objective"] == objective and selection["selected_label"] == winner
    assert selection["score_unit"] == ("points" if objective == "points" else "probability")
    rows = selection["training_score_table"]
    assert [row["mean_points"] for row in rows] == [12, 6.25, 11.25]
    assert [row["mean_score"] for row in rows] == {
        "points": [12, 6.25, 11.25], "win": [0, .25, 0], "podium": [0, .25, .75],
    }[objective]
    labels = ["reference"] if winner == "reference" else ["reference", winner]
    assert all(list(variants) == labels for variants in validation)
    assert all(seed in (72, 76) for seed, _ in calls)
    metrics = selection["validation_target_metrics"]
    assert metrics["mean_score_difference"] == (0 if objective == "points" else -1)
    assert metrics["score_difference_standard_error"] == (None if winner == "reference" else 0)
    assert metrics["mean_points_difference"] == (0 if winner == "reference" else -25)
    assert selection["seed_ranges"]["validation"]["first_seed"] == 76


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("objective", ["win", "podium"])
def test_constructor_event_counts_any_member_once_per_race(monkeypatch, weighted, objective):
    def outcomes(runner, result):
        planned = bool((runner.pit_plans or {}).get("A"))
        for index, race in enumerate(result.race_results):
            if objective == "podium":
                awards = {"A": 18, "B": 15} if not planned and index == 0 else (
                    {"A": 0, "B": 15} if planned else {"A": 0, "B": 0}
                )
            else:
                awards = {"A": 25, "B": 18} if not planned and index == 0 else (
                    {"A": 0, "B": 25} if planned else {"A": 0, "B": 0}
                )
            _set_classification(race, awards)

    _controlled_run(monkeypatch, outcomes)
    runner = _classification_runner(constructor=True)
    plans = {"reference": None, "planned": {"A": _plans()["win"], "B": _plans()["podium"]}}
    kwargs = dict(constructor_id="T", objective=objective,
                  training_simulations=4, validation_simulations=4)
    if weighted:
        outcome = rival_selector.evaluate_prepared_rival_pit_plan_selection(
            rival_selector.prepare_rival_pit_plan_selection(
                runner, 1, plans, "reference", {"rivals": {"weight": 1, "pit_plans": {}}},
                **kwargs,
            ),
        )
    else:
        outcome = selector.evaluate_pit_plan_selection(runner, 1, plans, "reference", **kwargs)
    selection = outcome["selection"]
    assert selection["selected_label"] == "planned"
    assert [row["mean_score"] for row in selection["training_score_table"]] == [.25, 1]
    assert "at least one constructor driver" in selection["objective_description"]
    assert selection["validation_target_metrics"]["mean_score_difference"] == .75
    assert selection["validation_target_metrics"]["score_difference_standard_error"] == .25


@pytest.mark.parametrize("objective", ["win", "podium"])
def test_probability_eligibility_matches_published_classification_statistics(objective):
    runner = _classification_runner()
    result = runner.run(5, parallel=False)
    eligibility = [(DriverStatus.DNF, True), (DriverStatus.DNF, False),
                   (DriverStatus.FINISHED, None), (DriverStatus.DNF, None),
                   (DriverStatus.FINISHED, False)]
    for race, (status, classified) in zip(result.race_results, eligibility):
        _set_classification(race, {"A": 25})
        row = next(row for row in race if row.driver_id == "A")
        row.status, row.classified = status, classified
    scores = selector._phase_scores({"plan": result}, {}, ["A"], objective)
    assert scores == {"plan": [1, 0, 1, 0, 0]}
    stats = runner._aggregate_statistics(result.race_results, result.qualifying_results)["A"]
    assert stats.wins == stats.podiums == sum(scores["plan"]) == 2


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("objective", ["win", "podium"])
def test_probability_success_cannot_hide_missing_teammate_evidence(
    monkeypatch, weighted, objective,
):
    def outcomes(runner, result):
        for race in result.race_results:
            _set_classification(race, {"A": 25, "B": 18})
            race[:] = [row for row in race if row.driver_id != "B"]

    _controlled_run(monkeypatch, outcomes)
    runner = _classification_runner(constructor=True)
    plans = {"reference": None, "same": None}
    kwargs = dict(constructor_id="T", objective=objective,
                  training_simulations=1, validation_simulations=1)
    with pytest.raises(ValueError, match="missing, duplicate, or malformed points outcome.*B"):
        if weighted:
            rival_selector.evaluate_prepared_rival_pit_plan_selection(
                rival_selector.prepare_rival_pit_plan_selection(
                    runner, 1, plans, "reference", {"rival": {"weight": 1, "pit_plans": {}}},
                    **kwargs,
                ),
            )
        else:
            selector.evaluate_pit_plan_selection(runner, 1, plans, "reference", **kwargs)


def test_weighted_probability_uncertainty_preserves_cross_scenario_covariance(monkeypatch):
    def outcomes(runner, result):
        planned = bool((runner.pit_plans or {}).get("A"))
        second = (runner.pit_plans or {}).get("B") == []
        if runner.base_seed == 72:
            values = [25, 25] if planned else [0, 0]
        elif second:
            values = [25, 0] if planned else [0, 25]
        else:
            values = [0, 25] if planned else [25, 0]
        for race, points in zip(result.race_results, values):
            _set_classification(race, {"A": points})

    _controlled_run(monkeypatch, outcomes)
    prepared = rival_selector.prepare_rival_pit_plan_selection(
        _classification_runner(), 1, {"reference": None, "planned": _plans()["win"]},
        "reference", {"first": {"weight": 3, "pit_plans": {}},
                      "second": {"weight": 1, "pit_plans": {"B": []}}},
        driver_id="A", objective="win", training_simulations=2, validation_simulations=2,
    )
    outcome = rival_selector.evaluate_prepared_rival_pit_plan_selection(prepared)
    metrics = outcome["selection"]["validation_target_metrics"]
    assert metrics["reference_mean_score"] == metrics["selected_mean_score"] == .5
    assert metrics["mean_score_difference"] == 0
    assert metrics["score_difference_standard_error"] == .5
    for row in outcome["selection"]["validation_scenario_metrics"].values():
        assert row["score_difference_standard_error"] == 1


def test_binary_scores_keep_exact_tie_evidence_with_tiny_rival_weight(monkeypatch):
    def outcomes(runner, result):
        planned = bool((runner.pit_plans or {}).get("A"))
        tiny_scenario = (runner.pit_plans or {}).get("B") == []
        for race in result.race_results:
            _set_classification(race, {"A": 25 if not tiny_scenario or planned else 0})

    _controlled_run(monkeypatch, outcomes)
    prepared = rival_selector.prepare_rival_pit_plan_selection(
        _classification_runner(), 1, {"reference": None, "planned": _plans()["win"]},
        "reference", {"large": {"weight": 1, "pit_plans": {}},
                      "tiny": {"weight": 1e-20, "pit_plans": {"B": []}}},
        driver_id="A", objective="win", training_simulations=1, validation_simulations=1,
    )
    selection = rival_selector.evaluate_prepared_rival_pit_plan_selection(prepared)["selection"]
    assert selection["selected_label"] == "planned"
    rows = selection["training_score_table"]
    assert [row["mean_score"] for row in rows] == [1, 1]
    assert rows[0]["mean_score_behind_selected"] == 1e-20
    assert [row["tied_for_best"] for row in rows] == [False, True]
    assert selection["validation_target_metrics"]["score_difference_standard_error"] is None


@pytest.mark.parametrize("objective", [None, True, {}, ["win"], "WIN", " win", "", "finish"])
def test_invalid_objectives_fail_before_inputs_or_trial_work(objective):
    kwargs = dict(driver_id="A", objective=objective,
                  training_simulations=1, validation_simulations=1)
    rivals = {"rival": {"weight": 1, "pit_plans": {}}}
    for call in (
        lambda: selector.evaluate_saved_pit_plan_selection("missing.json", _plans(), "reference",
                                                         **kwargs),
        lambda: selector.evaluate_pit_plan_selection(None, 1, _plans(), "reference", **kwargs),
        lambda: rival_selector.evaluate_saved_rival_pit_plan_selection(
            "missing.json", _plans(), "reference", rivals, **kwargs,
        ),
        lambda: rival_selector.prepare_rival_pit_plan_selection(
            None, 1, _plans(), "reference", rivals, **kwargs,
        ),
    ):
        with pytest.raises(ValueError, match="objective must be"):
            call()


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("objective", ["win", "podium"])
def test_saved_selection_preserves_objective_and_source_bytes(tmp_path, weighted, objective):
    runner = _classification_runner(constructor=True)
    source = Exporter(tmp_path).export_statistics_json(runner.run(1, parallel=False))
    before = source.read_bytes()
    plans = {"reference": None, "same": None}
    kwargs = dict(constructor_id="T", objective=objective,
                  training_simulations=1, validation_simulations=1)
    if weighted:
        outcome = rival_selector.evaluate_saved_rival_pit_plan_selection(
            source, plans, "reference", {"rival": {"weight": 1, "pit_plans": {}}}, **kwargs,
        )
    else:
        outcome = selector.evaluate_saved_pit_plan_selection(source, plans, "reference", **kwargs)
    selection = outcome["selection"]
    assert source.read_bytes() == before and selection["objective"] == objective
    assert selection["selected_label"] == "reference"
    assert selection["validation_target_metrics"]["score_difference_standard_error"] is None


def test_objective_report_keeps_aggregate_with_hostile_and_reserved_scenario_names():
    metrics = dict(reference_mean_score=.2, selected_mean_score=.1, mean_score_difference=-.1,
                   score_difference_standard_error=None, paired_races=1)
    selection = dict(objective="podium", objective_description="Team <T> podium",
                     selected_label="<script>candidate</script>", validation_status="evaluated",
                     training_score_table=[dict(label="plan", mean_score=.5, trials=2)],
                     training_scenario_score_tables={"Aggregate": {
                         "scores": [dict(label="<img>", mean_score=.75, trials=2)]}},
                     validation_target_metrics=metrics, validation_scenario_metrics={"Aggregate": {
                         **metrics, "selected_mean_score": .75}})
    html = _selection_objective_html(selection)
    assert "50.000%" in html and "75.000%" in html
    assert "-10.000 percentage points" in html
    assert "Not estimated (1 paired race)" in html
    assert "<script>candidate</script>" not in html and "&lt;script&gt;" in html
    assert "&lt;img&gt;" in html and "Team &lt;T&gt;" in html
    assert "training winner" in render_rival_strategy_selection_report({"selection": selection})


def test_exact_weighted_paired_score_standard_error_retains_small_differences():
    tiny = Fraction(1, 10**20)
    differences = [Fraction(1), Fraction(1) + tiny]
    metrics = selector._score_metrics([Fraction(0)] * 2, differences)
    assert metrics["mean_score_difference"] == float(mean(differences))
    assert metrics["score_difference_standard_error"] == pytest.approx(
        stdev(differences) / sqrt(2), abs=0, rel=1e-12,
    )
