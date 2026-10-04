"""Scenario regret uses predeclared mean outcomes and a frozen validation choice."""

from copy import deepcopy
from fractions import Fraction

import pytest
from test_rival_strategy_selection import (
    _HARD_STOP,
    _classification_runner,
    _controlled_run,
    _is_rival_no_stop,
    _set_classification,
)

from f1sim.analysis.rival_strategy_selection import (
    evaluate_prepared_rival_pit_plan_selection,
    evaluate_saved_rival_pit_plan_selection,
    prepare_rival_pit_plan_selection,
)
from f1sim.output.comparison import _selection_regret_html


def _plans(constructor=False):
    return {
        "reference": None,
        "risky": {"A": [], "B": []} if constructor else [],
        "balanced": {key: deepcopy(_HARD_STOP) for key in ("A", "B")}
        if constructor else deepcopy(_HARD_STOP),
    }


def _cases(constructor=False, reverse=False):
    rival = "C" if constructor else "B"
    cases = {
        "favorable": {"weight": 9, "pit_plans": {rival: None}},
        "adverse": {"weight": 1, "pit_plans": {rival: []}},
    }
    return dict(reversed(list(cases.items()))) if reverse else cases


def _label(runner):
    plan = (runner.pit_plans or {}).get("A")
    return "reference" if plan is None else "risky" if plan == [] else "balanced"


@pytest.mark.parametrize("constructor", [False, True])
@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_lower_weighted_mean_can_limit_scenario_shortfall_and_stays_frozen(
    monkeypatch, constructor, engine,
):
    calls = []
    rival = "C" if constructor else "B"

    def award(runner, result):
        label = _label(runner)
        adverse = _is_rival_no_stop(runner, rival)
        values = {"reference": (15, 0), "risky": (25, 0), "balanced": (18, 15)}
        extra = {"reference": (12, 2), "risky": (18, 1), "balanced": (15, 12)}
        for race in result.race_results:
            # Validation deliberately reverses the training preference. It
            # cannot pick the unused risky candidate or replace the choice.
            awards = {"A": 25 if label == "reference" else 0} if runner.base_seed >= 75 else {
                "A": values[label][adverse],
            }
            if constructor:
                awards["B"] = 0 if runner.base_seed >= 75 else extra[label][adverse]
            _set_classification(race, awards)

    _controlled_run(monkeypatch, award, calls)
    runner = _classification_runner(constructor=constructor)
    runner.race_engine = engine
    plans, cases = _plans(constructor), _cases(constructor)
    before = deepcopy((plans, cases))
    target = {"constructor_id": "T"} if constructor else {"driver_id": "A"}
    prepared = prepare_rival_pit_plan_selection(
        runner, 1, plans, "reference", cases, **target,
        training_simulations=3, validation_simulations=2, selection_method="minimax_regret",
    )
    outcome = evaluate_prepared_rival_pit_plan_selection(prepared)
    selection = outcome["selection"]
    assert selection["selected_label"] == "balanced"
    assert selection["tiebreak_applied"] == "unique_lowest_training_maximum_regret"
    assert selection["schema_version"] == 2
    assert selection["validation_target_metrics"]["mean_points_difference"] < 0
    assert (plans, cases) == before
    assert len(calls) == 10
    assert [seed for seed, _ in calls] == [72] * 6 + [75] * 4
    assert all(list(results) == ["reference", "balanced"]
               for results in outcome["validation_results"].values())
    means = {row["label"]: row for row in selection["training_score_table"]}
    assert means["balanced"]["mean_points"] < means["risky"]["mean_points"]
    assert means["risky"]["mean_points_behind_selected"] < 0
    regrets = {row["label"]: row for row in selection["training_regret_table"]}
    assert regrets["balanced"]["maximum_regret"] == (10 if constructor else 7)
    assert regrets["balanced"]["worst_scenarios"] == ["favorable"]
    assert regrets["risky"]["maximum_regret"] == (26 if constructor else 15)
    assert regrets["balanced"]["scenarios"]["adverse"]["regret"] == 0

    # Changing only the declared criterion selects the high-average plan.
    weighted = deepcopy(prepared)
    weighted["selection_method"] = "weighted_mean"
    assert evaluate_prepared_rival_pit_plan_selection(weighted)["selection"]["selected_label"] == (
        "risky"
    )


@pytest.mark.parametrize("objective", ["win", "podium"])
@pytest.mark.parametrize("constructor", [False, True])
def test_regret_uses_classified_probability_objective_and_team_event_once(
    monkeypatch, objective, constructor,
):
    rival = "C" if constructor else "B"

    def award(runner, result):
        label = _label(runner)
        adverse = _is_rival_no_stop(runner, rival)
        for index, race in enumerate(result.race_results):
            points = 0
            if label == "risky" and not adverse:
                points = 25
            elif label == "balanced":
                points = 25 if adverse or index == 0 else 18 if objective == "win" else 6
            _set_classification(race, {"A": points, **({"B": 0} if constructor else {})})

    _controlled_run(monkeypatch, award)
    target = {"constructor_id": "T"} if constructor else {"driver_id": "A"}
    outcome = evaluate_prepared_rival_pit_plan_selection(prepare_rival_pit_plan_selection(
        _classification_runner(constructor=constructor), 1, _plans(constructor), "reference",
        _cases(constructor), **target, objective=objective, selection_method="minimax_regret",
        training_simulations=3, validation_simulations=1,
    ))
    selection = outcome["selection"]
    assert selection["selected_label"] == "balanced"
    assert selection["score_unit"] == "probability"
    regrets = {row["label"]: row for row in selection["training_regret_table"]}
    assert regrets["balanced"]["maximum_regret"] == pytest.approx(2 / 3)
    assert regrets["balanced"]["scenarios"]["favorable"]["mean_score"] == pytest.approx(1 / 3)
    assert regrets["risky"]["maximum_regret"] == 1
    assert selection["validation_target_metrics"]["score_difference_standard_error"] is None
    html = _selection_regret_html(selection)
    assert "percentage points" in html and "66.667" in html


@pytest.mark.parametrize("reference_tied", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_fractional_regret_ties_ignore_weights_and_prefer_reference_then_plan_order(
    monkeypatch, reference_tied, reverse,
):
    plans = {"risky": [], "balanced": deepcopy(_HARD_STOP), "reference": None}
    cases = _cases(reverse=reverse)
    cases["favorable"]["weight"] = 1e-20
    patterns = {
        "risky": ((10, 0, 0), (6, 0, 0)),
        "balanced": ((6, 0, 0), (10, 0, 0)),
        "reference": ((10, 0, 0), (6, 0, 0)) if reference_tied else ((0, 0, 0), (0, 0, 0)),
    }

    def award(runner, result):
        values = patterns[_label(runner)][_is_rival_no_stop(runner)]
        for race, points in zip(result.race_results, values):
            _set_classification(race, {"A": points})

    _controlled_run(monkeypatch, award)
    selection = evaluate_prepared_rival_pit_plan_selection(prepare_rival_pit_plan_selection(
        _classification_runner(), 1, plans, "reference", cases, driver_id="A",
        selection_method="minimax_regret", training_simulations=3, validation_simulations=3,
    ))["selection"]
    assert selection["selected_label"] == ("reference" if reference_tied else "risky")
    assert selection["tiebreak_applied"] == (
        "reference_preferred_on_exact_tie" if reference_tied else "first_plan_order_on_exact_tie"
    )
    regrets = {row["label"]: row for row in selection["training_regret_table"]}
    assert regrets["risky"]["maximum_regret"] == float(Fraction(4, 3))
    assert regrets["balanced"]["maximum_regret"] == float(Fraction(4, 3))
    assert regrets["risky"]["tied_for_best"] and regrets["balanced"]["tied_for_best"]
    # Means are computed before regret. Per-trial hindsight would produce a
    # different scenario benchmark than the reported 10/3 average.
    assert regrets["balanced"]["scenarios"]["favorable"]["best_candidate_mean_score"] == (
        float(Fraction(10, 3))
    )


@pytest.mark.parametrize("method", [None, True, 1, {}, "", "minimax", "MINIMAX_REGRET"])
def test_invalid_method_rejected_before_saved_loading_or_any_runner_access(method):
    with pytest.raises(ValueError, match="selection_method"):
        evaluate_saved_rival_pit_plan_selection(
            "missing-source.json", {}, "reference", {}, selection_method=method,
        )
    with pytest.raises(ValueError, match="selection_method"):
        prepare_rival_pit_plan_selection(object(), 1, {}, "reference", {}, selection_method=method)


def test_mutated_prepared_method_is_revalidated_before_trials(monkeypatch):
    prepared = prepare_rival_pit_plan_selection(
        _classification_runner(), 1, _plans(), "reference", _cases(), driver_id="A",
        selection_method="minimax_regret", training_simulations=1, validation_simulations=1,
    )
    prepared["selection_method"] = "unknown"
    runner_type = type(next(iter(prepared["training_runners"]["favorable"].values())))
    monkeypatch.setattr(runner_type, "run", lambda *args, **kwargs: pytest.fail("invalid run"))
    with pytest.raises(ValueError, match="selection_method"):
        evaluate_prepared_rival_pit_plan_selection(prepared)


def test_regret_report_escapes_labels_and_keeps_missing_evidence_unknown():
    html = _selection_regret_html({
        "selection_method": "minimax_regret", "training_regret_table": [{
            "label": "<script>candidate</script>", "maximum_regret": None,
            "worst_scenarios": ["<img src=x>"], "trials_per_scenario": 1,
            "scenarios": {"<svg>": {"mean_score": True, "regret": None}},
        }],
    })
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<img" not in html and "&lt;img src=x&gt;" in html
    assert "<svg>" not in html and "&lt;svg&gt;" in html
    assert html.count("Not recorded") == 4
    assert "Maximum training shortfall" in html
    assert _selection_regret_html({"selection_method": "weighted_mean"}) == ""
