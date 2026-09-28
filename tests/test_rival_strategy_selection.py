"""Weighted pit-plan selection across explicit rival-plan assumptions."""

import json
from copy import deepcopy
from fractions import Fraction

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.rival_strategy_selection import (
    _paired_summary,
    evaluate_saved_rival_pit_plan_selection,
)
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter

_HARD_STOP = [{"lap": 2, "compound": "hard"}]


def _saved(tmp_path, *, constructor=False):
    if constructor:
        drivers = [
            Driver(id="A", name="A", team_id="T"),
            Driver(id="B", name="B", team_id="T"),
            Driver(id="C", name="C", team_id="U"),
        ]
    else:
        drivers = [
            Driver(id="A", name="A", team_id="T"),
            Driver(id="B", name="B", team_id="U"),
        ]
    ids = [driver.id for driver in drivers]
    cars = {
        team_id: Car(team_id=team_id, team_name=team_id)
        for team_id in {driver.team_id for driver in drivers}
    }
    inventory = {
        driver_id: [
            {"id": f"{driver_id}-medium", "compound": "medium", "age": 0},
            {"id": f"{driver_id}-hard", "compound": "hard", "age": 0},
        ]
        for driver_id in ids
    }
    pit_plans = {driver_id: deepcopy(_HARD_STOP) for driver_id in ids}
    result = MonteCarloRunner(
        drivers, cars,
        Track(id="t", name="Saved", country="T", total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=71,
        starting_tires={driver_id: "medium" for driver_id in ids},
        tire_inventory=inventory,
        pit_plans=pit_plans,
    ).run(1, parallel=False)
    return Exporter(tmp_path).export_statistics_json(result)


def _driver_plans():
    return {"reference": None, "planned": deepcopy(_HARD_STOP)}


def _constructor_plans():
    return {
        "reference": None,
        "planned": {"A": deepcopy(_HARD_STOP), "B": deepcopy(_HARD_STOP)},
    }


def _scenarios(first_weight=1, second_weight=1, *, constructor=False):
    rival = "C" if constructor else "B"
    return {
        "rival_auto": {"weight": first_weight, "pit_plans": {rival: None}},
        "rival_no_stop": {"weight": second_weight, "pit_plans": {rival: []}},
    }


def _set_points(result, driver_id, values):
    for race, points in zip(result.race_results, values):
        rows = [row for row in race if row.driver_id == driver_id]
        assert len(rows) == 1
        rows[0].points_awarded = points


def _controlled_run(monkeypatch, award, calls=None, *, qualify_rival_no_stop=False):
    original = MonteCarloRunner.run

    def run(self, count, *, parallel=False, max_workers=None):
        result = original(self, count, parallel=False, max_workers=None)
        if calls is not None:
            calls.append((self.base_seed, deepcopy(self.pit_plans)))
        award(self, result)
        if qualify_rival_no_stop:
            plans = self.pit_plans or {}
            if plans.get("B") == []:
                for rows in result.qualifying_results:
                    rows[0].q1_time += 0.1
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", run)


def _is_planned(runner):
    return (runner.pit_plans or {}).get("A") == _HARD_STOP


def _is_rival_no_stop(runner, rival="B"):
    return (runner.pit_plans or {}).get(rival) == []


def test_fractional_paired_summary_preserves_near_equal_seed_differences():
    summary = _paired_summary(
        [0, 0], [1, 1],
        differences=[Fraction(1), Fraction(1) + Fraction(1, 10**20)],
    )

    assert summary["mean_points_difference"] == 1.0
    assert summary["points_difference_standard_error"] == pytest.approx(
        5e-21, rel=1e-12, abs=0,
    )


def test_scenario_weights_change_training_winner_and_freeze_validation_set(
    tmp_path, monkeypatch,
):
    path = _saved(tmp_path)
    calls = []

    def favor_weighted_scenario(runner, result):
        planned = _is_planned(runner)
        no_stop = _is_rival_no_stop(runner)
        if runner.base_seed >= 73:
            planned_points = 0 if no_stop else 1
            automatic_points = 10 if no_stop else 9
        else:
            planned_points = 0 if no_stop else 10
            automatic_points = 10 if no_stop else 0
        _set_points(result, "A", [planned_points if planned else automatic_points])

    _controlled_run(monkeypatch, favor_weighted_scenario, calls)
    first = evaluate_saved_rival_pit_plan_selection(
        path, _driver_plans(), "reference", _scenarios(3, 1), driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    assert first["selection"]["selected_label"] == "planned"
    assert [row["normalized_weight"] for row in first["selection"]["rival_scenarios"]] == [
        0.75, 0.25,
    ]
    assert list(first["validation_results"]) == ["rival_auto", "rival_no_stop"]
    assert list(first["validation_results"]["rival_auto"]) == ["reference", "planned"]
    assert all(seed == 73 for seed, _ in calls[4:])
    assert first["selection"]["validation_target_metrics"]["mean_points_difference"] < 0

    calls.clear()
    second = evaluate_saved_rival_pit_plan_selection(
        path, _driver_plans(), "reference", _scenarios(1, 3), driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    assert second["selection"]["selected_label"] == "reference"
    assert all(list(results) == ["reference"] for results in second["validation_results"].values())
    assert all(seed == 73 for seed, _ in calls[4:])
    assert second["selection"]["validation_target_metrics"][
        "points_outcome_profile"
    ] is None
    assert all(
        metrics["points_outcome_profile"] is None
        for metrics in second["selection"]["validation_scenario_metrics"].values()
    )


def test_validation_standard_error_uses_weighted_per_seed_differences(
    tmp_path, monkeypatch,
):
    path = _saved(tmp_path)

    def controlled_points(runner, result):
        planned = _is_planned(runner)
        values = []
        for index in range(len(result.race_results)):
            seed = result.seed + index
            if seed < 74:
                difference = 10 if planned else 0
            else:
                difference = (0 if seed == 74 else 10) if planned else 0
            values.append(difference)
        _set_points(result, "A", values)

    calls = []
    _controlled_run(monkeypatch, controlled_points, calls)
    outcome = evaluate_saved_rival_pit_plan_selection(
        path, _driver_plans(), "reference", _scenarios(), driver_id="A",
        training_simulations=1, validation_simulations=2,
    )
    assert outcome["selection"]["selected_label"] == "planned"
    metrics = outcome["selection"]["validation_target_metrics"]
    assert metrics["mean_points_difference"] == 5
    assert metrics["points_difference_standard_error"] == 5
    for scenario_metrics in outcome["selection"]["validation_scenario_metrics"].values():
        assert scenario_metrics["mean_points_difference"] == 5
        assert scenario_metrics["points_difference_standard_error"] == 5


def test_opposing_rival_scenarios_cancel_within_each_validation_seed(
    tmp_path, monkeypatch,
):
    path = _saved(tmp_path)

    def controlled_points(runner, result):
        planned = _is_planned(runner)
        no_stop = _is_rival_no_stop(runner)
        values = []
        for index in range(len(result.race_results)):
            seed = result.seed + index
            if runner.base_seed == 72:
                points = 20 if planned else 10
            else:
                magnitude = 10 if seed == 73 else 2
                difference = -magnitude if no_stop else magnitude
                points = 10 + difference if planned else 10
            values.append(points)
        _set_points(result, "A", values)

    _controlled_run(monkeypatch, controlled_points)
    outcome = evaluate_saved_rival_pit_plan_selection(
        path, _driver_plans(), "reference", _scenarios(), driver_id="A",
        training_simulations=1, validation_simulations=2,
    )
    selection = outcome["selection"]
    profile = selection["validation_target_metrics"]["points_outcome_profile"]
    assert selection["selected_label"] == "planned"
    assert selection["seed_ranges"]["validation"] == {
        "first_seed": 73, "last_seed": 74, "trials": 2,
    }
    assert selection["validation_target_metrics"]["mean_points_difference"] == 0
    assert profile == {
        "paired_races": 2,
        "more_points_races": 0,
        "equal_points_races": 2,
        "fewer_points_races": 0,
        "mean_points_gain_when_ahead": None,
        "mean_points_loss_when_behind": None,
    }
    scenario_profiles = selection["validation_scenario_metrics"]
    assert scenario_profiles["rival_auto"]["points_outcome_profile"] == {
        "paired_races": 2,
        "more_points_races": 2,
        "equal_points_races": 0,
        "fewer_points_races": 0,
        "mean_points_gain_when_ahead": 6,
        "mean_points_loss_when_behind": None,
    }
    assert scenario_profiles["rival_no_stop"]["points_outcome_profile"] == {
        "paired_races": 2,
        "more_points_races": 0,
        "equal_points_races": 0,
        "fewer_points_races": 2,
        "mean_points_gain_when_ahead": None,
        "mean_points_loss_when_behind": 6,
    }


@pytest.mark.parametrize(
    "reference_points, planned_points",
    [
        ((0, 0, 9), (1, 1, 7)),  # weighted scores 3 and 2.9999999999999996 in float math
        ((0, 1, 2), (3, 0, 0)),  # delta terms 3, -1, -2 also cancel exactly
    ],
)
def test_equal_thirds_use_exact_weighted_differences_for_ties(
    tmp_path, monkeypatch, reference_points, planned_points,
):
    path = _saved(tmp_path)
    scenarios = {
        "first": {"weight": 1, "pit_plans": {"B": None}},
        "second": {"weight": 1, "pit_plans": {"B": []}},
        "third": {"weight": 1, "pit_plans": {"B": deepcopy(_HARD_STOP)}},
    }

    def controlled_points(runner, result):
        planned = _is_planned(runner)
        rival_plan = (runner.pit_plans or {}).get("B")
        scenario_index = 0 if rival_plan is None else 1 if rival_plan == [] else 2
        if runner.base_seed == 72:
            points = 10 if planned else 0
        else:
            points = (
                planned_points[scenario_index]
                if planned else reference_points[scenario_index]
            )
        _set_points(result, "A", [points] * len(result.race_results))

    calls = []
    _controlled_run(monkeypatch, controlled_points, calls)
    outcome = evaluate_saved_rival_pit_plan_selection(
        path, _driver_plans(), "reference", scenarios, driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    selection = outcome["selection"]
    metrics = selection["validation_target_metrics"]
    assert selection["selected_label"] == "planned"
    assert [seed for seed, _ in calls] == [72] * 6 + [73] * 6
    assert [seed for seed, _ in calls] == [72] * 6 + [73] * 6
    assert metrics["mean_points_difference"] == 0
    assert metrics["points_difference_standard_error"] is None
    assert metrics["points_outcome_profile"] == {
        "paired_races": 1,
        "more_points_races": 0,
        "equal_points_races": 1,
        "fewer_points_races": 0,
        "mean_points_gain_when_ahead": None,
        "mean_points_loss_when_behind": None,
    }


def test_tiny_nonzero_weighted_difference_is_not_rounded_to_a_tie(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    scenarios = {
        "first": {"weight": 1, "pit_plans": {"B": None}},
        "second": {"weight": 1, "pit_plans": {"B": []}},
        "third": {
            "weight": 1 + 1e-15,
            "pit_plans": {"B": deepcopy(_HARD_STOP)},
        },
    }

    def controlled_points(runner, result):
        planned = _is_planned(runner)
        rival_plan = (runner.pit_plans or {}).get("B")
        scenario_index = 0 if rival_plan is None else 1 if rival_plan == [] else 2
        if runner.base_seed == 72:
            points = 10 if planned else 0
        else:
            reference_points = (0, 0, 1)
            planned_points = (1, 0, 0)
            points = planned_points[scenario_index] if planned else reference_points[
                scenario_index
            ]
        _set_points(result, "A", [points] * len(result.race_results))

    calls = []
    _controlled_run(monkeypatch, controlled_points, calls)
    outcome = evaluate_saved_rival_pit_plan_selection(
        path, _driver_plans(), "reference", scenarios, driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    selection = outcome["selection"]
    metrics = selection["validation_target_metrics"]
    assert [seed for seed, _ in calls] == [72] * 6 + [73] * 6
    weights = {
        row["name"]: Fraction(row["normalized_weight"])
        for row in selection["rival_scenarios"]
    }
    expected_difference = weights["first"] - weights["third"]
    assert expected_difference != 0
    assert metrics["mean_points_difference"] == float(expected_difference)
    assert metrics["points_outcome_profile"]["fewer_points_races"] == 1
    assert metrics["points_outcome_profile"]["mean_points_loss_when_behind"] == float(
        -expected_difference,
    )


def test_constructor_target_sums_members_inside_each_seed_and_returns_scenario_results(
    tmp_path, monkeypatch,
):
    path = _saved(tmp_path, constructor=True)

    def controlled_points(runner, result):
        planned = _is_planned(runner)
        for driver_id, planned_points, reference_points in (
            ("A", 8, 1), ("B", 2, 1), ("C", 0, 0),
        ):
            _set_points(
                result, driver_id,
                [planned_points if planned else reference_points] * len(result.race_results),
            )

    _controlled_run(monkeypatch, controlled_points)
    outcome = evaluate_saved_rival_pit_plan_selection(
        path, _constructor_plans(), "reference", _scenarios(constructor=True),
        constructor_id="T", training_simulations=1, validation_simulations=1,
    )
    selection = outcome["selection"]
    assert selection["target_mode"] == "constructor"
    assert selection["target_member_ids"] == ["A", "B"]
    assert selection["selected_label"] == "planned"
    assert selection["training_score_table"][1]["mean_points"] == 10
    assert set(outcome["training_results"]) == {"rival_auto", "rival_no_stop"}
    assert all(set(results) == {"reference", "planned"}
               for results in outcome["training_results"].values())


@pytest.mark.parametrize(
    "scenarios, message",
    [
        (_scenarios(0, 1), "positive finite"),
        (_scenarios(float("nan"), 1), "positive finite"),
        (_scenarios(1e-300, 1e300), "normalize safely"),
        ({" ": {"weight": 1, "pit_plans": {}}}, "at most 80"),
        ({"extra": {"weight": 1, "pit_plans": {}, "surprise": True}}, "exactly weight"),
    ],
)
def test_invalid_weights_and_scenario_definitions_fail_before_running(
    tmp_path, monkeypatch, scenarios, message,
):
    path = _saved(tmp_path)
    calls = []
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match=message):
        evaluate_saved_rival_pit_plan_selection(
            path, _driver_plans(), "reference", scenarios, driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    assert calls == []


@pytest.mark.parametrize(
    "override, message",
    [
        ({"A": None}, "cannot override target"),
        ({"UNKNOWN": None}, "Unknown or nonrunnable"),
        ({"B": [{"lap": 6, "compound": "hard"}]}, "total_laps"),
    ],
)
def test_target_unknown_and_malformed_rival_plans_fail_before_any_trial(
    tmp_path, monkeypatch, override, message,
):
    path = _saved(tmp_path)
    calls = []
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *args, **kwargs: calls.append(args))
    scenarios = {
        "valid": {"weight": 1, "pit_plans": {"B": None}},
        "invalid": {"weight": 1, "pit_plans": override},
    }
    with pytest.raises(ValueError, match=message):
        evaluate_saved_rival_pit_plan_selection(
            path, _driver_plans(), "reference", scenarios, driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    assert calls == []


def test_cross_scenario_qualifying_mismatch_fails_before_validation(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    calls = []

    def simple_points(runner, result):
        _set_points(result, "A", [10 if _is_planned(runner) else 0])

    _controlled_run(monkeypatch, simple_points, calls, qualify_rival_no_stop=True)
    with pytest.raises(ValueError, match="qualifying results do not match across training"):
        evaluate_saved_rival_pit_plan_selection(
            path, _driver_plans(), "reference", _scenarios(), driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    assert len(calls) == 4
    assert all(seed == 72 for seed, _ in calls)


def test_seed_overflow_and_plan_preflight_happen_before_runs(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["seed"] = 2**32 - 1
    overflowing = tmp_path / "overflow.json"
    overflowing.write_text(json.dumps(payload), encoding="utf-8")
    calls = []
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="must not exceed"):
        evaluate_saved_rival_pit_plan_selection(
            overflowing, _driver_plans(), "reference", _scenarios(), driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    with pytest.raises(ValueError, match="total_laps"):
        evaluate_saved_rival_pit_plan_selection(
            path,
            {"reference": None, "late-invalid": [{"lap": 6, "compound": "hard"}]},
            "reference", _scenarios(), driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    assert calls == []
