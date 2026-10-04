"""Weighted pit-plan selection across explicit rival-plan assumptions."""

import json
from copy import deepcopy
from fractions import Fraction
from itertools import permutations
from threading import Event

import pytest

from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import _load_saved_runner
from f1sim.analysis.rival_strategy_selection import (
    _paired_summary,
    evaluate_prepared_rival_pit_plan_selection,
    evaluate_saved_rival_pit_plan_selection,
    prepare_rival_pit_plan_selection,
)
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter
from f1sim.simulation.race import DriverStatus
from f1sim.simulation.race_points import POINTS_SYSTEM, points_for_classification

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


def _classification_runner(*, constructor=False):
    """A full field lets controlled awards agree with finishing classifications."""
    drivers = [
        Driver(id=driver_id, name=driver_id,
               team_id="T" if driver_id == "A" or constructor and driver_id == "B"
               else f"team-{driver_id}")
        for driver_id in "ABCDEFGHIJKL"
    ]
    return MonteCarloRunner(
        drivers,
        {driver.team_id: Car(team_id=driver.team_id, team_name=driver.team_id)
         for driver in drivers},
        Track(id="t", name="Controlled", country="T", total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=71,
        starting_tires={driver.id: "medium" for driver in drivers},
    )


def _set_classification(race, target_awards):
    positions = {points: position for position, points in POINTS_SYSTEM.items()}
    available = set(range(1, len(race) + 1))
    assigned = {}
    for driver_id, points in target_awards.items():
        position = positions[points] if points else max(available)
        assert position in available
        assigned[driver_id] = position
        available.remove(position)
    for row in race:
        position = assigned.get(row.driver_id)
        if position is None:
            position = min(available)
            available.remove(position)
        row.position = position
        row.status = DriverStatus.FINISHED
        row.classified = True
        row.laps_completed = 5
        row.dnf_reason = None
        row.total_time = 450 + position
        row.gap_to_leader = position - 1
        row.points_awarded = points_for_classification(position, True, 5, 5, True)
    race.sort(key=lambda row: row.position)


def _assert_validation_cohort(outcome, selected):
    expected = ["reference"] if selected == "reference" else ["reference", selected]
    assert all(list(rows) == expected for rows in outcome["validation_results"].values())
    json.dumps(outcome["selection"], allow_nan=False)


@pytest.mark.parametrize("reverse_scenarios", [False, True])
@pytest.mark.parametrize("reference_tied", [False, True])
def test_prepared_weighted_exact_ties_follow_candidate_policy(
    monkeypatch, reverse_scenarios, reference_tied,
):
    plans = {"reference": None, "first_tied": [], "second_tied": deepcopy(_HARD_STOP)}
    scenarios = {
        "automatic": {"weight": 1, "pit_plans": {"B": None}},
        "no_stop": {"weight": 1, "pit_plans": {"B": []}},
        "planned": {"weight": 1, "pit_plans": {"B": deepcopy(_HARD_STOP)}},
    }
    if reverse_scenarios:
        scenarios = dict(reversed(list(scenarios.items())))

    def award(runner, result):
        target_plan = (runner.pit_plans or {}).get("A")
        label = "reference" if target_plan is None else (
            "first_tied" if target_plan == [] else "second_tied"
        )
        rival_plan = (runner.pit_plans or {}).get("B")
        index = 0 if rival_plan is None else 1 if rival_plan == [] else 2
        awards = {
            "reference": (0, 2, 10) if reference_tied else (0, 0, 0),
            "first_tied": (0, 2, 10),
            "second_tied": (0, 4, 8),
        }
        for race in result.race_results:
            _set_classification(race, {"A": awards[label][index]})

    _controlled_run(monkeypatch, award)
    prepared = prepare_rival_pit_plan_selection(
        _classification_runner(), 1, plans, "reference", scenarios, driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    outcome = evaluate_prepared_rival_pit_plan_selection(prepared)
    selection = outcome["selection"]
    selected = "reference" if reference_tied else "first_tied"
    assert selection["selected_label"] == selected
    assert selection["tiebreak_applied"] == (
        "reference_preferred_on_exact_tie" if reference_tied
        else "first_plan_order_on_exact_tie"
    )
    assert [row["mean_points"] for row in selection["training_score_table"]][1:] == [4, 4]
    assert [row["tied_for_best"] for row in selection["training_score_table"]] == [
        reference_tied, True, True,
    ]
    assert all(row["mean_points_behind_selected"] == 0
               for row in selection["training_score_table"][1:])
    _assert_validation_cohort(outcome, selected)


@pytest.mark.parametrize("reverse_scenarios", [False, True])
@pytest.mark.parametrize("tiny_direction, validation_direction", [(-1, -1), (1, 1), (1, -1)])
def test_prepared_selection_preserves_tiny_real_weighted_advantage(
    monkeypatch, reverse_scenarios, tiny_direction, validation_direction,
):
    scenarios = _scenarios(1, 1e-20)
    if reverse_scenarios:
        scenarios = dict(reversed(list(scenarios.items())))

    def award(runner, result):
        points = 25
        direction = tiny_direction if runner.base_seed == 72 else validation_direction
        if _is_rival_no_stop(runner):
            points = (25 if _is_planned(runner) else 18) if direction > 0 else (
                18 if _is_planned(runner) else 25
            )
        for race in result.race_results:
            _set_classification(race, {"A": points})

    _controlled_run(monkeypatch, award)
    prepared = prepare_rival_pit_plan_selection(
        _classification_runner(), 1, _driver_plans(), "reference", scenarios,
        driver_id="A", training_simulations=1, validation_simulations=1,
    )
    outcome = evaluate_prepared_rival_pit_plan_selection(prepared)
    selection = outcome["selection"]
    selected = "planned" if tiny_direction > 0 else "reference"
    assert selection["selected_label"] == selected
    assert selection["tiebreak_applied"] == "unique_highest_weighted_training_mean"
    assert [row["mean_points"] for row in selection["training_score_table"]] == [25, 25]
    expected_gap = float(7 * Fraction(prepared["normalized_weights"]["rival_no_stop"]))
    for row in selection["training_score_table"]:
        assert row["tied_for_best"] is (row["label"] == selected)
        assert row["mean_points_behind_selected"] == (
            0 if row["label"] == selected else expected_gap
        )
    _assert_validation_cohort(outcome, selected)
    metrics = selection["validation_target_metrics"]
    if tiny_direction > 0:
        expected_gain = float(7 * Fraction(prepared["normalized_weights"]["rival_no_stop"]))
        assert metrics["reference_mean_points"] == metrics["selected_mean_points"] == 25
        assert metrics["mean_points_difference"] == validation_direction * expected_gain
        profile = metrics["points_outcome_profile"]
        if validation_direction > 0:
            assert profile["more_points_races"] == 1
            assert profile["mean_points_gain_when_ahead"] == expected_gain
        else:
            assert profile["fewer_points_races"] == 1
            assert profile["mean_points_loss_when_behind"] == expected_gain
    else:
        assert metrics["mean_points_difference"] == 0
        assert metrics["points_outcome_profile"] is None


@pytest.mark.parametrize("scenario_order", list(permutations(("a", "b", "c"))))
def test_normalization_order_preserves_exact_tie_and_validation_cohort(monkeypatch, scenario_order):
    definitions = {
        "a": {"weight": 1, "pit_plans": {"B": None}},
        "b": {"weight": 2, "pit_plans": {"B": []}},
        "c": {"weight": 5, "pit_plans": {"B": deepcopy(_HARD_STOP)}},
    }

    def award(runner, result):
        rival = (runner.pit_plans or {}).get("B")
        index = 0 if rival is None else 1 if rival == [] else 2
        points = ((1, 2, 0) if _is_planned(runner) else (0, 0, 1))[index]
        for race in result.race_results:
            _set_classification(race, {"A": points})

    _controlled_run(monkeypatch, award)
    prepared = prepare_rival_pit_plan_selection(
        _classification_runner(), 1, _driver_plans(), "reference",
        {name: definitions[name] for name in scenario_order}, driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    assert prepared["normalized_weights"] == {"a": 1 / 8, "b": 2 / 8, "c": 5 / 8}
    outcome = evaluate_prepared_rival_pit_plan_selection(prepared)
    selection = outcome["selection"]
    assert selection["selected_label"] == "reference"
    assert selection["tiebreak_applied"] == "reference_preferred_on_exact_tie"
    assert [row["name"] for row in selection["rival_scenarios"]] == list(scenario_order)
    assert all(row["mean_points"] == 5 / 8 and row["tied_for_best"]
               and row["mean_points_behind_selected"] == 0
               for row in selection["training_score_table"])
    assert all("tied_for_best" not in row
               for table in selection["training_scenario_score_tables"].values()
               for row in table["scores"])
    _assert_validation_cohort(outcome, "reference")


def test_positive_exact_shortfall_below_float_precision_is_not_a_tie(monkeypatch):
    def award(runner, result):
        for index, race in enumerate(result.race_results):
            points = int(_is_planned(runner) and _is_rival_no_stop(runner) and index == 0)
            _set_classification(race, {"A": points})

    _controlled_run(monkeypatch, award)
    prepared = prepare_rival_pit_plan_selection(
        _classification_runner(), 1, _driver_plans(), "reference", _scenarios(1, 5e-324),
        driver_id="A", training_simulations=2, validation_simulations=1,
    )
    outcome = evaluate_prepared_rival_pit_plan_selection(prepared)
    selection = outcome["selection"]
    assert selection["selected_label"] == "planned"
    assert selection["tiebreak_applied"] == "unique_highest_weighted_training_mean"
    assert [row["mean_points"] for row in selection["training_score_table"]] == [0, 0]
    assert [row["mean_points_behind_selected"]
            for row in selection["training_score_table"]] == [0, 0]
    assert [row["tied_for_best"] for row in selection["training_score_table"]] == [False, True]
    _assert_validation_cohort(outcome, "planned")


def test_prepared_constructor_weights_sum_members_and_multiple_seeds(monkeypatch):
    def award(runner, result):
        for trial, race in enumerate(result.race_results):
            if _is_planned(runner):
                awards = (25, 18) if trial == 0 else (10, 8)
                if _is_rival_no_stop(runner, "C"):
                    awards = (15, 12) if trial == 0 else (6, 4)
            else:
                awards = (2, 1)
            _set_classification(race, dict(zip(("A", "B"), awards)))

    _controlled_run(monkeypatch, award)
    prepared = prepare_rival_pit_plan_selection(
        _classification_runner(constructor=True), 1, _constructor_plans(), "reference",
        _scenarios(3, 1, constructor=True), constructor_id="T",
        training_simulations=2, validation_simulations=2,
    )
    outcome = evaluate_prepared_rival_pit_plan_selection(prepared)
    selection = outcome["selection"]
    assert selection["selected_label"] == "planned"
    assert selection["tiebreak_applied"] == "unique_highest_weighted_training_mean"
    assert selection["training_score_table"] == [
        {"label": "reference", "total_points": 6, "mean_points": 3, "trials": 2,
         "total_score": 6, "mean_score": 3, "mean_score_behind_selected": 24.5,
         "mean_points_behind_selected": 24.5, "tied_for_best": False},
        {"label": "planned", "total_points": 55, "mean_points": 27.5, "trials": 2,
         "total_score": 55, "mean_score": 27.5, "mean_score_behind_selected": 0,
         "mean_points_behind_selected": 0, "tied_for_best": True},
    ]
    metrics = selection["validation_target_metrics"]
    assert metrics["reference_mean_points"] == 3
    assert metrics["selected_mean_points"] == 27.5
    assert metrics["mean_points_difference"] == 24.5
    assert metrics["points_difference_standard_error"] == 11.5
    assert metrics["points_outcome_profile"]["more_points_races"] == 2
    _assert_validation_cohort(outcome, "planned")


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
        ({"extra": {"weight": 1, "pit_plans": {}, "surprise": True}}, "must contain weight"),
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


def test_saved_adapter_and_prepared_in_memory_selector_have_matching_evidence(tmp_path):
    path = _saved(tmp_path)
    scenarios = _scenarios(3, 1)
    runner, saved_count = _load_saved_runner(path)

    saved = evaluate_saved_rival_pit_plan_selection(
        path, _driver_plans(), "reference", scenarios, driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    prepared = prepare_rival_pit_plan_selection(
        runner, saved_count, _driver_plans(), "reference", scenarios,
        driver_id="A", training_simulations=1, validation_simulations=1,
    )
    in_memory = evaluate_prepared_rival_pit_plan_selection(prepared)

    assert in_memory["selection"] == saved["selection"]
    for phase in ("training_results", "validation_results"):
        assert list(in_memory[phase]) == list(saved[phase])
        for scenario_name in saved[phase]:
            assert list(in_memory[phase][scenario_name]) == list(saved[phase][scenario_name])
            for label in saved[phase][scenario_name]:
                left = in_memory[phase][scenario_name][label]
                right = saved[phase][scenario_name][label]
                assert left.seed == right.seed
                assert left.num_simulations == right.num_simulations
                assert left.input_snapshot == right.input_snapshot


def test_prepared_rival_plans_keep_inherit_automatic_and_empty_distinct(tmp_path):
    path = _saved(tmp_path)
    runner, saved_count = _load_saved_runner(path)
    plans = _driver_plans()
    scenarios = {
        "inherit": {"weight": 1, "pit_plans": {}},
        "automatic": {"weight": 1, "pit_plans": {"B": None}},
        "no_stop": {"weight": 1, "pit_plans": {"B": []}},
    }

    prepared = prepare_rival_pit_plan_selection(
        runner, saved_count, plans, "reference", scenarios,
        driver_id="A", training_simulations=1, validation_simulations=1,
    )

    assert prepared["plans"] == plans
    assert prepared["plans_by_rival_scenario"]["inherit"]["reference"] == {
        "B": _HARD_STOP,
    }
    assert prepared["plans_by_rival_scenario"]["automatic"]["reference"] == {}
    assert prepared["plans_by_rival_scenario"]["no_stop"]["reference"] == {"B": []}
    assert prepared["plans_by_rival_scenario"]["inherit"]["planned"] == {
        "A": _HARD_STOP,
        "B": _HARD_STOP,
    }


def test_prepared_rival_evaluation_stops_between_runs_when_cancelled(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    runner, saved_count = _load_saved_runner(path)
    prepared = prepare_rival_pit_plan_selection(
        runner, saved_count, _driver_plans(), "reference", _scenarios(),
        driver_id="A", training_simulations=1, validation_simulations=1,
    )
    cancelled = Event()
    original_run = MonteCarloRunner.run
    calls = []

    def cancel_after_first_run(self, count, parallel=False, max_workers=None, *,
                               cancel_requested=None):
        assert cancel_requested is not None
        calls.append((self.base_seed, self.pit_plans))
        result = original_run(
            self, count, parallel=parallel, max_workers=max_workers,
            cancel_requested=cancel_requested,
        )
        cancelled.set()
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", cancel_after_first_run)
    with pytest.raises(SimulationCancelled):
        evaluate_prepared_rival_pit_plan_selection(
            prepared, cancel_requested=cancelled.is_set,
        )

    assert len(calls) == 1
    assert calls[0][0] == prepared["train_start"]
