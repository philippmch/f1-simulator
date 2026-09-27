"""Weighted pit-plan selection across explicit rival-plan assumptions."""

import json
from copy import deepcopy

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.rival_strategy_selection import (
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

    _controlled_run(monkeypatch, controlled_points)
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
