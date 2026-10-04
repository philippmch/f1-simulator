"""Saved pit-plan selection uses a complete training cohort and fresh validation seeds."""

import json
from copy import deepcopy

import pytest

from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.replay import _load_saved_runner
from f1sim.analysis.strategy_selection import (
    _points_outcome_profile,
    evaluate_pit_plan_selection,
    evaluate_saved_pit_plan_selection,
)
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output import Exporter


def _saved(tmp_path, *, engine="standard", constructor=False):
    drivers = [Driver(id="A", name="A", team_id="T")]
    if constructor:
        drivers.extend((
            Driver(id="B", name="B", team_id="T"),
            Driver(id="C", name="C", team_id="U"),
        ))
    ids = [driver.id for driver in drivers]
    inventory = {
        driver_id: [
            {"id": f"{driver_id}-medium", "compound": "medium", "age": 0},
            {"id": f"{driver_id}-hard", "compound": "hard", "age": 0},
        ]
        for driver_id in ids
    }
    cars = {team_id: Car(team_id=team_id, team_name=team_id)
            for team_id in {driver.team_id for driver in drivers}}
    pit_plans = {
        driver_id: [{"lap": 2, "compound": "hard"}]
        for driver_id in ids
    }
    result = MonteCarloRunner(
        drivers, cars,
        Track(id="t", name="Saved", country="T", total_laps=5, base_lap_time=90),
        Weather(change_probability=0), seed=71, race_engine=engine,
        starting_tires={driver_id: "medium" for driver_id in ids},
        tire_inventory=inventory,
        tire_warmup={"medium": 1.0, "hard": 0.5},
        pit_plans=pit_plans,
    ).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    return path


def _driver_plans():
    return {
        "automatic": None,
        "earlier": [{"lap": 2, "compound": "hard"}],
    }


@pytest.mark.parametrize("constructor", [False, True])
def test_saved_adapter_and_in_memory_selection_share_the_same_cohorts(
    tmp_path, constructor,
):
    path = _saved(tmp_path, constructor=constructor)
    plans = (
        {"automatic": None, "staggered": {
            "A": [{"lap": 2, "compound": "hard"}],
            "B": [{"lap": 3, "compound": "hard"}],
        }}
        if constructor else _driver_plans()
    )
    target = {"constructor_id": "T"} if constructor else {"driver_id": "A"}
    saved = evaluate_saved_pit_plan_selection(
        path, plans, "automatic", **target,
        training_simulations=2, validation_simulations=2,
    )
    runner, source_count = _load_saved_runner(path)
    in_memory = evaluate_pit_plan_selection(
        runner, source_count, plans, "automatic", **target,
        training_simulations=2, validation_simulations=2,
    )

    assert in_memory["selection"] == saved["selection"]
    for phase in ("training_results", "validation_results"):
        assert list(in_memory[phase]) == list(saved[phase])
        for label in in_memory[phase]:
            left = in_memory[phase][label]
            right = saved[phase][label]
            assert (left.seed, left.num_simulations, left.input_snapshot) == (
                right.seed, right.num_simulations, right.input_snapshot,
            )
            assert [
                [row.points_awarded for row in race] for race in left.race_results
            ] == [
                [row.points_awarded for row in race] for race in right.race_results
            ]


def _set_points(result, driver_id, values):
    for race, points in zip(result.race_results, values):
        rows = [row for row in race if row.driver_id == driver_id]
        assert len(rows) == 1
        rows[0].points_awarded = points


def _controlled_run(monkeypatch, award_function, calls=None):
    original = MonteCarloRunner.run

    def run(self, count, *, parallel=False, max_workers=None):
        result = original(self, count, parallel=False, max_workers=None)
        if calls is not None:
            calls.append((self.base_seed, deepcopy(self.pit_plans)))
        award_function(self, result)
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", run)


def test_points_outcome_profile_uses_conditional_gains_and_loss_magnitudes():
    assert _points_outcome_profile([10, 10, 10, 10], [15, 15, 10, 8]) == {
        "paired_races": 4,
        "more_points_races": 2,
        "equal_points_races": 1,
        "fewer_points_races": 1,
        "mean_points_gain_when_ahead": 5,
        "mean_points_loss_when_behind": 2,
    }
    assert _points_outcome_profile([0, 0], [3, 1]) == {
        "paired_races": 2,
        "more_points_races": 2,
        "equal_points_races": 0,
        "fewer_points_races": 0,
        "mean_points_gain_when_ahead": 2,
        "mean_points_loss_when_behind": None,
    }
    assert _points_outcome_profile([7], [4]) == {
        "paired_races": 1,
        "more_points_races": 0,
        "equal_points_races": 0,
        "fewer_points_races": 1,
        "mean_points_gain_when_ahead": None,
        "mean_points_loss_when_behind": 3,
    }


def test_frozen_training_winner_can_lose_on_heldout_and_keeps_fractional_mean(
    tmp_path, monkeypatch,
):
    path = _saved(tmp_path)

    def awards(runner, result):
        planned = bool((runner.pit_plans or {}).get("A"))
        if runner.base_seed == 72:
            values = [12, 13] if planned else [5, 5]
        else:
            values = [1, 0] if planned else [15, 14]
        _set_points(result, "A", values)

    _controlled_run(monkeypatch, awards)
    outcome = evaluate_saved_pit_plan_selection(
        path, _driver_plans(), "automatic", driver_id="A",
        training_simulations=2, validation_simulations=2,
    )
    selection = outcome["selection"]
    assert selection["selected_label"] == "earlier"
    assert selection["validation_status"] == "evaluated"
    assert selection["training_score_table"] == [
        {"label": "automatic", "total_points": 10, "mean_points": 5, "trials": 2,
         "total_score": 10, "mean_score": 5, "mean_score_behind_selected": 7.5,
         "tied_for_best": False},
        {"label": "earlier", "total_points": 25, "mean_points": 12.5, "trials": 2,
         "total_score": 25, "mean_score": 12.5, "mean_score_behind_selected": 0,
         "tied_for_best": True},
    ]
    assert selection["seed_ranges"] == {
        "source": {"first_seed": 71, "last_seed": 71, "trials": 1},
        "training": {"first_seed": 72, "last_seed": 73, "trials": 2},
        "validation": {"first_seed": 74, "last_seed": 75, "trials": 2},
    }
    assert list(outcome["validation_results"]) == ["automatic", "earlier"]
    assert selection["validation_target_metrics"]["mean_points_difference"] == -14
    assert selection["validation_target_metrics"]["points_difference_standard_error"] == 0
    assert selection["validation_target_metrics"]["points_outcome_profile"] == {
        "paired_races": 2,
        "more_points_races": 0,
        "equal_points_races": 0,
        "fewer_points_races": 2,
        "mean_points_gain_when_ahead": None,
        "mean_points_loss_when_behind": 14,
    }


def test_exact_ties_prefer_reference_then_remaining_ties_use_mapping_order(
    tmp_path, monkeypatch,
):
    path = _saved(tmp_path)

    def equal_awards(runner, result):
        _set_points(result, "A", [8])

    _controlled_run(monkeypatch, equal_awards)
    reference_outcome = evaluate_saved_pit_plan_selection(
        path, {"first": None, "reference": None}, "reference", driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    assert reference_outcome["selection"]["selected_label"] == "reference"
    assert (reference_outcome["selection"]["tiebreak_applied"]
            == "reference_preferred_on_exact_tie")
    assert list(reference_outcome["validation_results"]) == ["reference"]

    def ordered_awards(runner, result):
        plans = runner.pit_plans or {}
        points = 1 if not plans.get("A") else 9
        if plans.get("A") and plans["A"][0]["lap"] == 3:
            points = 9
        _set_points(result, "A", [points])

    _controlled_run(monkeypatch, ordered_awards)
    order_outcome = evaluate_saved_pit_plan_selection(
        path,
        {"first": [{"lap": 2, "compound": "hard"}], "automatic": None,
         "second": [{"lap": 3, "compound": "hard"}]},
        "automatic", driver_id="A", training_simulations=1, validation_simulations=1,
    )
    assert order_outcome["selection"]["selected_label"] == "first"
    assert order_outcome["selection"]["tiebreak_applied"] == "first_plan_order_on_exact_tie"


def test_constructor_scores_all_members_and_keeps_rival_out_of_target(tmp_path, monkeypatch):
    path = _saved(tmp_path, constructor=True)

    def awards(runner, result):
        candidate = bool((runner.pit_plans or {}).get("A"))
        if runner.base_seed == 72:
            _set_points(result, "A", [20 if candidate else 10])
            _set_points(result, "B", [0 if candidate else 10])
        else:
            _set_points(result, "A", [20 if candidate else 10])
            _set_points(result, "B", [0 if candidate else 10])
            _set_points(result, "C", [25])

    _controlled_run(monkeypatch, awards)
    constructor_plans = {
        "automatic": None,
        "staggered": {
            "A": [{"lap": 2, "compound": "hard"}],
            "B": [{"lap": 3, "compound": "hard"}],
        },
    }
    constructor_outcome = evaluate_saved_pit_plan_selection(
        path, constructor_plans, "automatic", constructor_id="T",
        training_simulations=1, validation_simulations=1,
    )
    selection = constructor_outcome["selection"]
    assert selection["target_member_ids"] == ["A", "B"]
    assert selection["selected_label"] == "automatic"
    assert selection["training_score_table"][0]["mean_points"] == 20
    assert selection["training_score_table"][1]["mean_points"] == 20
    assert selection["tiebreak_applied"] == "reference_preferred_on_exact_tie"
    assert list(constructor_outcome["validation_results"]) == ["automatic"]

    driver_outcome = evaluate_saved_pit_plan_selection(
        path, _driver_plans(), "automatic", driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    assert driver_outcome["selection"]["selected_label"] == "earlier"


def test_constructor_validation_sums_teammates_per_seed_before_standard_error(
    tmp_path, monkeypatch,
):
    path = _saved(tmp_path, constructor=True)

    def awards(runner, result):
        candidate = bool((runner.pit_plans or {}).get("A"))
        if runner.base_seed == 72:
            _set_points(result, "A", [20 if candidate else 10])
            _set_points(result, "B", [5 if candidate else 10])
        else:
            for index, race in enumerate(result.race_results):
                effective_seed = result.seed + index
                if effective_seed == 73:
                    points_a = 20 if candidate else 10
                    points_b = 0 if candidate else 10
                else:
                    points_a = 10 if candidate else 20
                    points_b = 20 if candidate else 0
                for row in race:
                    if row.driver_id == "A":
                        row.points_awarded = points_a
                    elif row.driver_id == "B":
                        row.points_awarded = points_b
        _set_points(result, "C", [7])

    _controlled_run(monkeypatch, awards)
    outcome = evaluate_saved_pit_plan_selection(
        path,
        {"automatic": None, "staggered": {
            "A": [{"lap": 2, "compound": "hard"}],
            "B": [{"lap": 3, "compound": "hard"}],
        }},
        "automatic", constructor_id="T", training_simulations=1,
        validation_simulations=2,
    )
    metrics = outcome["selection"]["validation_target_metrics"]
    assert outcome["selection"]["selected_label"] == "staggered"
    assert metrics["mean_points_difference"] == 5
    assert metrics["points_difference_standard_error"] == 5
    assert metrics["reference_mean_points"] == 20
    assert metrics["selected_mean_points"] == 25
    assert metrics["points_outcome_profile"] == {
        "paired_races": 2,
        "more_points_races": 1,
        "equal_points_races": 1,
        "fewer_points_races": 0,
        "mean_points_gain_when_ahead": 10,
        "mean_points_loss_when_behind": None,
    }


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_phases_preserve_saved_engine_inputs_and_rival_plans(tmp_path, engine):
    path = _saved(tmp_path, engine=engine, constructor=True)
    before = path.read_bytes()
    plans = {
        "automatic": None,
        "changed": {
            "A": [{"lap": 2, "compound": "hard"}],
            "B": [{"lap": 3, "compound": "hard"}],
        },
    }
    outcome = evaluate_saved_pit_plan_selection(
        path, plans, "automatic", constructor_id="T", training_simulations=2,
        validation_simulations=2,
    )
    assert list(outcome["training_results"]) == list(plans)
    assert outcome["training_results"]["automatic"].seed == 72
    assert outcome["validation_results"]["automatic"].seed == 74
    assert path.read_bytes() == before
    for phase_results in (outcome["training_results"], outcome["validation_results"]):
        for result in phase_results.values():
            assert result.race_engine == engine
            assert result.input_snapshot["starting_tires"] == {
                "A": "medium", "B": "medium", "C": "medium",
            }
            assert result.input_snapshot["tire_inventory"]["A"][0]["compound"] == "medium"
            assert result.input_snapshot["tire_warmup"] == {"medium": 1.0, "hard": 0.5}
            assert result.input_snapshot["pit_plans"]["C"] == [
                {"lap": 2, "compound": "hard"},
            ]


def test_no_change_runs_one_validation_variant_and_records_identity(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    calls = []

    def no_points(runner, result):
        _set_points(result, "A", [4])

    _controlled_run(monkeypatch, no_points, calls)
    outcome = evaluate_saved_pit_plan_selection(
        path, {"reference": None, "same": None}, "reference", driver_id="A",
        training_simulations=1, validation_simulations=1,
    )
    assert calls == [(72, None), (72, None), (73, None)]
    assert list(outcome["validation_results"]) == ["reference"]
    assert outcome["selection"]["validation_status"] == "no_change"
    assert outcome["selection"]["validation_target_metrics"]["mean_points_difference"] == 0
    assert outcome["selection"]["validation_target_metrics"][
        "points_outcome_profile"
    ] is None
    assert outcome["selection"]["validation_target_metrics"][
        "points_difference_standard_error"
    ] is None


def test_cancellation_after_first_selection_variant_stops_before_next(monkeypatch, tmp_path):
    path = _saved(tmp_path)
    runner, source_count = _load_saved_runner(path)
    original = MonteCarloRunner.run
    calls = []
    cancelled = False

    def run(self, count, *, parallel=False, max_workers=None, cancel_requested=None):
        nonlocal cancelled
        calls.append(self.base_seed)
        result = original(self, count, parallel=False, max_workers=None)
        cancelled = True
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", run)
    with pytest.raises(SimulationCancelled, match="selection was cancelled"):
        evaluate_pit_plan_selection(
            runner, source_count, _driver_plans(), "automatic", driver_id="A",
            training_simulations=1, validation_simulations=1,
            cancel_requested=lambda: cancelled,
        )
    assert calls == [72]


def test_invalid_late_plan_and_seed_overflow_fail_before_any_trial(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    calls = []
    monkeypatch.setattr(MonteCarloRunner, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="must not exceed"):
        _overflowing_call(path, tmp_path)
    assert calls == []

    with pytest.raises(ValueError, match="lap|compound"):
        evaluate_saved_pit_plan_selection(
            path,
            {"valid": None, "invalid": [{"lap": 6, "compound": "hard"}]},
            "valid", driver_id="A", training_simulations=1, validation_simulations=1,
        )
    assert calls == []


def _overflowing_call(path, tmp_path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["seed"] = 2**32 - 1
    changed = tmp_path / "overflow.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    return evaluate_saved_pit_plan_selection(
        changed, _driver_plans(), "automatic", driver_id="A",
        training_simulations=1, validation_simulations=1,
    )


@pytest.mark.parametrize("corruption", ["qualifying", "points"])
def test_incomplete_training_evidence_rejects_without_running_validation(
    tmp_path, monkeypatch, corruption,
):
    path = _saved(tmp_path)
    calls = []
    original = MonteCarloRunner.run

    def corrupt_training(self, count, *, parallel=False, max_workers=None):
        result = original(self, count, parallel=False, max_workers=None)
        calls.append(self.base_seed)
        if self.base_seed == 72 and (self.pit_plans or {}).get("A"):
            if corruption == "qualifying":
                result.qualifying_results[0].clear()
            else:
                result.race_results[0][0].points_awarded = 1.5
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", corrupt_training)
    with pytest.raises(ValueError, match="qualifying|points outcome"):
        evaluate_saved_pit_plan_selection(
            path, _driver_plans(), "automatic", driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    assert calls == [72, 72]


@pytest.mark.parametrize("field", ["engine", "opening"])
def test_incompatible_training_engine_or_opening_rejects_cohort(tmp_path, monkeypatch, field):
    path = _saved(tmp_path)
    calls = []
    original = MonteCarloRunner.run

    def mutate_inputs(self, count, *, parallel=False, max_workers=None):
        result = original(self, count, parallel=False, max_workers=None)
        calls.append(self.base_seed)
        if self.base_seed == 72 and (self.pit_plans or {}).get("A"):
            if field == "engine":
                result.race_engine = "chronological"
            else:
                result.input_snapshot["starting_tires"]["A"] = "hard"
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", mutate_inputs)
    message = "race engine" if field == "engine" else "opening tyre"
    with pytest.raises(ValueError, match=message):
        evaluate_saved_pit_plan_selection(
            path, _driver_plans(), "automatic", driver_id="A",
            training_simulations=1, validation_simulations=1,
        )
    assert calls == [72, 72]


def test_incomplete_constructor_trial_rejects_selection(tmp_path, monkeypatch):
    path = _saved(tmp_path, constructor=True)
    original = MonteCarloRunner.run

    def drop_teammate(self, count, *, parallel=False, max_workers=None):
        result = original(self, count, parallel=False, max_workers=None)
        if self.base_seed == 72 and (self.pit_plans or {}).get("A"):
            result.race_results[0][:] = [
                row for row in result.race_results[0] if row.driver_id != "B"
            ]
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", drop_teammate)
    with pytest.raises(ValueError, match="points outcome.*B"):
        evaluate_saved_pit_plan_selection(
            path, {"automatic": None, "changed": {
                "A": [{"lap": 2, "compound": "hard"}],
                "B": [{"lap": 3, "compound": "hard"}],
            }}, "automatic", constructor_id="T", training_simulations=1,
            validation_simulations=1,
        )


def test_incomplete_validation_evidence_fails_after_training_before_result(tmp_path, monkeypatch):
    path = _saved(tmp_path)
    original = MonteCarloRunner.run

    def drop_validation(self, count, *, parallel=False, max_workers=None):
        result = original(self, count, parallel=False, max_workers=None)
        candidate = bool((self.pit_plans or {}).get("A"))
        if self.base_seed == 72:
            for race in result.race_results:
                row = next(row for row in race if row.driver_id == "A")
                row.points_awarded = 20 if candidate else 1
        if self.base_seed == 74 and candidate:
            result.race_results[0].clear()
        return result

    monkeypatch.setattr(MonteCarloRunner, "run", drop_validation)
    with pytest.raises(ValueError, match="points outcome"):
        evaluate_saved_pit_plan_selection(
            path, _driver_plans(), "automatic", driver_id="A",
            training_simulations=2, validation_simulations=1,
        )


@pytest.mark.parametrize(
    "training,validation,workers",
    [(None, 1, None), (1, None, None), (True, 1, None), (1, False, None),
     (1, 1, 0)],
)
def test_invalid_counts_and_workers_are_rejected(tmp_path, training, validation, workers):
    path = _saved(tmp_path)
    with pytest.raises(ValueError):
        evaluate_saved_pit_plan_selection(
            path, _driver_plans(), "automatic", driver_id="A",
            training_simulations=training, validation_simulations=validation,
            max_workers=workers,
        )
