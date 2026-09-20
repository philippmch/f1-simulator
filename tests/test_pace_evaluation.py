"""Held-out pace must be independent of target times and later performance."""

import copy
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from f1sim.analysis.pace_evaluation import (
    _aggregate,
    _aggregate_team_metrics,
    _aggregate_teammate_gaps,
    _component_variant_metrics,
    _constructor_standings,
    _metrics,
    _neutral_driver_stats,
    _team_median_metrics,
    _teammate_gap_metrics,
    evaluate_qualifying_pace,
)
from f1sim.analysis.qualifying_history import (
    build_historical_q1_events,
    recent_team_q1_predictions,
)
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader

YEAR = datetime.now(timezone.utc).year


def row(round_number, code, team, q1, position):
    return {"round": round_number, "Driver": {"driverId": code.lower(), "code": code,
                                              "givenName": code, "familyName": "Racer"},
            "Constructor": {"constructorId": team, "name": team},
            "Q1": q1, "position": str(position), "status": "Finished",
            "FastestLap": {"AverageSpeed": {"speed": str(230 - position)},
                           "Time": {"time": "1:00.000"}}}


def standings(round_number):
    return {"MRData": {"total": "2", "offset": "0", "StandingsTable": {
        "season": str(YEAR), "round": str(round_number),
        "StandingsLists": [{"season": str(YEAR),
        "round": str(round_number), "ConstructorStandings": [
            {"Constructor": {"constructorId": "a", "name": "a"}, "points": "40"},
            {"Constructor": {"constructorId": "b", "name": "b"}, "points": "20"},
        ]}]}}}


@pytest.fixture
def fixture(monkeypatch):
    loader = CurrentSeasonDataLoader(current_year=YEAR, http_getter=lambda *a, **k: {})
    loader._calendar = [{"round": n, "race": f"Race {n}", "circuit_id": f"circuit{n}",
                         "completed": True} for n in (1, 2, 3)]
    qualifying = [row(n, code, team, f"1:{time:02}.000", pos)
                  for n in (1, 2, 3)
                  for pos, (code, team, time) in enumerate([
                      ("A1", "a", 10), ("A2", "a", 11), ("B1", "b", 12), ("B2", "b", 13),
                  ], 1)]
    results = copy.deepcopy(qualifying)
    calls = []

    def fetch(url):
        calls.append(url)
        assert f"/{YEAR}/1/constructorstandings.json?" in url or (
            f"/{YEAR}/2/constructorstandings.json?" in url
        )
        return standings(int(url.split(f"/{YEAR}/")[1].split("/")[0]))

    def forbidden(*args, **kwargs):
        pytest.fail("Holdout requested live standings, live roster, or target track calibration")

    monkeypatch.setattr(loader, "_fetch_json", fetch)
    monkeypatch.setattr(loader, "_season_data", lambda _: copy.deepcopy((results, qualifying)))
    monkeypatch.setattr(loader, "_standings", forbidden)
    monkeypatch.setattr(loader, "_round_data", forbidden)
    monkeypatch.setattr(loader, "get_official_roster", forbidden)
    return loader, results, qualifying, calls


def test_target_times_and_future_rows_do_not_change_predictions(fixture):
    loader, results, qualifying, calls = fixture
    original = evaluate_qualifying_pace(loader, YEAR, target_race=2)
    for rows in (results, qualifying):
        for record in rows:
            if record["round"] >= 2:
                record["Q1"] = "2:00.000"
                record["position"] = "22"
                record["FastestLap"] = {"AverageSpeed": {"speed": "999"},
                                        "Time": {"time": "0:01.000"}}
                record["points"] = "99999"
    changed = evaluate_qualifying_pace(loader, YEAR, target_race=2)
    before, after = original["folds"][0], changed["folds"][0]
    assert before["form_rounds"] == after["form_rounds"] == [1]
    assert before["standings_round"] == after["standings_round"] == 1
    assert before["static_reference_lap_seconds"] == after["static_reference_lap_seconds"] == 90
    for p, q in zip(before["predictions"], after["predictions"]):
        assert p["observed_q1_seconds"] != q["observed_q1_seconds"]
        assert {k: v for k, v in p.items() if k != "observed_q1_seconds"} == {
            k: v for k, v in q.items() if k != "observed_q1_seconds"
        }
    assert len(calls) == 2 and all(f"/{YEAR}/1/" in url for url in calls)


def test_prefix_changes_affect_prediction_and_standings_include_full_prefix(fixture):
    loader, _, qualifying, _ = fixture
    before = evaluate_qualifying_pace(loader, YEAR, target_race=3, form_races=1)["folds"][0]
    for record in qualifying:
        if record["round"] == 2 and record["Driver"]["code"] == "A1":
            record["Q1"] = "1:00.000"
    after = evaluate_qualifying_pace(loader, YEAR, target_race=3, form_races=1)["folds"][0]
    assert before["form_rounds"] == [2]
    assert before["standings_round"] == 2
    assert before["predictions"][0]["constructor_points"] == 40
    assert before["predictions"][0]["skill_rating"] != after["predictions"][0]["skill_rating"]


def test_unshared_later_form_sessions_do_not_create_prediction_advantages(fixture):
    loader, _, qualifying, _ = fixture
    before = evaluate_qualifying_pace(loader, YEAR, target_race=3)
    for record in qualifying:
        if record["round"] < 3 and record["Driver"]["code"] == "A1":
            # A1 alone reaches faster later sessions. Its teammate and the
            # broadly represented field still share only Q1 observations.
            record.update(Q2="1:00.000", Q3="0:59.000", bestTime="0:58.000")
    after = evaluate_qualifying_pace(loader, YEAR, target_race=3)
    assert before["folds"] == after["folds"]
    assert before["aggregate"] == after["aggregate"]


def test_first_round_has_no_standings_or_baseline_and_no_historical_requests(fixture):
    loader, _, _, calls = fixture
    fold = evaluate_qualifying_pace(loader, YEAR, target_race=1)["folds"][0]
    assert calls == []
    assert fold["form_rounds"] == [] and fold["standings_round"] is None
    assert fold["baseline_round"] is None
    assert fold["paired_comparison"]["previous_q1"]["rank_mae"] is None
    assert fold["model"]["drivers"] == 4


def test_result_sources_and_live_snapshot_caches_are_not_mutated(fixture):
    loader, results, qualifying, _ = fixture
    loader._driver_stats = {"sentinel": "live"}
    loader._track_stats = {"circuit2": "live target fastest lap"}
    loader._last_completed_rounds = [3]
    loader._last_qualifying_rounds = [3]
    before = copy.deepcopy((results, qualifying, loader._driver_stats, loader._track_stats))
    first = evaluate_qualifying_pace(loader, YEAR, target_race=2)
    second = evaluate_qualifying_pace(loader, YEAR, target_race=2)
    assert first == second
    assert (results, qualifying, loader._driver_stats, loader._track_stats) == before
    assert loader._last_completed_rounds == loader._last_qualifying_rounds == [3]


def test_missing_q1_labels_use_paired_denominators_and_never_other_sessions(fixture):
    loader, _, qualifying, _ = fixture
    for record in qualifying:
        if record["round"] == 1 and record["Driver"]["code"] == "A1":
            record["Q1"] = None
            record["Q3"] = "0:01.000"
        if record["round"] == 2 and record["Driver"]["code"] == "B2":
            record["Q1"] = None
            record["Q3"] = "0:01.000"
    fold = evaluate_qualifying_pace(loader, YEAR, target_race=2)["folds"][0]
    assert fold["model"]["drivers"] == 3
    assert fold["paired_comparison"]["driver_ids"] == ["A2", "B1"]
    assert fold["paired_comparison"]["model"]["drivers"] == 2
    assert fold["paired_comparison"]["previous_q1"]["drivers"] == 2


def test_entrant_team_is_from_target_identity_not_current_roster(fixture):
    loader, _, qualifying, _ = fixture
    for record in qualifying:
        if record["round"] == 2 and record["Driver"]["code"] == "A1":
            record["Constructor"] = {"constructorId": "b", "name": "b"}
    fold = evaluate_qualifying_pace(loader, YEAR, target_race=2)["folds"][0]
    assert fold["predictions"][0]["team_id"] == "b"


@pytest.mark.parametrize("problem", ["round", "season", "empty", "partial", "duplicate", "nan"])
def test_standings_must_prove_the_requested_cutoff(fixture, monkeypatch, problem):
    loader, _, _, _ = fixture
    payload = standings(1)
    section = payload["MRData"]["StandingsTable"]["StandingsLists"][0]
    if problem in ("round", "season"):
        section[problem] = "9999"
    elif problem == "empty":
        section["ConstructorStandings"] = []
    elif problem == "partial":
        payload["MRData"]["total"] = "3"
    elif problem == "duplicate":
        section["ConstructorStandings"][1] = copy.deepcopy(section["ConstructorStandings"][0])
    else:
        section["ConstructorStandings"][0]["points"] = "NaN"
    monkeypatch.setattr(loader, "_fetch_json", lambda _: payload)
    with pytest.raises(CurrentSeasonDataError):
        _constructor_standings(loader, YEAR, 1)


def test_conflicting_prefix_rows_fail_instead_of_arbitrary_selection(fixture):
    loader, _, qualifying, _ = fixture
    duplicate = dict(qualifying[0], Q1="0:01.000")
    qualifying.append(duplicate)
    with pytest.raises(CurrentSeasonDataError, match="Conflicting"):
        evaluate_qualifying_pace(loader, YEAR, target_race=2)


def test_conflicting_prefix_aliases_cannot_double_count_a_driver(fixture):
    loader, _, qualifying, _ = fixture
    duplicate = copy.deepcopy(qualifying[0])
    duplicate["Driver"]["driverId"] = "alternate-alias"
    qualifying.append(duplicate)
    with pytest.raises(CurrentSeasonDataError, match="Conflicting aliases"):
        evaluate_qualifying_pace(loader, YEAR, target_race=2)


@pytest.mark.parametrize("field,value", [("round", "9"), ("round", None), ("season", "2020")])
def test_top_level_standings_cutoff_is_required(fixture, monkeypatch, field, value):
    loader, _, _, _ = fixture
    payload = standings(1)
    payload["MRData"]["StandingsTable"][field] = value
    monkeypatch.setattr(loader, "_fetch_json", lambda _: payload)
    with pytest.raises(CurrentSeasonDataError):
        _constructor_standings(loader, YEAR, 1)


def test_partial_target_entrant_cohort_is_rejected(fixture):
    loader, _, qualifying, _ = fixture
    qualifying[:] = [record for record in qualifying
                    if record["round"] != 2 or record["Driver"]["code"] in {"A1", "A2"}]
    with pytest.raises(CurrentSeasonDataError, match="Incomplete target entrant coverage"):
        evaluate_qualifying_pace(loader, YEAR, target_race=2)


@pytest.mark.parametrize("value", [None, True, "NaN", "Infinity", -1, "0:00.000"])
def test_unusable_target_q1_is_missing_without_dropping_entrant_identity(fixture, value):
    loader, _, qualifying, _ = fixture
    for record in qualifying:
        if record["round"] == 2 and record["Driver"]["code"] == "A1":
            record["Q1"] = value
    fold = evaluate_qualifying_pace(loader, YEAR, target_race=2)["folds"][0]
    assert fold["entrants"] == 4
    assert fold["model"]["drivers"] == 3
    assert fold["predictions"][0]["observed_q1_seconds"] is None


def test_exact_duplicates_and_permutation_do_not_change_report(fixture):
    loader, results, qualifying, _ = fixture
    before = evaluate_qualifying_pace(loader, YEAR)
    results.append(copy.deepcopy(results[0]))
    qualifying.append(copy.deepcopy(qualifying[0]))
    results.reverse()
    qualifying.reverse()
    assert evaluate_qualifying_pace(loader, YEAR) == before


def test_metrics_handle_ties_missing_samples_and_scale_invariance():
    perfect = _metrics([1, 2, 3], [10, 20, 30])
    assert perfect["rank_mae"] == perfect["relative_pace_mae_pct"] == 0
    assert perfect["pairwise_concordance"] == 1
    tied = _metrics([1, 1, 1], [10, 20, 30])
    assert tied["pairwise_concordance"] == .5
    assert tied["rank_mae"] == pytest.approx(2 / 3)
    assert _metrics([1, 2], [10, 10])["pairwise_concordance"] is None
    assert _metrics([1], [10])["rank_mae"] is None
    aggregate = _aggregate([perfect, tied, _metrics([], [])])
    assert aggregate["scored_folds"] == 2 and aggregate["driver_observations"] == 6
    assert aggregate["pairwise_concordance"] == .75
    assert math.isfinite(aggregate["relative_pace_mae_pct"])


def _component_rows(
    values: list[tuple[str, str, float, float, float | None]],
) -> list[dict]:
    return [
        {
            "driver_id": driver_id,
            "team_id": team_id,
            "predicted_seconds": predicted,
            "observed_q1_seconds": observed,
            "previous_q1_seconds": previous,
        }
        for driver_id, team_id, predicted, observed, previous in values
    ]


def test_component_team_and_teammate_metrics_use_explicit_denominators():
    rows = _component_rows([
        ("A1", "A", 100.0, 100.0, 99.0),
        ("A2", "A", 101.0, 102.0, 101.0),
        ("A3", "A", 102.0, 104.0, 103.0),
        ("B1", "B", 98.0, 98.0, 97.0),
    ])
    teams = _team_median_metrics(rows)
    gaps = _teammate_gap_metrics(rows)
    assert teams["teams"] == 2
    assert gaps["teammate_pairs"] == 3
    assert gaps["comparable_pairs"] == 3
    assert gaps["pairwise_concordance"] == 1.0
    assert gaps["mean_abs_predicted_gap_pct"] == pytest.approx(
        4 / 100.5 * 100 / 3,
    )

    one_team = _team_median_metrics(rows[:3])
    one_team_gaps = _teammate_gap_metrics(rows[:3])
    assert one_team["teams"] == 1 and one_team["rank_mae"] is None
    assert one_team_gaps["teammate_pairs"] == 3
    aggregate_teams = _aggregate_team_metrics([teams, one_team])
    aggregate_gaps = _aggregate_teammate_gaps([gaps, one_team_gaps])
    assert aggregate_teams["team_observations"] == 2
    assert aggregate_gaps["teammate_pairs"] == 6


def test_component_teammate_metrics_handle_ties_scale_and_empty_cohorts():
    tied = _component_rows([
        ("A1", "A", 100.0, 100.0, None),
        ("A2", "A", 100.0, 102.0, None),
    ])
    scaled = _component_rows([
        (row["driver_id"], row["team_id"], row["predicted_seconds"] * 2,
         row["observed_q1_seconds"] * 2, None)
        for row in tied
    ])
    tied_metrics = _teammate_gap_metrics(tied)
    assert tied_metrics["pairwise_concordance"] == 0.5
    assert tied_metrics["gap_mae_pct"] == pytest.approx(
        tied_metrics["mean_abs_observed_gap_pct"],
    )
    assert _teammate_gap_metrics(scaled) == pytest.approx(tied_metrics)
    empty = _team_median_metrics([])
    assert empty["teams"] == 0 and empty["rank_mae"] is None
    assert _teammate_gap_metrics([]) == {
        "gap_mae_pct": None,
        "mean_abs_predicted_gap_pct": None,
        "mean_abs_observed_gap_pct": None,
        "teammate_pairs": 0,
        "comparable_pairs": 0,
        "pairwise_concordance": None,
    }


def test_component_variant_metrics_keep_paired_cohort_and_baseline_separate():
    rows = _component_rows([
        ("A1", "A", 100.0, 100.0, 101.0),
        ("A2", "A", 101.0, 102.0, None),
    ])
    metrics = _component_variant_metrics(rows, ["A1", "A2"])
    paired = metrics["paired_comparison"]
    assert paired["driver_ids"] == ["A1"]
    assert paired["model"]["drivers"] == 1
    assert paired["previous_q1"]["drivers"] == 1
    assert paired["paired_team_medians"]["teams"] == 1
    assert paired["previous_q1_team_medians"]["teams"] == 1
    assert paired["paired_teammate_gaps"]["teammate_pairs"] == 0
    assert paired["previous_q1_teammate_gaps"]["teammate_pairs"] == 0


def test_recent_team_q1_uses_fixed_window_and_native_fallback_without_future_data():
    history = [
        {"round": 1, "status": "scored", "teams": {
            "old": {"q1_count": 2, "residual": 0.01},
            "a": {"q1_count": 2, "residual": 0.01},
        }},
        {"round": 2, "status": "scored", "teams": {
            "a": {"q1_count": 2, "residual": 0.03},
        }},
        {"round": 3, "status": "scored", "teams": {
            "a": {"q1_count": 2, "residual": 0.02},
        }},
        {"round": 4, "status": "scored", "teams": {
            "a": {"q1_count": 2, "residual": 0.04},
        }},
        {"round": 6, "status": "scored", "teams": {
            "a": {"q1_count": 2, "residual": 0.90},
        }},
    ]
    native = [
        {"driver_id": "A1", "team_id": "a", "predicted_seconds": 103.0},
        {"driver_id": "A2", "team_id": "a", "predicted_seconds": 103.0},
        {"driver_id": "C1", "team_id": "new", "predicted_seconds": 97.0},
        {"driver_id": "C2", "team_id": "new", "predicted_seconds": 97.0},
    ]
    candidate, metadata = recent_team_q1_predictions(native, history, target_round=5)
    values = {row["driver_id"]: row["predicted_seconds"] for row in candidate}
    assert metadata["training_rounds"] == [2, 3, 4]
    assert metadata["source_coverage"]["a"]["rounds"] == [2, 3, 4]
    assert metadata["source_coverage"]["new"]["source"] == (
        "native_full_model_team_residual"
    )
    assert metadata["fallback_teams"] == ["new"]
    assert values == {"A1": 103.0, "A2": 103.0, "C1": 97.0, "C2": 97.0}

    future_changed = copy.deepcopy(history)
    future_changed[-1]["teams"]["a"]["residual"] = 9.0
    changed, changed_metadata = recent_team_q1_predictions(
        native, future_changed, target_round=5,
    )
    assert changed == candidate
    assert changed_metadata == metadata


def test_recent_team_q1_cold_start_returns_exact_native_predictions():
    native = [
        {"driver_id": "A1", "team_id": "a", "predicted_seconds": 100.123456789},
        {"driver_id": "A2", "team_id": "a", "predicted_seconds": 101.0},
    ]
    candidate, metadata = recent_team_q1_predictions(native, [], target_round=1)
    assert candidate == native
    assert metadata["training_rounds"] == []
    assert metadata["candidate_fallback"] is None
    assert metadata["source_coverage"]["a"]["source"] == (
        "native_full_model_team_residual"
    )


def test_historical_q1_event_gate_reports_coverage_and_q1_usability(fixture):
    loader, results, qualifying, _ = fixture
    events = build_historical_q1_events(loader, loader._calendar, results, qualifying)
    assert [event["round"] for event in events] == [1, 2, 3]
    assert all(event["status"] == "scored" for event in events)
    assert events[0]["usable_q1_count"] == 4
    assert events[0]["result_coverage"] is True
    assert events[0]["q1_coverage"] is True

    tied_qualifying = copy.deepcopy(qualifying)
    for record in tied_qualifying:
        if record["round"] == 1:
            record["Q1"] = "1:00.000"
    tied = build_historical_q1_events(loader, loader._calendar, results, tied_qualifying)
    assert tied[0]["status"] == "scored"
    assert tied[0]["teams"]["a"]["residual"] == 0

    one_q1 = copy.deepcopy(qualifying)
    for record in one_q1:
        if record["round"] == 1 and record["Driver"]["code"] != "A1":
            record["Q1"] = None
    incomplete = build_historical_q1_events(loader, loader._calendar, results, one_q1)
    assert incomplete[0]["status"] == "insufficient_q1_times"
    assert incomplete[0]["usable_q1_count"] == 1

    prefix = build_historical_q1_events(
        loader, loader._calendar, results, qualifying, before_round=3,
    )
    assert [event["round"] for event in prefix] == [1, 2]


def test_history_component_matches_single_target_and_isolation_rules(fixture):
    loader, results, qualifying, _ = fixture
    all_report = evaluate_qualifying_pace(loader, YEAR, include_components=True)
    target_report = evaluate_qualifying_pace(
        loader, YEAR, target_race=2, include_components=True,
    )
    all_fold = next(fold for fold in all_report["folds"] if fold["round"] == 2)
    target_fold = target_report["folds"][0]
    all_component = all_fold["components"]["recent_team_q1"]
    target_component = target_fold["components"]["recent_team_q1"]
    assert all_component == target_component

    before = copy.deepcopy(target_component)
    for rows in (results, qualifying):
        for record in rows:
            if record["round"] >= 2:
                record["Q1"] = "2:00.000"
                record["FastestLap"] = {"AverageSpeed": {"speed": "999"}}
    changed = evaluate_qualifying_pace(
        loader, YEAR, target_race=2, include_components=True,
    )["folds"][0]["components"]["recent_team_q1"]
    for before_row, changed_row in zip(before["predictions"], changed["predictions"]):
        assert {
            key: value for key, value in before_row.items()
            if key not in {"observed_q1_seconds", "previous_q1_seconds"}
        } == {
            key: value for key, value in changed_row.items()
            if key not in {"observed_q1_seconds", "previous_q1_seconds"}
        }
    assert before["forecast"] == changed["forecast"]

    for record in qualifying:
        if record["round"] == 2:
            record["Q1"] = None
    missing = evaluate_qualifying_pace(
        loader, YEAR, target_race=2, include_components=True,
    )["folds"][0]["components"]["recent_team_q1"]
    assert before["forecast"] == missing["forecast"]
    assert [row["predicted_seconds"] for row in before["predictions"]] == [
        row["predicted_seconds"] for row in missing["predictions"]
    ]


def test_history_component_does_not_change_default_report_or_fetches(fixture):
    loader, _, _, calls = fixture
    before = evaluate_qualifying_pace(loader, YEAR, target_race=2)
    before_calls = list(calls)
    with_components = evaluate_qualifying_pace(
        loader, YEAR, target_race=2, include_components=True,
    )
    after = evaluate_qualifying_pace(loader, YEAR, target_race=2)
    assert before == after
    assert "components" not in before["folds"][0]
    assert "components" in with_components["folds"][0]
    assert len(calls) == len(before_calls) + 2


def test_neutral_component_driver_assumptions_are_explicit_in_wet_conditions(fixture):
    loader, _, _, _ = fixture
    stats = loader._build_driver_stats(
        year=YEAR,
        target_event=loader._calendar[1],
        roster=[
            {"id": "A1", "name": "A1", "team_name": "a"},
            {"id": "A2", "name": "A2", "team_name": "a"},
        ],
        driver_standings=[], constructor_standings=[], race_rows=[], quali_rows=[],
        target_qualifying_rows=[], track_weight=0.0, form_weight=0.0, quali_weight=0.0,
    )
    neutral = _neutral_driver_stats(stats)
    assert {
        (
            item.driver_skill_rating, item.consistency_rating,
            item.wet_skill_modifier, item.overtaking_skill, item.tire_management,
        )
        for item in neutral.values()
    } == {(0.92, 0.968, 0.9936, 0.90, 0.968)}


def test_components_are_opt_in_and_do_not_add_fetches_or_mutate_loader(fixture):
    loader, results, qualifying, calls = fixture
    loader._driver_stats = {"sentinel": "live"}
    loader._track_stats = {"circuit2": "live target calibration"}
    before_sources = copy.deepcopy((results, qualifying, loader._driver_stats, loader._track_stats))
    report = evaluate_qualifying_pace(loader, YEAR, target_race=2, include_components=True)
    assert len(calls) == 1 and all(f"/{YEAR}/1/" in url for url in calls)
    assert (results, qualifying, loader._driver_stats, loader._track_stats) == before_sources
    fold = report["folds"][0]
    assert set(fold["components"]) == {
        "assumptions", "constructor_prior", "team_form", "full_model", "recent_team_q1",
    }
    assert fold["components"]["full_model"]["model"] == fold["model"]
    assert fold["components"]["full_model"]["predictions"] == fold["predictions"]
    paired_ids = fold["paired_comparison"]["driver_ids"]
    for name in ("constructor_prior", "team_form", "full_model", "recent_team_q1"):
        component = fold["components"][name]
        assert component["paired_comparison"]["driver_ids"] == paired_ids
        assert "paired_team_medians" in component["paired_comparison"]
        assert "previous_q1_teammate_gaps" in component["paired_comparison"]
    first = evaluate_qualifying_pace(loader, YEAR, target_race=1, include_components=True)
    first_components = first["folds"][0]["components"]
    assert first_components["recent_team_q1"]["predictions"] == (
        first_components["full_model"]["predictions"]
    )
    assert first_components["recent_team_q1"]["forecast"]["training_rounds"] == []


def test_component_predictions_ignore_target_and_future_performance(fixture):
    loader, results, qualifying, _ = fixture
    before = evaluate_qualifying_pace(loader, YEAR, target_race=2, include_components=True)
    for rows in (results, qualifying):
        for record in rows:
            if record["round"] >= 2:
                record["Q1"] = "2:00.000"
                record["position"] = "22"
                record["FastestLap"] = {"AverageSpeed": {"speed": "999"},
                                         "Time": {"time": "0:01.000"}}
    after = evaluate_qualifying_pace(loader, YEAR, target_race=2, include_components=True)
    for name in ("constructor_prior", "team_form", "full_model", "recent_team_q1"):
        first = before["folds"][0]["components"][name]["predictions"]
        second = after["folds"][0]["components"][name]["predictions"]
        for previous, current in zip(first, second):
            assert {
                key: value for key, value in previous.items()
                if key not in {"observed_q1_seconds", "previous_q1_seconds"}
            } == {
                key: value for key, value in current.items()
                if key not in {"observed_q1_seconds", "previous_q1_seconds"}
            }


def test_constructor_prior_stays_fixed_when_prior_pace_changes_but_team_reacts(
    fixture, monkeypatch,
):
    loader, results, qualifying, _ = fixture
    def equal_standings(url):
        payload = standings(1)
        for row_data in payload["MRData"]["StandingsTable"]["StandingsLists"][0][
            "ConstructorStandings"
        ]:
            row_data["points"] = "20"
        return payload

    monkeypatch.setattr(loader, "_fetch_json", equal_standings)
    before = evaluate_qualifying_pace(loader, YEAR, target_race=2, include_components=True)
    for rows in (results, qualifying):
        for record in rows:
            if record["round"] == 1 and record["Driver"]["code"] in {"A1", "A2"}:
                record["Q1"] = "2:00.000"
                record["FastestLap"] = {"AverageSpeed": {"speed": "50"},
                                         "Time": {"time": "0:01.000"}}
    after = evaluate_qualifying_pace(loader, YEAR, target_race=2, include_components=True)
    before_components = before["folds"][0]["components"]
    after_components = after["folds"][0]["components"]
    before_prior = [
        row["predicted_seconds"]
        for row in before_components["constructor_prior"]["predictions"]
    ]
    after_prior = [
        row["predicted_seconds"]
        for row in after_components["constructor_prior"]["predictions"]
    ]
    assert before_prior == after_prior
    assert [row["predicted_seconds"] for row in before_components["team_form"]["predictions"]] != [
        row["predicted_seconds"] for row in after_components["team_form"]["predictions"]
    ]


@pytest.mark.parametrize("form", [-1, True, 25, 1.5])
def test_form_validation_precedes_fetches(fixture, form):
    loader, _, _, calls = fixture
    with pytest.raises(ValueError, match="form_races"):
        evaluate_qualifying_pace(loader, YEAR, form_races=form)
    assert calls == []


@pytest.mark.parametrize("scenario", ["dry", "light_rain", "heavy_rain"])
def test_cli_emits_single_report_with_the_selected_scenario(fixture, monkeypatch, capsys, scenario):
    loader, _, _, _ = fixture
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_qualifying_pace.py"
    spec = importlib.util.spec_from_file_location("pace_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CurrentSeasonDataLoader", lambda **kwargs: loader)
    monkeypatch.setattr(sys, "argv", [str(path), "--race", "2", "--scenario", scenario])
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert report["weather_assumption"]["condition"] == scenario
    assert report["weather_assumption"]["change_probability"] == 0
    assert report["folds"][0]["round"] == 2


def test_cli_components_are_opt_in_and_report_all_variants(fixture, monkeypatch, capsys):
    loader, _, _, _ = fixture
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_qualifying_pace.py"
    spec = importlib.util.spec_from_file_location("pace_components_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CurrentSeasonDataLoader", lambda **kwargs: loader)
    monkeypatch.setattr(sys, "argv", [str(path), "--race", "2", "--components"])
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert set(report["folds"][0]["components"]) == {
        "assumptions", "constructor_prior", "team_form", "full_model", "recent_team_q1",
    }


def test_cli_default_report_has_no_component_diagnostic(fixture, monkeypatch, capsys):
    loader, _, _, _ = fixture
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_qualifying_pace.py"
    spec = importlib.util.spec_from_file_location("pace_default_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CurrentSeasonDataLoader", lambda **kwargs: loader)
    monkeypatch.setattr(sys, "argv", [str(path), "--race", "2"])
    module.main()
    report = json.loads(capsys.readouterr().out)
    assert "components" not in report["folds"][0]
    assert "components" not in report["aggregate"]


def test_component_forecasts_allow_an_empty_completed_season(fixture):
    loader, results, _, calls = fixture
    results.clear()
    report = evaluate_qualifying_pace(loader, YEAR, include_components=True)
    assert report["folds"] == []
    assert calls == []
    for name in ("constructor_prior", "team_form", "full_model", "recent_team_q1"):
        aggregate = report["aggregate"]["components"][name]
        assert aggregate["model"]["scored_folds"] == 0
        assert aggregate["model"]["rank_mae"] is None
        assert aggregate["teammate_gaps"]["teammate_pairs"] == 0
    json.dumps(report, allow_nan=False)


def test_cli_failure_prints_no_partial_json(fixture, monkeypatch, capsys):
    loader, _, qualifying, _ = fixture
    qualifying.clear()
    path = Path(__file__).resolve().parents[1] / "examples" / "evaluate_qualifying_pace.py"
    spec = importlib.util.spec_from_file_location("pace_failure_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CurrentSeasonDataLoader", lambda **kwargs: loader)
    monkeypatch.setattr(sys, "argv", [str(path), "--race", "2"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Evaluation failed" in captured.err
