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
    _constructor_standings,
    _metrics,
    evaluate_qualifying_pace,
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
