"""Shared model assembly keeps target labels out of holdout model inputs."""

import copy
from datetime import datetime, timezone

import pytest

from f1sim.analysis.holdout_folds import (
    HoldoutFoldInputs,
    HoldoutObservations,
    InsufficientTargetCoverage,
    assemble_holdout_fold,
)
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader

YEAR = datetime.now(timezone.utc).year


def _row(round_number, code, team, q1, position):
    return {
        "round": round_number,
        "Driver": {
            "driverId": code.lower(), "code": code,
            "givenName": code, "familyName": "Racer",
        },
        "Constructor": {"constructorId": team, "name": team},
        "Q1": q1,
        "position": str(position),
        "status": "Finished",
        "FastestLap": {"AverageSpeed": {"speed": str(230 - position)},
                       "Time": {"time": "1:00.000"}},
    }


def _standings(round_number):
    return {"MRData": {"total": "2", "offset": "0", "StandingsTable": {
        "season": str(YEAR), "round": str(round_number),
        "StandingsLists": [{"season": str(YEAR), "round": str(round_number),
        "ConstructorStandings": [
            {"Constructor": {"constructorId": "a", "name": "a"}, "points": "40"},
            {"Constructor": {"constructorId": "b", "name": "b"}, "points": "20"},
        ]}]}}}


@pytest.fixture
def fold_sources(monkeypatch):
    loader = CurrentSeasonDataLoader(current_year=YEAR, http_getter=lambda *a, **k: {})
    events = [
        {"round": n, "race": f"Race {n}", "circuit_id": f"circuit{n}", "completed": True}
        for n in (1, 2, 3)
    ]
    qualifying = [
        _row(n, code, team, f"1:{time:02}.000", pos)
        for n in (1, 2, 3)
        for pos, (code, team, time) in enumerate([
            ("A1", "a", 10), ("A2", "a", 11), ("B1", "b", 12), ("B2", "b", 13),
        ], 1)
    ]
    results = copy.deepcopy(qualifying)

    def fetch(url):
        round_number = int(url.split(f"/{YEAR}/")[1].split("/")[0])
        return _standings(round_number)

    monkeypatch.setattr(loader, "_fetch_json", fetch)
    monkeypatch.setattr(loader, "_standings", lambda *a, **k: pytest.fail("live standings"))
    monkeypatch.setattr(loader, "_round_data", lambda *a, **k: pytest.fail("live round data"))
    monkeypatch.setattr(
        loader, "get_official_roster", lambda *a, **k: pytest.fail("live roster"),
    )
    return loader, events, results, qualifying


def test_assembly_returns_identity_only_roster_and_separate_observations(
    fold_sources,
):
    loader, events, results, qualifying = fold_sources
    for row in qualifying:
        if row["round"] == 2 and row["Driver"]["code"] == "A1":
            row["Q1"] = None

    assembled, observations = assemble_holdout_fold(
        loader, YEAR, events[1], events, results, qualifying,
    )

    assert isinstance(assembled, HoldoutFoldInputs)
    assert isinstance(observations, HoldoutObservations)
    assert [row["id"] for row in assembled.roster] == ["A1", "A2", "B1", "B2"]
    assert [driver.id for driver in assembled.drivers] == ["A1", "A2", "B1", "B2"]
    assert set(assembled.cars) == {"a", "b"}
    assert assembled.track.base_lap_time == 90
    assert all(not {"Q1", "position", "FastestLap"} & row.keys()
               for row in assembled.roster)
    assert len(observations.target_qualifying_rows) == 4
    assert next(row for row in observations.target_qualifying_rows
                if row["Driver"]["code"] == "A1")["Q1"] is None
    assert assembled.metadata.training_cutoff_round == 1
    assert assembled.metadata.eligible_form_rounds == (1,)
    assert assembled.metadata.form_rounds == (1,)
    assert assembled.metadata.standings_round == assembled.metadata.baseline_round == 1
    assert assembled.metadata.coverage.qualifying_entrants == 4
    assert assembled.metadata.coverage.result_entrants == 4
    assert assembled.metadata.coverage.matched_result_entrants == 4


def test_target_and_future_performance_mutations_do_not_change_assembled_inputs(
    fold_sources,
):
    loader, events, results, qualifying = fold_sources
    before, observations_before = assemble_holdout_fold(
        loader, YEAR, events[1], events, results, qualifying,
    )

    for rows in (results, qualifying):
        for row in rows:
            if row["round"] == 2:
                row["Q1"] = "2:00.000"
                row["position"] = "22"
                row["FastestLap"] = {"AverageSpeed": {"speed": "999"},
                                     "Time": {"time": "0:01.000"}}
            elif row["round"] == 3:
                row["Q1"] = "3:00.000"
                row["position"] = "33"
                row["FastestLap"] = {"AverageSpeed": {"speed": "999"},
                                     "Time": {"time": "0:01.000"}}
    after, observations_after = assemble_holdout_fold(
        loader, YEAR, events[1], events, results, qualifying,
    )

    assert after == before
    assert observations_after != observations_before
    assert observations_before.target_qualifying_rows[0]["Q1"] != (
        observations_after.target_qualifying_rows[0]["Q1"]
    )


def test_insufficient_result_overlap_exposes_typed_coverage_context(fold_sources):
    loader, events, results, qualifying = fold_sources
    target_results = [row for row in results if row["round"] == 2]
    results[:] = [row for row in results if row is not target_results[-1]]

    with pytest.raises(InsufficientTargetCoverage) as caught:
        assemble_holdout_fold(loader, YEAR, events[1], events, results, qualifying)

    error = caught.value
    assert str(error) == "Incomplete target entrant coverage for round 2"
    assert error.target_round == 2
    assert len(error.roster) == len(error.target_qualifying_rows) == 4
    assert error.coverage.result_entrants == 3
    assert error.coverage.matched_result_entrants == 3
    assert error.coverage.expected_result_entrants == 4


def test_conflicting_target_result_aliases_are_fatal_before_coverage_exclusion(
    fold_sources,
):
    loader, events, results, qualifying = fold_sources
    target_a1 = next(row for row in results
                     if row["round"] == 2 and row["Driver"]["code"] == "A1")
    target_a1["Driver"]["givenName"] = "B2"

    with pytest.raises(CurrentSeasonDataError, match="Conflicting aliases") as caught:
        assemble_holdout_fold(loader, YEAR, events[1], events, results, qualifying)

    assert not isinstance(caught.value, InsufficientTargetCoverage)
