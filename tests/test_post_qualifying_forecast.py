"""Post-qualifying records freeze the published grid and score only future racing."""

from copy import deepcopy
from datetime import timedelta

import pytest
from test_recorded_forecast import AllocationForecastLoader, result

from f1sim.analysis.grid_winner_reference import score_saved_grid_references
from f1sim.analysis.recorded_forecast import (
    _digest,
    load_recorded_forecast,
    record_race_forecast,
    save_recorded_forecast,
    score_recorded_forecast,
    validate_recorded_forecast,
)
from f1sim.models import Driver


class PostQualifyingLoader(AllocationForecastLoader):
    def __init__(self, *, pit=False):
        super().__init__()
        qualifying = self.instant - timedelta(hours=3)
        race = self.instant + timedelta(days=1)
        self.event.update(date=race.date().isoformat(), time=race.time().isoformat() + "Z")
        self.event["sessions"]["Qualifying"] = {
            "date": qualifying.date().isoformat(), "time": qualifying.time().isoformat() + "Z",
        }
        self.event["sprint"] = True
        sprint_qualifying = qualifying - timedelta(days=1)
        self.event["sessions"]["SprintQualifying"] = {
            "date": sprint_qualifying.date().isoformat(),
            "time": sprint_qualifying.time().isoformat() + "Z",
        }
        self.qualifying_rows = [result(identity, i, qualifying=True)
                                for i, identity in enumerate(("BB", "AA"), 1)]
        for row in self.qualifying_rows:
            row["round"] = 3
        self.pit = pit

    def get_starting_grid(self, year, race, drivers):
        session = self.event["sessions"]["Qualifying"]
        self._race_grid = {
            "mode": "published", "year": year, "round": 3,
            "starting_grid": ["BB", "AA"], "pit_lane_starters": ["AA"] if self.pit else [],
            "source_url": f"https://www.formula1.com/en/results/{year}/races/1234/test/starting-grid",
            "fetched_at": self.instant.isoformat(),
            "qualifying_started_at": f'{session["date"]}T{session["time"]}',
        }
        return ["BB", "AA"]


def record(loader, **kwargs):
    return record_race_forecast(loader, loader.instant.year, 3, trials=2,
                                now=lambda: loader.instant, stage="post_qualifying", **kwargs)


@pytest.mark.parametrize("pit", [False, True])
def test_post_qualifying_round_trip_scores_frozen_references_without_scoring_observed_qualifying(
    tmp_path, monkeypatch, pit,
):
    loader = PostQualifyingLoader(pit=pit)
    saved = record(loader)
    assert saved["schema_version"] == 4
    assert saved["kind"] == "post_qualifying_race_forecast"
    assert saved["target_performance_used"] is True
    assert saved["target_race_performance_used"] is False
    assert saved["performance_rounds"] == [1, 2]
    assert saved["qualifying_observation_round"] == 3
    assert saved["simulation_inputs"]["starting_grid"] == ["BB", "AA"]
    assert saved["simulation_inputs"]["schema_version"] == (15 if pit else 14)
    assert "winner_estimate" not in saved
    assert saved["simulation_inputs"].get("winner_allocation") is None
    path = save_recorded_forecast(tmp_path / "after-grid.json", saved)
    assert load_recorded_forecast(path) == saved
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_recorded_forecast(path, saved)
    monkeypatch.setattr(loader, "get_starting_grid", lambda *a: pytest.fail("No updated grid"))
    monkeypatch.setattr(loader, "get_winner_allocation", lambda *a: pytest.fail("No refitting"))
    monkeypatch.setattr("f1sim.analysis.montecarlo.MonteCarloRunner.run",
                        lambda *a, **k: pytest.fail("No simulation while scoring"))
    observations = [result(identity, i) for i, identity in enumerate(("AA", "BB"), 1)]
    for row in observations:
        row["round"] = 3
    score = score_recorded_forecast(saved, loader, observations, loader.qualifying_rows)
    assert score["forecast_stage"] == "post_qualifying"
    assert score["observed_pole"] is None and score["pole_score"] is None
    assert score["qualifying_position_mae"] is None
    assert score["qualifying_scored_drivers"] == 0
    assert len(score["grid_references"]["references"]) == 3
    assert len(score["winner_error_comparisons"]) == 3
    assert path.read_bytes() == before


def test_new_post_qualifying_records_do_not_redistribute_grid_chances_to_points(monkeypatch):
    loader = PostQualifyingLoader()
    monkeypatch.setattr(loader, "get_winner_allocation",
                        lambda *a: pytest.fail("Confirmed-grid chances redistributed to points"))
    saved = record(loader)
    assert "winner_estimate" not in saved
    assert saved["simulation_inputs"].get("winner_allocation") is None


def test_saved_post_qualifying_point_policy_still_scores_without_refitting(monkeypatch):
    from f1sim.analysis.teammate_forecast import teammate_winner_forecast

    loader = PostQualifyingLoader()
    saved = record(loader)
    allocation = loader.get_winner_allocation(loader.instant.year, 3,
        [Driver.model_validate(d) for d in saved["simulation_inputs"]["drivers"]])
    saved["winner_estimate"] = teammate_winner_forecast(saved["winner_forecast"], allocation)
    saved["simulation_inputs"]["winner_allocation"] = deepcopy(allocation)
    saved["content_sha256"] = _digest({k: v for k, v in saved.items() if k != "content_sha256"})
    before = deepcopy(saved)
    monkeypatch.setattr(loader, "get_winner_allocation", lambda *a: pytest.fail("No refitting"))
    observations = [result(identity, i) for i, identity in enumerate(("AA", "BB"), 1)]
    for row in observations:
        row["round"] = 3
    score = score_recorded_forecast(saved, loader, observations, loader.qualifying_rows)
    assert score["winner_policy"] == saved["winner_estimate"]["policy"]
    assert "native_winner_score" in score
    assert saved == before


@pytest.mark.parametrize("change", ["race_result", "later_qualifying", "before_gp", "missing_grid",
                                   "no_race_time", "unfinished_gp"])
def test_unavailable_or_leaking_post_qualifying_record_fails_before_trials(monkeypatch, change):
    loader = PostQualifyingLoader()
    if change == "race_result":
        row = result("AA", 1)
        row["round"] = 3
        loader.rows.append(row)
    elif change == "later_qualifying":
        row = result("AA", 1, qualifying=True)
        row["round"] = 4
        loader.qualifying_rows.append(row)
    elif change == "missing_grid":
        loader.get_starting_grid = lambda *a: None
    elif change == "no_race_time":
        del loader.event["time"]
    else:
        gp = loader.instant + timedelta(hours=1) if change == "before_gp" else loader.instant
        loader.event["sessions"]["Qualifying"] = {
            "date": gp.date().isoformat(), "time": gp.time().isoformat() + "Z",
        }
    monkeypatch.setattr("f1sim.analysis.montecarlo.MonteCarloRunner.run",
                        lambda *a, **k: pytest.fail("Invalid record reached a trial"))
    with pytest.raises(ValueError):
        record(loader)


def test_finishing_after_race_start_does_not_produce_a_valid_record():
    loader = PostQualifyingLoader()
    times = iter([loader.instant, loader.instant, loader.instant + timedelta(days=2)])
    with pytest.raises(ValueError, match="race started"):
        record_race_forecast(loader, loader.instant.year, 3, trials=1,
                             now=lambda: next(times), stage="post_qualifying")


@pytest.mark.parametrize("change", ["probability", "grid", "pit", "kind", "target_result",
                                   "cutoff", "old_fetch", "qualifying_round", "schema",
                                   "race_date", "race_time", "diagnostic_scope"])
def test_resealing_cannot_hide_inconsistent_grid_probabilities_or_information_boundaries(change):
    loader = PostQualifyingLoader(pit=True)
    saved = record(loader)
    if change == "probability":
        reference = saved["grid_references"]["references"]["grid_rank_softmax_18_v1"]
        reference["probabilities"]["AA"] = .1
    elif change == "grid":
        saved["simulation_inputs"]["starting_grid"].reverse()
    elif change == "pit":
        saved["simulation_inputs"]["pit_lane_starters"] = []
    elif change == "kind":
        saved["kind"] = "pre_qualifying_race_forecast"
    elif change == "target_result":
        saved["target_race_performance_used"] = True
    elif change == "cutoff":
        saved["performance_rounds"].append(3)
    elif change == "old_fetch":
        evidence = saved["grid_references"]["evidence"]
        evidence["fetched_at"] = (loader.instant - timedelta(minutes=1)).isoformat()
        saved["provenance"]["race_grid"] = deepcopy(evidence)
    elif change == "qualifying_round":
        saved["qualifying_observation_round"] = 2
    elif change == "race_date":
        saved["event"]["date"] = (loader.instant + timedelta(days=2)).date().isoformat()
    elif change == "race_time":
        del saved["event"]["time"]
    elif change == "diagnostic_scope":
        saved["qualifying_forecast_scope"] = "independent_qualifying_forecast"
    else:
        saved["schema_version"] = 3
    saved["content_sha256"] = _digest({k: v for k, v in saved.items() if k != "content_sha256"})
    with pytest.raises(ValueError):
        validate_recorded_forecast(saved)


def test_grid_reference_ignores_an_edited_score_and_keeps_original_evidence():
    loader = PostQualifyingLoader()
    saved = record(loader)
    reference = saved["grid_references"]
    reference["references"]["grid_rank_softmax_18_v1"]["score"] = {"brier_score": 999}
    before = deepcopy(reference)
    scored = score_saved_grid_references(
        reference, ["AA", "BB"], year=loader.instant.year, target_round=3,
        recorded_at=saved["recorded_at"], qualifying_starts_at=saved["qualifying_starts_at"],
        race_starts_at=saved["race_starts_at"], observed_winner="AA",
    )
    assert scored["references"]["grid_rank_softmax_18_v1"]["score"]["brier_score"] != 999
    assert reference == before
