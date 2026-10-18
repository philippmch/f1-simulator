"""Strict causal inputs for the frozen after-qualifying inference schema."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from f1sim.analysis.learned_afterqual import AfterQualifyingFeatures, canonical_digest
from f1sim.data.confirmed_grid_forecast import (
    build_confirmed_grid_features,
    build_feature_snapshot,
    validate_confirmed_grid_feature_snapshot,
)
from f1sim.data.current import CurrentSeasonDataLoader
from f1sim.data.qualifying import QUALIFYING_CLASSIFICATION
from f1sim.models import Driver

INSTANT = datetime(2026, 10, 16, 12, tzinfo=timezone.utc)


def predictors(number=4):
    return {"year": 2026, "round": number, "roster": [
        {"driver": identity, "code": code, "team": "team_x" if i < 2 else "team_y",
         "qualifying_position": i + 1, "q1": 90. + i,
         "q2": 89. + i if i < 3 else None, "q3": 88. + i if i < 2 else None,
         "grid_position": [2, 1, 4, 3][i], "official_table_order_index": [2, 1, 4, 3][i],
         "pit_lane_start": i == 2}
        for i, (identity, code) in enumerate(zip("abcd", ("AA", "BB", "CC", "DD"), strict=True))]}


def history(number=1, *, retired=False, year=2026):
    return {"year": year, "round": number, "rows": [
        {"driver": identity, "team": "team_x" if i < 2 else "team_y",
         "position": i + 1, "qualifying_position": i + 1,
         "status": "Retired" if retired and i < 2 else "+1 Lap" if i == 3 else "Finished"}
        for i, identity in enumerate("abcd")]}


def test_exact_stage_units_grid_pits_and_last_three_nonmissing_history():
    target = predictors(6)
    earlier = [history(i, retired=i >= 3) for i in range(1, 6)]
    original = deepcopy((target, earlier))
    result = build_feature_snapshot(target, earlier)
    assert result["ids"] == list("abcd")
    assert result["codes"] == ["AA", "BB", "CC", "DD"]
    assert result["grid_ids"] == list("badc")
    a, c = result["x"][0], result["x"][2]
    assert a[0] == 1 - 1 / 3 and a[1] == 1
    assert a[3] == 100 * (91.5 - 90.) / 91.5
    assert a[9] == 100 * (88.5 - 88.) / 88.5
    assert c[2] == 1 and c[8] == 0 and c[5] == 0
    # Three newest team races are entirely retired; use older nonmissing
    # race levels, rather than replacing the frozen history by a3GP window.
    assert a[11] == pytest.approx((5 / 6 + 5 / 6 + 1) / 4)
    assert result["risk"][0] == [1., (3 + .3) / 5, (3 + .3) / 5]
    assert result["tree_x"][0] == a + [a[1] - a[0]]
    assert (target, earlier) == original
    AfterQualifyingFeatures.from_mapping(result)


def test_season_and_seat_histories_reset_and_target_labels_are_ignored():
    target = predictors(1)
    target["labels"] = {"a": "arbitrary future outcome"}
    result = build_feature_snapshot(target, [history(24, year=2025)])
    assert all(row[11:15] == [.5, .5, 0., 0.] for row in result["x"])
    assert all(row == [1., .15, .15] for row in result["risk"])
    target = predictors(2)
    target["roster"][0]["team"] = "team_y"
    result = build_feature_snapshot(target, [history(1)])
    assert result["x"][0][13:15] == [0., 0.]
    assert result["risk"][0][2] == .15


@pytest.mark.parametrize("earlier", [history(4), history(5), history(1, year=2027)])
def test_target_or_future_history_is_rejected(earlier):
    with pytest.raises(ValueError, match="strictly earlier"):
        build_feature_snapshot(predictors(), [earlier])


def test_missing_qualifying_is_explicit_and_history_requires_complete_race():
    target = predictors()
    for row in target["roster"]:
        row.update(q1=None, q2=None, q3=None)
    result = build_feature_snapshot(target, [])
    assert all(row[3:11] == [0.] * 8 for row in result["x"])
    invalid = history()
    invalid["rows"][1]["position"] = 1
    with pytest.raises(ValueError, match="permutation"):
        build_feature_snapshot(target, [invalid])


class Loader:
    def __init__(self):
        self.drivers = [Driver(id=row["code"], name=row["driver"], team_id=row["team"])
                        for row in predictors()["roster"]]
        self.grid, self.pits = ["BB", "AA", "DD", "CC"], ["CC"]
        qualifying = INSTANT - timedelta(days=1)
        race = INSTANT + timedelta(days=1)
        self.event = {"round": 4, "race": "Test Grand Prix", "circuit_id": "test",
            "date": race.date().isoformat(), "time": race.timetz().isoformat(),
            "sessions": {"Qualifying": {"date": qualifying.date().isoformat(),
                                        "time": qualifying.timetz().isoformat()}}}
        self.calendar = [{"round": i,
                          "date": (INSTANT - timedelta(days=7 * (4 - i))).date().isoformat(),
                          "time": "12:00:00Z"} for i in range(1, 4)] + [self.event]
        self.results, self.qualifying = [], []
        for number in range(1, 5):
            for i, row in enumerate(predictors()["roster"]):
                raw = {"round": number, "Driver": {"driverId": row["driver"], "code": row["code"]},
                    "Constructor": {"constructorId": row["team"], "name": row["team"]},
                    "position": str(i + 1), "status": "Finished"}
                if number < 4:
                    self.results.append(deepcopy(raw))
                raw.update(Q1=f"1:{30 + i:02d}.000", Q2=None, Q3=None)
                self.qualifying.append(raw)
        self.provenance = {"fetched_at": (INSTANT - timedelta(minutes=1)).isoformat(),
            "race_grid": {"mode": "published", "year": 2026, "round": 4,
                "starting_grid": self.grid[:], "pit_lane_starters": self.pits[:],
                "source_url": "https://www.formula1.com/en/results/2026/races/1234/test/starting-grid",
                "fetched_at": (INSTANT - timedelta(minutes=1)).isoformat(),
                "qualifying_started_at": qualifying.isoformat()}}

    def _assert_current_year(self, year):
        assert year == 2026

    def _event_for_race(self, year, target):
        assert target == 4
        return deepcopy(self.event)

    def _season_data(self, year):
        return deepcopy(self.results), deepcopy(self.qualifying)

    def _supplement_target_qualifying_times(self, year, event, rows):
        return deepcopy(rows)

    def get_event_schedule(self, year):
        return deepcopy(self.calendar)

    def get_provenance(self):
        return deepcopy(self.provenance)


def build(loader, **kwargs):
    return build_confirmed_grid_features(loader, 2026, 4, loader.drivers,
        loader.grid, loader.pits, **({"as_of": INSTANT} | kwargs))


def classified_loader():
    loader = Loader()
    rows = loader.qualifying[-4:]
    rows[3]["position"] = "3"  # Ambiguous raw source, independent of the grid.
    body = "<h1>2026 TEST GRAND PRIX - QUALIFYING</h1><table><tr>" + "".join(
        f"<th>{key}</th>" for key in ("Pos.", "No.", "Driver", "Team", "Q1", "Q2", "Q3", "Laps")
    ) + "</tr>"
    for index, row in enumerate(rows):
        body += "<tr>" + "".join(f"<td>{value}</td>" for value in (
            "DQ" if index == 2 else str(index + 1 if index < 2 else 3), str(index + 1),
            "Driver " + row["Driver"]["code"], row["Constructor"]["name"], row["Q1"], "", "", "5",
        )) + "</tr>"
    body += "</table>"
    url = "https://www.formula1.com/en/results/2026/races/1234/test/qualifying"
    corrected = CurrentSeasonDataLoader._parse_official_qualifying_classification(
        body, year=2026, event=loader.event, rows=rows, source_url=url,
    )
    loader.qualifying[-4:] = corrected
    source = corrected[0][QUALIFYING_CLASSIFICATION]["source"]
    loader.provenance["qualifying_timing_supplements"] = [{
        **source, "status": "available", "fetched_at": (INSTANT - timedelta(minutes=1)).isoformat(),
    }]
    return loader


def test_primary_dq_classification_keeps_grid_pit_queue_and_replays_saved_evidence():
    snapshot = build(classified_loader())
    dq = snapshot["codes"].index("CC")
    assert snapshot["grid_ids"] == list("badc")
    assert snapshot["x"][dq][2] == 1
    assert snapshot["x"][dq][1] == 0
    assert snapshot["x"][dq][3:9] == [0.] * 6
    source = snapshot["context"]["target_predictors"]["roster"][dq]
    assert source["qualifying_position"] == 3 and source["q1"] == 92.
    assert validate_confirmed_grid_feature_snapshot(snapshot) == snapshot


@pytest.mark.parametrize("change", [
    "receipt", "receipt_hash", "receipt_future", "partial", "status",
])
def test_primary_saved_qualifying_source_mutations_are_rejected(change):
    snapshot = build(classified_loader())
    context = snapshot["context"]
    roster = context["target_predictors"]["roster"]
    if change == "receipt":
        context["source_provenance"].pop("qualifying_timing_supplements")
    elif change == "receipt_hash":
        receipt = context["source_provenance"]["qualifying_timing_supplements"][0]
        receipt["content_sha256"] = "1" * 64
    elif change == "receipt_future":
        context["source_provenance"]["qualifying_timing_supplements"][0]["fetched_at"] = (
            INSTANT + timedelta(seconds=1)).isoformat()
    elif change == "partial":
        roster[0].pop(QUALIFYING_CLASSIFICATION)
    else:
        roster[2][QUALIFYING_CLASSIFICATION]["status"] = "classified"
        roster[2][QUALIFYING_CLASSIFICATION]["position"] = 3
    context["target_predictors_sha256"] = canonical_digest(context["target_predictors"])
    snapshot["content_sha256"] = canonical_digest({
        key: value for key, value in snapshot.items() if key != "content_sha256"})
    with pytest.raises(ValueError):
        validate_confirmed_grid_feature_snapshot(snapshot)


def test_live_snapshot_detached_ordered_sealed_and_keeps_whole_previous_field():
    loader = Loader()
    # One older driver was replaced; the original whole-field denominator
    # remains part of that constructor's history and is not filtered away.
    loader.results[0]["Driver"] = {"driverId": "old_a", "code": "OA"}
    loader.qualifying[0]["Driver"] = {"driverId": "old_a", "code": "OA"}
    snapshot = build(loader)
    assert snapshot["ids"] == list("abcd") and snapshot["grid_ids"] == list("badc")
    assert snapshot["as_of"] == INSTANT.isoformat()
    assert snapshot["history_event_keys"] == [[2026, 1], [2026, 2], [2026, 3]]
    assert len(snapshot["context"]["completed_history"][0]["rows"]) == 4
    assert snapshot["context"]["completed_history"][0]["rows"][-1]["driver"] == "old_a"
    assert snapshot["context"]["target_race_performance_used"] is False
    assert snapshot["content_sha256"] == canonical_digest(
        {k: v for k, v in snapshot.items() if k != "content_sha256"})
    AfterQualifyingFeatures.from_mapping(snapshot)
    before = deepcopy(snapshot)
    loader.results[0]["position"] = "99"
    loader.provenance["race_grid"]["starting_grid"].reverse()
    assert snapshot == before


@pytest.mark.parametrize("failure", ["target_race", "later_qualifying", "missing_entrant",
    "duplicate_code", "duplicate_provider", "team", "qualifying_rank", "wrong_grid_source",
    "wrong_pit_queue", "future_fetch", "future_timing", "future_history_date", "partial_history"])
def test_live_invalid_or_leaking_sources_are_rejected(failure):
    loader = Loader()
    target = loader.qualifying[-4:]
    if failure == "target_race":
        loader.results.append({**deepcopy(loader.results[0]), "round": 4})
    elif failure == "later_qualifying":
        loader.qualifying.append({**deepcopy(target[0]), "round": 5})
    elif failure == "missing_entrant":
        loader.qualifying.pop()
    elif failure == "duplicate_code":
        target[1]["Driver"]["code"] = "AA"
    elif failure == "duplicate_provider":
        target[1]["Driver"]["driverId"] = "a"
    elif failure == "team":
        target[0]["Constructor"] = {"constructorId": "team_y", "name": "team_y"}
    elif failure == "qualifying_rank":
        target[0]["position"] = "2"
    elif failure == "wrong_grid_source":
        loader.provenance["race_grid"]["round"] = 3
    elif failure == "wrong_pit_queue":
        loader.provenance["race_grid"]["pit_lane_starters"] = []
    elif failure == "future_fetch":
        loader.provenance["fetched_at"] = (INSTANT + timedelta(seconds=1)).isoformat()
    elif failure == "future_timing":
        loader.provenance["qualifying_timing_supplements"] = [
            {"fetched_at": (INSTANT + timedelta(seconds=1)).isoformat()}]
    elif failure == "future_history_date":
        loader.calendar[0].update(date=loader.event["date"], time=loader.event["time"])
    elif failure == "partial_history":
        loader.results.pop(0)
    with pytest.raises(ValueError):
        build(loader)


@pytest.mark.parametrize("instant", [INSTANT - timedelta(days=2), INSTANT + timedelta(days=1),
                                     INSTANT.replace(tzinfo=None)])
def test_qualifying_and_race_deadlines_are_strict(instant):
    with pytest.raises(ValueError):
        build(Loader(), as_of=instant)


def test_default_cutoff_is_frozen_after_required_source_reads(monkeypatch):
    loader = Loader()
    read = []
    original = loader._season_data

    def fetch(year):
        read.append(True)
        loader.provenance["fetched_at"] = INSTANT.isoformat()
        return original(year)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            assert read
            return INSTANT + timedelta(seconds=1)

    monkeypatch.setattr(loader, "_season_data", fetch)
    monkeypatch.setattr("f1sim.data.confirmed_grid_forecast.datetime", Clock)
    snapshot = build_confirmed_grid_features(loader, 2026, 4, loader.drivers,
                                            loader.grid, loader.pits)
    assert snapshot["as_of"] == (INSTANT + timedelta(seconds=1)).isoformat()


@pytest.mark.parametrize("change", ["matrix", "grid", "source_digest", "target_history",
    "future_calendar", "future_source", "seat", "cutoff", "invalid_seal"])
def test_saved_feature_evidence_is_rebuilt_and_resealed_poisoning_rejected(change):
    snapshot = build(Loader())
    if change == "matrix":
        snapshot["x"][0][11] += .01
        snapshot["tree_x"][0][11] += .01
    elif change == "grid":
        snapshot["grid_ids"].reverse()
    elif change == "source_digest":
        snapshot["context"]["target_predictors"]["roster"][0]["q1"] += 1.
    elif change == "target_history":
        snapshot["context"]["completed_history"][-1]["round"] = snapshot["round"]
        snapshot["context"]["completed_history_sha256"] = canonical_digest(
            snapshot["context"]["completed_history"])
    elif change == "future_calendar":
        snapshot["context"]["prior_race_starts_at"]["1"] = snapshot["context"]["race_starts_at"]
    elif change == "future_source":
        snapshot["context"]["source_provenance"]["fetched_at"] = (
            INSTANT + timedelta(seconds=1)).isoformat()
    elif change == "seat":
        snapshot["context"]["modeled_seats"]["AA"] = "team_y"
    elif change == "cutoff":
        snapshot["as_of"] = (INSTANT + timedelta(seconds=1)).isoformat()
    elif change == "invalid_seal":
        snapshot["as_of"] = (INSTANT + timedelta(seconds=1)).isoformat()
        with pytest.raises(ValueError, match="seal"):
            validate_confirmed_grid_feature_snapshot(snapshot)
        return
    snapshot["content_sha256"] = canonical_digest(
        {k: v for k, v in snapshot.items() if k != "content_sha256"})
    with pytest.raises(ValueError):
        validate_confirmed_grid_feature_snapshot(snapshot)


def test_offline_feature_rebuild_uses_no_loader_and_returns_a_detached_copy():
    snapshot = build(Loader())
    validated = validate_confirmed_grid_feature_snapshot(snapshot)
    assert validated == snapshot and validated is not snapshot
    validated["context"]["completed_history"][0]["rows"].clear()
    assert len(snapshot["context"]["completed_history"][0]["rows"]) == 4
    normalized = AfterQualifyingFeatures.from_mapping(snapshot).to_dict()
    assert "content_sha256" not in normalized
    assert validate_confirmed_grid_feature_snapshot(normalized) == normalized
