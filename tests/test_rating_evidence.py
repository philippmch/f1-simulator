"""Missing evidence and qualifying progression cannot manufacture rating gaps."""

import copy
from datetime import datetime, timezone

import pytest

from f1sim.data import CurrentSeasonDataLoader

YEAR = datetime.now(timezone.utc).year


def fixture():
    roster = [{"id": f"{team}{i}", "name": f"Driver {team}{i}", "team_name": team}
              for team in "ABCDEFGH" for i in (1, 2)]
    rows = [{"round": 1, "Driver": {"code": d["id"], "driverId": d["id"]},
             "Constructor": {"constructorId": d["team_name"], "name": d["team_name"]},
             "Q1": 90, "FastestLap": {"AverageSpeed": {"speed": 200}}}
            for d in roster]
    return dict(year=YEAR, target_event={"round": 2}, roster=roster,
                driver_standings=[], constructor_standings=[
                    {"Constructor": {"constructorId": t, "name": t},
                     "points": {"A": 100, "B": 90}.get(t, 0)} for t in "ABCDEFGH"],
                race_rows=[], quali_rows=rows, target_qualifying_rows=[],
                track_weight=0, form_weight=0, quali_weight=1)


def build(inputs):
    def no_network(*args, **kwargs):
        pytest.fail("Pure model assembly must not fetch")
    return CurrentSeasonDataLoader(current_year=YEAR, http_getter=no_network)._build_driver_stats(
        **inputs,
    )


def teams(stats):
    return {key: value.team_pace_rating for key, value in stats.items()}


@pytest.mark.parametrize("bucket,weight", [("quali_rows", "quali_weight"),
                                          ("target_qualifying_rows", "track_weight"),
                                          ("race_rows", "form_weight")])
def test_missing_neutral_team_evidence_does_not_rescale_constructor_anchor(bucket, weight):
    data = fixture()
    rows = data.pop("quali_rows")
    data["quali_rows"] = []
    data[bucket] = rows
    if bucket == "target_qualifying_rows":
        for row in rows:
            row["round"] = 2
    data.update(track_weight=0, form_weight=0, quali_weight=0)
    data[weight] = 1
    complete = build(data)
    data[bucket] = [r for r in rows if r["Constructor"]["name"] != "B"]
    missing = build(data)
    assert teams(complete) == teams(missing)
    assert complete["A1"].team_pace_rating == 1
    assert complete["B1"].team_pace_rating == pytest.approx(.976)


def test_zero_weight_signal_cannot_change_constructor_anchor():
    data = fixture()
    data["quali_weight"] = 0
    initial = teams(build(data))
    data["quali_rows"][2]["Q1"] = 60
    assert teams(build(data)) == initial


def test_fully_observed_signed_formula_matches_previous_common_denominator():
    data = fixture()
    for row in data["quali_rows"]:
        row["Q1"] = {"A": 89, "B": 92}.get(row["Constructor"]["name"], 90)
    stats = build(data)
    old_a = (1 + 1 / 90) / 2
    old_b = (.9 - 2 / 90) / 2
    assert stats["B1"].team_pace_rating == pytest.approx(.76 + old_b / old_a * .24)
    assert stats["B1"].team_pace_rating < .976


def test_later_uniform_sessions_do_not_create_team_or_teammate_gaps():
    data = fixture()
    baseline = build(data)
    data["quali_rows"][0].update(Q2=88, Q3=86)
    data["quali_rows"][1].update(Q2=88)
    changed = build(data)
    assert teams(changed) == teams(baseline)
    assert changed["A1"].driver_skill_rating == changed["A2"].driver_skill_rating == .92


def test_latest_shared_session_preserves_real_driver_gap_and_q1_team_reference():
    data = fixture()
    data["quali_rows"][0].update(Q2=88, Q3=70)
    data["quali_rows"][1].update(Q2=89)
    stats = build(data)
    assert stats["A1"].driver_skill_rating == pytest.approx(.92 + .5 / 88.5 * 8)
    assert stats["A2"].driver_skill_rating == pytest.approx(.92 - .5 / 88.5 * 8)
    assert stats["B1"].team_pace_rating == pytest.approx(.976)


@pytest.mark.parametrize("session", ["Q2", "Q3"])
def test_broad_later_session_is_used_when_earlier_session_unavailable(session):
    data = fixture()
    for row in data["quali_rows"]:
        row[session] = row.pop("Q1")
    data["quali_rows"][2][session] = data["quali_rows"][3][session] = 92
    assert build(data)["B1"].team_pace_rating < .976


def test_sparse_later_session_does_not_create_team_signal():
    data = fixture()
    for row in data["quali_rows"]:
        del row["Q1"]
    data["quali_rows"][0]["Q3"] = 70
    data["quali_rows"][1]["Q3"] = 70
    assert build(data)["B1"].team_pace_rating == pytest.approx(.976)


def test_singleton_and_unknown_session_times_do_not_invent_driver_comparisons():
    data = fixture()
    data["quali_rows"][0]["Q1"] = 80
    del data["quali_rows"][1]["Q1"]
    data["quali_rows"][1]["bestTime"] = 80
    stats = build(data)
    assert stats["A1"].driver_skill_rating == stats["A2"].driver_skill_rating == .9
    assert stats["A2"].qualifying_samples == 1


@pytest.mark.parametrize("value", [True, float("inf"), float("nan"), -1, 0,
                                  {"time": True}, "Infinity"])
def test_invalid_session_times_are_omitted(value):
    data = fixture()
    data["quali_rows"][0]["Q1"] = value
    stats = build(data)
    assert stats["A1"].driver_skill_rating == stats["A2"].driver_skill_rating == .9
    assert stats["B1"].team_pace_rating == pytest.approx(.976)


def test_input_order_and_lowercase_session_keys_do_not_change_ratings():
    data = fixture()
    data["quali_rows"][0].update(q2=88, q3=87)
    data["quali_rows"][1].update(q2=89, q3=88)
    before = copy.deepcopy(data)
    baseline = build(data)
    reordered = copy.deepcopy(data)
    for key in ("roster", "constructor_standings", "quali_rows"):
        reordered[key].reverse()
    changed = build(reordered)
    assert teams(changed) == teams(baseline)
    assert {d: s.driver_skill_rating for d, s in changed.items()} == {
        d: s.driver_skill_rating for d, s in baseline.items()
    }
    assert data == before
