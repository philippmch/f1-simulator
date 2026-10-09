"""Regression checks for live post-qualifying timing units and signal weights."""

from copy import deepcopy

import pytest

from f1sim.data.current import CurrentSeasonDataLoader


def inputs():
    roster = [{"id": f"{team}{seat}", "code": f"{team}{seat}",
               "name": f"Driver {team}{seat}", "team_name": f"Team {team}"}
              for team in ("A", "B", "C") for seat in (1, 2)]
    qualifying = [{"round": 3, "Driver": {"code": d["id"]},
                   "Constructor": {"name": d["team_name"]}, "Q1": "1:40.000"}
                  for d in roster]
    races = [{"round": 1, "Driver": {"code": d["id"]},
              "Constructor": {"name": d["team_name"]}, "status": "Finished",
              "FastestLap": {"Time": {"time": str(t)}}}
             for d, t in zip(roster, (80, 84, 90, 94, 100, 104), strict=True)]
    return roster, qualifying, races


def build(roster, qualifying, races, *, standings=(), track_weight=0.0, form_weight=1.0):
    loader = CurrentSeasonDataLoader(current_year=2026)
    return loader._build_driver_stats(
        year=2026, target_event={"round": 3, "circuit_id": "monza"}, roster=roster,
        driver_standings=[], constructor_standings=list(standings), race_rows=races,
        quali_rows=[], target_qualifying_rows=qualifying,
        track_weight=track_weight, form_weight=form_weight, quali_weight=0.0,
    )


def assert_same_stats(a, b):
    assert set(a) == set(b)
    for code in a:
        for key, value in a[code].model_dump().items():
            expected = b[code].model_dump()[key]
            if isinstance(value, float):
                assert value == pytest.approx(expected)
            else:
                assert value == expected


def test_available_lap_times_restore_team_and_teammate_form_after_qualifying():
    roster, qualifying, races = inputs()
    stats = build(roster, qualifying, races)
    assert (
        stats["A1"].team_pace_rating > stats["B1"].team_pace_rating > stats["C1"].team_pace_rating
    )
    assert stats["A1"].driver_skill_rating > stats["A2"].driver_skill_rating


def test_time_fallback_matches_event_relative_speeds_across_different_circuits():
    roster, qualifying, races = inputs()
    second = deepcopy(races)
    for row in second:
        row["round"] = "2"
        row["FastestLap"]["Time"]["time"] = str(float(row["FastestLap"]["Time"]["time"])*1.3)
    timed = races+second
    measured = deepcopy(timed)
    for row in measured:
        distance = 5.0 if int(row["round"]) == 1 else 3.0
        time = float(row["FastestLap"]["Time"]["time"])
        row["FastestLap"]["AverageSpeed"] = {"speed": str(3600*distance/time)}
    mixed = deepcopy(timed)
    for i, row in enumerate(measured[:len(races)]):
        mixed[i] = row
    expected = build(roster, qualifying, measured)
    assert_same_stats(build(roster, qualifying, timed), expected)
    assert_same_stats(build(roster, qualifying, mixed), expected)


def test_partial_speed_event_never_mixes_inverse_seconds_with_supplied_speed():
    roster, qualifying, races = inputs()
    races[0]["FastestLap"]["AverageSpeed"] = {"speed": "225"}
    without_times = deepcopy(races)
    for row in without_times:
        row["FastestLap"].pop("Time")
    assert_same_stats(build(roster, qualifying, races), build(roster, qualifying, without_times))


@pytest.mark.parametrize("invalid", [None, True, "nan", "inf", "-1", "0", "bad"])
def test_invalid_lap_times_do_not_create_form(invalid):
    roster, qualifying, races = inputs()
    for row in races:
        row["FastestLap"]["Time"]["time"] = invalid
    without_times = deepcopy(races)
    for row in without_times:
        row["FastestLap"].pop("Time")
    assert_same_stats(build(roster, qualifying, races), build(roster, qualifying, without_times))


def test_time_fallback_is_not_applied_before_qualifying():
    roster, _, races = inputs()
    without_times = deepcopy(races)
    for row in without_times:
        row["FastestLap"].pop("Time")
    assert_same_stats(build(roster, [], races), build(roster, [], without_times))


def test_observed_qualifying_can_overcome_a_small_constructor_points_gap():
    roster, qualifying, _ = inputs()
    for row in qualifying:
        row["Q1"] = {"A": "1:40.000", "B": "1:39.000", "C": "1:42.000"}[row["Driver"]["code"][0]]
    standings = [{"Constructor": {"name": f"Team {team}"}, "points": points}
                 for team, points in (("A", 100), ("B", 95), ("C", 20))]
    observed = build(roster, qualifying, [], standings=standings, track_weight=1.0, form_weight=0.0)
    disabled = build(roster, qualifying, [], standings=standings, track_weight=0.0, form_weight=0.0)
    assert observed["B1"].team_pace_rating > observed["A1"].team_pace_rating
    assert disabled["A1"].team_pace_rating > disabled["B1"].team_pace_rating
