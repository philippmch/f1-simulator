"""Constructor-anchor selection preserves timing cutoffs and model boundaries."""
from copy import deepcopy

import pytest

from f1sim.data.current import CurrentSeasonDataLoader, _utc_now


def fixture_arguments(*, teams=3):
    roster = [{"id": f"{team}{seat}", "code": f"{team}{seat}",
               "name": f"Driver {team}{seat}", "team_name": f"Team {team}"}
              for team in range(teams) for seat in (1, 2)]
    qualifying = [{"round": 3, "Driver": {"code": row["id"]},
                   "Constructor": {"name": row["team_name"]},
                   "Q1": 80.+index*.2}
                  for index, row in enumerate(roster)]
    points = [{"Constructor": {"name": f"Team {team}"}, "points": 10.*(team+1)**2}
              for team in range(teams)]
    return {"year": _utc_now().year, "target_event": {"round": 3, "circuit_id": "monza"},
            "roster": roster, "driver_standings": [], "constructor_standings": points,
            "race_rows": [], "quali_rows": [], "target_qualifying_rows": qualifying,
            "track_weight": .5, "form_weight": .3, "quali_weight": .2}


def loader():
    def unexpected(*args, **kwargs):
        pytest.fail("Supplied policy evidence must not fetch network data")
    return CurrentSeasonDataLoader(current_year=_utc_now().year, http_getter=unexpected)


def ratings(stats):
    return {driver: row.team_pace_rating for driver, row in stats.items()}


def test_qualifying_led_pace_ignores_constructor_points_but_retains_standings():
    adapter = loader()
    arguments = fixture_arguments()
    before = deepcopy(arguments)
    old = adapter._build_driver_stats(**arguments, postqual_pace_policy="qualifying_led")
    changed = deepcopy(arguments)
    for row in changed["constructor_standings"]:
        row["points"] = 120-row["points"]
    new = adapter._build_driver_stats(**changed, postqual_pace_policy="qualifying_led")
    assert ratings(old) == ratings(new)
    assert adapter.create_drivers_from_stats(old) == adapter.create_drivers_from_stats(new)
    assert {k: v.constructor_points for k, v in old.items()} != {
        k: v.constructor_points for k, v in new.items()}
    assert arguments == before
    assert adapter._driver_stats == {}
    assert adapter._postqual_pace_policy is None


def test_legacy_default_keeps_constructor_anchor_and_driver_traits_unchanged():
    adapter = loader()
    arguments = fixture_arguments()
    default = adapter._build_driver_stats(**arguments)
    explicit = adapter._build_driver_stats(**arguments, postqual_pace_policy="season_points")
    timing = adapter._build_driver_stats(**arguments, postqual_pace_policy="qualifying_led")
    assert default == explicit
    assert ratings(default) != ratings(timing)
    assert adapter.create_drivers_from_stats(default) == adapter.create_drivers_from_stats(timing)
    old_cars = adapter.create_cars_from_stats(default)
    new_cars = adapter.create_cars_from_stats(timing)
    for team in old_cars:
        differences = {key for key, value in old_cars[team].model_dump().items()
                       if value != new_cars[team].model_dump()[key]}
        assert differences <= {"base_pace", "wet_performance"}
    assert {k: v.constructor_points for k, v in default.items()} == {
        k: v.constructor_points for k, v in timing.items()}


@pytest.mark.parametrize("case", ["missing", "untimed", "incomplete", "zero_weights", "neutral"])
def test_unsupported_current_qualifying_uses_exact_legacy_fallback(case):
    adapter = loader()
    arguments = fixture_arguments()
    if case == "missing":
        arguments["target_qualifying_rows"] = []
    elif case == "untimed":
        for row in arguments["target_qualifying_rows"]:
            row["Q1"] = None
    elif case == "incomplete":
        arguments["target_qualifying_rows"] = arguments["target_qualifying_rows"][:4]
    elif case == "neutral":
        for row in arguments["target_qualifying_rows"]:
            row["Q1"] = 80.
    else:
        arguments.update(track_weight=0., form_weight=0., quali_weight=0.)
    assert adapter._build_driver_stats(**arguments, postqual_pace_policy="qualifying_led") == (
        adapter._build_driver_stats(**arguments, postqual_pace_policy="season_points"))


def test_team_without_any_weighted_timing_keeps_constructor_fallback():
    adapter = loader()
    arguments = fixture_arguments(teams=5)
    arguments["target_qualifying_rows"] = arguments["target_qualifying_rows"][:8]
    assert len(arguments["target_qualifying_rows"]) == .8*len(arguments["roster"])
    first = adapter._build_driver_stats(**arguments, postqual_pace_policy="qualifying_led")
    changed = deepcopy(arguments)
    changed["constructor_standings"][-1]["points"] = 100.
    second = adapter._build_driver_stats(**changed, postqual_pace_policy="qualifying_led")
    assert first["41"].team_pace_rating != second["41"].team_pace_rating
    assert first["41"].driver_skill_rating == second["41"].driver_skill_rating


@pytest.mark.parametrize("policy", ["", "unknown", None, True, []])
def test_invalid_policy_rejected_before_network_or_assembly(policy):
    adapter = loader()
    with pytest.raises(ValueError, match="postqual_pace_policy"):
        adapter.get_weighted_driver_stats(_utc_now().year, 3, postqual_pace_policy=policy)
    with pytest.raises(ValueError, match="postqual_pace_policy"):
        adapter._build_driver_stats(**fixture_arguments(), postqual_pace_policy=policy)


def test_public_selection_records_provenance_and_refresh_clears_it(monkeypatch):
    adapter = loader()
    arguments = fixture_arguments()
    monkeypatch.setattr(adapter, "_event_for_race", lambda *args: arguments["target_event"])
    monkeypatch.setattr(adapter, "get_official_roster", lambda *args: arguments["roster"])
    monkeypatch.setattr(adapter, "_load_season_rows", lambda *args, **kwargs: (
        arguments["driver_standings"], arguments["constructor_standings"], [], []))
    monkeypatch.setattr(adapter, "_season_data", lambda *args: (
        [], arguments["target_qualifying_rows"]))
    actual = adapter.get_weighted_driver_stats(_utc_now().year, 3, quali_weight=0.,
                                               postqual_pace_policy="qualifying_led")
    expected = adapter._build_driver_stats(**{**arguments, "quali_weight": 0.},
                                           postqual_pace_policy="qualifying_led")
    assert actual == expected
    assert adapter.get_provenance()["postqual_pace_policy"] == "qualifying_led"
    adapter.refresh()
    assert "postqual_pace_policy" not in adapter.get_provenance()
