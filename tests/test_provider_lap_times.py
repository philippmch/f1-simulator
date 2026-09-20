"""Provider duration and numeric fields fail closed on malformed values."""

from math import inf, isfinite, nan

import pytest

import f1sim.data.current as current_module
from f1sim.data import CurrentSeasonDataLoader

CURRENT_YEAR = current_module._utc_now().year


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (90, 90.0),
        (90.5, 90.5),
        (" 90.5 ", 90.5),
        ("1e2", 100.0),
        ("1:20", 80.0),
        ("1:20.000", 80.0),
        ("1:02:03.5", 3723.5),
        ("100:00", 6000.0),
        ({"time": "" , "Time": "1:20.000"}, 80.0),
        ({"time": "1:20.000"}, 80.0),
        ({"Time": "1:20.000"}, 80.0),
    ],
)
def test_parse_time_seconds_accepts_provider_duration_forms(raw, expected):
    assert current_module._parse_time_seconds(raw) == pytest.approx(expected)


@pytest.mark.parametrize(
    "raw",
    [
        True,
        False,
        0,
        -1,
        inf,
        nan,
        "",
        "+1:20",
        "-1:30",
        "1:99",
        "1:60:00",
        "1:02:60",
        "1:2e1",
        "1:2:3e1",
        "1:2:3:4",
        "1:NaN",
        {"time": True},
        {"Time": inf},
        [],
    ],
)
def test_parse_time_seconds_rejects_nonfinite_signed_and_malformed_values(raw):
    assert current_module._parse_time_seconds(raw) is None


def test_parse_time_seconds_rejects_overflow_without_raising():
    assert current_module._parse_time_seconds(10**1000) is None
    assert current_module._parse_time_seconds("9" * 1000 + ":00") is None


@pytest.mark.parametrize("raw", [True, False, inf, -inf, nan, 10**1000])
def test_as_float_returns_default_for_invalid_provider_numbers(raw):
    assert current_module._as_float(raw, 7.5) == 7.5


def test_as_float_preserves_finite_signed_values_for_coordinates_and_valid_numbers():
    assert current_module._as_float(-12.5) == -12.5
    assert current_module._as_float("1e3") == 1000.0


def _track_loader(rows):
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    loader._calendar = [{
        "round": 1,
        "race": "Parser Grand Prix",
        "circuit_id": "parser_test",
        "circuit_name": "Parser Test Circuit",
        "country": "Testland",
        "completed": True,
    }]
    profile = {
        "lap": 90.0,
        "laps": 50,
        "pit": 20.0,
        "overtake": .5,
        "tire": .5,
        "sc": .3,
        "weather": .2,
        "active_aero": 2,
    }
    loader._venue_profile = lambda _event: profile
    loader._round_data = lambda _year, _round: (rows, [])
    return loader


def test_invalid_fastest_lap_uses_venue_profile_fallback():
    loader = _track_loader([{"FastestLap": {"Time": True}}])

    stats = loader.get_track_stats(CURRENT_YEAR, "Parser Grand Prix")

    assert stats.fastest_lap == 90.0


def test_mixed_fastest_laps_use_the_valid_provider_value():
    loader = _track_loader([
        {"FastestLap": {"Time": True}},
        {"FastestLap": {"Time": "1:20.000"}},
        {"FastestLap": {"Time": inf}},
    ])

    stats = loader.get_track_stats(CURRENT_YEAR, "Parser Grand Prix")

    assert stats.fastest_lap == 80.0


def test_qualifying_coverage_ignores_boolean_and_nonfinite_generic_times():
    rows = [
        {
            "round": 1,
            "Driver": {"code": "A"},
            "Q1": True,
            "Q2": "1:23.000",
        },
        {
            "round": 1,
            "Driver": {"code": "B"},
            "time": True,
            "Q1": inf,
        },
        {
            "round": 1,
            "Driver": {"code": "C"},
            "Q3": "1:20.000",
        },
    ]

    observed = CurrentSeasonDataLoader._round_driver_ids(
        rows,
        {"a": "A", "b": "B", "c": "C"},
        qualifying=True,
    )

    assert observed == {1: {"A", "C"}}


def test_race_speed_and_constructor_points_ignore_invalid_numeric_provider_values(
    monkeypatch,
):
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(loader, "_venue_profile", lambda _event: {
        "lap": 90.0,
        "laps": 50,
        "pit": 20.0,
        "overtake": .5,
        "tire": .5,
        "sc": .3,
        "weather": .2,
        "active_aero": 2,
    })
    stats = loader._build_driver_stats(
        year=CURRENT_YEAR,
        target_event={"round": 2},
        roster=[
            {"id": "A", "name": "Alice", "team_name": "Team A"},
            {"id": "B", "name": "Bob", "team_name": "Team A"},
        ],
        driver_standings=[],
        constructor_standings=[{
            "points": True,
            "Constructor": {"constructorId": "team_a", "name": "Team A"},
        }],
        race_rows=[
            {
                "round": 1,
                "position": 1,
                "Driver": {"code": "A"},
                "Constructor": {"name": "Team A"},
                "FastestLap": {"AverageSpeed": {"speed": True}},
            },
            {
                "round": 1,
                "position": 2,
                "Driver": {"code": "B"},
                "Constructor": {"name": "Team A"},
                "FastestLap": {"AverageSpeed": {"speed": "220"}},
            },
        ],
        quali_rows=[],
        target_qualifying_rows=[],
        track_weight=.5,
        form_weight=1.0,
        quali_weight=.2,
    )

    assert all(isfinite(item.team_pace_rating) for item in stats.values())
    assert all(item.constructor_points == 0.0 for item in stats.values())
    assert current_module._as_float(True, 0.0) == 0.0
