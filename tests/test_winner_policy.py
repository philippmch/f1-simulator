"""The practice-era reporting choice must respect information and replay boundaries."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from f1sim.analysis.practice_qualifying import calibrate_practice_qualifying_drivers
from f1sim.analysis.winner_policy import simulation_winner_allocation
from f1sim.data.current import CurrentSeasonDataLoader, DriverStats
from f1sim.models import Track, Weather


@pytest.fixture
def practice_context(monkeypatch):
    before = datetime(2026, 3, 28, 13, tzinfo=timezone.utc)
    monkeypatch.setattr("f1sim.data.current._utc_now", lambda: before)
    loader = CurrentSeasonDataLoader(current_year=2026, http_getter=lambda *a, **k: {})
    stats = {code: DriverStats(driver_id=code, driver_name=code, team_id=team, team_name=team)
             for code, team in (("A1", "a"), ("A2", "a"), ("B1", "b"), ("B2", "b"))}
    drivers = loader.create_drivers_from_stats(stats)
    cars = loader.create_cars_from_stats(stats)
    track = Track(id="test", name="Test", country="Test", total_laps=8, base_lap_time=90)
    practice = {"year": 2026, "round": 2, "session_number": 3, "rows": [
        {"driver": d.id, "position": i, "lap_seconds": 89+i*.2, "laps": 12}
        for i, d in enumerate(drivers, 1)
    ]}
    fitted, forecast = calibrate_practice_qualifying_drivers(
        drivers, cars, track, [], practice, year=2026, target_round=2, circuit="test",
    )
    assert not forecast.get("candidate_fallback")
    loader._qualifying_forecast = forecast
    loader._driver_stats = {d.id: stats[d.id].model_copy(update={
        "qualifying_pace_adjustment": d.qualifying_pace_adjustment,
        "qualifying_pace_source": "current_practice",
    }) for d in fitted}
    event = {"round": 2, "sessions": {"Qualifying": {
        "date": "2026-03-28", "time": "14:00:00Z",
    }}}
    monkeypatch.setattr(loader, "_event_for_race", lambda *a: event)
    return loader, fitted, event, Weather(change_probability=0)


def test_valid_current_practice_keeps_native_chances_without_reading_points(
    practice_context, monkeypatch,
):
    loader, drivers, _, weather = practice_context
    before = deepcopy((loader._qualifying_forecast, loader._driver_stats, drivers))
    monkeypatch.setattr(loader, "get_winner_allocation",
                        lambda *a: pytest.fail("native practice forecast redistributed to points"))
    assert simulation_winner_allocation(loader, 2026, 2, drivers, weather) is None
    assert (loader._qualifying_forecast, loader._driver_stats, drivers) == before


@pytest.mark.parametrize("context", [
    "rain", "wet_surface", "changing_weather", "qualifying_weather", "weather_schedule",
    "known_grid", "older_policy", "wrong_year", "wrong_round", "fallback", "missing_stats",
    "partial_roster", "different_skill", "different_qualifying", "qualifying_started",
    "unknown_qualifying_time", "sprint_unknown_cutoff", "sprint_started",
])
def test_other_contexts_keep_existing_allocation(practice_context, monkeypatch, context):
    loader, drivers, event, weather = practice_context
    kwargs = {}
    if context == "rain":
        weather.rain_intensity = .1
    elif context == "wet_surface":
        weather.track_wetness = .1
    elif context == "changing_weather":
        weather.change_probability = .1
    elif context == "qualifying_weather":
        kwargs["qualifying_weather"] = {"Q1": [{"lap": 1, "rain_intensity": 0.}]}
    elif context == "weather_schedule":
        kwargs["weather_schedule"] = [{"lap": 5, "rain_intensity": .2}]
    elif context == "known_grid":
        kwargs["starting_grid"] = [d.id for d in drivers]
    elif context == "older_policy":
        loader._qualifying_forecast["policy"] = "earlier_team_q1"
    elif context == "wrong_year":
        loader._qualifying_forecast["year"] = 2025
    elif context == "wrong_round":
        loader._qualifying_forecast["target_round"] = 1
    elif context == "fallback":
        loader._qualifying_forecast["candidate_fallback"] = "qualifying_physics_floor"
    elif context == "missing_stats":
        loader._driver_stats = None
    elif context == "partial_roster":
        drivers = drivers[:3]
    elif context == "different_skill":
        drivers[0].skill_rating -= .01
    elif context == "different_qualifying":
        drivers[0].qualifying_pace_adjustment += .001
    elif context == "qualifying_started":
        monkeypatch.setattr("f1sim.data.current._utc_now",
                            lambda: datetime(2026, 3, 28, 14, tzinfo=timezone.utc))
    elif context == "unknown_qualifying_time":
        del event["sessions"]["Qualifying"]["time"]
    else:
        event["sprint"] = True
        if context == "sprint_started":
            event["sessions"]["SprintQualifying"] = {
                "date": "2026-03-27", "time": "12:00:00Z",
            }
    sentinel = object()
    calls = []
    monkeypatch.setattr(loader, "get_winner_allocation",
                        lambda *args: (calls.append(args), sentinel)[1])
    assert simulation_winner_allocation(loader, 2026, 2, drivers, weather, **kwargs) is sentinel
    assert calls == [(2026, 2, drivers)]


def test_legacy_adapter_retains_three_argument_hook():
    calls = []

    class Legacy:
        def get_winner_allocation(self, year, race, drivers):
            calls.append((year, race, drivers))
            return "legacy"

    assert simulation_winner_allocation(Legacy(), 2026, 2, [], Weather()) == "legacy"
    assert calls == [(2026, 2, [])]
