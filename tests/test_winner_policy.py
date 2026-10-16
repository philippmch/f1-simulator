"""The practice-era reporting choice must respect information and replay boundaries."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

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


@pytest.fixture
def published_grid_context(practice_context, monkeypatch):
    loader, drivers, event, weather = practice_context
    now = datetime(2026, 3, 28, 16, tzinfo=timezone.utc)
    monkeypatch.setattr("f1sim.data.current._utc_now", lambda: now)
    event.update(date="2026-03-29", time="14:00:00Z")
    loader._qualifying_forecast = None
    loader._race_grid = {
        "mode": "published", "year": 2026, "round": 2,
        "source_url": "https://www.formula1.com/en/results/2026/races/1234/test/starting-grid",
        "starting_grid": [d.id for d in drivers], "pit_lane_starters": [drivers[-1].id],
        "fetched_at": now.isoformat(), "qualifying_started_at": "2026-03-28T14:00:00Z",
    }
    return loader, drivers, event, weather


@pytest.mark.parametrize("wet", [False, True])
def test_verified_published_grid_keeps_native_driver_chances(
    published_grid_context, monkeypatch, wet,
):
    loader, drivers, _, weather = published_grid_context
    if wet:
        weather.rain_intensity = .3
        weather.track_wetness = .2
    monkeypatch.setattr(loader, "get_winner_allocation",
                        lambda *a: pytest.fail("Published-grid chances redistributed to points"))
    before = deepcopy((loader._driver_stats, loader._race_grid, drivers))
    assert simulation_winner_allocation(
        loader, 2026, 2, drivers, weather, starting_grid=[d.id for d in drivers],
    ) is None
    assert (loader._driver_stats, loader._race_grid, drivers) == before


@pytest.mark.parametrize("context", [
    "wrong_grid", "partial_field", "wrong_round", "wrong_year", "unpublished",
    "before_qualifying", "future_fetch", "no_race_time", "wrong_race_year",
    "race_before_qualifying", "wrong_qualifying", "different_driver_model",
    "invalid_pit_queue", "wrong_source", "untimed_fetch", "single_entrant",
])
def test_unverified_grid_context_retains_existing_policy(
    published_grid_context, monkeypatch, context,
):
    loader, drivers, event, weather = published_grid_context
    grid = [d.id for d in drivers]
    if context == "wrong_grid":
        grid.reverse()
    elif context == "partial_field":
        drivers = drivers[:-1]
    elif context in ("wrong_round", "wrong_year"):
        key = "round" if context == "wrong_round" else "year"
        loader._race_grid[key] -= 1
    elif context == "unpublished":
        loader._race_grid["mode"] = "manual"
    elif context in ("before_qualifying", "future_fetch"):
        loader._race_grid["fetched_at"] = (
            "2026-03-28T13:00:00Z" if context == "before_qualifying" else "2026-03-28T17:00:00Z"
        )
    elif context == "no_race_time":
        del event["time"]
    elif context == "wrong_race_year":
        event["date"] = "2025-03-29"
    elif context == "race_before_qualifying":
        event.update(date="2026-03-28", time="14:30:00Z")
    elif context == "wrong_qualifying":
        loader._race_grid["qualifying_started_at"] = "2026-03-28T13:00:00Z"
    elif context == "different_driver_model":
        drivers[0].skill_rating -= .01
    elif context == "wrong_source":
        loader._race_grid["source_url"] = "https://example.com/starting-grid"
    elif context == "untimed_fetch":
        loader._race_grid["fetched_at"] = "2026-03-28T16:00:00"
    elif context == "single_entrant":
        drivers = drivers[:1]
        grid = grid[:1]
        loader._race_grid.update(starting_grid=grid, pit_lane_starters=[])
        loader._driver_stats = {d.id: loader._driver_stats[d.id] for d in drivers}
    else:
        loader._race_grid["pit_lane_starters"] = [grid[0]]
    sentinel = object()
    monkeypatch.setattr(loader, "get_winner_allocation", lambda *a: sentinel)
    assert simulation_winner_allocation(
        loader, 2026, 2, drivers, weather, starting_grid=grid,
    ) is sentinel


@pytest.mark.parametrize("elapsed,fresh_lookup", [(0, False), (2, False), (2, True)])
def test_official_grid_native_results_do_not_change_at_race_deadline(
    published_grid_context, monkeypatch, elapsed, fresh_lookup,
):
    from f1sim.analysis.grid_winner_reference import build_grid_winner_references
    from f1sim.analysis.montecarlo import MonteCarloRunner
    from f1sim.analysis.teammate_forecast import build_teammate_allocation

    loader, drivers, event, weather = published_grid_context
    event["round"] = loader._race_grid["round"] = 3
    grid = [d.id for d in drivers]
    deadline = datetime(2026, 3, 29, 14, tzinfo=timezone.utc)
    legacy = build_teammate_allocation(
        {d.id: d.team_id for d in drivers},
        [{"round": number, "driver_id": d.id, "team_id": d.team_id,
          "points": 25 if d.id.endswith("2") else 0}
         for number in (1, 2) for d in drivers], cutoff_round=2,
    )
    monkeypatch.setattr(loader, "get_winner_allocation", lambda *a: legacy)

    def run():
        allocation = simulation_winner_allocation(
            loader, 2026, 3, drivers, weather, starting_grid=grid,
        )
        return MonteCarloRunner(
            deepcopy(drivers), loader.create_cars_from_stats(loader._driver_stats),
            Track(id="test", name="Test", country="Test", total_laps=8, base_lap_time=90),
            deepcopy(weather), seed=42, starting_grid=grid,
            pit_lane_starters=loader.get_pit_lane_starters(), winner_allocation=allocation,
        ).run(4, parallel=False)

    provenance = loader.get_provenance()
    before = run()
    assert before.winner_allocation is None
    assert loader.get_provenance() == provenance
    redistributed = deepcopy(before)
    redistributed.winner_allocation = legacy
    assert redistributed.get_win_probabilities() != before.get_win_probabilities()
    after = deadline + timedelta(days=elapsed)
    monkeypatch.setattr("f1sim.data.current._utc_now", lambda: after)
    if fresh_lookup:
        loader._race_grid["fetched_at"] = after.isoformat()
    expected_provenance = loader.get_provenance()
    completed = run()
    assert completed.winner_allocation is None
    assert completed.get_winner_forecast() is None
    assert completed.get_win_probabilities() == before.get_win_probabilities()
    assert completed.race_results == before.race_results
    assert completed.input_snapshot == before.input_snapshot
    assert loader.get_provenance() == expected_provenance
    # Scenario reporting is independent of the stricter prospective record contract.
    with pytest.raises(ValueError, match="before the race"):
        build_grid_winner_references(
            grid, loader._race_grid, year=2026, target_round=3,
            recorded_at=after.isoformat(), qualifying_starts_at="2026-03-28T14:00:00Z",
            race_starts_at=deadline.isoformat(),
        )
