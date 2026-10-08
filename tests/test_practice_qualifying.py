"""Causal practice forecasts must reach real physics and the live adapter."""

from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import pytest

from f1sim.analysis.practice_qualifying import (
    calibrate_practice_qualifying_drivers,
    load_practice_qualifying_model,
    predict_practice_qualifying,
)
from f1sim.data.current import CurrentSeasonDataError, CurrentSeasonDataLoader, DriverStats
from f1sim.data.practice import (
    build_current_qualifying_history,
    eligible_practice_sessions,
    fetch_current_practice,
    parse_practice_table,
)
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator


def inputs():
    drivers = [Driver(id=code, name=code, team_id=team)
               for code, team in (("A1", "a"), ("A2", "a"), ("B1", "b"), ("B2", "b"))]
    cars = {team: Car(team_id=team, team_name=team) for team in ("a", "b")}
    track = Track(id="test", name="Test", country="Test", total_laps=8, base_lap_time=90)
    practice = {"year": 2026, "round": 2, "session_number": 3, "rows": [
        {"driver": driver.id, "position": i, "lap_seconds": 89 + i * .2, "laps": 12}
        for i, driver in enumerate(drivers, 1)
    ]}
    return drivers, cars, track, practice


def test_real_laps_follow_practice_clock_without_changing_race_pace_or_compounding():
    drivers, cars, track, practice = inputs()
    originals = deepcopy((drivers, cars, track, practice))
    fitted, evidence = calibrate_practice_qualifying_drivers(
        drivers, cars, track, [], practice, year=2026, target_round=2, circuit="test",
    )
    again, _ = calibrate_practice_qualifying_drivers(
        fitted, cars, track, [], practice, year=2026, target_round=2, circuit="test",
    )
    assert fitted == again
    offset = load_practice_qualifying_model(2026)["practice_clock_offsets_percent"]["3"]
    assert evidence["predicted_median_seconds"] == pytest.approx(89.5 * (1 + offset / 100))
    lap = LapSimulator(np.random.default_rng(0))
    for before, after in zip(drivers, fitted):
        actual = min(lap.calculate_qualifying_lap(
            after, cars[after.team_id], track, tire, Weather(change_probability=0),
            sample_variation=False,
        ) for tire in TIRE_COMPOUNDS.values())
        assert actual == pytest.approx(evidence["predicted_seconds"][after.id], abs=1e-10)
        for tire in TIRE_COMPOUNDS.values():
            assert lap.calculate_lap_time(before, cars[before.team_id], track, tire, Weather(),
                                          3, 8, sample_variation=False) == lap.calculate_lap_time(
                after, cars[after.team_id], track, tire, Weather(), 3, 8, sample_variation=False,
            )
    assert (drivers, cars, track, practice) == originals


@pytest.mark.parametrize("change", ["future_history", "wrong_season", "boolean_round",
                                   "duplicate_driver", "missing_coverage", "nan_lap"])
def test_forecast_rejects_leaks_and_invalid_evidence(change):
    drivers, _, _, practice = inputs()
    roster = {d.id: d.team_id for d in drivers}
    history = []
    if change == "future_history":
        history = [{"round": 2, "circuit": "test", "rows": []}]
    elif change == "wrong_season":
        practice["year"] = 2025
    elif change == "boolean_round":
        practice["round"] = True
    elif change == "duplicate_driver":
        practice["rows"][1]["driver"] = "A1"
    elif change == "missing_coverage":
        practice["rows"] = practice["rows"][:1]
    else:
        practice["rows"][0]["lap_seconds"] = float("nan")
    with pytest.raises(ValueError):
        predict_practice_qualifying(roster, history, practice, year=2026,
                                   target_round=2, circuit="test")


def event():
    return {"round": 2, "date": "2026-03-29", "circuit_id": "test", "sessions": {
        "FirstPractice": {"date": "2026-03-27", "time": "10:00:00Z"},
        "SecondPractice": {"date": "2026-03-27", "time": "14:00:00Z"},
        "ThirdPractice": {"date": "2026-03-28", "time": "10:00:00Z"},
        "Qualifying": {"date": "2026-03-28", "time": "14:00:00Z"},
    }}


def test_practice_cutoff_uses_first_sprint_qualifying_and_completed_sessions():
    target = event()
    now = datetime(2026, 3, 28, 16, tzinfo=timezone.utc)
    assert eligible_practice_sessions(target, now) == [3, 2, 1]
    target["sprint"] = True
    target["sessions"]["SprintQualifying"] = {"date": "2026-03-27", "time": "12:00:00Z"}
    assert eligible_practice_sessions(target, now) == [1]
    assert eligible_practice_sessions(target, datetime(2026, 3, 27, 10, 30,
                                                      tzinfo=timezone.utc)) == []
    del target["sessions"]["SprintQualifying"]
    assert eligible_practice_sessions(target, now) == []


def table(number=3, reserve=False):
    rows = [("Reserve RES", "a", "54.000", "15")] if reserve else []
    rows += [(code, team, ("1:29.000" if not reserve else "+0.200s") if i == 0
              else f"+{.2 * (i + 1):.3f}s", "12")
             for i, (code, team) in enumerate((("A1", "a"), ("A2", "a"),
                                                ("B1", "b"), ("B2", "b")))]
    return f"<h1>Test 2026 - PRACTICE {number}</h1><table>" + "".join(
        f"<tr><td>{i}</td><td>1</td><td>{code}</td><td>{team}</td><td>{lap}</td>"
        f"<td>{laps}</td></tr>" for i, (code, team, lap, laps) in enumerate(rows, 1)
    ) + "</table>"


def loader(monkeypatch, getter):
    # Production parameters are deliberately scoped to 2026, irrespective of test run year.
    monkeypatch.setattr("f1sim.data.current._utc_now",
                        lambda: datetime(2026, 3, 28, 16, tzinfo=timezone.utc))
    return CurrentSeasonDataLoader(current_year=2026, http_getter=getter)


def test_parser_accounts_for_reserve_fastest_lap_but_keeps_modeled_roster(monkeypatch):
    adapter = loader(monkeypatch, lambda _: "")
    drivers = inputs()[0]
    rows = parse_practice_table(table(reserve=True), year=2026, number=3,
                                loader=adapter, drivers=drivers)
    assert {r["driver"] for r in rows} == {d.id for d in drivers}
    assert rows[0]["lap_seconds"] == pytest.approx(54.2)
    with pytest.raises(CurrentSeasonDataError):
        parse_practice_table(table().replace("<td>a</td>", "<td>b</td>", 1), year=2026,
                             number=3, loader=adapter, drivers=drivers)


def test_live_calibration_reaches_stats_and_offline_holdouts_never_fetch_practice(monkeypatch):
    calls = []
    index = ('<h1>2026 RACE RESULTS</h1><table><tr><td>Test</td><td>29 Mar</td>'
             '<td><a href="/en/results/2026/races/1234/test/race-result">Test</a></td>'
             '</tr></table>')

    def getter(url, **kwargs):
        calls.append(url)
        assert kwargs["headers"]["Cache-Control"] == "no-cache, no-store, max-age=0"
        return index if url.endswith("/races") else table()

    adapter = loader(monkeypatch, getter)
    drivers = inputs()[0]
    stats = {d.id: DriverStats(driver_id=d.id, driver_name=d.name, team_id=d.team_id,
                               team_name=d.team_id) for d in drivers}
    calibrated, evidence = adapter._calibrate_qualifying_stats(
        stats, 2026, event(), [], [], [event()], fetch_practice=True,
    )
    assert evidence["policy"] == "current_season_practice_q1_v1"
    assert all(s.qualifying_pace_source == "current_practice" for s in calibrated.values())
    assert len(calls) == 2
    calls.clear()
    native, _ = adapter._calibrate_qualifying_stats(stats, 2026, event(), [], [], [event()])
    assert native == stats and calls == []
    target = event()
    target["sessions"]["ThirdPractice"]["date"] = "2026-03-29"
    target["sessions"]["SecondPractice"]["date"] = "2026-03-29"
    target["sessions"]["FirstPractice"]["date"] = "2026-03-29"
    assert fetch_current_practice(adapter, 2026, target, drivers,
                                  now=datetime(2026, 3, 28, tzinfo=timezone.utc)) is None
    assert calls == []


def test_historical_adapter_excludes_target_and_later_labels(monkeypatch):
    adapter = loader(monkeypatch, lambda _: "")
    events = [{"round": n, "circuit_id": "test"} for n in (1, 2, 3)]
    results = [{"round": n, "Driver": {"code": d.id, "givenName": d.id,
                                        "familyName": "Racer"},
                "Constructor": {"name": d.team_id}, "points": 25 - i,
                "Q1": "1:29.000", "position": i + 1}
               for n in (1, 2, 3) for i, d in enumerate(inputs()[0])]
    before = build_current_qualifying_history(adapter, events, results, results, before_round=2)
    for row in results:
        if row["round"] >= 2:
            row.update(Q1="2:00.000", points=100, position=30)
    assert before == build_current_qualifying_history(
        adapter, events, results, results, before_round=2,
    )
    assert [r["round"] for r in before] == [1]


@pytest.mark.parametrize("circuit,race,slug", [
    ("marina_bay", "Singapore Grand Prix", "singapore"),
    ("sepang", "Bahrain Grand Prix in Malaysia", "bahrain"),
])
def test_unfinished_race_resolves_observed_navigation_without_a_winner_table(
    monkeypatch, circuit, race, slug,
):
    calls = []
    index = ('<h1>2026 RACE RESULTS</h1>'
             f'<a href="/en/results/2026/races/1296/{slug}/race-result">{race}</a>'
             f'<a href="/en/results/2025/races/999/{slug}/race-result">Older season</a>')

    def getter(url, **kwargs):
        calls.append(url)
        return index if url.endswith("/races") else table()

    adapter = loader(monkeypatch, getter)
    target = event()
    target.update(circuit_id=circuit, race=race)
    practice = fetch_current_practice(adapter, 2026, target, inputs()[0],
                                      now=datetime(2026, 3, 28, 16, tzinfo=timezone.utc))
    assert practice["source_url"].endswith(f"/2026/races/1296/{slug}/practice/3")
    assert len(calls) == 2
