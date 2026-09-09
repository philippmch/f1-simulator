"""Contract tests for the live current-season data adapter."""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit

import pytest

import f1sim.data.current as current_module
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader

CURRENT_YEAR = datetime.now(timezone.utc).year


def _calendar(year: int = CURRENT_YEAR) -> dict:
    return {
        "MRData": {
            "total": "4",
            "RaceTable": {
                "season": str(year),
                "Races": [
                    {
                        "season": str(year),
                        "round": "1",
                        "raceName": "Old Venue Grand Prix",
                        "Circuit": {
                            "circuitId": "old_venue",
                            "circuitName": "Old Venue",
                            "Location": {
                                "lat": "1",
                                "long": "2",
                                "locality": "Oldtown",
                                "country": "Testland",
                            },
                        },
                        "date": f"{year}-03-01",
                    },
                    {
                        "season": str(year),
                        "round": "1",
                        "raceName": "Replacement Grand Prix",
                        "Circuit": {
                            "circuitId": "replacement",
                            "circuitName": "Replacement Circuit",
                            "Location": {
                                "lat": "3",
                                "long": "4",
                                "locality": "Newtown",
                                "country": "Testland",
                            },
                        },
                        "date": f"{year}-03-08",
                        "Sprint": {"date": f"{year}-03-07", "time": "12:00:00Z"},
                    },
                    {
                        "season": str(year),
                        "round": "2",
                        "raceName": "Cancelled Grand Prix",
                        "cancelled": True,
                        "Circuit": {"circuitId": "cancelled", "Location": {}},
                        "date": f"{year}-04-01",
                    },
                    {
                        "season": str(year),
                        "round": "3",
                        "raceName": "Pre-Season Testing",
                        "Circuit": {"circuitId": "testing", "Location": {}},
                        "date": f"{year}-05-01",
                        "eventFormat": "testing",
                    },
                    {
                        "season": str(year),
                        "round": "4",
                        "raceName": "Future Grand Prix",
                        "Circuit": {
                            "circuitId": "future_venue",
                            "circuitName": "Future Circuit",
                            "Location": {
                                "lat": "5",
                                "long": "6",
                                "locality": "Futuretown",
                                "country": "Testland",
                            },
                        },
                        "date": f"{year + 1}-01-01",
                    },
                ],
            },
        }
    }


def _roster_html() -> str:
    return """
    <main>
      <section><a href='/en/teams/team-a'>Team A</a>
        <a href='/en/drivers/alice-racer'>Alice Racer</a>
        <a href='/en/drivers/bob-racer'>Bob Racer</a>
      </section>
      <section><a href='/en/teams/team-b'>Team B</a>
        <a href='/en/drivers/former-racer'>Former Racer</a>
        <a href='/en/drivers/reserve-racer'>Reserve Racer</a>
      </section>
    </main>
    """


class FakeGetter:
    def __init__(self, year: int = CURRENT_YEAR) -> None:
        self.year = year
        self.urls: list[str] = []

    def __call__(self, url: str, **_: object) -> object:
        self.urls.append(url)
        parsed = urlsplit(url)
        path = parsed.path
        if path.endswith("/races/"):
            return _calendar(self.year)
        if path.endswith("/teams"):
            return _roster_html()
        if path.endswith("/driverstandings.json"):
            return {
                "MRData": {
                    "total": "3",
                    "StandingsTable": {
                        "season": str(self.year),
                        "StandingsLists": [
                            {
                                "DriverStandings": [
                                    {
                                        "points": "20",
                                        "Driver": {
                                            "driverId": "alice-racer",
                                            "code": "ALC",
                                            "givenName": "Alice",
                                            "familyName": "Racer",
                                        },
                                    },
                                    {
                                        "points": "10",
                                        "Driver": {
                                            "driverId": "bob-racer",
                                            "code": "BOB",
                                            "givenName": "Bob",
                                            "familyName": "Racer",
                                        },
                                    },
                                    {
                                        "points": "99",
                                        "Driver": {
                                            "driverId": "former-racer",
                                            "code": "OLD",
                                            "givenName": "Former",
                                            "familyName": "Racer",
                                        },
                                    },
                                ]
                            }
                        ]
                    },
                }
            }
        if path.endswith("/constructorstandings.json"):
            return {
                "MRData": {
                    "StandingsTable": {
                        "season": str(self.year),
                        "StandingsLists": [
                            {
                                "ConstructorStandings": [
                                    {
                                        "points": "30",
                                        "Constructor": {
                                            "constructorId": "team-a",
                                            "name": "Team A",
                                        },
                                    },
                                    {
                                        "points": "10",
                                        "Constructor": {
                                            "constructorId": "team-b",
                                            "name": "Team B",
                                        },
                                    },
                                ]
                            }
                        ]
                    }
                }
            }
        if path.endswith("/results.json") or path.endswith("/qualifying.json"):
            # Future round calls are deliberately not expected; if a caller
            # makes one, returning an empty page still keeps the fixture valid.
            return {
                "MRData": {
                    "total": "0",
                    "RaceTable": {"season": str(self.year), "Races": []},
                }
            }
        raise AssertionError(f"unexpected URL: {url}")


def test_rejects_non_current_year_before_network() -> None:
    getter = FakeGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)

    with pytest.raises(ValueError, match="current UTC season"):
        loader.list_available_events(CURRENT_YEAR - 1)
    assert getter.urls == []

    with pytest.raises(ValueError, match="runtime UTC season"):
        CurrentSeasonDataLoader(current_year=CURRENT_YEAR - 1, http_getter=getter)


@pytest.mark.parametrize("feed", ["calendar", "results", "qualifying", "standings"])
def test_provider_payload_for_another_season_is_rejected(feed: str) -> None:
    wrong_year = CURRENT_YEAR - 1

    def getter(url: str, **_: object) -> object:
        path = urlsplit(url).path
        if path.endswith("/races/"):
            return {
                "MRData": {
                    "RaceTable": {"season": str(wrong_year), "Races": []},
                }
            }
        if path.endswith("/driverstandings.json"):
            return {
                "MRData": {
                    "StandingsTable": {
                        "season": str(wrong_year),
                        "StandingsLists": [{"DriverStandings": []}],
                    }
                }
            }
        if path.endswith("/results.json") or path.endswith("/qualifying.json"):
            return {
                "MRData": {
                    "RaceTable": {"season": str(wrong_year), "Races": []},
                }
            }
        raise AssertionError(f"unexpected URL: {url}")

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    with pytest.raises(CurrentSeasonDataError, match="declares season"):
        if feed == "calendar":
            loader.get_event_schedule(CURRENT_YEAR)
        elif feed == "standings":
            loader._standings(CURRENT_YEAR)
        else:
            loader._fetch_season_collection(CURRENT_YEAR, feed, ("Results", "QualifyingResults"))


@pytest.mark.parametrize("feed", ["calendar", "results", "qualifying", "standings"])
def test_jolpica_payload_without_season_metadata_is_rejected(feed: str) -> None:
    def getter(url: str, **_: object) -> object:
        path = urlsplit(url).path
        if path.endswith("/races/"):
            return {"MRData": {"RaceTable": {"Races": []}}}
        if path.endswith("/driverstandings.json"):
            return {"MRData": {"StandingsTable": {"StandingsLists": [{"DriverStandings": []}]}}}
        if path.endswith("/results.json") or path.endswith("/qualifying.json"):
            return {"MRData": {"RaceTable": {"Races": []}}}
        raise AssertionError(f"unexpected URL: {url}")

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    with pytest.raises(CurrentSeasonDataError, match="missing season metadata"):
        if feed == "calendar":
            loader.get_event_schedule(CURRENT_YEAR)
        elif feed == "standings":
            loader._standings(CURRENT_YEAR)
        else:
            loader._fetch_season_collection(CURRENT_YEAR, feed, ("Results", "QualifyingResults"))


def test_loader_rechecks_runtime_year_after_utc_rollover(monkeypatch) -> None:
    initial = datetime(CURRENT_YEAR, 12, 31, tzinfo=timezone.utc)
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=FakeGetter())
    monkeypatch.setattr(
        current_module,
        "_utc_now",
        lambda: initial.replace(year=CURRENT_YEAR + 1, day=1, month=1),
    )

    with pytest.raises(ValueError, match="stale UTC season"):
        loader.get_event_schedule(CURRENT_YEAR)
    with pytest.raises(ValueError, match="stale UTC season"):
        loader.get_provenance()


def test_calendar_filters_cancelled_testing_and_keeps_replacement_metadata() -> None:
    getter = FakeGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)

    events = loader.list_available_events(CURRENT_YEAR)

    assert [event["round"] for event in events] == [1, 4]
    replacement = events[0]
    assert replacement["race"] == "Replacement Grand Prix"
    assert replacement["location"] == "Newtown"
    assert replacement["latitude"] == 3.0
    assert replacement["longitude"] == 4.0
    assert replacement["sprint"] is True
    assert replacement["sessions"]["Sprint"]["time"] == "12:00:00Z"


def test_include_testing_is_not_cached_into_default_calendar() -> None:
    first = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=FakeGetter())
    with_testing = first.get_event_schedule(CURRENT_YEAR, include_testing=True)
    without_testing = first.get_event_schedule(CURRENT_YEAR, include_testing=False)

    assert [event["round"] for event in with_testing] == [1, 3, 4]
    assert [event["round"] for event in without_testing] == [1, 4]
    assert with_testing[1]["testing"] is True

    second = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=FakeGetter())
    assert [event["round"] for event in second.get_event_schedule(CURRENT_YEAR)] == [1, 4]
    assert [event["round"] for event in second.get_event_schedule(CURRENT_YEAR, True)] == [1, 3, 4]


def test_race_resolution_prefers_exact_normalized_aliases() -> None:
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    loader._calendar = [
        {
            "round": 1,
            "race": "Las Vegas Grand Prix",
            "location": "Las Vegas",
            "country": "USA",
        },
        {"round": 2, "race": "Las", "location": "Somewhere", "country": "Testland"},
    ]

    assert loader.resolve_race_identifier(CURRENT_YEAR, "LAS") == 2


def test_race_resolution_rejects_ambiguous_fuzzy_aliases() -> None:
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    loader._calendar = [
        {
            "round": 1,
            "race": "Alpha Grand Prix",
            "location": "Alpha",
            "country": "Testland",
        },
        {
            "round": 2,
            "race": "Beta Grand Prix",
            "location": "Beta",
            "country": "Testland",
        },
    ]

    with pytest.raises(ValueError, match="ambiguous"):
        loader.resolve_race_identifier(CURRENT_YEAR, "Grand Prix")


def test_official_roster_is_active_only_and_teams_are_official() -> None:
    getter = FakeGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)

    roster = loader.get_official_roster(CURRENT_YEAR)

    assert [entry["name"] for entry in roster] == ["Alice Racer", "Bob Racer"]
    assert {entry["team_name"] for entry in roster} == {"Team A"}


def test_formula1_team_card_markup_is_supported() -> None:
    page = """
    <h1>F1 Teams {CURRENT_YEAR}</h1>
    <a href="/en/teams/team-a">
      <span><p>Team A</p>
        <span class="body-xs-regular">Alice</span><span class="body-xs-bold">Racer</span>
        <span class="body-xs-regular">Bob</span><span class="body-xs-bold">Racer</span>
      </span>
    </a>
    """

    entries = CurrentSeasonDataLoader._team_card_roster(page)

    assert [(entry["name"], entry["team_name"]) for entry in entries] == [
        ("Alice Racer", "Team A"),
        ("Bob Racer", "Team A"),
    ]


def test_driver_aliases_match_short_and_extended_given_names() -> None:
    official = {"id": "kimi_antonelli", "name": "Kimi Antonelli"}
    provider = {
        "Driver": {
            "driverId": "antonelli",
            "code": "ANT",
            "givenName": "Andrea Kimi",
            "familyName": "Antonelli",
        }
    }

    assert CurrentSeasonDataLoader._driver_aliases(official) & (
        CurrentSeasonDataLoader._driver_aliases(provider)
    ) == {"antonelli"}
    assert CurrentSeasonDataLoader._standings_driver_map([provider])["antonelli"] == provider


def test_current_roster_alias_collisions_remove_surname_but_keep_unique_aliases() -> None:
    roster = [
        {"id": "ALC", "name": "Alice Smith", "team_name": "Team A"},
        {"id": "BOB", "name": "Bob Smith", "team_name": "Team B"},
        {"id": "CAR", "name": "Cara Jones", "team_name": "Team C"},
    ]
    active, aliases_to_id = CurrentSeasonDataLoader._build_active_driver_map(roster, {})

    assert set(active) == {"ALC", "BOB", "CAR"}
    assert "smith" not in aliases_to_id
    assert aliases_to_id["alice smith"] == "ALC"
    assert aliases_to_id["alice_smith"] == "ALC"
    assert aliases_to_id["alc"] == "ALC"
    assert aliases_to_id["bob smith"] == "BOB"
    assert CurrentSeasonDataLoader._resolve_row_driver(
        {"Driver": {"familyName": "Smith"}}, aliases_to_id
    ) is None


def test_roster_dedupe_rejects_conflicting_non_unknown_team_assignments() -> None:
    with pytest.raises(CurrentSeasonDataError, match="Conflicting non-unknown teams"):
        CurrentSeasonDataLoader._dedupe_roster(
            [
                {"id": "driver-1", "name": "Alex Driver", "team_name": "Team A"},
                {"id": "driver-1", "name": "Alex Driver", "team_name": "Team B"},
            ]
        )


def test_calendar_completion_waits_for_result_feed_and_calendar_call_stays_light() -> None:
    today = datetime.now(timezone.utc).date().isoformat()
    payload = _calendar()
    payload["MRData"]["RaceTable"]["Races"][1]["date"] = today

    class CompletionGetter(FakeGetter):
        def __call__(self, url: str, **kwargs: object) -> object:
            path = urlsplit(url).path
            if path.endswith("/races/"):
                self.urls.append(url)
                return payload
            if path.endswith("/teams"):
                self.urls.append(url)
                return _roster_html()
            if path.endswith("/results.json"):
                self.urls.append(url)
                return {
                    "MRData": {
                        "total": "2",
                        "RaceTable": {
                            "season": str(CURRENT_YEAR),
                            "Races": [
                                {
                                    "round": "1",
                                    "Results": [
                                        {
                                            "position": "1",
                                            "Driver": {
                                                "driverId": "alice-racer",
                                                "code": "ALC",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                        },
                                        {
                                            "position": "2",
                                            "Driver": {
                                                "driverId": "bob-racer",
                                                "code": "BOB",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                        },
                                    ],
                                }
                            ],
                        },
                    }
                }
            if path.endswith("/qualifying.json"):
                self.urls.append(url)
                return {
                    "MRData": {
                        "total": "2",
                        "RaceTable": {
                            "season": str(CURRENT_YEAR),
                            "Races": [
                                {
                                    "round": "1",
                                    "QualifyingResults": [
                                        {
                                            "position": "1",
                                            "Driver": {
                                                "driverId": "alice-racer",
                                                "code": "ALC",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                            "Q3": "1:10.000",
                                        },
                                        {
                                            "position": "2",
                                            "Driver": {
                                                "driverId": "bob-racer",
                                                "code": "BOB",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                            "Q3": "1:12.000",
                                        },
                                    ],
                                }
                            ],
                        },
                    }
                }
            raise AssertionError(f"unexpected URL: {url}")

    getter = CompletionGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    events = loader.get_event_schedule(CURRENT_YEAR)

    assert events[0]["date"] == today
    assert events[0]["completed"] is None
    assert not any(urlsplit(url).path.endswith("/results.json") for url in getter.urls)

    loader.get_official_roster(CURRENT_YEAR)
    loader._season_data(CURRENT_YEAR)
    assert loader.get_event_schedule(CURRENT_YEAR)[0]["completed"] is True


def test_invalid_utf8_is_a_current_season_data_error() -> None:
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: b"\xff",
    )

    with pytest.raises(CurrentSeasonDataError, match="UTF-8"):
        loader._fetch_json("https://example.invalid/data.json")
    with pytest.raises(CurrentSeasonDataError, match="UTF-8"):
        loader._fetch_text("https://example.invalid/data.html")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), 121])
def test_rejects_invalid_or_excessive_request_timeout(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        CurrentSeasonDataLoader(current_year=CURRENT_YEAR, timeout=timeout)


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), 301])
def test_rejects_invalid_or_excessive_live_fetch_budget(budget: float) -> None:
    with pytest.raises(ValueError, match="fetch_budget"):
        CurrentSeasonDataLoader(current_year=CURRENT_YEAR, fetch_budget=budget)


def test_live_fetch_budget_caps_timeout_and_aborts_before_next_request(monkeypatch) -> None:
    now = [100.0]
    monkeypatch.setattr(current_module.time, "monotonic", lambda: now[0])
    calls: list[tuple[str, float]] = []

    def getter(url: str, **kwargs: object) -> object:
        calls.append((url, float(kwargs["timeout"])))
        now[0] += 2.0
        return {"ok": True}

    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=getter,
        timeout=10.0,
        fetch_budget=3.0,
    )
    assert loader._fetch_json("https://example.invalid/one") == {"ok": True}
    now[0] = loader._fetch_budget_deadline
    with pytest.raises(CurrentSeasonDataError, match="budget exhausted"):
        loader._fetch_json("https://example.invalid/two")

    assert len(calls) == 1
    assert calls[0][1] == 3.0


def test_live_fetch_budget_checks_custom_getter_after_it_returns(monkeypatch) -> None:
    now = [10.0]
    monkeypatch.setattr(current_module.time, "monotonic", lambda: now[0])
    calls: list[str] = []

    def getter(url: str, **_: object) -> object:
        calls.append(url)
        now[0] = 20.0
        return {"ok": True}

    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=getter,
        fetch_budget=5.0,
    )
    with pytest.raises(CurrentSeasonDataError, match="budget exhausted"):
        loader._fetch_json("https://example.invalid/slow")
    assert calls == ["https://example.invalid/slow"]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("Finished", True),
        ("Lapped", True),
        ("+1 Lap", True),
        ("+12 Laps", True),
        ("Engine", False),
        ("Gearbox", False),
        ("Hydraulics", False),
        ("Collision", False),
        ("Accident", False),
    ],
)
def test_classification_uses_explicit_finish_and_failure_statuses(
    status: str, expected: bool
) -> None:
    assert CurrentSeasonDataLoader._classified({"status": status, "position": "4"}) is expected


def test_faster_teammate_gets_higher_skill_rating() -> None:
    class PaceGetter(FakeGetter):
        def __call__(self, url: str, **kwargs: object) -> object:
            path = urlsplit(url).path
            if path.endswith("/results.json"):
                return {
                    "MRData": {
                        "total": "2",
                        "RaceTable": {
                            "season": str(CURRENT_YEAR),
                            "Races": [
                                {
                                    "round": "1",
                                    "Results": [
                                        {
                                            "position": "1",
                                            "status": "Finished",
                                            "Driver": {
                                                "driverId": "alice-racer",
                                                "code": "ALC",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                            "FastestLap": {"AverageSpeed": {"speed": "220"}},
                                        },
                                        {
                                            "position": "2",
                                            "status": "Finished",
                                            "Driver": {
                                                "driverId": "bob-racer",
                                                "code": "BOB",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                            "FastestLap": {"AverageSpeed": {"speed": "210"}},
                                        },
                                    ],
                                }
                            ]
                        },
                    }
                }
            if path.endswith("/qualifying.json"):
                return {
                    "MRData": {
                        "total": "2",
                        "RaceTable": {
                            "season": str(CURRENT_YEAR),
                            "Races": [
                                {
                                    "round": "1",
                                    "QualifyingResults": [
                                        {
                                            "position": "1",
                                            "Driver": {
                                                "driverId": "alice-racer",
                                                "code": "ALC",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                            "Q3": "1:10.000",
                                        },
                                        {
                                            "position": "2",
                                            "Driver": {
                                                "driverId": "bob-racer",
                                                "code": "BOB",
                                            },
                                            "Constructor": {
                                                "constructorId": "team-a",
                                                "name": "Team A",
                                            },
                                            "Q3": "1:12.000",
                                        },
                                    ],
                                }
                            ]
                        },
                    }
                }
            return super().__call__(url, **kwargs)

    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=PaceGetter(),
    )
    stats = loader.get_weighted_driver_stats(
        CURRENT_YEAR,
        "Future Grand Prix",
        form_races=1,
        track_weight=0.0,
        form_weight=0.7,
        quali_weight=0.3,
    )

    assert stats["ALC"].driver_skill_rating > stats["BOB"].driver_skill_rating


def test_failed_season_feeds_fail_closed_once_and_are_not_provenance_successes() -> None:
    class FailingGetter(FakeGetter):
        def __call__(self, url: str, **kwargs: object) -> object:
            path = urlsplit(url).path
            if path.endswith("/results.json") or path.endswith("/qualifying.json"):
                self.urls.append(url)
                raise OSError("feed unavailable")
            return super().__call__(url, **kwargs)

    getter = FailingGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)

    with pytest.raises(RuntimeError, match="feed unavailable"):
        loader.get_weighted_driver_stats(CURRENT_YEAR, "Future Grand Prix")
    first_call_count = len(getter.urls)
    with pytest.raises(RuntimeError, match="feed unavailable"):
        loader.get_weighted_driver_stats(CURRENT_YEAR, "Future Grand Prix")

    result_urls = [url for url in getter.urls if urlsplit(url).path.endswith("/results.json")]
    qualifying_urls = [
        url for url in getter.urls if urlsplit(url).path.endswith("/qualifying.json")
    ]
    assert len(result_urls) == 1
    assert len(qualifying_urls) == 1
    assert len(getter.urls) == first_call_count
    provenance = loader.get_provenance()
    assert all(url not in provenance["urls"] for url in result_urls + qualifying_urls)
    assert provenance["fetched_at"]
    assert provenance["fresh_fetch"] is False


def test_live_snapshot_excludes_selected_target_and_honors_form_races(monkeypatch) -> None:
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    loader._calendar = [
        {"round": 1, "completed": True},
        {"round": 2, "completed": True},
        {"round": 3, "completed": True},
    ]
    loader._roster = [
        {"id": "A", "name": "Alice One", "team_name": "Team A"},
        {"id": "B", "name": "Bob Two", "team_name": "Team A"},
    ]
    monkeypatch.setattr(loader, "_standings", lambda _year: ([], []))
    monkeypatch.setattr(
        loader,
        "_season_data",
        lambda _year: (
            [
                {
                    "round": round_number,
                    "position": str(position),
                    "Driver": {"code": code},
                    "Constructor": {"name": "Team A"},
                }
                for round_number in (1, 2, 3)
                for code, position in (("A", 1), ("B", 2))
            ],
            [
                {
                    "round": round_number,
                    "Driver": {"code": code},
                    "Constructor": {"name": "Team A"},
                    "Q3": time_value,
                }
                for round_number in (1, 2, 3)
                for code, time_value in (("A", "1:10.000"), ("B", "1:11.000"))
            ],
        ),
    )

    _, _, race_rows, qualifying_rows = loader._load_season_rows(
        CURRENT_YEAR,
        target_round=3,
        form_races=1,
    )

    assert [row["round"] for row in race_rows] == [2, 2]
    assert [row["round"] for row in qualifying_rows] == [2, 2]
    assert loader.provenance["completed_rounds"] == [2]


def test_live_snapshot_uses_latest_completed_rounds_for_a_past_target(monkeypatch) -> None:
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    loader._calendar = [
        {"round": round_number, "completed": False} for round_number in (1, 2, 3, 4)
    ]
    loader._roster = [
        {"id": "A", "name": "Alice One", "team_name": "Team A"},
        {"id": "B", "name": "Bob Two", "team_name": "Team A"},
    ]
    monkeypatch.setattr(loader, "_standings", lambda _year: ([], []))
    monkeypatch.setattr(
        loader,
        "_season_data",
        lambda _year: (
            [
                {
                    "round": round_number,
                    "position": str(position),
                    "Driver": {"code": code},
                    "Constructor": {"name": "Team A"},
                }
                for round_number in (1, 2, 3, 4, 9)
                for code, position in (("A", 1), ("B", 2))
            ],
            [
                {
                    "round": round_number,
                    "Driver": {"code": code},
                    "Constructor": {"name": "Team A"},
                    "Q3": time_value,
                }
                for round_number in (1, 2, 3, 4, 9)
                for code, time_value in (("A", "1:10.000"), ("B", "1:11.000"))
            ],
        ),
    )

    _, _, race_rows, qualifying_rows = loader._load_season_rows(
        CURRENT_YEAR,
        target_round=2,
        form_races=2,
    )

    # Round 2 is the selected target, while rounds 3 and 4 are newer live
    # samples and therefore remain eligible for this current snapshot.
    assert [row["round"] for row in race_rows] == [3, 3, 4, 4]
    assert [row["round"] for row in qualifying_rows] == [3, 3, 4, 4]
    assert loader.provenance["completed_rounds"] == [3, 4]


def test_zeroed_signal_weights_do_not_change_team_pace(monkeypatch) -> None:
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    roster = [
        {"id": "A1", "name": "Alice One", "team_name": "Team A"},
        {"id": "A2", "name": "Avery Two", "team_name": "Team A"},
        {"id": "B1", "name": "Bella One", "team_name": "Team B"},
        {"id": "B2", "name": "Boris Two", "team_name": "Team B"},
    ]
    loader._calendar = [
        {"round": 1, "race": "Round One", "completed": True},
        {"round": 2, "race": "Target Grand Prix", "completed": True},
    ]
    monkeypatch.setattr(loader, "get_official_roster", lambda _year: roster)
    monkeypatch.setattr(loader, "_standings", lambda _year: ([], []))
    monkeypatch.setattr(
        loader,
        "_season_data",
        lambda _year: (
            [
                    {
                        "round": 1,
                        "position": "1",
                        "Driver": {"code": code},
                        "Constructor": {
                            "name": "Team A" if code.startswith("A") else "Team B"
                        },
                        "status": "Finished",
                    "FastestLap": {"AverageSpeed": {"speed": str(speed)}},
                }
                for code, speed in (("A1", 220), ("A2", 210), ("B1", 220), ("B2", 210))
            ],
            [
                {
                    "round": 1,
                    "Driver": {"code": code},
                    "Constructor": {
                        "name": "Team A" if code.startswith("A") else "Team B"
                    },
                    "Q3": time_value,
                }
                for code, time_value in (
                    ("A1", "1:10.000"),
                    ("A2", "1:11.000"),
                    ("B1", "1:10.000"),
                    ("B2", "1:11.000"),
                )
            ]
            + [
                {
                    "round": 2,
                    "Driver": {"code": code},
                    "Constructor": {
                        "name": "Team A" if code.startswith("A") else "Team B"
                    },
                    "Q3": time_value,
                }
                for code, time_value in (
                    ("A1", "1:10.000"),
                    ("A2", "1:11.000"),
                    ("B1", "1:15.000"),
                    ("B2", "1:16.000"),
                )
            ],
        ),
    )

    def weighted(**weights):
        return loader.get_weighted_driver_stats(
            CURRENT_YEAR,
            "Target Grand Prix",
            form_races=1,
            **weights,
        )

    zeroed = weighted(track_weight=0.0, form_weight=0.0, quali_weight=0.0)
    assert zeroed["A1"].team_pace_rating == zeroed["B1"].team_pace_rating

    # The only unequal bucket is target qualifying. It changes team pace only
    # when the explicit track weight enables that bucket.
    target_qualifying = weighted(track_weight=1.0, form_weight=0.0, quali_weight=0.0)
    assert target_qualifying["A1"].team_pace_rating > target_qualifying["B1"].team_pace_rating

    target_qualifying_disabled = weighted(track_weight=0.0, form_weight=1.0, quali_weight=1.0)
    assert (
        target_qualifying_disabled["A1"].team_pace_rating
        == target_qualifying_disabled["B1"].team_pace_rating
    )


def test_form_rows_use_their_own_constructor_for_transferred_teammates(monkeypatch) -> None:
    roster = [
        {"id": "ALC", "name": "Alice Current", "team_name": "Team B"},
        {"id": "BOB", "name": "Bob Current", "team_name": "Team B"},
        {"id": "CAR", "name": "Cara Current", "team_name": "Team A"},
        {"id": "DAN", "name": "Dan Current", "team_name": "Team A"},
    ]

    def make_loader(alice_qualifying: str) -> CurrentSeasonDataLoader:
        loader = CurrentSeasonDataLoader(
            current_year=CURRENT_YEAR,
            http_getter=lambda *_args, **_kwargs: {},
        )
        loader._calendar = [
            {"round": 1, "race": "Form Grand Prix"},
            {"round": 2, "race": "Target Grand Prix"},
        ]
        monkeypatch.setattr(loader, "get_official_roster", lambda _year: roster)
        monkeypatch.setattr(loader, "_standings", lambda _year: ([], []))
        result_rows = [
            {
                "round": 1,
                "position": str(position),
                "Driver": {"code": code},
                "Constructor": {"name": team},
                "FastestLap": {"AverageSpeed": {"speed": "220"}},
            }
            for position, code, team in (
                (1, "ALC", "Team A"),
                (2, "BOB", "Team B"),
                (3, "CAR", "Team A"),
                (4, "DAN", "Team A"),
            )
        ]
        qualifying_rows = [
            {
                "round": 1,
                "position": str(position),
                "Driver": {"code": code},
                "Constructor": {"name": team},
                "Q3": value,
            }
            for position, code, team, value in (
                (1, "ALC", "Team A", alice_qualifying),
                (2, "BOB", "Team B", "1:20.000"),
                (3, "CAR", "Team A", "1:10.000"),
                (4, "DAN", "Team A", "1:11.000"),
            )
        ]
        qualifying_rows.extend(
            {
                "round": 2,
                "position": str(position),
                "Driver": {"code": code},
                "Constructor": {"name": team},
                "Q3": "1:10.000",
            }
            for position, code, team in (
                (1, "ALC", "Team B"),
                (2, "BOB", "Team B"),
                (3, "CAR", "Team A"),
                (4, "DAN", "Team A"),
            )
        )
        monkeypatch.setattr(
            loader,
            "_season_data",
            lambda _year: (result_rows, qualifying_rows),
        )
        return loader

    stats_slow_alice = make_loader("1:30.000").get_weighted_driver_stats(
        CURRENT_YEAR,
        "Target Grand Prix",
        form_races=1,
        track_weight=0.0,
        form_weight=0.0,
        quali_weight=1.0,
    )
    stats_fast_alice = make_loader("1:00.000").get_weighted_driver_stats(
        CURRENT_YEAR,
        "Target Grand Prix",
        form_races=1,
        track_weight=0.0,
        form_weight=0.0,
        quali_weight=1.0,
    )

    # Bob is currently on Team B but has no same-team row in the historical
    # round. Alice's old Team A result must not become Bob's teammate signal.
    assert stats_slow_alice["BOB"].driver_skill_rating == pytest.approx(
        stats_fast_alice["BOB"].driver_skill_rating
    )
    assert (
        stats_slow_alice["ALC"].driver_skill_rating
        != stats_fast_alice["ALC"].driver_skill_rating
    )


@pytest.mark.parametrize("non_finish", ["Retired", "Collision", "Did not start", "Engine"])
def test_finish_evidence_stays_with_its_constructor_and_is_not_mechanical_risk(
    monkeypatch, non_finish,
) -> None:
    roster = [
        {"id": "ALC", "name": "Alice Current", "team_name": "Team B"},
        {"id": "BOB", "name": "Bob Current", "team_name": "Team B"},
        {"id": "CAR", "name": "Cara Current", "team_name": "Team A"},
        {"id": "DAN", "name": "Dan Current", "team_name": "Team A"},
    ]
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    loader._calendar = [
        {"round": 1, "race": "Form Grand Prix"},
        {"round": 2, "race": "Target Grand Prix"},
    ]
    monkeypatch.setattr(loader, "get_official_roster", lambda _year: roster)
    monkeypatch.setattr(loader, "_standings", lambda _year: ([], []))

    result_rows = []
    for position, code, team, status in (
        (1, "ALC", "Team A", non_finish),
        (2, "BOB", "Team B", "Finished"),
        (3, "CAR", "Team A", "Finished"),
        (4, "DAN", "Team A", "Finished"),
    ):
        result_rows.append(
            {
                "round": 1,
                "position": str(position),
                "Driver": {"code": code},
                "Constructor": {"name": team},
                "status": status,
                "FastestLap": {"AverageSpeed": {"speed": "220"}},
            }
        )
    qualifying_rows = [
        {
            "round": 1,
            "position": str(position),
            "Driver": {"code": code},
            "Constructor": {"name": team},
            "Q3": "1:10.000",
        }
        for position, code, team in (
            (1, "ALC", "Team A"),
            (2, "BOB", "Team B"),
            (3, "CAR", "Team A"),
            (4, "DAN", "Team A"),
        )
    ]
    monkeypatch.setattr(
        loader,
        "_season_data",
        lambda _year: (result_rows, qualifying_rows),
    )

    stats = loader.get_weighted_driver_stats(
        CURRENT_YEAR,
        "Target Grand Prix",
        form_races=1,
    )
    cars = loader.create_cars_from_stats(stats)

    assert stats["ALC"].team_finish_rate == pytest.approx(1.0)
    assert stats["BOB"].team_finish_rate == pytest.approx(1.0)
    assert stats["CAR"].team_finish_rate == pytest.approx(2 / 3)
    assert stats["DAN"].team_finish_rate == pytest.approx(2 / 3)
    assert stats["ALC"].team_result_count == 1
    assert stats["CAR"].team_result_count == 3
    assert stats["CAR"].team_finished_count == 2
    assert stats["ALC"].dnf_rate == 1.0  # Personal all-cause evidence remains available.
    for item in stats.values():
        assert item.team_reliability_source == "model_prior"
        assert item.team_reliability == pytest.approx(0.95)
    for car in cars.values():
        assert car.reliability == pytest.approx(0.95)
        assert set(car.component_reliability_map().values()) == {0.95}

    no_form = loader.get_weighted_driver_stats(CURRENT_YEAR, "Target Grand Prix", form_races=0)
    for item in no_form.values():
        assert item.team_finish_rate is None
        assert item.team_result_count == item.team_finished_count == 0
        assert item.team_reliability == 0.95
        assert item.team_reliability_source == "model_prior"


def test_explicit_mechanical_reliability_inputs_still_create_configured_cars():
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=lambda *a, **k: {})
    stats = current_module.DriverStats(
        driver_id="A", driver_name="A", team_id="team", team_name="Team",
        team_reliability=0.8, team_finish_rate=0.1,
    )
    car = loader.create_cars_from_stats({"A": stats})["team"]
    assert stats.team_reliability_source == "provided"
    assert car.reliability == 0.8
    assert set(car.component_reliability_map().values()) == {0.8}


def test_incomplete_or_identity_less_rounds_do_not_enter_form_or_provenance(monkeypatch) -> None:
    roster = [
        {"id": "A", "name": "Alice One", "team_name": "Team A"},
        {"id": "B", "name": "Bob Two", "team_name": "Team A"},
        {"id": "C", "name": "Cara Three", "team_name": "Team B"},
        {"id": "D", "name": "Dan Four", "team_name": "Team B"},
    ]
    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    loader._calendar = [
        {"round": round_number, "race": name}
        for round_number, name in (
            (1, "Usable Grand Prix"),
            (2, "Target Grand Prix"),
            (3, "Short Result Grand Prix"),
            (4, "Short Qualifying Grand Prix"),
            (5, "Identity-less Grand Prix"),
        )
    ]
    monkeypatch.setattr(loader, "get_official_roster", lambda _year: roster)
    monkeypatch.setattr(loader, "_standings", lambda _year: ([], []))

    def result_row(round_number: int, code: str, position: int) -> dict[str, object]:
        team = "Team A" if code in {"A", "B"} else "Team B"
        return {
            "round": round_number,
            "position": str(position),
            "Driver": {"code": code},
            "Constructor": {"name": team},
            "status": "Finished",
            "FastestLap": {"AverageSpeed": {"speed": "220"}},
        }

    def qualifying_row(round_number: int, code: str, position: int) -> dict[str, object]:
        team = "Team A" if code in {"A", "B"} else "Team B"
        return {
            "round": round_number,
            "position": str(position),
            "Driver": {"code": code},
            "Constructor": {"name": team},
            "Q3": "1:10.000",
        }

    complete_codes = ("A", "B", "C", "D")
    result_rows = [result_row(1, code, index) for index, code in enumerate(complete_codes, 1)]
    result_rows.append(result_row(3, "A", 1))
    result_rows.extend(result_row(4, code, index) for index, code in enumerate(complete_codes, 1))
    result_rows.extend(
        {
            "round": 5,
            "position": str(index),
            "Driver": {},
            "Constructor": {"name": "Team A"},
        }
        for index in range(1, 5)
    )
    qualifying_rows = [
        qualifying_row(1, code, index) for index, code in enumerate(complete_codes, 1)
    ]
    qualifying_rows.extend(
        qualifying_row(2, code, index) for index, code in enumerate(complete_codes, 1)
    )
    qualifying_rows.extend(
        qualifying_row(3, code, index) for index, code in enumerate(complete_codes, 1)
    )
    qualifying_rows.append(qualifying_row(4, "A", 1))
    qualifying_rows.extend(
        {
            "round": 5,
            "position": str(index),
            "Driver": {},
            "Constructor": {"name": "Team A"},
            "Q3": "1:10.000",
        }
        for index in range(1, 5)
    )
    monkeypatch.setattr(loader, "_season_data", lambda _year: (result_rows, qualifying_rows))

    stats = loader.get_weighted_driver_stats(
        CURRENT_YEAR,
        "Target Grand Prix",
        form_races=10,
        track_weight=0.0,
        form_weight=1.0,
        quali_weight=1.0,
    )

    assert all(item.current_season_starts == 1 for item in stats.values())
    assert all(item.qualifying_samples == 1 for item in stats.values())
    assert loader.provenance["completed_rounds"] == [1]
    assert loader.provenance["qualifying_rounds"] == [1, 2, 3]


def test_paginated_rows_merge_all_pages() -> None:
    calls: list[str] = []

    def getter(url: str, **_: object) -> object:
        calls.append(url)
        offset = int(parse_qs(urlsplit(url).query).get("offset", ["0"])[0])
        rows = list(range(100)) if offset == 0 else [100, 101]
        return {"MRData": {"total": "102", "Rows": rows}}

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    rows = loader._fetch_paginated("https://example.invalid/rows", ("Rows",))

    assert rows == list(range(102))
    assert len(calls) == 2
    assert "offset=100" in calls[1]


def test_paginated_rows_fail_closed_when_total_exceeds_received_rows() -> None:
    def getter(url: str, **_: object) -> object:
        return {"MRData": {"total": "3", "Rows": [1, 2]}}

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    with pytest.raises(CurrentSeasonDataError, match="declared 3"):
        loader._fetch_paginated("https://example.invalid/rows", ("Rows",))


def test_paginated_rows_fail_closed_on_repeated_page() -> None:
    calls: list[str] = []

    def getter(url: str, **_: object) -> object:
        calls.append(url)
        return {"MRData": {"Rows": list(range(100))}}

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    with pytest.raises(CurrentSeasonDataError, match="repeated"):
        loader._fetch_paginated("https://example.invalid/rows", ("Rows",))
    assert len(calls) == 2


def test_paginated_rows_have_a_finite_page_cap_without_totals() -> None:
    calls: list[str] = []

    def getter(url: str, **_: object) -> object:
        calls.append(url)
        offset = int(parse_qs(urlsplit(url).query).get("offset", ["0"])[0])
        return {"MRData": {"Rows": list(range(offset, offset + 100))}}

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    with pytest.raises(CurrentSeasonDataError, match="page cap"):
        loader._fetch_paginated("https://example.invalid/rows", ("Rows",))
    assert len(calls) == 100


def test_paginated_rows_without_totals_still_accept_a_short_final_page() -> None:
    calls: list[str] = []

    def getter(url: str, **_: object) -> object:
        calls.append(url)
        offset = int(parse_qs(urlsplit(url).query).get("offset", ["0"])[0])
        rows = list(range(100)) if offset == 0 else [100, 101]
        return {"MRData": {"Rows": rows}}

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    assert loader._fetch_paginated("https://example.invalid/rows", ("Rows",)) == list(range(102))
    assert len(calls) == 2


def test_season_result_pages_merge_a_round_split_across_pages() -> None:
    def getter(url: str, **_: object) -> object:
        offset = int(parse_qs(urlsplit(url).query).get("offset", ["0"])[0])
        rows = [
            {"position": str(offset + index + 1), "Driver": {"code": f"D{offset + index:02d}"}}
            for index in range(100 if offset == 0 else 2)
        ]
        return {
            "MRData": {
                "total": "102",
                "RaceTable": {
                    "season": str(CURRENT_YEAR),
                    "Races": [{"round": "1", "Results": rows}],
                },
            }
        }

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    rows = loader._fetch_season_collection(
        CURRENT_YEAR,
        "results",
        ("Results", "results"),
    )

    assert len(rows) == 102
    assert {row["round"] for row in rows} == {1}


def test_season_result_pagination_fails_closed_when_total_exceeds_received_rows() -> None:
    def getter(url: str, **_: object) -> object:
        return {
            "MRData": {
                "total": "3",
                "RaceTable": {
                    "season": str(CURRENT_YEAR),
                    "Races": [
                        {
                            "round": "1",
                            "Results": [{"position": "1"}, {"position": "2"}],
                        }
                    ]
                },
            }
        }

    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    with pytest.raises(CurrentSeasonDataError, match="declared 3"):
        loader._fetch_season_collection(CURRENT_YEAR, "results", ("Results",))


def test_future_event_builds_models_without_target_results() -> None:
    getter = FakeGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)

    track_stats = loader.get_track_stats(CURRENT_YEAR, "Future Grand Prix")
    driver_stats = loader.get_weighted_driver_stats(CURRENT_YEAR, "Future Grand Prix")
    drivers = loader.create_drivers_from_stats(driver_stats)
    cars = loader.create_cars_from_stats(driver_stats)
    track = loader.create_track_from_stats(track_stats)

    assert len(driver_stats) == 2
    assert {driver.id for driver in drivers} == {"ALC", "BOB"}
    assert set(cars) == {"team_a"}
    assert track.id == "future_venue"
    assert track.total_laps > 0
    assert not any("/4/results" in url or "/4/qualifying" in url for url in getter.urls)
    provenance = loader.get_provenance()
    assert provenance["source"] == "jolpica+formula1.com"
    assert provenance["season"] == CURRENT_YEAR
    assert provenance["fetched_at"]


def test_completed_track_keeps_scheduled_lap_count_after_shortened_result() -> None:
    class ShortenedRaceGetter(FakeGetter):
        def __call__(self, url: str, **kwargs: object) -> object:
            path = urlsplit(url).path
            if path.endswith("/results.json"):
                return {
                    "MRData": {
                        "total": "1",
                        "RaceTable": {
                            "season": str(CURRENT_YEAR),
                            "Races": [
                                {
                                    "round": "1",
                                    "Results": [
                                        {
                                            "laps": "68",
                                            "FastestLap": {"Time": "1:20.000"},
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                }
            if path.endswith("/qualifying.json"):
                return {
                    "MRData": {
                        "total": "0",
                        "RaceTable": {"season": str(CURRENT_YEAR), "Races": []},
                    }
                }
            return super().__call__(url, **kwargs)

    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=ShortenedRaceGetter(),
    )
    loader._calendar = [
        {
            "round": 1,
            "race": "Canadian Grand Prix",
            "circuit_id": "montreal",
            "circuit_name": "Circuit Gilles Villeneuve",
            "country": "Canada",
            "completed": True,
        }
    ]

    stats = loader.get_track_stats(CURRENT_YEAR, "Canadian Grand Prix")

    assert stats.total_laps == 70
    assert stats.fastest_lap == 80.0


def test_current_venue_profiles_use_active_aero_and_neutral_fallback() -> None:
    profiles = CurrentSeasonDataLoader.VENUE_PROFILES
    assert not {"bahrain", "sakhir", "jeddah", "imola"}.intersection(profiles)
    assert profiles["monaco"]["active_aero"] == 0
    assert set(profiles["monaco"]) == {
        "lap",
        "laps",
        "pit",
        "overtake",
        "tire",
        "sc",
        "weather",
        "active_aero",
    }

    loader = CurrentSeasonDataLoader(
        current_year=CURRENT_YEAR,
        http_getter=lambda *_args, **_kwargs: {},
    )
    fallback = loader._venue_profile({"circuit_id": "a-current-replacement"})
    assert fallback["active_aero"] == 2


def test_supplied_driver_evidence_assembly_is_pure_and_matches_live(monkeypatch):
    import copy

    getter = FakeGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    assembled = loader._build_driver_stats
    captured = {}

    def capture(**kwargs):
        captured.update(copy.deepcopy(kwargs))
        return assembled(**kwargs)

    monkeypatch.setattr(loader, "_build_driver_stats", capture)
    live = loader.get_weighted_driver_stats(CURRENT_YEAR, 4)
    before_inputs = copy.deepcopy(captured)
    cache_names = ("_driver_stats", "_track_stats", "_calendar", "_roster",
                   "_driver_standings", "_constructor_standings", "_round_results",
                   "_round_qualifying", "_season_results", "_season_qualifying", "_fetched_at")
    before_caches = {name: copy.deepcopy(getattr(loader, name)) for name in cache_names}
    before_urls = list(getter.urls)

    def unexpected(*args, **kwargs):
        pytest.fail("Supplied evidence assembly must not fetch or read live evidence")

    for name in ("_event_for_race", "get_official_roster", "_load_season_rows", "_season_data",
                 "_round_data", "_standings"):
        monkeypatch.setattr(loader, name, unexpected)
    direct = assembled(**captured)
    assert direct == live
    assert captured == before_inputs
    assert {name: getattr(loader, name) for name in cache_names} == before_caches
    assert getter.urls == before_urls
    first_id = next(iter(direct))
    direct[first_id].driver_skill_rating = .01
    assert assembled(**captured) == live
    assert loader._driver_stats == live


def test_static_track_assembly_ignores_primed_live_cache_and_fetching(monkeypatch):
    import copy

    getter = FakeGetter()
    loader = CurrentSeasonDataLoader(current_year=CURRENT_YEAR, http_getter=getter)
    event = loader._event_for_race(CURRENT_YEAR, 4)
    event["completed"] = True
    profile_lap = float(loader._venue_profile(event)["lap"])
    calibrated = loader._track_stats_from_event(CURRENT_YEAR, event, fastest_lap=profile_lap - 10)
    loader._track_stats[calibrated.track_id] = calibrated
    before_event = copy.deepcopy(event)
    before_cache = copy.deepcopy(loader._track_stats)
    before_urls = list(getter.urls)

    def unexpected(*args, **kwargs):
        pytest.fail("Static venue assembly must not read target results")

    monkeypatch.setattr(loader, "_round_data", unexpected)
    static = loader._track_stats_from_event(CURRENT_YEAR, event)
    assert static.fastest_lap == static.avg_lap_time == profile_lap
    assert static != calibrated
    assert loader._track_stats == before_cache
    assert event == before_event
    assert getter.urls == before_urls
    static.fastest_lap = 1
    assert loader._track_stats_from_event(CURRENT_YEAR, event).fastest_lap == profile_lap
