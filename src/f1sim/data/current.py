"""Live, current-season Formula 1 data.

This module deliberately has a small, explicit live-data contract.  A
:class:`CurrentSeasonDataLoader` is a live-data
adapter: it only accepts the UTC calendar year in which it is running, keeps
responses in memory for the lifetime of the adapter, and never writes a cache
or substitutes another season.

The public methods retain the names used by the simulator's original data
adapter where that is useful.  They return models built from the current
official roster, current-season Jolpica results, and a small venue-physics
configuration (the latter is configuration, not archived result data).
"""

from __future__ import annotations

import copy
import html
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timezone
from html.parser import HTMLParser
from statistics import median
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from f1sim.models import ActiveAeroZone, Car, Driver, Sector, Track

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
FORMULA1_TEAMS_URL = "https://www.formula1.com/en/teams"
DEFAULT_HTTP_TIMEOUT = 20.0
JOLPICA_PAGE_SIZE = 100
# A current F1 season is comfortably below these limits.  Keeping explicit
# bounds here prevents a broken provider (or an ignored ``offset`` parameter)
# from turning one live request into an unbounded loop or memory allocation.
MAX_PAGINATION_PAGES = 100
MAX_PAGINATION_ROWS = 10_000
DEFAULT_LIVE_FETCH_BUDGET = 60.0
MAX_LIVE_FETCH_BUDGET = 300.0
MAX_HTTP_TIMEOUT = 120.0
RESULT_ROUND_COMPLETENESS = 0.8


class CurrentSeasonDataError(RuntimeError):
    """Raised when live current-season data cannot be loaded safely."""


class DriverStats(BaseModel):
    """Current-season driver summary consumed by the simulator.

    The first fields intentionally match the simulator's existing model so
    callers that only need basic timing statistics can migrate without a
    second translation layer.  The additional fields make the provenance and
    the pace decomposition explicit: ``driver_skill_rating`` is teammate
    relative, while ``team_pace_rating`` comes from constructor and team-level
    race/qualifying data.
    """

    driver_id: str
    driver_name: str
    team_id: str
    team_name: str
    avg_lap_time: float = Field(default=90.0, gt=0)
    lap_time_std: float = Field(default=0.3, ge=0)
    avg_sector1: float = Field(default=30.0, ge=0)
    avg_sector2: float = Field(default=30.0, ge=0)
    avg_sector3: float = Field(default=30.0, ge=0)
    pit_stop_avg: float = Field(default=2.5, ge=0)
    pit_stop_std: float = Field(default=0.3, ge=0)
    dnf_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    sample_size: int = Field(default=0, ge=0)

    # Explicitly separated pace components.
    driver_skill_rating: float = Field(default=0.9, ge=0.0, le=1.0)
    team_pace_rating: float = Field(default=0.85, ge=0.0, le=1.0)
    consistency_rating: float = Field(default=0.94, ge=0.0, le=1.0)
    wet_skill_modifier: float = Field(default=1.0, ge=0.5, le=1.5)
    overtaking_skill: float = Field(default=0.8, ge=0.0, le=1.0)
    tire_management: float = Field(default=0.8, ge=0.0, le=1.0)
    team_reliability: float = Field(default=0.95, ge=0.0, le=1.0)
    constructor_points: float = Field(default=0.0, ge=0)
    current_season_starts: int = Field(default=0, ge=0)
    classified_finishes: int = Field(default=0, ge=0)
    qualifying_samples: int = Field(default=0, ge=0)
    season: int | None = None
    source: str = "jolpica+formula1.com"
    fetched_at: str | None = None


class TrackStats(BaseModel):
    """Current-season event and venue configuration summary."""

    track_id: str
    track_name: str
    country: str = "Unknown"
    total_laps: int = Field(default=57, gt=0)
    fastest_lap: float = Field(default=90.0, gt=0)
    avg_lap_time: float = Field(default=90.0, gt=0)
    sector1_avg: float = Field(default=30.0, gt=0)
    sector2_avg: float = Field(default=30.0, gt=0)
    sector3_avg: float = Field(default=30.0, gt=0)
    pit_lane_time: float = Field(default=20.0, gt=0)
    safety_car_rate: float = Field(default=0.3, ge=0.0, le=1.0)
    active_aero_zones: int = Field(default=2, ge=0)
    overtake_difficulty: float = Field(default=0.5, ge=0.0, le=1.0)
    tire_stress: float = Field(default=0.5, ge=0.0, le=1.0)
    weather_variability: float = Field(default=0.2, ge=0.0, le=1.0)
    latitude: float | None = None
    longitude: float | None = None
    event_date: str | None = None
    sprint: bool = False
    season: int | None = None
    source: str = "jolpica+formula1.com"
    fetched_at: str | None = None


HttpGetter = Callable[..., Any]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _slug(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", text)


def _normalise_text(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None or value == "":
            return default
        if isinstance(value, bool):
            return default
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _validated_duration(value: Any, name: str, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0 or parsed > maximum:
        raise ValueError(f"{name} must be finite, greater than zero, and at most {maximum:g}")
    return parsed


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _parse_time_seconds(value: Any) -> float | None:
    """Parse Jolpica's ``M:SS.mmm`` and ``H:M:S.mmm`` time values."""

    if value is None:
        return None
    if isinstance(value, Mapping):
        value = value.get("time") or value.get("Time")
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    text = str(value).strip()
    if not text or text.startswith("+") or text.lower() in {"nan", "none"}:
        return None
    try:
        parts = text.split(":")
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        return float(text)
    except ValueError:
        return None


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for key in keys:
        if key.lower() in lowered:
            return lowered[key.lower()]
    return default


def _find_lists(node: Any, names: Sequence[str]) -> list[Any]:
    """Find the first list stored under one of ``names`` in nested JSON."""

    wanted = {name.lower() for name in names}
    if isinstance(node, Mapping):
        for key, value in node.items():
            if str(key).lower() in wanted and isinstance(value, list):
                return value
        for value in node.values():
            found = _find_lists(value, names)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_lists(value, names)
            if found:
                return found
    return []


def _find_total(node: Any) -> int | None:
    if isinstance(node, Mapping):
        for key, value in node.items():
            if str(key).lower() == "total":
                parsed = _as_int(value)
                if parsed is not None:
                    return parsed
        for value in node.values():
            found = _find_total(value)
            if found is not None:
                return found
    return None


def _declared_seasons(node: Any) -> tuple[tuple[int, ...], bool]:
    """Return explicit provider season metadata found in a response.

    Jolpica/Ergast payloads put ``season`` on different wrapper objects
    depending on the endpoint (``MRData``, ``RaceTable``, or individual
    races).  Walk the complete decoded tree, but only inspect season-shaped
    keys so ordinary event years/dates do not accidentally become metadata.
    The boolean reports a present-but-invalid declaration; treating that as
    an error is safer than silently accepting an untrustworthy feed.
    """

    seasons: list[int] = []
    invalid = False

    def visit(value: Any) -> None:
        nonlocal invalid
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
                if key_normalized in {"season", "seasonyear"}:
                    candidate = child
                    if isinstance(candidate, Mapping):
                        candidate = _first(candidate, "year", "season", "value", default=None)
                    parsed = _as_int(candidate)
                    if parsed is None:
                        if str(candidate or "").strip():
                            invalid = True
                    else:
                        seasons.append(parsed)
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(node)
    return tuple(seasons), invalid


class _RosterHTMLParser(HTMLParser):
    """Collect driver/team links from the Formula1.com teams page.

    The Formula1 site changes its React markup periodically.  Keeping this
    parser intentionally structural (link paths and visible text) makes it
    useful for both the current page and small fixture pages in tests.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: list[dict[str, str]] = []
        self._anchor: dict[str, Any] | None = None
        self._anchor_depth = 0
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._depth += 1
        if tag.lower() != "a":
            return
        attr_map = {str(key).lower(): value or "" for key, value in attrs}
        self._anchor = {"href": attr_map.get("href", ""), "text": []}
        self._anchor_depth = self._depth

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor["text"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._anchor is not None:
            self.anchors.append(
                {
                    "href": str(self._anchor.get("href", "")),
                    "text": re.sub(r"\s+", " ", " ".join(self._anchor.get("text", []))).strip(),
                }
            )
            self._anchor = None
        self._depth = max(0, self._depth - 1)


class CurrentSeasonDataLoader:
    """Load only fresh data for the current UTC F1 season.

    ``http_getter`` is intentionally injectable.  It receives ``url``,
    ``headers`` and ``timeout`` keyword arguments and may return decoded JSON,
    text, or bytes.  A one-argument getter is also accepted for straightforward
    tests.  The default getter uses :mod:`urllib.request` and sends explicit
    no-cache headers on every request.
    """

    SOURCE = "jolpica+formula1.com"
    JOLPICA_BASE_URL = JOLPICA_BASE_URL
    FORMULA1_TEAMS_URL = FORMULA1_TEAMS_URL

    # Configuration for physics, not an archived result set. Every known
    # location has an entry and the fallback is deliberately conservative.
    VENUE_PROFILES: Mapping[str, Mapping[str, float | int]] = MappingProxyType(
        {
            "albert_park": {
                "lap": 81.0,
                "laps": 58,
                "pit": 21.0,
                "overtake": 0.55,
                "tire": 0.55,
                "sc": 0.32,
                "weather": 0.35,
                "active_aero": 4,
            },
            "melbourne": {
                "lap": 81.0,
                "laps": 58,
                "pit": 21.0,
                "overtake": 0.55,
                "tire": 0.55,
                "sc": 0.32,
                "weather": 0.35,
                "active_aero": 4,
            },
            "villeneuve": {
                "lap": 75.0,
                "laps": 70,
                "pit": 19.0,
                "overtake": 0.4,
                "tire": 0.48,
                "sc": 0.42,
                "weather": 0.4,
                "active_aero": 2,
            },
            "shanghai": {
                "lap": 96.0,
                "laps": 56,
                "pit": 22.0,
                "overtake": 0.42,
                "tire": 0.58,
                "sc": 0.28,
                "weather": 0.28,
                "active_aero": 2,
            },
            "suzuka": {
                "lap": 91.0,
                "laps": 53,
                "pit": 20.0,
                "overtake": 0.62,
                "tire": 0.68,
                "sc": 0.26,
                "weather": 0.36,
                "active_aero": 2,
            },
            "miami": {
                "lap": 91.0,
                "laps": 57,
                "pit": 21.0,
                "overtake": 0.48,
                "tire": 0.58,
                "sc": 0.34,
                "weather": 0.32,
                "active_aero": 3,
            },
            "monaco": {
                "lap": 74.0,
                "laps": 78,
                "pit": 19.0,
                "overtake": 0.95,
                "tire": 0.4,
                "sc": 0.52,
                "weather": 0.28,
                "active_aero": 0,
            },
            "barcelona": {
                "lap": 80.0,
                "laps": 66,
                "pit": 21.0,
                "overtake": 0.7,
                "tire": 0.62,
                "sc": 0.24,
                "weather": 0.2,
                "active_aero": 2,
            },
            "catalunya": {
                "lap": 80.0,
                "laps": 66,
                "pit": 21.0,
                "overtake": 0.7,
                "tire": 0.62,
                "sc": 0.24,
                "weather": 0.2,
                "active_aero": 2,
            },
            "montreal": {
                "lap": 75.0,
                "laps": 70,
                "pit": 19.0,
                "overtake": 0.4,
                "tire": 0.48,
                "sc": 0.42,
                "weather": 0.4,
                "active_aero": 2,
            },
            "spielberg": {
                "lap": 69.0,
                "laps": 71,
                "pit": 19.0,
                "overtake": 0.4,
                "tire": 0.45,
                "sc": 0.25,
                "weather": 0.32,
                "active_aero": 3,
            },
            "austria": {
                "lap": 69.0,
                "laps": 71,
                "pit": 19.0,
                "overtake": 0.4,
                "tire": 0.45,
                "sc": 0.25,
                "weather": 0.32,
                "active_aero": 3,
            },
            "red_bull_ring": {
                "lap": 69.0,
                "laps": 71,
                "pit": 19.0,
                "overtake": 0.4,
                "tire": 0.45,
                "sc": 0.25,
                "weather": 0.32,
                "active_aero": 3,
            },
            "silverstone": {
                "lap": 90.0,
                "laps": 52,
                "pit": 22.0,
                "overtake": 0.5,
                "tire": 0.72,
                "sc": 0.24,
                "weather": 0.45,
                "active_aero": 2,
            },
            "spa": {
                "lap": 106.0,
                "laps": 44,
                "pit": 22.0,
                "overtake": 0.3,
                "tire": 0.65,
                "sc": 0.3,
                "weather": 0.5,
                "active_aero": 2,
            },
            "hungaroring": {
                "lap": 78.0,
                "laps": 70,
                "pit": 22.0,
                "overtake": 0.82,
                "tire": 0.62,
                "sc": 0.26,
                "weather": 0.28,
                "active_aero": 2,
            },
            "hungary": {
                "lap": 78.0,
                "laps": 70,
                "pit": 22.0,
                "overtake": 0.82,
                "tire": 0.62,
                "sc": 0.26,
                "weather": 0.28,
                "active_aero": 2,
            },
            "zandvoort": {
                "lap": 72.0,
                "laps": 72,
                "pit": 21.0,
                "overtake": 0.72,
                "tire": 0.7,
                "sc": 0.32,
                "weather": 0.38,
                "active_aero": 2,
            },
            "monza": {
                "lap": 82.0,
                "laps": 53,
                "pit": 21.0,
                "overtake": 0.25,
                "tire": 0.38,
                "sc": 0.3,
                "weather": 0.2,
                "active_aero": 2,
            },
            "baku": {
                "lap": 105.0,
                "laps": 51,
                "pit": 21.0,
                "overtake": 0.36,
                "tire": 0.44,
                "sc": 0.48,
                "weather": 0.2,
                "active_aero": 2,
            },
            "singapore": {
                "lap": 99.0,
                "laps": 62,
                "pit": 24.0,
                "overtake": 0.86,
                "tire": 0.58,
                "sc": 0.62,
                "weather": 0.36,
                "active_aero": 3,
            },
            "marina_bay": {
                "lap": 99.0,
                "laps": 62,
                "pit": 24.0,
                "overtake": 0.86,
                "tire": 0.58,
                "sc": 0.62,
                "weather": 0.36,
                "active_aero": 3,
            },
            "austin": {
                "lap": 96.0,
                "laps": 56,
                "pit": 22.0,
                "overtake": 0.45,
                "tire": 0.6,
                "sc": 0.35,
                "weather": 0.3,
                "active_aero": 2,
            },
            "americas": {
                "lap": 96.0,
                "laps": 56,
                "pit": 22.0,
                "overtake": 0.45,
                "tire": 0.6,
                "sc": 0.35,
                "weather": 0.3,
                "active_aero": 2,
            },
            "cota": {
                "lap": 96.0,
                "laps": 56,
                "pit": 22.0,
                "overtake": 0.45,
                "tire": 0.6,
                "sc": 0.35,
                "weather": 0.3,
                "active_aero": 2,
            },
            "mexico": {
                "lap": 79.0,
                "laps": 71,
                "pit": 20.0,
                "overtake": 0.4,
                "tire": 0.44,
                "sc": 0.26,
                "weather": 0.25,
                "active_aero": 3,
            },
            "rodriguez": {
                "lap": 79.0,
                "laps": 71,
                "pit": 20.0,
                "overtake": 0.4,
                "tire": 0.44,
                "sc": 0.26,
                "weather": 0.25,
                "active_aero": 3,
            },
            "interlagos": {
                "lap": 72.0,
                "laps": 71,
                "pit": 20.0,
                "overtake": 0.4,
                "tire": 0.52,
                "sc": 0.36,
                "weather": 0.42,
                "active_aero": 2,
            },
            "sao_paulo": {
                "lap": 72.0,
                "laps": 71,
                "pit": 20.0,
                "overtake": 0.4,
                "tire": 0.52,
                "sc": 0.36,
                "weather": 0.42,
                "active_aero": 2,
            },
            "las_vegas": {
                "lap": 95.0,
                "laps": 50,
                "pit": 20.0,
                "overtake": 0.34,
                "tire": 0.3,
                "sc": 0.38,
                "weather": 0.12,
                "active_aero": 2,
            },
            "vegas": {
                "lap": 95.0,
                "laps": 50,
                "pit": 20.0,
                "overtake": 0.34,
                "tire": 0.3,
                "sc": 0.38,
                "weather": 0.12,
                "active_aero": 2,
            },
            "lusail": {
                "lap": 83.0,
                "laps": 57,
                "pit": 23.0,
                "overtake": 0.58,
                "tire": 0.82,
                "sc": 0.26,
                "weather": 0.08,
                "active_aero": 2,
            },
            "losail": {
                "lap": 83.0,
                "laps": 57,
                "pit": 23.0,
                "overtake": 0.58,
                "tire": 0.82,
                "sc": 0.26,
                "weather": 0.08,
                "active_aero": 2,
            },
            "qatar": {
                "lap": 83.0,
                "laps": 57,
                "pit": 23.0,
                "overtake": 0.58,
                "tire": 0.82,
                "sc": 0.26,
                "weather": 0.08,
                "active_aero": 2,
            },
            "yas_marina": {
                "lap": 87.0,
                "laps": 58,
                "pit": 22.0,
                "overtake": 0.45,
                "tire": 0.55,
                "sc": 0.28,
                "weather": 0.08,
                "active_aero": 3,
            },
            "abu_dhabi": {
                "lap": 87.0,
                "laps": 58,
                "pit": 22.0,
                "overtake": 0.45,
                "tire": 0.55,
                "sc": 0.28,
                "weather": 0.08,
                "active_aero": 3,
            },
            "madrid": {
                "lap": 92.0,
                "laps": 57,
                "pit": 22.0,
                "overtake": 0.55,
                "tire": 0.58,
                "sc": 0.42,
                "weather": 0.3,
                "active_aero": 3,
            },
            "ifema": {
                "lap": 92.0,
                "laps": 57,
                "pit": 22.0,
                "overtake": 0.55,
                "tire": 0.58,
                "sc": 0.42,
                "weather": 0.3,
                "active_aero": 3,
            },
            "madring": {
                "lap": 92.0,
                "laps": 57,
                "pit": 22.0,
                "overtake": 0.55,
                "tire": 0.58,
                "sc": 0.42,
                "weather": 0.3,
                "active_aero": 3,
            },
            "sepang": {
                "lap": 94.0,
                "laps": 56,
                "pit": 23.0,
                "overtake": 0.38,
                "tire": 0.72,
                "sc": 0.3,
                "weather": 0.48,
                "active_aero": 2,
            },
            "kuala_lumpur": {
                "lap": 94.0,
                "laps": 56,
                "pit": 23.0,
                "overtake": 0.38,
                "tire": 0.72,
                "sc": 0.3,
                "weather": 0.48,
                "active_aero": 2,
            },
        }
    )

    def __init__(
        self,
        current_year: int | None = None,
        http_getter: HttpGetter | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        fetch_budget: float = DEFAULT_LIVE_FETCH_BUDGET,
    ) -> None:
        runtime_year = _utc_now().year
        if current_year is None:
            current_year = runtime_year
        if isinstance(current_year, bool) or not isinstance(current_year, int):
            raise TypeError("current_year must be an integer UTC calendar year")
        if current_year != runtime_year:
            raise ValueError(
                f"current_year must match the runtime UTC season ({runtime_year}); "
                f"got {current_year}"
            )
        self.current_year = current_year
        self.timeout = _validated_duration(timeout, "timeout", MAX_HTTP_TIMEOUT)
        self.fetch_budget = _validated_duration(
            fetch_budget,
            "fetch_budget",
            MAX_LIVE_FETCH_BUDGET,
        )
        self._fetch_budget_started = time.monotonic()
        self._fetch_budget_deadline = self._fetch_budget_started + self.fetch_budget
        self._http_getter = http_getter or self._default_http_get
        # Strict roster shape is enforced for real Formula1.com traffic.  A
        # custom getter is an explicit test/integration boundary and may use a
        # reduced fixture while still exercising every active-seat rule.
        self._strict_roster = http_getter is None
        self._min_request_interval = 0.26 if http_getter is None else 0.0
        self._last_request_monotonic = 0.0
        self._headers = {
            "Cache-Control": "no-cache, no-store, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "User-Agent": "f1sim-current-season/1.0",
            "Accept": "application/json, text/html;q=0.9, */*;q=0.8",
        }

        self._calendar: list[dict[str, Any]] | None = None
        self._roster: list[dict[str, Any]] | None = None
        self._driver_standings: list[dict[str, Any]] | None = None
        self._constructor_standings: list[dict[str, Any]] | None = None
        self._round_results: dict[int, list[dict[str, Any]]] = {}
        self._round_qualifying: dict[int, list[dict[str, Any]]] = {}
        self._season_results: list[dict[str, Any]] | None = None
        self._season_qualifying: list[dict[str, Any]] | None = None
        self._season_results_attempted = False
        self._season_qualifying_attempted = False
        self._season_results_error: Exception | None = None
        self._season_qualifying_error: Exception | None = None
        self._driver_stats: dict[str, DriverStats] = {}
        self._track_stats: dict[str, TrackStats] = {}
        self._fetched_at: str | None = None
        self._fetched_urls: list[str] = []
        self._failed_urls: list[str] = []
        self._last_completed_rounds: list[int] = []
        self._last_qualifying_rounds: list[int] = []

    # ------------------------------------------------------------------
    # HTTP and season validation
    # ------------------------------------------------------------------
    @staticmethod
    def _default_http_get(url: str, *, headers: Mapping[str, str], timeout: float) -> bytes:
        request = Request(url, headers=dict(headers), method="GET")
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL is a fixed API/page endpoint
            return response.read()

    def _assert_current_year(self, year: int) -> None:
        runtime_year = _utc_now().year
        if self.current_year != runtime_year:
            raise ValueError(
                "This loader was initialized for a stale UTC season "
                f"({self.current_year}); create a new loader for {runtime_year}"
            )
        if isinstance(year, bool) or not isinstance(year, int):
            raise ValueError(f"Only the current UTC season ({runtime_year}) is available")
        if year != runtime_year:
            raise ValueError(
                f"Only the current UTC season ({runtime_year}) is available; got {year}"
            )

    def _assert_runtime_year(self) -> None:
        """Reject cached models after a loader crosses a UTC New Year boundary."""

        runtime_year = _utc_now().year
        if self.current_year != runtime_year:
            raise ValueError(
                "This loader was initialized for a stale UTC season "
                f"({self.current_year}); create a new loader for {runtime_year}"
            )

    def _validate_payload_season(self, payload: Any, expected_year: int, url: str) -> None:
        """Fail closed when a live provider explicitly identifies another season."""

        seasons, invalid = _declared_seasons(payload)
        if not seasons and not invalid:
            if url not in self._failed_urls:
                self._failed_urls.append(url)
            raise CurrentSeasonDataError(f"Live response is missing season metadata: {url}")
        if invalid:
            if url not in self._failed_urls:
                self._failed_urls.append(url)
            raise CurrentSeasonDataError(f"Invalid season metadata in live response: {url}")
        mismatched = sorted({season for season in seasons if season != expected_year})
        if mismatched:
            if url not in self._failed_urls:
                self._failed_urls.append(url)
            listed = ", ".join(str(season) for season in mismatched)
            raise CurrentSeasonDataError(
                f"Live response declares season {listed}; expected current UTC season "
                f"{expected_year}: {url}"
            )

    def _remaining_fetch_budget(self, url: str) -> float:
        remaining = self._fetch_budget_deadline - time.monotonic()
        if not math.isfinite(remaining) or remaining <= 0.0:
            self._mark_failed_url(url)
            raise CurrentSeasonDataError(f"Live fetch budget exhausted before {url}")
        return min(self.timeout, remaining)

    def _call_getter(self, url: str) -> Any:
        getter = self._http_getter
        remaining = self._remaining_fetch_budget(url)
        if self._min_request_interval:
            elapsed = time.monotonic() - self._last_request_monotonic
            pacing_delay = max(0.0, self._min_request_interval - elapsed)
            if pacing_delay >= remaining:
                self._mark_failed_url(url)
                raise CurrentSeasonDataError(
                    f"Live fetch budget exhausted before pacing request: {url}"
                )
            if pacing_delay:
                time.sleep(pacing_delay)
            # Pacing itself consumes the shared budget.  Fail before invoking
            # the provider rather than sleeping through the remaining budget.
            remaining = self._remaining_fetch_budget(url)
            self._last_request_monotonic = time.monotonic()
        # The documented shape is (url, *, headers, timeout).  Small tests
        # often use a one-argument lambda, so gracefully support that too.
        try:
            payload = getter(
                url,
                headers=self._headers,
                timeout=remaining,
            )
        except TypeError as first_error:
            try:
                payload = getter(
                    url,
                    self._headers,
                    self._remaining_fetch_budget(url),
                )
            except TypeError:
                try:
                    self._remaining_fetch_budget(url)
                    payload = getter(url)
                except TypeError:
                    self._mark_failed_url(url)
                    raise first_error
                except Exception:
                    self._mark_failed_url(url)
                    raise
            except Exception:
                self._mark_failed_url(url)
                raise
        except CurrentSeasonDataError:
            raise
        except Exception:
            self._mark_failed_url(url)
            raise
        # A custom getter may ignore the timeout argument. Check the shared
        # wall-clock budget immediately so it cannot consume the next page's
        # entire allowance unnoticed.
        self._remaining_fetch_budget(url)
        self._fetched_at = _iso_now()
        self._fetched_urls.append(url)
        return payload

    def _fetch_json(self, url: str) -> Any:
        try:
            payload = self._call_getter(url)
        except CurrentSeasonDataError:
            raise
        except Exception as exc:
            raise CurrentSeasonDataError(f"Live data request failed: {url}") from exc
        if isinstance(payload, (Mapping, list)):
            return payload
        if isinstance(payload, bytes):
            try:
                payload = payload.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                if url not in self._failed_urls:
                    self._failed_urls.append(url)
                raise CurrentSeasonDataError(f"Invalid UTF-8 response from {url}") from exc
        if not isinstance(payload, str):
            raise CurrentSeasonDataError(f"Expected JSON response from {url}")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            if url not in self._failed_urls:
                self._failed_urls.append(url)
            raise CurrentSeasonDataError(f"Invalid JSON response from {url}") from exc

    def _fetch_text(self, url: str) -> str:
        try:
            payload = self._call_getter(url)
        except CurrentSeasonDataError:
            raise
        except Exception as exc:
            raise CurrentSeasonDataError(f"Live data request failed: {url}") from exc
        if isinstance(payload, bytes):
            try:
                return payload.decode("utf-8-sig")
            except UnicodeDecodeError as exc:
                if url not in self._failed_urls:
                    self._failed_urls.append(url)
                raise CurrentSeasonDataError(f"Invalid UTF-8 response from {url}") from exc
        if isinstance(payload, str):
            return payload
        if isinstance(payload, Mapping):
            # A structured fixture is useful when exercising roster parsing.
            return json.dumps(payload)
        if url not in self._failed_urls:
            self._failed_urls.append(url)
        raise CurrentSeasonDataError(f"Expected text response from {url}")

    @staticmethod
    def _with_pagination(url: str, offset: int) -> str:
        split = urlsplit(url)
        query = [(key, value) for key, value in parse_qsl(split.query, keep_blank_values=True)]
        query = [(key, value) for key, value in query if key.lower() not in {"limit", "offset"}]
        query.extend([("limit", str(JOLPICA_PAGE_SIZE)), ("offset", str(offset))])
        return urlunsplit(
            (split.scheme, split.netloc, split.path, urlencode(query), split.fragment)
        )

    @staticmethod
    def _page_signature(rows: Sequence[Any]) -> str:
        """Build a stable, order-independent fingerprint for one page."""

        try:
            encoded = [
                json.dumps(row, sort_keys=True, separators=(",", ":"), default=repr)
                for row in rows
            ]
            return "\x1f".join(sorted(encoded))
        except (TypeError, ValueError):
            return repr(sorted((repr(row) for row in rows)))

    def _mark_failed_url(self, url: str) -> None:
        if url not in self._failed_urls:
            self._failed_urls.append(url)

    def _fetch_paginated(
        self,
        url: str,
        names: Sequence[str],
        *,
        expected_season: int | None = None,
    ) -> list[Any]:
        """Fetch pages while detecting truncation, repetition, and runaway feeds."""

        rows: list[Any] = []
        offset = 0
        total: int | None = None
        page_signatures: set[str] = set()
        for page_number in range(MAX_PAGINATION_PAGES):
            page_url = self._with_pagination(url, offset)
            page = self._fetch_json(page_url)
            if expected_season is not None:
                self._validate_payload_season(page, expected_season, page_url)
            page_rows = _find_lists(page, names)
            page_total = _find_total(page)
            if page_total is not None:
                if page_total < 0 or page_total > MAX_PAGINATION_ROWS:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Invalid or excessive row total ({page_total}) in live response: "
                        f"{page_url}"
                    )
                if total is not None and page_total != total:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Live pagination total changed from {total} to {page_total}: {page_url}"
                    )
                total = page_total
            if not page_rows:
                if total is not None and len(rows) < total:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Live pagination ended at {len(rows)} rows but declared {total}: "
                        f"{page_url}"
                    )
                break
            signature = self._page_signature(page_rows)
            if signature in page_signatures:
                self._mark_failed_url(page_url)
                raise CurrentSeasonDataError(f"Live pagination repeated a page: {page_url}")
            page_signatures.add(signature)
            if len(rows) + len(page_rows) > MAX_PAGINATION_ROWS:
                self._mark_failed_url(page_url)
                raise CurrentSeasonDataError(f"Live pagination exceeded row cap: {page_url}")
            rows.extend(page_rows)
            if total is not None and len(rows) >= total:
                break
            if len(page_rows) < JOLPICA_PAGE_SIZE:
                if total is not None and len(rows) < total:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Live pagination ended at {len(rows)} rows but declared {total}: "
                        f"{page_url}"
                    )
                break
            if page_number + 1 >= MAX_PAGINATION_PAGES:
                self._mark_failed_url(page_url)
                raise CurrentSeasonDataError(f"Live pagination exceeded page cap: {page_url}")
            offset += JOLPICA_PAGE_SIZE
        else:  # pragma: no cover - the loop always raises on its final page
            raise CurrentSeasonDataError(f"Live pagination exceeded page cap: {url}")
        if total is not None and len(rows) < total:
            self._mark_failed_url(url)
            raise CurrentSeasonDataError(
                f"Live pagination received {len(rows)} rows but declared {total}: {url}"
            )
        return rows

    def refresh(self) -> None:
        """Discard in-memory responses so the next operation fetches live data."""

        self._calendar = None
        self._roster = None
        self._driver_standings = None
        self._constructor_standings = None
        self._round_results.clear()
        self._round_qualifying.clear()
        self._season_results = None
        self._season_qualifying = None
        self._season_results_attempted = False
        self._season_qualifying_attempted = False
        self._season_results_error = None
        self._season_qualifying_error = None
        self._driver_stats.clear()
        self._track_stats.clear()
        self._fetched_at = None
        self._fetched_urls.clear()
        self._failed_urls.clear()
        self._last_completed_rounds.clear()
        self._last_qualifying_rounds.clear()
        self._last_request_monotonic = 0.0
        self._fetch_budget_started = time.monotonic()
        self._fetch_budget_deadline = self._fetch_budget_started + self.fetch_budget

    @property
    def provenance(self) -> dict[str, Any]:
        """Return live-fetch metadata suitable for an API response."""

        self._assert_runtime_year()
        return {
            "source": self.SOURCE,
            "season": self.current_year,
            "fetched_at": self._fetched_at,
            "cache": "disabled",
            "fresh_fetch": bool(self._fetched_urls) and not self._failed_urls,
            "urls": list(dict.fromkeys(self._fetched_urls)),
            "completed_rounds": list(self._last_completed_rounds),
            "qualifying_rounds": list(self._last_qualifying_rounds),
        }

    def get_provenance(self) -> dict[str, Any]:
        return copy.deepcopy(self.provenance)

    # ------------------------------------------------------------------
    # Current calendar
    # ------------------------------------------------------------------
    def _calendar_url(self, year: int) -> str:
        # Jolpica's stable ``races`` resource is the canonical current
        # calendar. The alpha schedule resource has changed shape in the
        # past; using the stable endpoint keeps the live feed deterministic.
        return f"{self.JOLPICA_BASE_URL}/{year}/races/"

    @staticmethod
    def _event_cancelled(raw: Mapping[str, Any]) -> bool:
        for key in ("cancelled", "canceled", "isCancelled", "isCanceled"):
            value = _first(raw, key)
            if value is True or str(value).lower() in {"true", "yes", "cancelled", "canceled"}:
                return True
        for key in ("status", "eventStatus", "raceStatus"):
            if "cancel" in str(_first(raw, key, default="")).lower():
                return True
        labels = " ".join(
            str(_first(raw, key, default=""))
            for key in ("raceName", "eventName", "officialEventName", "meetingName", "EventName")
        ).lower()
        return "cancelled" in labels or "canceled" in labels

    @staticmethod
    def _is_testing_event(raw: Mapping[str, Any]) -> bool:
        labels = " ".join(
            str(_first(raw, key, default=""))
            for key in (
                "raceName",
                "eventName",
                "officialEventName",
                "meetingName",
                "EventName",
                "eventFormat",
            )
        ).lower()
        return "testing" in labels or "test event" in labels or "pre season test" in labels

    @staticmethod
    def _normalise_session(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, Mapping):
            return None
        result: dict[str, Any] = {}
        for key in ("date", "time", "url", "sessionName", "name"):
            if key in value:
                result[key] = value[key]
        # Preserve provider additions (e.g. sprint shootout naming) without
        # retaining a whole unbounded response tree.
        for key, item in value.items():
            if key not in result and isinstance(item, (str, int, float, bool)):
                result[str(key)] = item
        return result or None

    def _normalise_event(self, raw: Mapping[str, Any], round_number: int) -> dict[str, Any]:
        circuit = _first(raw, "Circuit", "circuit", default={})
        if not isinstance(circuit, Mapping):
            circuit = {}
        location = _first(circuit, "Location", "location", default={})
        if not isinstance(location, Mapping):
            location = {}

        race_name = str(
            _first(
                raw, "raceName", "eventName", "EventName", "name", default=f"Round {round_number}"
            )
        ).strip()
        official_name = str(
            _first(raw, "officialEventName", "OfficialEventName", "meetingName", default=race_name)
        ).strip()
        circuit_id = _slug(_first(circuit, "circuitId", "id", "circuitName", default=race_name))
        circuit_name = str(_first(circuit, "circuitName", "name", default=race_name)).strip()
        country = str(
            _first(
                location,
                "country",
                "Country",
                default=_first(raw, "country", "Country", default="Unknown"),
            )
        ).strip()
        locality = str(
            _first(
                location,
                "locality",
                "city",
                "Location",
                default=_first(raw, "location", "Location", default=""),
            )
        ).strip()
        latitude = _as_float(
            _first(location, "lat", "latitude", default=_first(raw, "lat", "latitude"))
        )
        longitude = _as_float(
            _first(
                location,
                "long",
                "lng",
                "lon",
                "longitude",
                default=_first(raw, "long", "lon", "longitude"),
            )
        )
        event_date = str(_first(raw, "date", "eventDate", "Date", default="") or "")[:10]
        sessions: dict[str, dict[str, Any]] = {}
        session_keys = (
            "FirstPractice",
            "SecondPractice",
            "ThirdPractice",
            "Sprint",
            "SprintQualifying",
            "SprintShootout",
            "Qualifying",
            "Race",
            "firstPractice",
            "secondPractice",
            "thirdPractice",
            "sprint",
            "sprintQualifying",
            "qualifying",
            "race",
        )
        seen_session_names: set[str] = set()
        for key in session_keys:
            if key.lower() in seen_session_names:
                continue
            value = _first(raw, key)
            normalised = self._normalise_session(value)
            if normalised is not None:
                sessions[key] = normalised
                seen_session_names.add(key.lower())
        sprint = bool(
            any(key.lower() in {"sprint", "sprintqualifying", "sprintshootout"} for key in sessions)
            or "sprint" in str(_first(raw, "eventFormat", "EventFormat", default="")).lower()
        )
        status = str(
            _first(raw, "status", "eventStatus", "raceStatus", default="scheduled") or "scheduled"
        )
        # Calendar dates only describe scheduling. Completion remains unknown
        # until a sufficiently complete current-season result feed has been
        # loaded; callers must not present an unfetched status as ``False``.
        completed: bool | None = None
        testing = self._is_testing_event(raw)
        return {
            "round": round_number,
            "race": race_name,
            "race_name": race_name,
            "official_name": official_name,
            "country": country,
            "location": locality,
            "latitude": latitude,
            "longitude": longitude,
            "lat": latitude,
            "lon": longitude,
            "circuit_id": circuit_id,
            "circuit_name": circuit_name,
            "date": event_date or None,
            "time": _first(raw, "time", "Time", default=None),
            "sessions": sessions,
            "sprint": sprint,
            "status": status,
            "completed": completed,
            "testing": testing,
            "slug": _slug(race_name),
            "url": _first(raw, "url", "URL", default=None),
        }

    def _sync_calendar_completion(self) -> None:
        """Mark events completed only when current-season result rows confirm them."""

        if self._calendar is None or self._season_results is None or self._roster is None:
            return
        standings_map = self._standings_driver_map(self._driver_standings or [])
        active, aliases_to_id = self._build_active_driver_map(self._roster, standings_map)
        result_ids = self._round_driver_ids(
            self._season_results,
            aliases_to_id,
            qualifying=False,
        )
        completed_rounds = {
            round_number
            for round_number, driver_ids in result_ids.items()
            if self._near_complete(len(driver_ids), len(active))
        }
        for event in self._calendar:
            round_number = _as_int(event.get("round"))
            event["completed"] = round_number is not None and round_number in completed_rounds

    def get_event_schedule(self, year: int, include_testing: bool = False) -> list[dict[str, Any]]:
        """Fetch and normalize the live Jolpica calendar for this season."""

        self._assert_current_year(year)
        if self._calendar is None:
            raw_events = self._fetch_paginated(
                self._calendar_url(year),
                ("Races", "races", "Events", "events"),
                expected_season=year,
            )
            by_round: dict[int, dict[str, Any]] = {}
            for raw in raw_events:
                if not isinstance(raw, Mapping) or self._event_cancelled(raw):
                    continue
                round_number = _as_int(_first(raw, "round", "Round", "RoundNumber"))
                if round_number is None or round_number <= 0:
                    continue
                # A replacement can reuse a round number.  Jolpica returns the
                # replacement later in the schedule; retaining the last row
                # preserves the current event identity.
                by_round[round_number] = self._normalise_event(raw, round_number)
            self._calendar = [by_round[key] for key in sorted(by_round)]
        self._sync_calendar_completion()
        events = self._calendar
        if not include_testing:
            events = [event for event in events if not event.get("testing", False)]
        return copy.deepcopy(events)

    def list_available_events(self, year: int) -> list[dict[str, Any]]:
        """Return frontend-friendly current events with session metadata."""

        return self.get_event_schedule(year)

    def resolve_race_identifier(self, year: int, race: str | int) -> int:
        """Resolve a round number, slug, country, or event name."""

        events = self.get_event_schedule(year)
        numeric = _as_int(race)
        if numeric is not None and str(race).strip() == str(numeric):
            if any(event["round"] == numeric for event in events):
                return numeric
            raise ValueError(f"Round '{race}' not found in {year} calendar")
        target = _normalise_text(race)
        if not target:
            raise ValueError(f"Race '{race}' not found in {year} calendar")
        target_words = set(target.split())
        aliases_by_event: list[tuple[dict[str, Any], set[str]]] = []
        for event in events:
            aliases = {
                _normalise_text(event.get("race")),
                _normalise_text(event.get("race_name")),
                _normalise_text(event.get("official_name")),
                _normalise_text(event.get("country")),
                _normalise_text(event.get("location")),
                _normalise_text(str(event.get("circuit_id", "")).replace("_", " ")),
                _normalise_text(str(event.get("slug", "")).replace("-", " ")),
            }
            aliases_by_event.append((event, {alias for alias in aliases if alias}))

        # Exact normalized aliases always win over fuzzy containment.  This
        # avoids a short query such as "Las" being captured by whichever
        # longer event happens to appear first in the schedule.
        exact = [event for event, aliases in aliases_by_event if target in aliases]
        exact_rounds = {int(event["round"]) for event in exact}
        if len(exact_rounds) == 1:
            return int(exact[0]["round"])
        if len(exact_rounds) > 1:
            names = ", ".join(str(event.get("race")) for event in exact)
            raise ValueError(f"Race '{race}' is ambiguous; matches: {names}")

        fuzzy: list[dict[str, Any]] = []
        for event, aliases in aliases_by_event:
            if any(
                target in alias
                or alias in target
                or target_words.issubset(set(alias.split()))
                for alias in aliases
            ):
                fuzzy.append(event)
        fuzzy_rounds = {int(event["round"]) for event in fuzzy}
        if len(fuzzy_rounds) == 1:
            return fuzzy[0]["round"]
        if len(fuzzy_rounds) > 1:
            names = ", ".join(str(event.get("race")) for event in fuzzy)
            raise ValueError(f"Race '{race}' is ambiguous; matches: {names}")
        available = ", ".join(str(event["race"]) for event in events)
        raise ValueError(
            f"Race '{race}' not found in {year} calendar. Available events: {available}"
        )

    def _event_for_race(self, year: int, race: str | int) -> dict[str, Any]:
        round_number = self.resolve_race_identifier(year, race)
        for event in self.get_event_schedule(year):
            if event["round"] == round_number:
                return event
        raise ValueError(f"Round '{race}' not found in {year} calendar")

    # ------------------------------------------------------------------
    # Official roster and Jolpica data
    # ------------------------------------------------------------------
    @staticmethod
    def _is_reserve_candidate(item: Mapping[str, Any]) -> bool:
        values = []
        for key in ("role", "status", "type", "driverType", "category", "availability"):
            value = _first(item, key)
            if value is not None:
                values.append(str(value).lower())
        for key in ("isReserve", "reserve", "isTest", "isDevelopment", "active"):
            value = _first(item, key)
            truthy = str(value).lower() in {"true", "yes", "1"}
            falsey = str(value).lower() in {"false", "no", "0"}
            if key.lower() in {"isreserve", "reserve", "istest", "isdevelopment"} and (
                value is True or truthy
            ):
                return True
            if key.lower() == "active" and (value is False or falsey):
                return True
        return any(
            any(
                term in value
                for term in ("reserve", "test driver", "development", "academy", "former")
            )
            for value in values
        )

    @staticmethod
    def _is_obvious_non_race_driver(item: Mapping[str, Any]) -> bool:
        """Filter labels that Formula1.com sometimes exposes beside seats."""

        labels = " ".join(
            str(_first(item, key, default=""))
            for key in ("id", "driver_id", "name", "fullName", "displayName", "href")
        ).lower()
        return any(
            term in labels
            for term in (
                "reserve",
                "test-driver",
                "test driver",
                "development",
                "academy",
                "former",
            )
        )

    @staticmethod
    def _driver_name(item: Mapping[str, Any]) -> str:
        given = _first(item, "givenName", "firstName", "forename", "first")
        family = _first(item, "familyName", "lastName", "surname", "last")
        if given and family:
            return f"{given} {family}".strip()
        value = _first(item, "name", "fullName", "displayName", "driverName", "title", default="")
        if isinstance(value, Mapping):
            given = _first(value, "givenName", "firstName", "given")
            family = _first(value, "familyName", "lastName", "family")
            if given or family:
                return f"{given or ''} {family or ''}".strip()
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @staticmethod
    def _team_name(item: Mapping[str, Any], context: str = "") -> str:
        value = _first(
            item,
            "teamName",
            "team_name",
            "constructorName",
            "team",
            "constructor",
            "teamTitle",
            "teamId",
            "constructorId",
            "constructor_id",
        )
        if isinstance(value, Mapping):
            value = _first(value, "name", "title", "teamName", default="")
        return str(value or context or "Unknown").strip()

    @classmethod
    def _structured_roster(cls, payload: Any) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []

        def walk(node: Any, team_context: str = "") -> None:
            if isinstance(node, list):
                for child in node:
                    walk(child, team_context)
                return
            if not isinstance(node, Mapping):
                return

            own_team = cls._team_name(node, team_context)
            nested_driver_lists = []
            for key, value in node.items():
                if str(key).lower() in {
                    "drivers",
                    "driver",
                    "roster",
                    "lineup",
                    "members",
                    "pilots",
                }:
                    nested_driver_lists.append(value)
            for child in nested_driver_lists:
                walk(child, own_team if own_team != "Unknown" else team_context)

            driver_id = _first(
                node, "abbreviation", "code", "driverCode", "shortCode", "driverId", "id"
            )
            name = cls._driver_name(node)
            url = str(_first(node, "url", "href", "profileUrl", default=""))
            has_driver_shape = bool(
                driver_id
                or _first(node, "givenName", "familyName", "firstName", "lastName")
                or "/driver" in url.lower()
                or "/drivers" in url.lower()
            )
            # A team itself often has only an id/name; do not mistake it for a
            # driver merely because it has nested data.
            if has_driver_shape and name and not cls._is_reserve_candidate(node):
                found.append(
                    {
                        "id": str(driver_id or _slug(name)),
                        "name": name,
                        "team_name": cls._team_name(node, team_context),
                        "team_id": _slug(cls._team_name(node, team_context)),
                        "active": True,
                    }
                )

            # Also inspect arbitrary nested objects (Next.js data commonly
            # wraps the useful list several levels deep).
            for key, value in node.items():
                if key not in {"drivers", "driver", "roster", "lineup", "members", "pilots"}:
                    walk(value, own_team if own_team != "Unknown" else team_context)

        walk(payload)
        return found

    @classmethod
    def _html_roster(cls, page: str) -> list[dict[str, Any]]:
        parser = _RosterHTMLParser()
        parser.feed(page)
        anchors = parser.anchors
        teams = [
            (index, anchor)
            for index, anchor in enumerate(anchors)
            if re.search(r"/teams?/", anchor["href"], flags=re.I)
            and anchor["text"]
            and _normalise_text(anchor["text"]) not in {"teams", "all teams"}
        ]
        drivers = [
            (index, anchor)
            for index, anchor in enumerate(anchors)
            if re.search(r"/drivers?/", anchor["href"], flags=re.I)
        ]
        found: list[dict[str, Any]] = []
        for index, anchor in drivers:
            href = html.unescape(anchor["href"])
            slug_match = re.search(r"/drivers?/([^/?#]+)", href, flags=re.I)
            driver_slug = slug_match.group(1) if slug_match else ""
            name = anchor["text"]
            if not name or len(_normalise_text(name).split()) < 2:
                name = driver_slug.replace("-", " ").replace("_", " ").title()
            team_name = "Unknown"
            if teams:
                # On the current page a team link is immediately before its
                # driver cards.  Nearest-link fallback handles card layouts
                # where it follows the cards.
                preceding = [item for item in teams if item[0] <= index]
                following = [item for item in teams if item[0] > index]
                # Driver links on list/card layouts follow their team link;
                # preferring the preceding section avoids assigning the final
                # driver of one team to the next team's heading merely because
                # that heading is one anchor closer.
                candidate = (preceding[-1] if preceding else following[0])[1]
                team_name = candidate["text"]
            found.append(
                {
                    "id": _slug(driver_slug or name),
                    "name": re.sub(r"\s+", " ", name).strip(),
                    "team_name": team_name,
                    "team_id": _slug(team_name),
                    "active": True,
                }
            )

        # Embedded __NEXT_DATA__ is more authoritative than link proximity
        # when available; merge both sources and let structured rows win team
        # assignments.
        script_matches = re.findall(
            r"<script[^>]+(?:id=[\"']__NEXT_DATA__[\"']|type=[\"']application/json[\"'])[^>]*>(.*?)</script>",
            page,
            flags=re.I | re.S,
        )
        for script in script_matches:
            try:
                found.extend(cls._structured_roster(json.loads(html.unescape(script))))
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
        # Formula1.com's current teams page renders each team as one link.  A
        # team-card link contains the team label and two name pairs but does
        # not necessarily contain nested ``/drivers/`` links, so the generic
        # anchor pass above intentionally cannot see those seats.
        found.extend(cls._team_card_roster(page))
        return found

    @staticmethod
    def _visible_html_text(fragment: str) -> str:
        text = re.sub(r"<script\b[^>]*>.*?</script>", " ", fragment, flags=re.I | re.S)
        text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.I | re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", html.unescape(text)).strip()

    @classmethod
    def _team_card_roster(cls, page: str) -> list[dict[str, Any]]:
        """Parse current Formula1.com cards with one team link per card."""

        # This deliberately accepts attributes in either order and all
        # whitespace/newline styles emitted by the site's server rendering.
        card_re = re.compile(
            r"<a\b(?=[^>]*\bhref\s*=\s*[\"'][^\"']*/teams?/[^\"']+[\"'])[^>]*>(.*?)</a>",
            flags=re.I | re.S,
        )
        found: list[dict[str, Any]] = []
        for match in card_re.finditer(page):
            body = match.group(1)
            href_match = re.search(
                r"\bhref\s*=\s*[\"']([^\"']*/teams?/[^\"']+)[\"']", match.group(0), flags=re.I
            )
            if href_match is None:
                continue
            href = html.unescape(href_match.group(1))
            team_match = re.search(r"<p\b[^>]*>(.*?)</p>", body, flags=re.I | re.S)
            team_name = cls._visible_html_text(team_match.group(1)) if team_match else ""
            if not team_name:
                # Card links generally start with the team label; use the
                # first visible segment before the driver name pairs.
                team_name = cls._visible_html_text(body).split("\n", 1)[0].strip()
            if not team_name:
                slug_match = re.search(r"/teams?/([^/?#]+)", href, flags=re.I)
                team_name = (
                    slug_match.group(1).replace("-", " ").title() if slug_match else "Unknown"
                )

            # In the live page each seat is a regular first-name span followed
            # by a bold family-name span.  Permit class ordering and additional
            # utility classes while retaining a fallback for equivalent text
            # markup used by future page revisions.
            span_re = re.compile(
                r"<span\b[^>]*class\s*=\s*[\"'][^\"']*body-xs-regular[^\"']*[\"'][^>]*>(.*?)</span>\s*"
                r"<span\b[^>]*class\s*=\s*[\"'][^\"']*body-xs-bold[^\"']*[\"'][^>]*>(.*?)</span>",
                flags=re.I | re.S,
            )
            names = [
                f"{cls._visible_html_text(first)} {cls._visible_html_text(last)}".strip()
                for first, last in span_re.findall(body)
            ]
            if len(names) < 2:
                spans = re.findall(
                    r"<span\b[^>]*class\s*=\s*[\"'][^\"']*(?:body-xs-regular|body-xs-bold)[^\"']*[\"'][^>]*>(.*?)</span>",
                    body,
                    flags=re.I | re.S,
                )
                visible_spans = [cls._visible_html_text(value) for value in spans]
                visible_spans = [value for value in visible_spans if value]
                names = [
                    " ".join(visible_spans[index : index + 2]).strip()
                    for index in range(0, len(visible_spans) - 1, 2)
                ]
            for name in names[:2]:
                if name and len(_normalise_text(name).split()) >= 2:
                    found.append(
                        {
                            "id": _slug(name),
                            "name": name,
                            "team_name": team_name,
                            "team_id": cls._team_id(team_name),
                            "active": True,
                        }
                    )
        return found

    @classmethod
    def _dedupe_roster(cls, entries: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        identity_index: dict[str, dict[str, Any]] = {}

        def team_key(value: Any) -> str | None:
            normalized = _normalise_text(value)
            if not normalized or normalized in {"unknown", "unassigned", "none", "na", "n a"}:
                return None
            return cls._team_id(str(value))

        for raw in entries:
            if (
                not isinstance(raw, Mapping)
                or cls._is_reserve_candidate(raw)
                or cls._is_obvious_non_race_driver(raw)
            ):
                continue
            name = cls._driver_name(raw)
            if not name:
                continue
            identifier = str(_first(raw, "id", "driver_id", "code", "abbreviation", default=""))
            identity_keys = {
                key
                for value in (identifier, name)
                for key in (_normalise_text(value), _slug(value))
                if key
            }
            existing_rows = {
                id(identity_index[key]): identity_index[key]
                for key in identity_keys
                if key in identity_index
            }
            if len(existing_rows) > 1:
                raise CurrentSeasonDataError(
                    f"Formula1.com roster aliases collide for driver '{name}'"
                )
            existing = next(iter(existing_rows.values()), None)
            team_name = cls._team_name(raw)
            row = {
                "id": identifier or _slug(name),
                "name": name,
                "team_name": team_name,
                "team_id": cls._team_id(team_name),
                "active": True,
            }
            if existing is None:
                output.append(row)
                existing = row
            else:
                old_team = team_key(existing.get("team_name"))
                new_team = team_key(team_name)
                if old_team is not None and new_team is not None and old_team != new_team:
                    raise CurrentSeasonDataError(
                        f"Conflicting non-unknown teams for roster driver '{name}'"
                    )
                if old_team is None and new_team is not None:
                    existing.update(
                        {
                            "team_name": team_name,
                            "team_id": cls._team_id(team_name),
                        }
                    )
            for key in identity_keys:
                identity_index[key] = existing
        return output

    @staticmethod
    def _team_id(team_name: str) -> str:
        cleaned = _normalise_text(team_name)
        mappings = (
            (("red bull",), "red_bull"),
            (("mclaren",), "mclaren"),
            (("ferrari",), "ferrari"),
            (("mercedes",), "mercedes"),
            (("aston martin",), "aston_martin"),
            (("alpine",), "alpine"),
            (("williams",), "williams"),
            (("racing bulls", "rb"), "rb"),
            (("audi",), "audi"),
            (("haas",), "haas"),
            (("cadillac",), "cadillac"),
        )
        for needles, value in mappings:
            if any(needle in cleaned for needle in needles):
                return value
        return _slug(team_name) or "unknown"

    def get_official_roster(self, year: int) -> list[dict[str, Any]]:
        """Fetch active race drivers from Formula1.com's teams page."""

        self._assert_current_year(year)
        if self._roster is None:
            page = self._fetch_text(self.FORMULA1_TEAMS_URL)
            if self._strict_roster and not re.search(rf"(?<!\d){self.current_year}(?!\d)", page):
                raise CurrentSeasonDataError(
                    "Formula1.com teams page did not identify the requested current season"
                )
            payload: Any = page
            # Structured fixture support, and support for a page that is
            # actually a JSON response despite a text content type.
            try:
                payload = json.loads(page)
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
            entries = (
                self._structured_roster(payload)
                if isinstance(payload, (Mapping, list))
                else self._html_roster(page)
            )
            roster = self._dedupe_roster(entries)
            if not roster:
                raise CurrentSeasonDataError(
                    "Formula1.com returned no active current-season drivers"
                )
            if self._strict_roster:
                teams = {
                    self._team_id(str(entry.get("team_name") or "Unknown")) for entry in roster
                }
                if len(roster) != 22 or len(teams) != 11:
                    raise CurrentSeasonDataError(
                        "Formula1.com current roster must contain exactly 22 active "
                        "drivers across 11 teams"
                    )
            self._roster = roster
        return copy.deepcopy(self._roster)

    # Compatibility aliases used by integrations.
    get_driver_roster = get_official_roster
    fetch_roster = get_official_roster

    def _standings(self, year: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._assert_current_year(year)
        if self._driver_standings is None:
            url = f"{self.JOLPICA_BASE_URL}/{year}/driverstandings.json"
            self._driver_standings = [
                row
                for row in self._fetch_paginated(
                    url,
                    ("DriverStandings", "driverStandings"),
                    expected_season=year,
                )
                if isinstance(row, Mapping)
            ]
        if self._constructor_standings is None:
            url = f"{self.JOLPICA_BASE_URL}/{year}/constructorstandings.json"
            self._constructor_standings = [
                row
                for row in self._fetch_paginated(
                    url,
                    ("ConstructorStandings", "constructorStandings"),
                    expected_season=year,
                )
                if isinstance(row, Mapping)
            ]
        return copy.deepcopy(self._driver_standings), copy.deepcopy(self._constructor_standings)

    @staticmethod
    def _extract_driver(item: Mapping[str, Any]) -> Mapping[str, Any]:
        nested = _first(item, "Driver", "driver", default=None)
        return nested if isinstance(nested, Mapping) else item

    @classmethod
    def _driver_aliases(cls, item: Mapping[str, Any]) -> set[str]:
        driver = cls._extract_driver(item)
        full_name = cls._driver_name(driver)
        family_name = _first(driver, "familyName", "family_name", "lastName", "surname")
        if not family_name and full_name:
            family_name = full_name.rsplit(" ", 1)[-1]
        values = [
            _first(driver, "code", "Code", "abbreviation"),
            _first(driver, "driverId", "id"),
            _first(driver, "permanentNumber"),
            family_name,
            full_name,
        ]
        aliases = {_normalise_text(value) for value in values if value}
        aliases.update(_slug(value) for value in values if value)
        return {alias for alias in aliases if alias}

    def _season_url(self, year: int, kind: str) -> str:
        return f"{self.JOLPICA_BASE_URL}/{year}/{kind}.json"

    @staticmethod
    def _extract_season_rows(payload: Any, names: Sequence[str]) -> list[dict[str, Any]]:
        """Flatten a season response while retaining each row's round.

        Ergast-compatible season endpoints wrap result rows as
        ``RaceTable.Races[].Results`` (or ``QualifyingResults``).  A page can
        end halfway through a round, so callers must merge rows from all
        pages before grouping by round.
        """

        races = _find_lists(payload, ("Races", "races", "Events", "events"))
        flattened: list[dict[str, Any]] = []
        for race in races:
            if not isinstance(race, Mapping):
                continue
            round_number = _as_int(_first(race, "round", "Round", "RoundNumber"))
            rows = _first(race, *names, default=None)
            if not isinstance(rows, list):
                continue
            for row in rows:
                if isinstance(row, Mapping):
                    flattened.append({"round": round_number, **dict(row)})
        if flattened:
            return flattened
        # Some Jolpica-compatible test/proxy payloads return rows directly;
        # preserve those too.  Their round can be supplied in each row.
        return [dict(row) for row in _find_lists(payload, names) if isinstance(row, Mapping)]

    def _fetch_season_collection(
        self, year: int, kind: str, names: Sequence[str]
    ) -> list[dict[str, Any]]:
        """Fetch one season result feed with 100-row pagination."""

        url = self._season_url(year, kind)
        rows: list[dict[str, Any]] = []
        offset = 0
        total: int | None = None
        page_signatures: set[str] = set()
        for page_number in range(MAX_PAGINATION_PAGES):
            page_url = self._with_pagination(url, offset)
            page = self._fetch_json(page_url)
            self._validate_payload_season(page, year, page_url)
            page_rows = self._extract_season_rows(page, names)
            page_total = _find_total(page)
            if page_total is not None:
                if page_total < 0 or page_total > MAX_PAGINATION_ROWS:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Invalid or excessive row total ({page_total}) in live response: "
                        f"{page_url}"
                    )
                if total is not None and page_total != total:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Live pagination total changed from {total} to {page_total}: {page_url}"
                    )
                total = page_total
            if not page_rows:
                if total is not None and len(rows) < total:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Live pagination ended at {len(rows)} rows but declared {total}: "
                        f"{page_url}"
                    )
                break
            signature = self._page_signature(page_rows)
            if signature in page_signatures:
                self._mark_failed_url(page_url)
                raise CurrentSeasonDataError(f"Live pagination repeated a page: {page_url}")
            page_signatures.add(signature)
            if len(rows) + len(page_rows) > MAX_PAGINATION_ROWS:
                self._mark_failed_url(page_url)
                raise CurrentSeasonDataError(f"Live pagination exceeded row cap: {page_url}")
            rows.extend(page_rows)
            if total is not None and len(rows) >= total:
                break
            if len(page_rows) < JOLPICA_PAGE_SIZE:
                if total is not None and len(rows) < total:
                    self._mark_failed_url(page_url)
                    raise CurrentSeasonDataError(
                        f"Live pagination ended at {len(rows)} rows but declared {total}: "
                        f"{page_url}"
                    )
                break
            if page_number + 1 >= MAX_PAGINATION_PAGES:
                self._mark_failed_url(page_url)
                raise CurrentSeasonDataError(f"Live pagination exceeded page cap: {page_url}")
            offset += JOLPICA_PAGE_SIZE
        else:  # pragma: no cover - the loop always raises on its final page
            raise CurrentSeasonDataError(f"Live pagination exceeded page cap: {url}")
        if total is not None and len(rows) < total:
            self._mark_failed_url(url)
            raise CurrentSeasonDataError(
                f"Live pagination received {len(rows)} rows but declared {total}: {url}"
            )
        # Keep the first occurrence of a row if a provider repeats the final
        # partial round on the next page, while retaining all distinct rows.
        deduplicated: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for row in rows:
            driver = self._extract_driver(row)
            key = (
                row.get("round"),
                _first(driver, "driverId", "code", "id", "givenName", default=""),
                _first(row, "position", "grid", "number", default=""),
            )
            if key not in seen:
                seen.add(key)
                deduplicated.append(row)
        return deduplicated

    def _season_data(self, year: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._assert_current_year(year)
        # Mark each feed attempted before starting pagination.  A failed feed
        # is not retried on every completed round (or on a later API call); a
        # live snapshot is either complete or explicitly unavailable.
        if not self._season_results_attempted:
            self._season_results_attempted = True
            try:
                self._season_results = self._fetch_season_collection(
                    year,
                    "results",
                    ("Results", "results", "RaceResults", "raceResults"),
                )
                self._sync_calendar_completion()
            except Exception as exc:
                self._season_results_error = exc
        if not self._season_qualifying_attempted:
            self._season_qualifying_attempted = True
            try:
                self._season_qualifying = self._fetch_season_collection(
                    year,
                    "qualifying",
                    ("QualifyingResults", "qualifyingResults", "Qualifying", "qualifying"),
                )
            except Exception as exc:
                self._season_qualifying_error = exc
        if self._season_results_error is not None or self._season_qualifying_error is not None:
            failed = []
            if self._season_results_error is not None:
                failed.append("race results")
            if self._season_qualifying_error is not None:
                failed.append("qualifying results")
            raise CurrentSeasonDataError(
                "Current-season "
                + " and ".join(failed)
                + " feed unavailable; refusing partial data"
            )
        return copy.deepcopy(self._season_results), copy.deepcopy(self._season_qualifying)

    def _round_data(
        self, year: int, round_number: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self._assert_current_year(year)
        if round_number not in self._round_results:
            raw_results, raw_qualifying = self._season_data(year)
            self._round_results[round_number] = [
                {key: value for key, value in row.items() if key != "round"}
                for row in raw_results
                if _as_int(row.get("round")) == round_number
            ]
            self._round_qualifying[round_number] = [
                {key: value for key, value in row.items() if key != "round"}
                for row in raw_qualifying
                if _as_int(row.get("round")) == round_number
            ]
        if round_number not in self._round_qualifying:
            # The branch is only reachable if a caller populated the race
            # cache manually; keep the invariant without reintroducing
            # per-round network requests.
            _, raw_qualifying = self._season_data(year)
            self._round_qualifying[round_number] = [
                {key: value for key, value in row.items() if key != "round"}
                for row in raw_qualifying
                if _as_int(row.get("round")) == round_number
            ]
        return copy.deepcopy(self._round_results[round_number]), copy.deepcopy(
            self._round_qualifying[round_number]
        )

    @staticmethod
    def _row_position(row: Mapping[str, Any]) -> int | None:
        return _as_int(_first(row, "position", "Position"))

    @classmethod
    def _row_race_metric(cls, row: Mapping[str, Any]) -> tuple[float | None, float | None]:
        fastest = _first(row, "FastestLap", "fastestLap", default={})
        if not isinstance(fastest, Mapping):
            fastest = {}
        fastest_time = _parse_time_seconds(_first(fastest, "Time", "time"))
        speed = _first(fastest, "AverageSpeed", "averageSpeed", "speed", default=None)
        if isinstance(speed, Mapping):
            speed = _first(speed, "speed", "value")
        return fastest_time, _as_float(speed)

    @classmethod
    def _row_qualifying_time(cls, row: Mapping[str, Any]) -> float | None:
        times = [
            _parse_time_seconds(_first(row, key))
            for key in ("Q1", "q1", "Q2", "q2", "Q3", "q3", "bestTime", "BestTime", "time", "Time")
        ]
        values = [value for value in times if value is not None and value > 0]
        return min(values) if values else None

    @classmethod
    def _classified(cls, row: Mapping[str, Any]) -> bool:
        status = (
            re.sub(r"\s+", " ", str(_first(row, "status", "Status", default=""))).strip().lower()
        )
        if status in {"finished", "classified", "lapped"} or re.fullmatch(
            r"\+\s*\d+\s+laps?", status
        ):
            return True
        non_finish_markers = (
            "retired",
            "did not start",
            "did not finish",
            "dnf",
            "dns",
            "accident",
            "collision",
            "crash",
            "engine",
            "gearbox",
            "hydraulic",
            "electrical",
            "mechanical",
            "failure",
            "suspension",
            "puncture",
            "wheel",
            "overheating",
            "transmission",
            "clutch",
            "brake",
            "fuel",
            "damage",
            "spun",
            "disqualified",
            "not classified",
        )
        if any(token in status for token in non_finish_markers):
            return False
        # A provider occasionally omits status. A numeric classified position
        # is useful in that case, but only after explicit failure labels above
        # have been ruled out.
        return not status and cls._row_position(row) is not None

    # ------------------------------------------------------------------
    # Stats decomposition and models
    # ------------------------------------------------------------------
    def _load_season_rows(
        self,
        year: int,
        target_round: int | None = None,
        form_races: int | None = None,
        official_roster: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]
    ]:
        """Return the latest live completed-round window from the season feeds.

        The result feed is the source of truth for which current-season rounds
        are complete.  This keeps a race-day event without published results
        from displacing a real form sample.  ``form_races`` selects the latest
        completed rounds at fetch time; only the selected target round is
        removed, so a past target still benefits from the live current form.
        """

        driver_standings, constructor_standings = self._standings(year)
        season_results, season_qualifying = self._season_data(year)
        calendar_rounds = {
            int(event["round"])
            for event in self.get_event_schedule(year)
            if _as_int(event.get("round")) is not None
        }
        roster = official_roster if official_roster is not None else self._roster
        self._last_qualifying_rounds = []
        if roster:
            standings_map = self._standings_driver_map(driver_standings)
            active, aliases_to_id = self._build_active_driver_map(roster, standings_map)
            result_ids = self._round_driver_ids(
                season_results,
                aliases_to_id,
                qualifying=False,
            )
            qualifying_ids = self._round_driver_ids(
                season_qualifying,
                aliases_to_id,
                qualifying=True,
            )
            result_rounds = {
                round_number
                for round_number, driver_ids in result_ids.items()
                if self._near_complete(len(driver_ids), len(active))
            }
            qualifying_rounds = {
                round_number
                for round_number, driver_ids in qualifying_ids.items()
                if self._near_complete(len(driver_ids), len(active))
            }
            self._last_qualifying_rounds = sorted(qualifying_rounds)
            completed_rounds = sorted(result_rounds & qualifying_rounds & calendar_rounds)
        else:
            # Without the official current field there is no safe denominator
            # for feed completeness, so do not let partial rows affect form.
            completed_rounds = []
        if target_round is not None:
            completed_rounds = [
                round_number for round_number in completed_rounds if round_number != target_round
            ]
        if form_races is not None:
            if isinstance(form_races, bool) or not isinstance(form_races, int) or form_races < 0:
                raise ValueError("form_races must be a non-negative integer")
            completed_rounds = completed_rounds[-form_races:] if form_races else []
        allowed_rounds = set(completed_rounds)
        all_results = [row for row in season_results if _as_int(row.get("round")) in allowed_rounds]
        all_qualifying = [
            row for row in season_qualifying if _as_int(row.get("round")) in allowed_rounds
        ]
        self._last_completed_rounds = completed_rounds
        return driver_standings, constructor_standings, all_results, all_qualifying

    @classmethod
    def _standings_driver_map(
        cls, rows: Sequence[Mapping[str, Any]]
    ) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        ambiguous: set[str] = set()
        for row in rows:
            for alias in cls._driver_aliases(row):
                existing = result.get(alias)
                if existing is not None and existing is not row:
                    ambiguous.add(alias)
                elif alias not in ambiguous:
                    result[alias] = row
        for alias in ambiguous:
            result.pop(alias, None)
        return result

    @classmethod
    def _constructor_map(cls, rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            constructor = (
                row.get("Constructor") if isinstance(row.get("Constructor"), Mapping) else row
            )
            if not isinstance(constructor, Mapping):
                continue
            values = [
                _first(constructor, "constructorId", "id", "name", default=""),
                _first(constructor, "name", default=""),
            ]
            for value in values:
                key = _normalise_text(value)
                if key:
                    result[key] = row
                    result[_slug(value)] = row
        return result

    @classmethod
    def _build_active_driver_map(
        cls,
        roster: Sequence[Mapping[str, Any]],
        standings_map: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
        """Build an active driver map while removing colliding aliases."""

        active: dict[str, dict[str, Any]] = {}
        aliases_by_driver: dict[str, set[str]] = {}
        for entry in roster:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            candidate = dict(entry)
            entry_aliases = cls._driver_aliases(entry) | {
                _normalise_text(name),
                _slug(name),
            }
            standings_rows = {
                id(standings_map[alias]): standings_map[alias]
                for alias in entry_aliases
                if alias in standings_map
            }
            if len(standings_rows) > 1:
                raise CurrentSeasonDataError(
                    f"Current roster driver '{name}' maps to conflicting standings identities"
                )
            standings_row = next(iter(standings_rows.values()), None)
            if standings_row is not None:
                provider_driver = cls._extract_driver(standings_row)
                code = _first(provider_driver, "code", "abbreviation", default=None)
                if code:
                    candidate["id"] = str(code).upper()
            driver_id = str(candidate.get("id") or _slug(name))
            if driver_id in active:
                raise CurrentSeasonDataError(
                    f"Current roster contains duplicate driver identity '{driver_id}'"
                )
            active[driver_id] = candidate
            aliases_by_driver[driver_id] = (
                entry_aliases | cls._driver_aliases(candidate)
            )

        aliases_to_drivers: dict[str, set[str]] = defaultdict(set)
        for driver_id, aliases in aliases_by_driver.items():
            for alias in aliases:
                if alias:
                    aliases_to_drivers[alias].add(driver_id)
        # An ambiguous surname (or any other collision) is intentionally
        # removed. Unique codes/full names remain usable for row matching.
        aliases_to_id = {
            alias: next(iter(driver_ids))
            for alias, driver_ids in aliases_to_drivers.items()
            if len(driver_ids) == 1
        }
        return active, aliases_to_id

    @classmethod
    def _row_team_id(cls, row: Mapping[str, Any]) -> str | None:
        """Return the constructor/team carried by one provider result row."""

        constructor = _first(
            row,
            "Constructor",
            "constructor",
            "Team",
            "team",
            "constructorData",
            default=None,
        )
        if isinstance(constructor, Mapping):
            value = _first(
                constructor,
                "name",
                "teamName",
                "constructorName",
                "constructorId",
                "teamId",
                "id",
                default=None,
            )
        else:
            value = constructor
        if value is None or not str(value).strip():
            value = _first(
                row,
                "constructorName",
                "constructorId",
                "teamName",
                "teamId",
                default=None,
            )
        if value is None or not str(value).strip():
            return None
        return cls._team_id(str(value))

    @classmethod
    def _resolve_row_driver(
        cls,
        row: Mapping[str, Any],
        aliases_to_id: Mapping[str, str],
    ) -> str | None:
        matches = {
            aliases_to_id[alias]
            for alias in cls._driver_aliases(row)
            if alias in aliases_to_id
        }
        return next(iter(matches)) if len(matches) == 1 else None

    @classmethod
    def _strong_driver_identity(cls, row: Mapping[str, Any]) -> str | None:
        """Return a provider identity strong enough for feed completeness."""

        driver = cls._extract_driver(row)
        for key in ("driverId", "id", "code", "abbreviation", "permanentNumber"):
            value = _normalise_text(_first(driver, key, default=""))
            if value:
                return f"{key}:{value}"
        given = _first(driver, "givenName", "firstName", "forename", "first")
        family = _first(driver, "familyName", "lastName", "surname", "last")
        full_name = _normalise_text(f"{given or ''} {family or ''}")
        return f"name:{full_name}" if given and family and full_name else None

    @classmethod
    def _round_driver_ids(
        cls,
        rows: Sequence[Mapping[str, Any]],
        aliases_to_id: Mapping[str, str],
        *,
        qualifying: bool,
    ) -> dict[int, set[str]]:
        by_round: dict[int, set[str]] = defaultdict(set)
        for row in rows:
            if cls._strong_driver_identity(row) is None:
                continue
            driver_id = cls._resolve_row_driver(row, aliases_to_id)
            round_number = _as_int(row.get("round"))
            if driver_id is None or round_number is None:
                continue
            if qualifying:
                if cls._row_qualifying_time(row) is None:
                    continue
            elif cls._row_position(row) is None:
                continue
            by_round[round_number].add(driver_id)
        return by_round

    @staticmethod
    def _near_complete(count: int, field_size: int) -> bool:
        required = max(1, math.ceil(field_size * RESULT_ROUND_COMPLETENESS))
        return count >= required

    def get_weighted_driver_stats(
        self,
        year: int,
        target_race: str | int,
        form_races: int = 3,
        track_weight: float = 0.5,
        form_weight: float = 0.3,
        quali_weight: float = 0.2,
    ) -> dict[str, DriverStats]:
        """Build current-season stats for every active official driver.

        Team pace is derived from live constructor standings plus separately
        weighted target-qualifying, recent-race, and recent-qualifying team
        signals.  Driver skill uses the residual to teammates, preventing the
        same car pace from being counted twice.  The snapshot uses the latest
        completed rounds available when fetched; ``form_races`` limits that
        window and the selected target round is excluded.  ``track_weight``
        applies only to target qualifying (when published), ``form_weight`` to
        recent race form, and ``quali_weight`` to recent qualifying.  Target
        race results are never used for driver pace.
        """

        self._assert_current_year(year)
        target_event = self._event_for_race(year, target_race)
        target_round = int(target_event["round"])
        roster = self.get_official_roster(year)
        driver_standings, constructor_standings, race_rows, quali_rows = self._load_season_rows(
            year,
            target_round,
            form_races,
            official_roster=roster,
        )
        _, all_qualifying_rows = self._season_data(year)
        target_qualifying_rows = [
            row for row in all_qualifying_rows if _as_int(row.get("round")) == target_round
        ]
        # Map each current roster seat to a stable provider-independent key.
        standings_map = self._standings_driver_map(driver_standings)
        active, aliases_to_id = self._build_active_driver_map(roster, standings_map)
        target_qualifying_ids = self._round_driver_ids(
            target_qualifying_rows,
            aliases_to_id,
            qualifying=True,
        ).get(target_round, set())
        if not self._near_complete(len(target_qualifying_ids), len(active)):
            target_qualifying_rows = []

        # Collect rows by active driver, resolving Jolpica aliases to the
        # official roster and intentionally ignoring former/reserve drivers.
        driver_races: dict[str, list[dict[str, Any]]] = defaultdict(list)
        driver_qualifying: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in race_rows:
            driver_id = self._resolve_row_driver(row, aliases_to_id)
            if driver_id:
                driver_races[driver_id].append(row)
        for row in quali_rows:
            driver_id = self._resolve_row_driver(row, aliases_to_id)
            if driver_id:
                driver_qualifying[driver_id].append(row)

        # Team-level metrics.  Qualifying and race measurements are first
        # normalized within each event; only the team residual is used here.
        team_target_qual: dict[str, list[float]] = defaultdict(list)
        team_form_race: dict[str, list[float]] = defaultdict(list)
        team_recent_qual: dict[str, list[float]] = defaultdict(list)
        driver_qual_delta: dict[str, list[float]] = defaultdict(list)
        driver_track_delta: dict[str, list[float]] = defaultdict(list)
        driver_race_delta: dict[str, list[float]] = defaultdict(list)
        team_points: dict[str, float] = defaultdict(float)
        constructor_map = self._constructor_map(constructor_standings)

        for entry in active.values():
            team_id = self._team_id(str(entry.get("team_name", "Unknown")))
            # Match official team assignment to constructor points by readable
            # name/id, but never replace the official roster assignment.
            constructor_row = constructor_map.get(_normalise_text(team_id))
            if constructor_row is None:
                constructor_row = next(
                    (
                        row
                        for key, row in constructor_map.items()
                        if team_id.replace("_", " ") in key
                    ),
                    None,
                )
            if constructor_row is not None:
                team_points[team_id] = max(
                    team_points[team_id],
                    _as_float(_first(constructor_row, "points", "Points", default=0), 0.0) or 0.0,
                )

        def group_relative(
            rows: Sequence[Mapping[str, Any]],
            value_fn: Callable[[Mapping[str, Any]], float | None],
            inverse: bool,
            target: dict[str, list[float]],
            team_target: dict[str, list[float]],
        ) -> None:
            grouped: dict[tuple[int, str], list[tuple[str, float]]] = defaultdict(list)
            for row in rows:
                driver_id = self._resolve_row_driver(row, aliases_to_id)
                if not driver_id:
                    continue
                value = value_fn(row)
                if value is None or value <= 0:
                    continue
                team = self._row_team_id(row)
                if team is None:
                    # Without an explicit row constructor there is no safe
                    # way to compare a transferred driver to current mates.
                    continue
                round_number = _as_int(row.get("round"), 0) or 0
                grouped[(round_number, team)].append((driver_id, value))
            by_round: dict[int, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
            for (round_number, team), values in grouped.items():
                by_round[round_number][team] = values
            for teams in by_round.values():
                event_reference = median(value for values in teams.values() for _, value in values)
                if event_reference <= 0:
                    continue
                for team, values in teams.items():
                    team_reference = median(value for _, value in values)
                    if team_reference <= 0:
                        continue
                    team_delta = (
                        (event_reference - team_reference) / event_reference
                        if inverse
                        else (team_reference - event_reference) / event_reference
                    )
                    team_target[team].append(team_delta)
                    for driver_id, value in values:
                        driver_delta = (
                            (team_reference - value) / team_reference
                            if inverse
                            else (value - team_reference) / team_reference
                        )
                        target[driver_id].append(driver_delta)

        group_relative(
            quali_rows,
            self._row_qualifying_time,
            True,
            driver_qual_delta,
            team_recent_qual,
        )
        group_relative(
            target_qualifying_rows,
            self._row_qualifying_time,
            True,
            driver_track_delta,
            team_target_qual,
        )

        def race_speed(row: Mapping[str, Any]) -> float | None:
            _, speed = self._row_race_metric(row)
            return speed

        group_relative(race_rows, race_speed, False, driver_race_delta, team_form_race)

        # Convert team residuals (higher means faster) and constructor points
        # to ratings with a realistic field spread.
        team_ids = {
            self._team_id(str(entry.get("team_name", "Unknown"))) for entry in active.values()
        }
        team_raw: dict[str, float] = {}
        max_points = max(team_points.values(), default=0.0)
        for team_id in team_ids:
            constructor_score = (
                team_points.get(team_id, 0.0) / max_points if max_points > 0 else 0.5
            )
            # Keep each live signal in its own bucket.  The public weights
            # control both driver and team pace, and an unavailable or zeroed
            # bucket contributes nothing rather than an implicit zero score.
            weighted_total = constructor_score
            total_weight = 1.0
            for bucket, weight in (
                (team_target_qual, max(0.0, float(track_weight))),
                (team_form_race, max(0.0, float(form_weight))),
                (team_recent_qual, max(0.0, float(quali_weight))),
            ):
                values = bucket.get(team_id)
                if values and weight > 0.0:
                    weighted_total += median(values) * weight
                    total_weight += weight
            team_raw[team_id] = weighted_total / total_weight
        raw_values = list(team_raw.values())
        raw_min, raw_max = (min(raw_values), max(raw_values)) if raw_values else (0.0, 0.0)
        team_rating: dict[str, float] = {}
        for team_id, raw in team_raw.items():
            if raw_max > raw_min:
                team_rating[team_id] = 0.76 + (raw - raw_min) / (raw_max - raw_min) * 0.24
            else:
                team_rating[team_id] = 0.86

        # Reliability belongs to the constructor that entered each car in
        # the sampled round. A transferred driver's earlier failure must not
        # be reassigned to the team they drive for today.
        team_starts: dict[str, int] = defaultdict(int)
        team_classified_finishes: dict[str, int] = defaultdict(int)
        for row in race_rows:
            row_team_id = self._row_team_id(row)
            if row_team_id is None:
                continue
            team_starts[row_team_id] += 1
            if self._classified(row):
                team_classified_finishes[row_team_id] += 1

        # Use the target venue profile as the reference only; no target-race
        # rows are required for a future event.
        profile = self._venue_profile(target_event)
        reference_lap = float(profile["lap"])
        stats: dict[str, DriverStats] = {}
        for driver_id, entry in active.items():
            team_name = str(entry.get("team_name") or "Unknown")
            team_id = self._team_id(team_name)
            qual_deltas = driver_qual_delta.get(driver_id, [])
            race_deltas = driver_race_delta.get(driver_id, [])
            components: list[float] = []
            weights: list[float] = []
            track_deltas = driver_track_delta.get(driver_id, [])
            if track_deltas:
                components.append(float(median(track_deltas)))
                weights.append(max(0.0, float(track_weight)))
            if race_deltas:
                components.append(float(median(race_deltas)))
                weights.append(max(0.0, float(form_weight)))
            if qual_deltas:
                components.append(float(median(qual_deltas)))
                weights.append(max(0.0, float(quali_weight)))
            relative_delta = (
                sum(component * weight for component, weight in zip(components, weights))
                / sum(weights)
                if components and sum(weights) > 0
                else 0.0
            )
            # Teammate relative spread is approximately 1.5% from best to
            # worst.  Keep all drivers distinct but avoid over-fitting sparse
            # current-season samples.
            driver_skill = max(0.78, min(1.0, 0.92 + relative_delta * 8.0))
            if not components:
                driver_skill = 0.9
            all_deltas = qual_deltas + race_deltas
            delta_std = 0.004 if len(all_deltas) < 2 else max(0.001, _population_std(all_deltas))
            consistency = max(0.82, min(1.0, 1.0 - delta_std * 8.0))

            rows = driver_races.get(driver_id, [])
            starts = len(rows)
            classified = sum(1 for row in rows if self._classified(row))
            dnf_rate = (starts - classified) / starts if starts else 0.0
            constructor_starts = team_starts.get(team_id, 0)
            team_reliability = (
                team_classified_finishes.get(team_id, 0) / constructor_starts
                if constructor_starts
                else 0.95
            )
            constructor_row = constructor_map.get(_normalise_text(team_id))
            constructor_points = team_points.get(team_id, 0.0)
            if constructor_row is not None:
                constructor_points = (
                    _as_float(
                        _first(constructor_row, "points", "Points", default=constructor_points),
                        constructor_points,
                    )
                    or constructor_points
                )
            avg_lap = reference_lap * (
                1.0 + (1.0 - team_rating.get(team_id, 0.86)) * 0.04 + (1.0 - driver_skill) * 0.012
            )
            lap_std = max(0.08, avg_lap * delta_std)
            stats[driver_id] = DriverStats(
                driver_id=driver_id,
                driver_name=str(entry.get("name") or driver_id),
                team_id=team_id,
                team_name=team_name,
                avg_lap_time=float(avg_lap),
                lap_time_std=float(lap_std),
                avg_sector1=float(avg_lap / 3.0),
                avg_sector2=float(avg_lap / 3.0),
                avg_sector3=float(avg_lap / 3.0),
                pit_stop_avg=2.5,
                pit_stop_std=0.3,
                dnf_rate=float(max(0.0, min(1.0, dnf_rate))),
                sample_size=starts,
                driver_skill_rating=float(driver_skill),
                team_pace_rating=float(team_rating.get(team_id, 0.86)),
                consistency_rating=float(consistency),
                wet_skill_modifier=float(max(0.95, min(1.05, 0.98 + (consistency - 0.9) * 0.2))),
                overtaking_skill=float(max(0.65, min(1.0, driver_skill - 0.02))),
                # Pace is already represented by driver skill; tyre management
                # follows consistency only to avoid double-counting speed.
                tire_management=float(max(0.7, min(1.0, consistency))),
                team_reliability=float(team_reliability),
                constructor_points=float(constructor_points),
                current_season_starts=starts,
                classified_finishes=classified,
                qualifying_samples=len(driver_qualifying.get(driver_id, [])),
                season=year,
                source=self.SOURCE,
                fetched_at=self._fetched_at,
            )
        self._driver_stats = stats
        return copy.deepcopy(stats)

    def get_driver_lap_stats(
        self, year: int, races: list[str | int] | None = None
    ) -> dict[str, DriverStats]:
        """Compatibility wrapper over current-season weighted stats."""

        events = self.get_event_schedule(year)
        if not events:
            raise CurrentSeasonDataError("Current-season calendar contains no race events")
        target = races[-1] if races else events[-1]["round"]
        return self.get_weighted_driver_stats(year, target)

    def get_track_stats(self, year: int, race: str | int) -> TrackStats:
        """Build a track model from live event metadata and venue config.

        Completed target results are used opportunistically to calibrate the
        fastest-lap value.  A future event never requests its result endpoint.
        """

        self._assert_current_year(year)
        event = self._event_for_race(year, race)
        track_id = str(event.get("circuit_id") or event.get("slug") or f"round_{event['round']}")
        if track_id in self._track_stats:
            return copy.deepcopy(self._track_stats[track_id])
        profile = self._venue_profile(event)
        fastest = float(profile["lap"])
        total_laps = int(profile["laps"])
        if event.get("completed"):
            try:
                rows, _ = self._round_data(year, int(event["round"]))
                lap_times = [
                    time_value
                    for row in rows
                    for time_value in [
                        _parse_time_seconds(
                            _first(
                                _first(row, "FastestLap", "fastestLap", default={})
                                if isinstance(
                                    _first(row, "FastestLap", "fastestLap", default={}), Mapping
                                )
                                else {},
                                "Time",
                                "time",
                            )
                        )
                    ]
                    if time_value is not None and time_value > 0
                ]
                if lap_times:
                    fastest = min(lap_times)
            except CurrentSeasonDataError:
                raise
            except Exception:
                # Configuration remains sufficient for a just-published or
                # temporarily unavailable result endpoint.
                pass
        avg_lap = float(fastest if fastest > 0 else profile["lap"])
        stats = TrackStats(
            track_id=track_id,
            track_name=str(event.get("circuit_name") or event.get("race") or track_id),
            country=str(event.get("country") or "Unknown"),
            total_laps=max(1, total_laps),
            fastest_lap=fastest,
            avg_lap_time=avg_lap,
            sector1_avg=avg_lap / 3.0,
            sector2_avg=avg_lap / 3.0,
            sector3_avg=avg_lap / 3.0,
            pit_lane_time=float(profile["pit"]),
            safety_car_rate=float(profile["sc"]),
            active_aero_zones=int(profile["active_aero"]),
            overtake_difficulty=float(profile["overtake"]),
            tire_stress=float(profile["tire"]),
            weather_variability=float(profile["weather"]),
            latitude=event.get("latitude"),
            longitude=event.get("longitude"),
            event_date=event.get("date"),
            sprint=bool(event.get("sprint")),
            season=year,
            source=self.SOURCE,
            fetched_at=self._fetched_at,
        )
        self._track_stats[track_id] = stats
        return copy.deepcopy(stats)

    def _venue_profile(self, event: Mapping[str, Any]) -> Mapping[str, float | int]:
        candidates = [
            _slug(event.get("circuit_id")),
            _slug(event.get("circuit_name")),
            _slug(event.get("location")),
            _slug(event.get("race")),
        ]
        for candidate in candidates:
            if candidate in self.VENUE_PROFILES:
                return self.VENUE_PROFILES[candidate]
            for known, profile in self.VENUE_PROFILES.items():
                if candidate and (known in candidate or candidate in known):
                    return profile
        return {
            "lap": 90.0,
            "laps": 57,
            "pit": 21.0,
            "overtake": 0.5,
            "tire": 0.5,
            "sc": 0.3,
            "weather": 0.25,
            "active_aero": 2,
        }

    def create_drivers_from_stats(
        self, stats: dict[str, DriverStats] | None = None
    ) -> list[Driver]:
        """Create deterministic Driver models from current-season stats."""

        self._assert_runtime_year()
        stats = stats if stats is not None else self._driver_stats
        if not stats:
            raise ValueError("No current-season driver stats available")
        return [
            Driver(
                id=item.driver_id,
                name=item.driver_name,
                team_id=item.team_id,
                skill_rating=item.driver_skill_rating,
                consistency=item.consistency_rating,
                wet_skill_modifier=item.wet_skill_modifier,
                overtaking_skill=item.overtaking_skill,
                tire_management=item.tire_management,
            )
            for item in stats.values()
        ]

    def create_cars_from_stats(self, stats: dict[str, DriverStats] | None = None) -> dict[str, Car]:
        """Create deterministic team cars without reusing driver pace."""

        self._assert_runtime_year()
        stats = stats if stats is not None else self._driver_stats
        if not stats:
            raise ValueError("No current-season driver stats available")
        grouped: dict[str, list[DriverStats]] = defaultdict(list)
        for item in stats.values():
            grouped[item.team_id].append(item)
        cars: dict[str, Car] = {}
        for team_id, members in grouped.items():
            first = members[0]
            team_pace = sum(member.team_pace_rating for member in members) / len(members)
            reliability = sum(member.team_reliability for member in members) / len(members)
            cars[team_id] = Car(
                team_id=team_id,
                team_name=first.team_name,
                base_pace=max(0.7, min(1.0, team_pace)),
                downforce_level=0.84,
                straight_line_speed=0.84,
                reliability=max(0.7, min(1.0, reliability)),
                engine_reliability=max(0.7, min(1.0, reliability)),
                gearbox_reliability=max(0.7, min(1.0, reliability)),
                brakes_reliability=max(0.7, min(1.0, reliability)),
                electrical_reliability=max(0.7, min(1.0, reliability)),
                cooling_reliability=max(0.7, min(1.0, reliability)),
                tire_degradation_factor=1.0,
                wet_performance=max(0.7, min(1.0, team_pace)),
                pit_stop_avg=first.pit_stop_avg,
                pit_stop_std=max(0.1, first.pit_stop_std),
            )
        return cars

    def create_track_from_stats(
        self, stats: TrackStats | None = None, track_id: str | None = None
    ) -> Track:
        """Create a Track model for any dynamically fetched current venue."""

        self._assert_runtime_year()
        if stats is None:
            if track_id and track_id in self._track_stats:
                stats = self._track_stats[track_id]
            else:
                raise ValueError("No current-season track stats available")
        return Track(
            id=stats.track_id,
            name=stats.track_name,
            country=stats.country,
            total_laps=stats.total_laps,
            base_lap_time=stats.avg_lap_time,
            pit_lane_delta=stats.pit_lane_time,
            sectors=[
                Sector(number=1, base_time=stats.sector1_avg),
                Sector(number=2, base_time=stats.sector2_avg),
                Sector(number=3, base_time=stats.sector3_avg),
            ],
            active_aero_zones=[
                ActiveAeroZone(zone_id=index + 1, sector=min(3, index % 3 + 1), time_gain=0.25)
                for index in range(stats.active_aero_zones)
            ],
            overtake_difficulty=stats.overtake_difficulty,
            tire_stress=stats.tire_stress,
            safety_car_probability=stats.safety_car_rate,
            weather_variability=stats.weather_variability,
        )


def _population_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = sum(values) / len(values)
    return (sum((value - average) ** 2 for value in values) / len(values)) ** 0.5


__all__ = [
    "CurrentSeasonDataError",
    "CurrentSeasonDataLoader",
    "DriverStats",
    "TrackStats",
]
