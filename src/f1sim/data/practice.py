"""Fresh official practice evidence for the current-season qualifying model."""

from __future__ import annotations

import math
import re
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

from f1sim.data.current import CurrentSeasonDataError, _as_float, _as_int, _parse_time_seconds

_SESSIONS = {1: "FirstPractice", 2: "SecondPractice", 3: "ThirdPractice"}
# Calendar identities mapped to official results navigation, including races
# whose result table is still empty. These are venue names, never archived grids.
_OFFICIAL_SLUGS = {
    "albert_park": "australia", "shanghai": "china", "suzuka": "japan",
    "bahrain": "bahrain", "jeddah": "saudi-arabia", "miami": "miami",
    "villeneuve": "canada", "monaco": "monaco", "catalunya": "barcelona-catalunya",
    "red_bull_ring": "austria", "silverstone": "great-britain", "spa": "belgium",
    "hungaroring": "hungary", "zandvoort": "netherlands", "monza": "italy",
    "madring": "spain", "madrid": "spain", "baku": "azerbaijan",
    "marina_bay": "singapore", "americas": "united-states", "rodriguez": "mexico",
    "interlagos": "brazil", "vegas": "las-vegas", "losail": "qatar",
    "yas_marina": "abu-dhabi",
}


class _ResultsTable(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.headings = [], []
        self.event_links = set()
        self._row = self._cell = self._heading = None
        self._links = []

    def handle_starttag(self, tag, attrs):
        if tag == "h1":
            self._heading = []
        elif tag == "tr":
            self._row, self._links = [], []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "a":
            link = dict(attrs).get("href", "")
            if isinstance(link, str) and re.fullmatch(
                r"/en/results/\d{4}/races/\d+/[a-z0-9-]+/race-result", link,
            ):
                self.event_links.add(link)
            if self._row is not None:
                self._links.append(link)

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)
        if self._heading is not None:
            self._heading.append(data)

    def handle_endtag(self, tag):
        if tag == "h1" and self._heading is not None:
            self.headings.append(" ".join(self._heading).strip())
            self._heading = None
        elif tag in ("td", "th") and self._cell is not None:
            self._row.append(" ".join(" ".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if len(self.rows) >= 1000:
                raise CurrentSeasonDataError("Too many official practice table rows")
            self.rows.append((self._row, self._links))
            self._row = self._cell = None


def _timestamp(session):
    if not isinstance(session, dict) or not session.get("date") or not session.get("time"):
        return None
    try:
        text = f"{session['date']}T{session['time']}".replace("Z", "+00:00")
        value = datetime.fromisoformat(text)
        return value.astimezone(timezone.utc) if value.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def eligible_practice_sessions(event, now):
    """Completed practice before the first qualifying, including sprint qualifying.

    Unknown session times provide no evidence of completion. The one-hour
    practice duration also prevents fetching a classification during a session.
    """
    sessions = event.get("sessions", {})
    qualifying = [_timestamp(sessions.get(name))
                  for name in ("Qualifying", "SprintQualifying", "SprintShootout")]
    if (event.get("sprint") and not any(
        _timestamp(sessions.get(name)) for name in ("SprintQualifying", "SprintShootout")
    )):
        return []
    known = [value for value in qualifying if value is not None]
    if not known:
        return []
    cutoff = min(known)
    candidates = []
    for number, name in _SESSIONS.items():
        start = _timestamp(sessions.get(name))
        if start is not None and start + timedelta(hours=1) <= min(cutoff, now):
            candidates.append((start, number))
    return [number for _, number in sorted(candidates, reverse=True)]


def _official_event_base(text, year, event):
    table = _ResultsTable()
    table.feed(text)
    if not any(f"{year} RACE RESULTS" in heading.upper() for heading in table.headings):
        raise CurrentSeasonDataError("Official results index does not match the requested season")
    try:
        race_date = datetime.fromisoformat(event["date"]).date()
    except (KeyError, TypeError, ValueError) as error:
        raise CurrentSeasonDataError("Practice event lacks a race date") from error
    matches = set()
    expected_slug = _OFFICIAL_SLUGS.get(event.get("circuit_id"))
    if expected_slug is None:
        # Rescheduled races may keep their Grand Prix name at a different venue.
        # Match the explicit calendar name to an observed navigation slug.
        prefix = re.split(r"\s+grand\s+prix\b", str(event.get("race", "")), flags=re.I)[0]
        named_slug = re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-")
        if any(link.startswith(f"/en/results/{year}/")
               and link.split("/")[-2] == named_slug for link in table.event_links):
            expected_slug = named_slug
    for cells, links in table.rows:
        if len(cells) < 2:
            continue
        try:
            displayed = datetime.strptime(f"{cells[1]} {year}", "%d %b %Y").date()
        except ValueError:
            continue
        for link in links:
            if not re.fullmatch(rf"/en/results/{year}/races/\d+/[a-z0-9-]+/race-result", link):
                continue
            vegas = (event.get("circuit_id") == "vegas" and "/las-vegas/" in link
                     and displayed + timedelta(days=1) == race_date)
            if (displayed == race_date or vegas) and (
                expected_slug is None or link.split("/")[-2] == expected_slug
            ):
                matches.add("https://www.formula1.com" + link.rsplit("/", 1)[0])
    if not matches and expected_slug is not None:
        # Before the race, the winner/date table can omit this event entirely.
        # Resolve its observed current-year navigation link by calendar venue.
        matches = {"https://www.formula1.com" + link.rsplit("/", 1)[0]
                   for link in table.event_links
                   if link.startswith(f"/en/results/{year}/")
                   and link.split("/")[-2] == expected_slug}
    if len(matches) != 1:
        raise CurrentSeasonDataError("Official practice event date is missing or ambiguous")
    return matches.pop()


def parse_practice_table(text, *, year, number, loader, drivers):
    """Resolve official driver codes and teams without adding reserve drivers."""
    table = _ResultsTable()
    table.feed(text)
    if not any(str(year) in heading and f"PRACTICE {number}" in heading.upper()
               for heading in table.headings):
        raise CurrentSeasonDataError("Official practice classification has a different session")
    roster = [{"id": driver.id, "name": driver.name, "code": driver.id,
               "team_name": driver.team_id} for driver in drivers]
    _, aliases = loader._build_active_driver_map(roster, {})
    teams = {driver.id: driver.team_id for driver in drivers}
    output, identities, best = [], set(), None
    for cells, _ in table.rows:
        if len(cells) != 6 or not cells[0].isdecimal():
            continue
        position, laps = _as_int(cells[0]), _as_int(cells[5])
        value = cells[4].removesuffix("s").strip()
        if value.startswith("+"):
            gap = _as_float(value[1:])
            seconds = best + gap if best is not None and gap is not None and gap >= 0 else None
        else:
            seconds = _parse_time_seconds(value)
            if seconds is not None and best is None:
                best = seconds
        if (seconds is None or not math.isfinite(seconds) or not 30 <= seconds <= 240
                or position is None or not 1 <= position <= 30
                or laps is None or not 0 <= laps <= 200):
            continue
        code = cells[2].split()[-1] if cells[2].split() else ""
        driver = loader._resolve_row_driver({"Driver": {"code": code}}, aliases)
        if driver is None:
            continue
        if driver in identities or loader._team_id(cells[3]) != teams[driver]:
            raise CurrentSeasonDataError("Conflicting official practice driver or team identity")
        identities.add(driver)
        output.append({"driver": driver, "position": position, "lap_seconds": seconds,
                       "laps": laps})
    return output


def fetch_current_practice(loader, year, event, drivers, *, now):
    """Fetch only a completed target practice; keep all HTTP work in loader budgets."""
    loader._assert_current_year(year)
    choices = eligible_practice_sessions(event, now)
    if not choices:
        return None
    index_url = f"https://www.formula1.com/en/results/{year}/races"
    base = _official_event_base(loader._fetch_text(index_url), year, event)
    for number in choices:
        url = f"{base}/practice/{number}"
        try:
            rows = parse_practice_table(loader._fetch_text(url), year=year, number=number,
                                        loader=loader, drivers=drivers)
        except CurrentSeasonDataError:
            continue
        minimum = .5 if event.get("sprint") and number == 1 else .8
        if len(rows) >= max(2, math.ceil(minimum * len(drivers))):
            start = _timestamp(event["sessions"][_SESSIONS[number]])
            return {"year": year, "round": int(event["round"]), "session_number": number,
                    "source_url": url, "rows": rows,
                    "practice_started_at": start.isoformat(),
                    "fetched_at": now.isoformat(), "identity_coverage": len(rows) / len(drivers)}
    return None


def build_current_qualifying_history(loader, events, results, qualifying, *, before_round):
    """Normalize earlier current-season evidence on each event's own GP roster."""
    from f1sim.analysis.qualifying_history import _roster, _unique_rows

    output = []
    for event in sorted(events, key=lambda row: int(row["round"])):
        number = int(event["round"])
        if not 1 <= number < before_round:
            continue
        r_rows = _unique_rows(loader, [r for r in results if _as_int(r.get("round")) == number])
        if len(r_rows) < 2:
            continue
        q_rows = _unique_rows(loader, [r for r in qualifying if _as_int(r.get("round")) == number])
        roster, aliases = loader._build_active_driver_map(_roster(loader, r_rows), {})
        q_by_driver = {}
        for row in q_rows:
            name = loader._resolve_row_driver(row, aliases)
            if name is None:
                continue
            if name in q_by_driver:
                raise CurrentSeasonDataError("Conflicting earlier qualifying driver aliases")
            q_by_driver[name] = row
        rows = []
        for result in r_rows:
            name = loader._resolve_row_driver(result, aliases)
            if name is None or name not in roster:
                raise CurrentSeasonDataError("Earlier GP has ambiguous qualifying driver identity")
            q = q_by_driver.get(name, {})
            time_value = _parse_time_seconds(q.get("Q1"))
            if time_value is not None and not 30 <= time_value <= 240:
                raise CurrentSeasonDataError("Earlier Q1 time is out of range")
            rows.append({"driver": name, "team": loader._row_team_id(result),
                         "qualifying_position": loader._row_position(q), "q1": time_value,
                         "points": _as_float(result.get("points"))})
        output.append({"round": number, "circuit": event["circuit_id"], "rows": rows})
    return output
