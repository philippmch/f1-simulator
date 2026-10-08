"""Fresh published race grids, available only after Grand Prix qualifying."""

import re
from datetime import timedelta

from f1sim.data.current import CurrentSeasonDataError, _normalise_text
from f1sim.data.practice import _official_event_base, _ResultsTable, _timestamp
from f1sim.simulation.execution import validate_starting_grid


class _GridTable(_ResultsTable):
    def __init__(self):
        super().__init__()
        self.text = []
        self.paragraphs = []
        self._paragraph = None
        self._hidden = 0

    def handle_starttag(self, tag, attrs):
        super().handle_starttag(tag, attrs)
        if tag in ("script", "style"):
            self._hidden += 1
        elif tag == "p" and not self._hidden:
            self._paragraph = []

    def handle_endtag(self, tag):
        super().handle_endtag(tag)
        if tag in ("script", "style"):
            self._hidden = max(0, self._hidden - 1)
        elif tag == "p" and self._paragraph is not None:
            self.paragraphs.append(" ".join(self._paragraph))
            self._paragraph = None

    def handle_data(self, data):
        super().handle_data(data)
        if not self._hidden:
            self.text.append(data)
            if self._paragraph is not None:
                self._paragraph.append(data)


def _pit_lane_notes(table, drivers):
    """Resolve explicit instructions, including subjects carried across sentences."""
    aliases = {}
    for driver in drivers:
        for label in (driver.id, driver.name, driver.name.split()[-1]):
            aliases.setdefault(_normalise_text(label), set()).add(driver.id)

    def identities(subject):
        subject = _normalise_text(subject)
        found = set()
        for label, ids in aliases.items():
            if re.search(rf"(?<!\w){re.escape(label)}(?!\w)", subject):
                if len(ids) != 1:
                    raise CurrentSeasonDataError("Ambiguous pit-lane start identity")
                found.update(ids)
        return found

    required = set()
    instructions = 0
    for paragraph in table.paragraphs:
        previous = set()
        for sentence in re.split(r"[.;]", paragraph):
            matches = identities(sentence)
            if re.search(r"\bstart\w*\b.{0,100}\bpit[\s-]+lane\b", sentence, re.I):
                # Only the subject preceding the instruction identifies starters.
                subject = re.split(r"\b(?:required|must|will|starts?|starting)\b",
                                   sentence, maxsplit=1, flags=re.I)[0]
                subject = re.sub(r"^\s*Note\s*[-:]\s*", "", subject, flags=re.I).strip()
                starters = identities(subject) if subject else previous
                for part in re.split(r",|\band\b|&", subject, flags=re.I):
                    if part.strip() and not identities(part):
                        raise CurrentSeasonDataError("Unresolved official pit-lane start identity")
                if not starters:
                    raise CurrentSeasonDataError("Unresolved official pit-lane start instruction")
                required.update(starters)
                instructions += 1
            if matches:
                previous = matches
    if (not instructions and re.search(r"\bstart\w*\b.{0,100}\bpit[\s-]+lane\b",
                                       " ".join(table.text), re.I)):
        raise CurrentSeasonDataError("Unresolved official pit-lane start instruction")
    return required


def parse_starting_grid(text, *, year, loader, drivers, include_pit_lane=False):
    """Require the complete published roster and resolve separate pit-lane starts.

    List-only callers still reject pit starts, so they cannot silently treat a
    pit starter as an ordinary last grid slot. Context callers receive both.
    """
    table = _GridTable()
    table.feed(text)
    if not any(str(year) in heading and "STARTING GRID" in heading.upper()
               for heading in table.headings):
        raise CurrentSeasonDataError("Official starting grid has a different season or session")
    pit_lane = _pit_lane_notes(table, drivers)
    roster = [{"id": d.id, "name": d.name, "code": d.id, "team_name": d.team_id}
              for d in drivers]
    _, aliases = loader._build_active_driver_map(roster, {})
    teams = {d.id: d.team_id for d in drivers}
    slots, identities, pit_rows = {}, set(), []
    for cells, _ in table.rows:
        if len(cells) not in (4, 5) or cells[0].lower().startswith("pos"):
            continue
        code = cells[2].split()[-1] if cells[2].split() else ""
        driver = loader._resolve_row_driver({"Driver": {"code": code}}, aliases)
        if driver is None:
            raise CurrentSeasonDataError("Official starting grid contains an unknown driver")
        pit_slot = cells[0].upper().replace(" ", "") in ("PL", "PITLANE")
        if pit_slot:
            pit_lane.add(driver)
        elif not cells[0].isdecimal() or not 1 <= int(cells[0]) <= len(drivers):
            raise CurrentSeasonDataError("Official starting grid has an unsupported starting slot")
        slot = None if pit_slot else int(cells[0])
        if (slot in slots or driver in identities
                or loader._team_id(cells[3]) != teams[driver]):
            raise CurrentSeasonDataError("Conflicting official starting grid driver or team")
        if slot is not None:
            slots[slot] = driver
        if driver in pit_lane:
            pit_rows.append((slot, driver))
        identities.add(driver)
    if (identities != set(teams)
            or not pit_lane and set(slots) != set(range(1, len(drivers) + 1))):
        raise CurrentSeasonDataError("Official starting grid is incomplete for the modeled roster")
    if pit_lane and not include_pit_lane:
        raise CurrentSeasonDataError("Pit-lane starting grids require their separate start context")
    if pit_lane == identities:
        raise CurrentSeasonDataError("A published grid must include ordinary grid starters")
    ordinary = [slots[p] for p in sorted(slots) if slots[p] not in pit_lane]
    # Numeric published slots establish the initial pit queue; PL rows preserve
    # their published table order. A later race-day queue change is unobserved.
    pit_order = [driver for _, driver in sorted(pit_rows, key=lambda row: (
        row[0] is None, row[0] if row[0] is not None else 0,
    ))]
    try:
        grid = validate_starting_grid(ordinary + pit_order, teams)
        return {"starting_grid": grid, "pit_lane_starters": pit_order} if include_pit_lane else grid
    except ValueError as error:
        raise CurrentSeasonDataError(str(error)) from error


def fetch_current_starting_grid(loader, year, event, drivers, *, now):
    """Keep fresh GETs within the loader budget; never use sprint qualifying."""
    loader._assert_current_year(year)
    qualifying = _timestamp(event.get("sessions", {}).get("Qualifying"))
    if qualifying is None or qualifying + timedelta(hours=1) > now:
        return None
    index_url = f"https://www.formula1.com/en/results/{year}/races"
    base = _official_event_base(loader._fetch_text(index_url), year, event)
    url = f"{base}/starting-grid"
    grid = parse_starting_grid(loader._fetch_text(url), year=year, loader=loader,
                              drivers=drivers, include_pit_lane=True)
    return {**grid, "source_url": url, "fetched_at": now.isoformat(),
            "qualifying_started_at": qualifying.isoformat(),
            "year": year, "round": int(event["round"])}
