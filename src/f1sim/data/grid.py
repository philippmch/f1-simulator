"""Fresh published race grids, available only after Grand Prix qualifying."""

import re
from datetime import timedelta

from f1sim.data.current import CurrentSeasonDataError
from f1sim.data.practice import _official_event_base, _ResultsTable, _timestamp
from f1sim.simulation.execution import validate_starting_grid


class _GridTable(_ResultsTable):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        super().handle_data(data)
        self.text.append(data)


def parse_starting_grid(text, *, year, loader, drivers):
    """Require a complete, unambiguous modeled roster with ordinary grid slots.

    Pit-lane starts cannot yet be represented by the race engines. Incomplete
    grids and pit-lane classifications therefore fail rather than being placed
    into invented slots. Qualifying lap times are not treated as observations.
    """
    table = _GridTable()
    table.feed(text)
    if not any(str(year) in heading and "STARTING GRID" in heading.upper()
               for heading in table.headings):
        raise CurrentSeasonDataError("Official starting grid has a different season or session")
    if re.search(r"\bstart\w*\b.{0,100}\bpit[\s-]+lane\b", " ".join(table.text), re.I):
        raise CurrentSeasonDataError("Official starting grid includes unsupported pit-lane starts")
    roster = [{"id": d.id, "name": d.name, "code": d.id, "team_name": d.team_id}
              for d in drivers]
    _, aliases = loader._build_active_driver_map(roster, {})
    teams = {d.id: d.team_id for d in drivers}
    slots, identities = {}, set()
    for cells, _ in table.rows:
        if len(cells) not in (4, 5) or cells[0].lower().startswith("pos"):
            continue
        code = cells[2].split()[-1] if cells[2].split() else ""
        driver = loader._resolve_row_driver({"Driver": {"code": code}}, aliases)
        if driver is None:
            raise CurrentSeasonDataError("Official starting grid contains an unknown driver")
        if not cells[0].isdecimal() or not 1 <= int(cells[0]) <= 30:
            raise CurrentSeasonDataError("Official starting grid has an unsupported starting slot")
        slot = int(cells[0])
        if (slot in slots or driver in identities
                or loader._team_id(cells[3]) != teams[driver]):
            raise CurrentSeasonDataError("Conflicting official starting grid driver or team")
        slots[slot] = driver
        identities.add(driver)
    if set(slots) != set(range(1, len(drivers) + 1)):
        raise CurrentSeasonDataError("Official starting grid is incomplete for the modeled roster")
    try:
        return validate_starting_grid([slots[p] for p in sorted(slots)], teams)
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
                               drivers=drivers)
    return {"starting_grid": grid, "source_url": url, "fetched_at": now.isoformat(),
            "qualifying_started_at": qualifying.isoformat(),
            "year": year, "round": int(event["round"])}
