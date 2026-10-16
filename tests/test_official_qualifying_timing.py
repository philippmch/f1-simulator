"""Repair untimed current GP classifications without importing race outcomes."""

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

import f1sim.data.current as current
from f1sim.data.current import CurrentSeasonDataLoader
from f1sim.data.practice import _ResultsTable

YEAR = datetime.now(timezone.utc).year
HTML = (Path(__file__).parent / "fixtures/qualifying-miami-2025.html").read_text(encoding="utf-8")
HTML = HTML.replace("2025", str(YEAR))


def event():
    return {"round": 6, "race": "Miami Grand Prix", "circuit_id": "miami",
            "date": f"{YEAR}-05-04", "sessions": {
                "Qualifying": {"date": f"{YEAR}-05-03", "time": "20:00:00Z"}}}


def qualifying():
    table = _ResultsTable()
    table.feed(HTML)
    output = []
    for cells, _ in table.rows:
        if len(cells) != 8 or not cells[0].isdigit():
            continue
        code = cells[2].split()[-1]
        output.append({"round": 6, "number": cells[1], "position": cells[0],
                       "Driver": {"driverId": f"stable-{code}", "code": code},
                       "Constructor": {"name": "Sauber" if "Sauber" in cells[3] else cells[3]},
                       "Q1": "", "Q2": "", "Q3": ""})
    return output


def loader(monkeypatch, *, page=HTML):
    monkeypatch.setattr(current, "_utc_now", lambda: datetime(YEAR, 5, 4, tzinfo=timezone.utc))
    requests = []

    def getter(url, **kwargs):
        requests.append(url)
        if url == f"https://www.formula1.com/en/results/{YEAR}/races":
            return (f"<h1>{YEAR} RACE RESULTS</h1><table><tr><td>Miami</td><td>04 May</td>"
                    f'<td><a href="/en/results/{YEAR}/races/9999/miami/race-result">GP</a>'
                    "</td></tr></table>")
        assert url == f"https://www.formula1.com/en/results/{YEAR}/races/9999/miami/qualifying"
        if isinstance(page, Exception):
            raise page
        return page

    return CurrentSeasonDataLoader(current_year=YEAR, http_getter=getter), requests


def test_complete_stage_table_repairs_times_preserving_identity_position_and_order(monkeypatch):
    adapter, requests = loader(monkeypatch)
    rows = qualifying()[::-1]  # Prove the join does not assume table/result order.
    original = deepcopy(rows)
    result = adapter._supplement_target_qualifying_times(YEAR, event(), rows)
    assert rows == original
    assert len(requests) == 2
    assert [r["Driver"] for r in result] == [r["Driver"] for r in rows]
    assert [r["Constructor"] for r in result] == [r["Constructor"] for r in rows]
    assert [r["position"] for r in result] == [r["position"] for r in rows]
    assert next(r for r in result if r["Driver"]["code"] == "VER")["Q3"] == "1:26.204"
    assert sum(bool(r["Q1"]) for r in result) == 20
    assert sum(bool(r["Q2"]) for r in result) == 15
    assert sum(bool(r["Q3"]) for r in result) == 10
    evidence = adapter.get_provenance()["qualifying_timing_supplements"][0]
    assert evidence["status"] == "available"
    assert evidence["source_url"].endswith("/9999/miami/qualifying")
    assert evidence["qualifying_started_at"] == f"{YEAR}-05-03T20:00:00+00:00"
    assert adapter._season_qualifying is None
    assert adapter._supplement_target_qualifying_times(YEAR, event(), rows) == result
    assert len(requests) == 2  # Reuse this loader's accepted source, not another GET.


def test_generic_best_time_does_not_replace_explicit_stage_evidence(monkeypatch):
    adapter, requests = loader(monkeypatch)
    rows = qualifying()
    rows[0]["bestTime"] = "1:26.204"
    result = adapter._supplement_target_qualifying_times(YEAR, event(), rows)
    assert result[0]["bestTime"] == "1:26.204"
    assert result[0]["Q1"] == "1:26.870"
    assert result[0]["Q2"] == "1:26.643"
    assert result[0]["Q3"] == "1:26.204"
    assert len(requests) == 2


def test_cached_supplement_preserves_new_row_order_and_isolates_returned_values(monkeypatch):
    adapter, requests = loader(monkeypatch)
    rows = qualifying()
    expected = adapter._supplement_target_qualifying_times(YEAR, event(), rows)
    original = deepcopy(expected)
    expected[0]["Q1"] = "9:59.999"
    expected[0]["Driver"]["code"] = "XXX"
    reordered = adapter._supplement_target_qualifying_times(YEAR, event(), rows[::-1])
    assert reordered == original[::-1]
    reordered[0]["Constructor"]["name"] = "Changed"
    assert adapter._supplement_target_qualifying_times(YEAR, event(), rows) == original
    assert len(requests) == 2


def test_generic_best_times_cannot_make_an_empty_official_stage_table_complete(monkeypatch):
    page = HTML
    table = _ResultsTable()
    table.feed(page)
    for cells, _ in table.rows:
        if len(cells) == 8 and cells[0].isdigit():
            for value in cells[4:7]:
                if value:
                    page = page.replace(f"<td>{value}</td>", "<td></td>")
    adapter, requests = loader(monkeypatch, page=page)
    rows = qualifying()
    for row in rows:
        row["bestTime"] = "1:26.204"
    assert adapter._supplement_target_qualifying_times(YEAR, event(), rows) == rows
    assert adapter.get_provenance()["qualifying_timing_supplements"][0]["status"] == "unavailable"
    assert len(requests) == 2


@pytest.mark.parametrize("case", ["no_rows", "timed", "partial", "pending", "unknown_time"])
def test_normal_elimination_and_uncompleted_sessions_never_trigger_fetch(monkeypatch, case):
    adapter, requests = loader(monkeypatch)
    rows, selected = qualifying(), event()
    if case == "no_rows":
        rows = []
    elif case in ("timed", "partial"):
        rows[0]["Q1"] = "1:26.870"
    elif case == "pending":
        selected["sessions"]["Qualifying"] = {
            "date": f"{YEAR}-05-04", "time": "00:00:00Z"}
    else:
        selected["sessions"] = {}
    assert adapter._supplement_target_qualifying_times(YEAR, selected, rows) == rows
    assert requests == []


@pytest.mark.parametrize("change", ["sprint", "year", "event", "stages", "duplicate", "unknown",
                                    "position", "number", "constructor", "invalid_time", "short"])
def test_rejected_official_source_preserves_blank_rows_and_records_reason(monkeypatch, change):
    page = HTML
    if change == "sprint":
        page = page.replace(" - QUALIFYING", " - SPRINT QUALIFYING")
    elif change == "year":
        page = page.replace(str(YEAR), str(YEAR-1))
    elif change == "event":
        page = page.replace("MIAMI", "MONACO")
    elif change == "stages":
        page = page.replace("<th>Q2</th>", "<th>Time</th>")
    elif change == "duplicate":
        page = page.replace("Lando Norris NOR", "Lando Norris VER")
    elif change == "unknown":
        page = page.replace("Lando Norris NOR", "Unknown Driver XXX")
    elif change == "position":
        page = page.replace("<td>1</td>", "<td>20</td>", 1)
    elif change == "number":
        page = page.replace("<td>1</td><td>1</td>", "<td>1</td><td>99</td>", 1)
    elif change == "constructor":
        page = page.replace("Red Bull Racing", "McLaren", 1)
    elif change == "invalid_time":
        page = page.replace("1:26.204", "nan")
    else:
        page = page[:page.rfind("<tr>")]+"</table>"
    adapter, requests = loader(monkeypatch, page=page)
    rows = qualifying()
    assert adapter._supplement_target_qualifying_times(YEAR, event(), rows) == rows
    assert adapter.get_provenance()["qualifying_timing_supplements"][0]["status"] == "unavailable"
    assert adapter._supplement_target_qualifying_times(YEAR, event(), rows) == rows
    assert len(requests) == 2


def test_offline_failure_is_graceful_and_refresh_clears_attempt(monkeypatch):
    adapter, requests = loader(monkeypatch, page=ConnectionError("offline"))
    rows = qualifying()
    assert adapter._supplement_target_qualifying_times(YEAR, event(), rows) == rows
    assert adapter._supplement_target_qualifying_times(YEAR, event(), rows) == rows
    assert len(requests) == 2
    adapter.refresh()
    assert "qualifying_timing_supplements" not in adapter.get_provenance()
    adapter._supplement_target_qualifying_times(YEAR, event(), rows)
    assert len(requests) == 4


def test_public_weighted_builder_consumes_supplement_without_race_fetch(monkeypatch):
    adapter, requests = loader(monkeypatch)
    rows = qualifying()
    selected = event()
    monkeypatch.setattr(adapter, "_event_for_race", lambda *args: selected)
    roster = [{"id": r["Driver"]["code"], "code": r["Driver"]["code"],
               "name": r["Driver"]["code"], "team_name": r["Constructor"]["name"]} for r in rows]
    monkeypatch.setattr(adapter, "get_official_roster", lambda *args: roster)
    monkeypatch.setattr(adapter, "_load_season_rows", lambda *args, **kwargs: ([], [], [], []))
    monkeypatch.setattr(adapter, "_season_data", lambda *args: ([], deepcopy(rows)))
    captured = []
    build = adapter._build_driver_stats

    def capture(**kwargs):
        captured.append(kwargs["target_qualifying_rows"])
        return build(**kwargs)

    monkeypatch.setattr(adapter, "_build_driver_stats", capture)
    stats = adapter.get_weighted_driver_stats(YEAR, 6, quali_weight=0.,
                                               postqual_pace_policy="qualifying_led")
    assert len(stats) == 20
    assert all(CurrentSeasonDataLoader._row_qualifying_time(r) for r in captured[0])
    assert len(requests) == 2 and all(not u.endswith("/race-result") for u in requests)


def test_current_year_guard_is_retained_before_official_fetch(monkeypatch):
    adapter, requests = loader(monkeypatch)
    with pytest.raises(ValueError, match="current UTC season"):
        adapter._supplement_target_qualifying_times(YEAR-1, event(), qualifying())
    assert requests == []
