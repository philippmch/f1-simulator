"""Qualifying status remains separate from measured pace and the official grid."""

from copy import deepcopy
from datetime import datetime, timezone

import pytest

import f1sim.data.current as current
from f1sim.analysis.qualifying_history import _q1_time
from f1sim.data.confirmed_grid_forecast import build_feature_snapshot
from f1sim.data.current import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.data.practice import build_current_qualifying_history
from f1sim.data.qualifying import (
    QUALIFYING_CLASSIFICATION as META,
)
from f1sim.data.qualifying import (
    classification,
    complete_classification,
    effective_qualifying,
)

YEAR = datetime.now(timezone.utc).year
URL = f"https://www.formula1.com/en/results/{YEAR}/races/9999/test/qualifying"


def event(number=1):
    return {"round": number, "race": "Test Grand Prix", "circuit_id": "test",
            "date": f"{YEAR}-03-10", "sessions": {
                "Qualifying": {"date": f"{YEAR}-03-09", "time": "12:00:00Z"}}}


def rows():
    output = []
    for rank, (driver, code, name, team, number) in enumerate([
        ("hamilton", "HAM", "Lewis Hamilton", "mercedes", 44),
        ("verstappen", "VER", "Max Verstappen", "red_bull", 1),
        ("bottas", "BOT", "Valtteri Bottas", "mercedes", 77),
        ("perez", "PER", "Sergio Perez", "red_bull", 11),
    ], 1):
        first, last = name.split()
        output.append({"round": 1, "number": str(number), "position": str(min(rank, 3)),
            "Driver": {"driverId": driver, "code": code, "givenName": first, "familyName": last},
            "Constructor": {"constructorId": team, "name": team},
            "Q1": f"1:{29 + rank:02d}.000", "Q2": "", "Q3": ""})
    return output


def page(*, last="NC", missing=False, note=True):
    table = [f"<h1>{YEAR} TEST GRAND PRIX - QUALIFYING</h1><table>",
             "<tr>" + "".join(f"<th>{c}</th>" for c in
                              ("Pos.", "No.", "Driver", "Team", "Q1", "Q2", "Q3", "Laps"))
             + "</tr>"]
    for position, number, driver, team, q1 in [
        ("1", "44", "Lewis Hamilton HAM", "Mercedes", "1:30.000"),
        ("2", "1", "Max Verstappen VER", "Red Bull Racing", "1:31.000"),
        ("DQ", "77", "Valtteri Bottas BOT", "Mercedes", "1:29.000"),
        (last, "11", "Sergio Perez PER", "Red Bull Racing", "1:33.000"),
    ][:3 if missing else 4]:
        table.append("<tr>" + "".join(f"<td>{c}</td>" for c in
            (position, number, driver, team, q1, "", "", "5")) + "</tr>")
    table.append("</table>")
    if missing and note:
        table.append("<p>Note - Perez did not take part in qualifying and races at the "
                     "stewards' discretion.</p>")
    return "".join(table)


def enriched(*, text=None):
    return CurrentSeasonDataLoader._parse_official_qualifying_classification(
        page() if text is None else text, year=YEAR, event=event(), rows=rows(), source_url=URL,
    )


def normalized(raw):
    return {"driver": raw["Driver"]["driverId"], "code": raw["Driver"]["code"],
            "team": raw["Constructor"]["constructorId"],
            "qualifying_position": int(raw["position"]),
            **{q.lower(): current._parse_time_seconds(raw[q]) for q in ("Q1", "Q2", "Q3")},
            **({META: deepcopy(raw[META])} if META in raw else {})}


def test_dq_rank_and_times_are_missing_but_nc_keeps_observed_pace():
    original = rows()
    result = enriched()
    assert rows() == original
    assert [{k: v for k, v in row.items() if k != META} for row in result] == original
    assert complete_classification(result)
    dq, nc = result[2:]
    assert effective_qualifying(dq, year=YEAR, round_number=1) == {
        "qualifying_position": None, "q1": None, "q2": None, "q3": None}
    assert CurrentSeasonDataLoader._row_qualifying_time(dq) is None
    assert CurrentSeasonDataLoader._row_qualifying_stage(dq, "Q1") is None
    assert _q1_time(dq) is None
    assert CurrentSeasonDataLoader._row_qualifying_position(nc) is None
    assert CurrentSeasonDataLoader._row_qualifying_time(nc) == 93.
    assert _q1_time(nc) == 93.
    assert dq[META]["reported"]["Q1"] == "1:32.000"
    assert dq[META]["official"]["Q1"] == "1:29.000"


def test_explicit_official_no_participation_note_preserves_untimed_entrant():
    result = enriched(text=page(missing=True))
    assert len(result) == 4
    assert result[3][META]["status"] == "did_not_start"
    assert CurrentSeasonDataLoader._row_qualifying_time(result[3]) is None
    with pytest.raises(CurrentSeasonDataError, match="without explicit status"):
        enriched(text=page(missing=True, note=False))
    with pytest.raises(CurrentSeasonDataError, match="without explicit status"):
        enriched(text=page(missing=True).replace("Perez did not", "Hamilton did not"))


def test_absence_note_uses_one_normalized_identity_rule_for_underscores_and_accents():
    raw = rows()
    raw[3]["Driver"]["driverId"] = "sergio_perez"
    result = CurrentSeasonDataLoader._parse_official_qualifying_classification(
        page(missing=True).replace("Perez did not", "Pérez did not"),
        year=YEAR, event=event(), rows=raw, source_url=URL,
    )
    assert result[3][META]["official"]["matched_label"] == "perez"
    assert classification(result[3])["status"] == "did_not_start"


def test_calendar_circuit_alias_is_bound_to_the_official_url_and_heading():
    selected = event() | {"race": "Dutch Grand Prix", "circuit_id": "zandvoort"}
    source = page().replace("TEST GRAND PRIX", "NETHERLANDS GRAND PRIX")
    url = URL.replace("/test/", "/netherlands/")
    result = CurrentSeasonDataLoader._parse_official_qualifying_classification(
        source, year=YEAR, event=selected, rows=rows(), source_url=url,
    )
    assert complete_classification(result)
    with pytest.raises(CurrentSeasonDataError, match="calendar circuit"):
        CurrentSeasonDataLoader._parse_official_qualifying_classification(
            source, year=YEAR, event=selected | {"circuit_id": "monza"},
            rows=rows(), source_url=url,
        )


@pytest.mark.parametrize("constructor,display", [
    ("alfa", "Alfa Romeo Ferrari"), ("haas", "Haas Ferrari"), ("sauber", "Kick Sauber Ferrari"),
])
def test_primary_chassis_name_precedes_its_ferrari_engine_suffix(constructor, display):
    raw = rows()
    raw[2]["Constructor"] = {"constructorId": constructor, "name": constructor}
    source = page().replace("Valtteri Bottas BOT</td><td>Mercedes",
                            f"Valtteri Bottas BOT</td><td>{display}")
    result = CurrentSeasonDataLoader._parse_official_qualifying_classification(
        source, year=YEAR, event=event(), rows=raw, source_url=URL,
    )
    assert classification(result[2])["status"] == "disqualified"
    with pytest.raises(CurrentSeasonDataError, match="constructor"):
        CurrentSeasonDataLoader._parse_official_qualifying_classification(
            source.replace(display, "Ferrari"), year=YEAR, event=event(), rows=raw, source_url=URL,
        )


@pytest.mark.parametrize("change", ["status", "rank", "time", "validity", "reported_rank",
    "reported_time", "official_rank", "official_time", "identity", "team", "year", "round",
    "source_session", "source_url"])
def test_metadata_cannot_be_resealed_to_promote_invalid_laps_or_change_evidence(change):
    row = enriched()[2]
    m = row[META]
    if change == "status":
        m["status"], m["position"] = "classified", 3
    elif change == "rank":
        m["position"] = 3
    elif change == "time":
        m["stage_times"]["Q1"], m["stage_validity"]["Q1"] = 89., True
    elif change == "validity":
        m["stage_validity"]["Q1"] = True
    elif change == "reported_rank":
        m["reported"]["position"] = "4"
    elif change == "reported_time":
        m["reported"]["Q1"] = "1:31.000"
    elif change == "official_rank":
        m["official"]["position"] = "3"
    elif change == "official_time":
        row = enriched()[3]  # NC has a valid observed time, unlike DQ.
        row[META]["official"]["Q1"] = "1:31.000"
    elif change == "identity":
        m["identity"]["driver"] = "hamilton"
    elif change == "team":
        m["official"]["team"] = "McLaren"
    elif change in {"year", "round"}:
        row[change] = YEAR - 1 if change == "year" else 2
    elif change == "source_session":
        m["source"]["session"] = "SprintQualifying"
    else:
        m["source"]["source_url"] = URL.replace("/qualifying", "/starting-grid")
    with pytest.raises(ValueError):
        classification(row)


def test_partial_primary_classification_requires_one_whole_joined_field():
    result = enriched()
    assert complete_classification(result)
    mixed = deepcopy(result)
    del mixed[0][META]
    assert not complete_classification(mixed)
    result[0][META]["source"]["content_sha256"] = "1" * 64
    assert not complete_classification(result)


def test_feature_rank_timing_and_prior_form_share_validity_without_changing_grid_or_risk():
    raw = enriched()
    roster = [normalized(row) | {"grid_position": i + 1, "official_table_order_index": i + 1,
                                "pit_lane_start": i == 2} for i, row in enumerate(raw)]
    target = {"year": YEAR, "round": 2, "roster": deepcopy(roster)}
    # This target uses the preceding event's enriched source solely as history.
    for index, row in enumerate(target["roster"]):
        del row[META]
        row["qualifying_position"] = index + 1
    # Explicit new event metadata must never be transplanted between rounds.
    with pytest.raises(ValueError, match="event identity"):
        build_feature_snapshot({"year": YEAR, "round": 2, "roster": roster}, [])
    history = {"year": YEAR, "round": 1, "rows": [
        normalized(row) | {"position": i + 1, "status": "Finished"}
        for i, row in enumerate(raw)]}
    before = deepcopy(history)
    # Raw legacy sources remain numerically unchanged, including duplicate ranks.
    legacy = deepcopy(history)
    for row in legacy["rows"]:
        del row[META]
    fixed = build_feature_snapshot(target, [history])
    old = build_feature_snapshot(target, [legacy])
    assert fixed["grid_ids"] == old["grid_ids"]
    assert fixed["risk"] == old["risk"]
    assert fixed["x"][0][12] != old["x"][0][12]
    assert history == before
    same_event = build_feature_snapshot({"year": YEAR, "round": 1, "roster": roster}, [])
    dq = same_event["ids"].index("bottas")
    assert same_event["x"][dq][1] == 0
    assert same_event["x"][dq][3:9] == [0.] * 6
    assert same_event["x"][dq][2] == 1


def test_practice_history_uses_effective_q1_and_rank_and_keeps_entire_field():
    qualifying = enriched()
    results = []
    for i, row in enumerate(rows()):
        result = deepcopy(row)
        result.update(position=str(i + 1), points="0", status="Finished")
        results.append(result)
    loader = CurrentSeasonDataLoader(current_year=YEAR)
    history = build_current_qualifying_history(loader, [event()], results, qualifying,
                                               before_round=2)
    assert len(history[0]["rows"]) == 4
    dq = next(row for row in history[0]["rows"] if row["driver"] == "BOT")
    nc = next(row for row in history[0]["rows"] if row["driver"] == "PER")
    assert dq["q1"] is None and dq["qualifying_position"] is None
    assert nc["q1"] == 93. and nc["qualifying_position"] is None


def test_ambiguous_live_ranks_trigger_once_with_existing_times_and_fail_closed(monkeypatch):
    monkeypatch.setattr(current, "_utc_now", lambda: datetime(YEAR, 3, 10, tzinfo=timezone.utc))
    requests = []

    def getter(url, **kwargs):
        requests.append(url)
        if url.endswith(f"/{YEAR}/races"):
            return (f"<h1>{YEAR} RACE RESULTS</h1><table><tr><td>Test</td><td>10 Mar</td>"
                    f'<td><a href="/en/results/{YEAR}/races/9999/test/race-result">GP</a>'
                    "</td></tr></table>")
        return page()

    loader = CurrentSeasonDataLoader(current_year=YEAR, http_getter=getter)
    result = loader._supplement_target_qualifying_times(YEAR, event(), rows())
    assert result == enriched()
    assert len(requests) == 2
    assert loader._supplement_target_qualifying_times(YEAR, event(), rows()[::-1]) == result[::-1]
    assert len(requests) == 2
    unavailable = CurrentSeasonDataLoader(current_year=YEAR,
        http_getter=lambda *args, **kwargs: "<h1>Wrong session</h1>")
    with pytest.raises(CurrentSeasonDataError, match="lacks verified source"):
        unavailable._supplement_target_qualifying_times(YEAR, event(), rows())
    with pytest.raises(CurrentSeasonDataError, match="lacks verified source"):
        unavailable._supplement_target_qualifying_times(YEAR, event(), rows())
