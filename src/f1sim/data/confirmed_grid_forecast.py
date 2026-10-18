"""Strictly earlier raw GP history for the frozen after-qualifying features."""

import re
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timedelta, timezone

import numpy as np

from f1sim.analysis.learned_afterqual import (
    FEATURE_SCHEMA,
    AfterQualifyingFeatures,
    canonical_digest,
)
from f1sim.data.current import CurrentSeasonDataLoader, _parse_time_seconds
from f1sim.data.grid import _verify_published_grid_lookup
from f1sim.data.qualifying import (
    QUALIFYING_CLASSIFICATION,
    complete_classification,
    effective_qualifying,
)
from f1sim.simulation.execution import validate_pit_lane_starters, validate_starting_grid


def _integer(value, name, *, optional=False):
    if optional and value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number < 1 or str(value).strip() != str(number):
        raise ValueError(f"{name} must be a positive integer")
    return number


def _identity(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} needs an unambiguous identity")
    return value


def _instant(value, name):
    if isinstance(value, dict):
        value = f"{value.get('date', '')}T{value.get('time', '')}"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{name} needs a dated timezone-aware timestamp") from error
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{name} needs a dated timezone-aware timestamp")
    return value.astimezone(timezone.utc)


def _finished(status):
    if not isinstance(status, str) or not status.strip():
        raise ValueError("Completed race history requires explicit statuses")
    text = status.strip().lower()
    return text in {"finished", "classified", "lapped"} or re.fullmatch(
        r"\+\s*\d+\s+laps?", text,
    ) is not None


def build_feature_snapshot(predictors, completed_history):
    """Build the unchanged 15 pace/3 risk columns and one rank interaction.

    Inputs use stable provider driver/constructor IDs. History includes every
    completed earlier GP, even without a usable historical starting grid.
    Target labels are neither accepted as history nor consulted for features.
    """
    if not isinstance(predictors, dict) or not isinstance(completed_history, list):
        raise ValueError("Feature predictors and completed history must be structured objects")
    target = deepcopy(predictors)
    year = _integer(target.get("year"), "year")
    number = _integer(target.get("round"), "round")
    roster = target.get("roster")
    if not isinstance(roster, list) or not 2 <= len(roster) <= 30:
        raise ValueError("After-qualifying features require the complete entrant field")
    if not all(isinstance(row, dict) for row in roster):
        raise ValueError("Every qualifying entrant must be an object")
    ids = [_identity(row.get("driver"), "driver") for row in roster]
    codes = [_identity(row.get("code"), "code") for row in roster]
    if len(set(ids)) != len(ids) or len(set(codes)) != len(codes):
        raise ValueError("Provider identities and physical codes must be one-to-one")
    for row in roster:
        _identity(row.get("team"), "team")
        effective = effective_qualifying(row, year=year, round_number=number)
        if effective is not None:
            row.update(effective)
        row["qualifying_position"] = _integer(
            row.get("qualifying_position"), "qualifying position", optional=True,
        )
        for name in ("grid_position", "official_table_order_index"):
            row[name] = _integer(row.get(name), name, optional=True)
        if row.get("pit_lane_start") is not None and type(row["pit_lane_start"]) is not bool:
            raise ValueError("Pit status must be explicit boolean or unknown")
        row.setdefault("pit_lane_start", None)
        for stage in ("q1", "q2", "q3"):
            value = row.get(stage)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not np.isfinite(value) or value <= 0
            ):
                raise ValueError("Qualifying stage times must be positive finite seconds")
            row.setdefault(stage, None)
    n = len(roster)
    grid_rows = sorted(roster, key=lambda row: (
        bool(row.get("pit_lane_start")), row.get("grid_position") or n + 1,
        row.get("official_table_order_index") or n + 1,
    ))
    grid_rank = {row["driver"]: i for i, row in enumerate(grid_rows)}
    team_history, driver_history = defaultdict(list), defaultdict(list)
    keys = []
    if not all(isinstance(event, dict) for event in completed_history):
        raise ValueError("Every completed historical GP must be an object")
    history = sorted(deepcopy(completed_history), key=lambda event: (
        _integer(event.get("year"), "history year"),
        _integer(event.get("round"), "history round"),
    ))
    for event in history:
        key = (_integer(event["year"], "history year"),
               _integer(event["round"], "history round"))
        if key >= (year, number) or key in keys:
            raise ValueError("Feature history must contain unique strictly earlier Grand Prix")
        keys.append(key)
        rows = event.get("rows")
        if not isinstance(rows, list) or len(rows) < 2:
            raise ValueError("Completed race history requires the whole race field")
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError("Every historical race entrant must be an object")
        if len({row.get("driver") for row in rows}) != len(rows):
            raise ValueError("Completed race history contains duplicate drivers")
        if sorted(_integer(row.get("position"), "race position") for row in rows) != list(
            range(1, len(rows) + 1),
        ):
            raise ValueError("Completed race positions must form a full permutation")
        by_team = defaultdict(list)
        for row in rows:
            _identity(row.get("driver"), "historical driver")
            _identity(row.get("team"), "historical constructor")
            row["position"] = _integer(row["position"], "historical position")
            effective = effective_qualifying(row, year=key[0], round_number=key[1])
            if effective is not None:
                row["qualifying_position"] = effective["qualifying_position"]
            row["qualifying_position"] = _integer(
                row.get("qualifying_position"), "historical qualifying", optional=True,
            )
            row["finished"] = _finished(row.get("status"))
            by_team[row["team"]].append(row)
        for team, team_rows in by_team.items():
            race_values = [1 - (row["position"] - 1) / (len(rows) - 1)
                           for row in team_rows if row["finished"]]
            quali_values = [1 - (row["qualifying_position"] - 1) / (len(rows) - 1)
                            for row in team_rows if row["qualifying_position"]]
            race = float(np.mean(race_values)) if race_values else None
            quali = float(np.mean(quali_values)) if quali_values else None
            team_history[team].append({"year": key[0], "race": race, "qualifying": quali,
                "nonfinish": float(np.mean([not row["finished"] for row in team_rows]))})
            for row in team_rows:
                driver_race = (1 - (row["position"] - 1) / (len(rows) - 1)
                               if row["finished"] else None)
                driver_quali = (1 - (row["qualifying_position"] - 1) / (len(rows) - 1)
                                if row["qualifying_position"] else None)
                driver_history[row["driver"], team].append({"year": key[0],
                    "race_residual": driver_race - race
                    if driver_race is not None and race is not None else None,
                    "qualifying_residual": driver_quali - quali
                    if driver_quali is not None and quali is not None else None,
                    "nonfinish": float(not row["finished"])})

    def recent(records, field, prior):
        values = [row[field] for row in records
                  if row["year"] == year and row[field] is not None][-3:]
        return (sum(values) + 2 * prior) / (len(values) + 2)

    median = {stage: float(np.median([row[stage] for row in roster if row[stage]]))
              for stage in ("q1", "q2", "q3") if any(row[stage] for row in roster)}
    gaps = {stage: {row["driver"]: float(np.clip(
        100 * (reference - row[stage]) / reference, -5, 5,
    )) for row in roster if row[stage]} for stage, reference in median.items()}
    x, risk = [], []
    for row in roster:
        driver, team = row["driver"], row["team"]
        mates = [mate for mate in roster if mate["team"] == team]
        shared = next((stage for stage in ("q3", "q2", "q1")
                       if len(mates) >= 2 and all(mate[stage] for mate in mates)), None)
        mate_delta = (100 * (float(np.median([mate[shared] for mate in mates])) - row[shared])
                      / median[shared]) if shared else 0.
        team_q1 = [gaps.get("q1", {}).get(mate["driver"]) for mate in mates]
        team_q1 = [value for value in team_q1 if value is not None]
        x.append([1 - grid_rank[driver] / (n - 1),
            1 - (row["qualifying_position"] - 1) / (n - 1)
            if row["qualifying_position"] else 0., float(bool(row["pit_lane_start"])),
            *[gaps.get(stage, {}).get(driver, 0.) for stage in ("q1", "q2", "q3")],
            *[float(bool(row[stage])) for stage in ("q1", "q2", "q3")],
            float(np.clip(mate_delta, -5, 5)), float(np.mean(team_q1)) if team_q1 else 0.,
            recent(team_history[team], "race", .5),
            recent(team_history[team], "qualifying", .5),
            recent(driver_history[driver, team], "race_residual", 0.),
            recent(driver_history[driver, team], "qualifying_residual", 0.)])
        risk.append([1., recent(team_history[team], "nonfinish", .15),
                     recent(driver_history[driver, team], "nonfinish", .15)])
    values = np.asarray(x)
    return {"feature_schema": FEATURE_SCHEMA, "year": year, "round": number,
        "ids": ids, "codes": codes, "grid_ids": [row["driver"] for row in grid_rows],
        "x": values.tolist(), "tree_x": np.c_[values, values[:, 1] - values[:, 0]].tolist(),
        "risk": risk, "history_event_keys": [list(key) for key in keys]}


def _provider(row):
    if not isinstance(row, dict):
        raise ValueError("Provider rows must be objects")
    driver, constructor = row.get("Driver", {}), row.get("Constructor", {})
    if not isinstance(driver, dict) or not isinstance(constructor, dict):
        raise ValueError("Provider driver and constructor must be explicit objects")
    return (_identity(driver.get("driverId"), "provider driver"),
            _identity(driver.get("code"), "physical code"),
            _identity(constructor.get("constructorId"), "provider constructor"))


def validate_confirmed_grid_feature_snapshot(snapshot):
    """Rebuild saved live features and verify their source/cutoff evidence offline.

    Content seals detect accidental changes, not independent publication. A
    re-sealed matrix must still agree with the saved raw qualifying/history.
    """
    if not isinstance(snapshot, dict):
        raise ValueError("Confirmed-grid feature snapshot must be an object")
    value = deepcopy(snapshot)
    if "content_sha256" in value and value["content_sha256"] != canonical_digest(
        {key: item for key, item in value.items() if key != "content_sha256"},
    ):
        raise ValueError("Confirmed-grid feature snapshot content seal mismatch")
    context = value.get("context")
    if not isinstance(context, dict):
        raise ValueError("Confirmed-grid features require saved source context")
    predictors, history = context.get("target_predictors"), context.get("completed_history")
    if not isinstance(predictors, dict) or not isinstance(history, list):
        raise ValueError("Saved target qualifying and earlier history must be present")
    if (context.get("target_predictors_sha256") != canonical_digest(predictors)
            or context.get("completed_history_sha256") != canonical_digest(history)):
        raise ValueError("Saved qualifying/history source digests disagree")
    rebuilt = build_feature_snapshot(predictors, history)
    for key in ("feature_schema", "year", "round", "ids", "codes", "grid_ids",
                "history_event_keys"):
        if value.get(key) != rebuilt[key]:
            raise ValueError(f"Saved feature identity/order disagrees with its source: {key}")
    AfterQualifyingFeatures.from_mapping(value)
    for key in ("x", "tree_x", "risk"):
        actual = np.asarray(value.get(key), dtype=float)
        expected = np.asarray(rebuilt[key], dtype=float)
        if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f"Saved feature matrix disagrees with qualifying/history: {key}")
    cutoff = _instant(value.get("as_of"), "saved as_of")
    if _instant(context.get("as_of"), "context as_of") != cutoff:
        raise ValueError("Saved feature cutoffs disagree")
    event = context.get("event")
    if not isinstance(event, dict) or _integer(event.get("round"), "event round") != value["round"]:
        raise ValueError("Saved feature event identity disagrees")
    qualifying = _instant(event.get("sessions", {}).get("Qualifying"), "GP qualifying")
    race = _instant(event, "GP race")
    if (not qualifying + timedelta(hours=1) <= cutoff < race or race.year != value["year"]
            or _instant(context.get("qualifying_starts_at"), "qualifying starts") != qualifying
            or _instant(context.get("race_starts_at"), "race starts") != race
            or context.get("target_race_performance_used") is not False):
        raise ValueError("Saved features violate the after-qualifying race cutoff")
    provenance = context.get("source_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Saved features require source provenance")
    fetched = _instant(provenance.get("fetched_at"), "provider fetched_at")
    if (fetched > cutoff or
            _instant(context.get("provider_fetched_at"), "provider fetched_at") != fetched):
        raise ValueError("Saved provider source was fetched after the cutoff")
    for source in provenance.get("qualifying_timing_supplements", []):
        if (source.get("fetched_at")
                and _instant(source["fetched_at"], "timing fetched_at") > cutoff):
            raise ValueError("Saved timing source was fetched after the cutoff")
    receipts = provenance.get("qualifying_timing_supplements", [])
    classified_fields = [(predictors["year"], predictors["round"], predictors["roster"])] + [
        (row["year"], row["round"], row["rows"]) for row in history]
    for source_year, source_round, field in classified_fields:
        enriched = [row for row in field if QUALIFYING_CLASSIFICATION in row]
        if not enriched:
            continue
        if not complete_classification(field):
            raise ValueError("Saved primary qualifying requires the entire joined field")
        for row in enriched:
            effective_qualifying(row, year=source_year, round_number=source_round)
            source = row[QUALIFYING_CLASSIFICATION]["source"]
            if not any(receipt.get("status") == "available"
                       and receipt.get("year") == source_year
                       and receipt.get("round") == source_round
                       and receipt.get("source_url") == source["source_url"]
                       and receipt.get("content_sha256") == source["content_sha256"]
                       and receipt.get("fetched_at")
                       and _instant(receipt["fetched_at"], "classification fetched_at") <= cutoff
                       for receipt in receipts):
                raise ValueError("Saved qualifying classification lacks a matching source receipt")
    mapping = dict(zip(value["ids"], value["codes"], strict=True))
    coded_grid = [mapping[driver] for driver in value["grid_ids"]]
    grid, pits = _verify_published_grid_lookup(
        provenance.get("race_grid"), value["codes"], year=value["year"],
        target_round=value["round"], qualifying=qualifying, started=qualifying, now=cutoff,
    )
    expected_pits = [row["code"] for row in sorted(
        predictors["roster"], key=lambda row: coded_grid.index(row["code"]),
    ) if row["pit_lane_start"]]
    if grid != coded_grid or pits != expected_pits:
        raise ValueError("Saved published grid/pit evidence disagrees with feature sources")
    roster = predictors["roster"]
    if not complete_classification(roster):
        raise ValueError("Saved target qualifying must form a complete classification")
    seats = context.get("modeled_seats")
    if not isinstance(seats, dict) or set(seats) != set(value["codes"]):
        raise ValueError("Saved features require all current modeled seats")
    for row in roster:
        team = CurrentSeasonDataLoader._row_team_id({"Constructor": {"constructorId": row["team"]}})
        if seats[row["code"]] != team:
            raise ValueError("Saved qualifying constructor disagrees with modeled seat")
    starts = context.get("prior_race_starts_at")
    if not isinstance(starts, dict) or set(starts) != {str(row["round"]) for row in history}:
        raise ValueError("Saved history requires a dated calendar entry for every GP")
    if any(row["year"] != value["year"] for row in history):
        raise ValueError("Live feature history must be from the target season")
    if any((_instant(instant, "earlier GP race") >= qualifying
            or _instant(instant, "earlier GP race").year != value["year"])
           for instant in starts.values()):
        raise ValueError("Saved history includes a race after target qualifying")
    return value


def build_confirmed_grid_features(
    loader, year, target_race, drivers, starting_grid, pit_lane_starters, *, as_of=None,
):
    """Freeze a live complete GP qualifying/grid and only already-finished GPs.

    Default cutoff is captured after required provider reads. An explicit
    cutoff is strict and is never advanced to accommodate a later fetch.
    """
    explicit_cutoff = _instant(as_of, "as_of") if as_of is not None else None
    loader._assert_current_year(year)
    event = deepcopy(loader._event_for_race(year, target_race))
    number = _integer(event.get("round"), "target round")
    ids = [driver.id for driver in drivers]
    grid = validate_starting_grid(starting_grid, ids)
    if grid is None or len(ids) < 2:
        raise ValueError("The learned forecast requires a complete confirmed grid")
    pits = validate_pit_lane_starters(pit_lane_starters, grid)
    qualifying = _instant(event.get("sessions", {}).get("Qualifying"), "GP qualifying")
    race = _instant(event, "GP race")
    results, quali = loader._season_data(year)
    target_rows = [row for row in quali if _integer(row.get("round"), "round") == number]
    target_rows = loader._supplement_target_qualifying_times(year, event, target_rows)
    calendar = {_integer(row.get("round"), "calendar round"): row
                for row in loader.get_event_schedule(year)}
    # Complete every source read before freezing provenance and the default
    # cutoff. Prior qualifying must obey the same explicit-status policy.
    previous_rounds = {_integer(row.get("round"), "result round") for row in results
                       if _integer(row.get("round"), "result round") < number}
    for previous in sorted(previous_rounds):
        indices = [i for i, row in enumerate(quali)
                   if _integer(row.get("round"), "qualifying round") == previous]
        earlier = [quali[i] for i in indices]
        if earlier and not complete_classification(earlier):
            scheduled = calendar.get(previous)
            if scheduled is None or _instant(scheduled, "earlier GP race") >= qualifying:
                raise ValueError("Earlier qualifying lacks an earlier dated GP calendar entry")
            corrected = loader._supplement_target_qualifying_times(year, scheduled, earlier)
            for index, row in zip(indices, corrected, strict=True):
                quali[index] = row
    provenance = deepcopy(loader.get_provenance())
    cutoff = explicit_cutoff or datetime.now(timezone.utc)
    if not qualifying + timedelta(hours=1) <= cutoff < race or race.year != year:
        raise ValueError("The learned forecast requires completed GP qualifying before the race")
    verified_grid, verified_pits = _verify_published_grid_lookup(
        provenance.get("race_grid"), ids, year=year, target_round=number,
        qualifying=qualifying, started=qualifying, now=cutoff,
    )
    if verified_grid != grid or verified_pits != pits:
        raise ValueError("Feature grid and pit queue must match published source evidence")
    fetched = _instant(provenance.get("fetched_at"), "provider fetched_at")
    if fetched > cutoff:
        raise ValueError("Feature sources were fetched after the information cutoff")
    for source in provenance.get("qualifying_timing_supplements", []):
        if (source.get("fetched_at")
                and _instant(source["fetched_at"], "timing fetched_at") > cutoff):
            raise ValueError("Qualifying timing source was fetched after the cutoff")
    if any(_integer(row.get("round"), "result round") >= number for row in results):
        raise ValueError("Target or later race performance cannot enter a confirmed-grid forecast")
    if any(_integer(row.get("round"), "qualifying round") > number for row in quali):
        raise ValueError("Later qualifying cannot enter a confirmed-grid forecast")
    modeled = {driver.id: driver for driver in drivers}
    roster, target_ids = [], set()
    rank = {code: i + 1 for i, code in enumerate(grid)}
    for row in target_rows:
        provider, code, team = _provider(row)
        if code not in modeled or provider in target_ids:
            raise ValueError("Target qualifying must match the complete modeled roster once")
        canonical_team = CurrentSeasonDataLoader._row_team_id(
            {"Constructor": {"constructorId": team}},
        )
        if (CurrentSeasonDataLoader._row_team_id(row) != modeled[code].team_id
                or canonical_team != modeled[code].team_id):
            raise ValueError("Qualifying constructor disagrees with the current modeled seat")
        target_ids.add(provider)
        roster.append({"driver": provider, "code": code, "team": team,
            "qualifying_position": (CurrentSeasonDataLoader._row_position(row)
                if QUALIFYING_CLASSIFICATION in row else
                _integer(row.get("position"), "qualifying position", optional=True)),
            **{stage.lower(): _parse_time_seconds(row.get(stage)) for stage in ("Q1", "Q2", "Q3")},
            **({QUALIFYING_CLASSIFICATION: deepcopy(row[QUALIFYING_CLASSIFICATION])}
               if QUALIFYING_CLASSIFICATION in row else {}),
            "grid_position": rank[code], "official_table_order_index": rank[code],
            "pit_lane_start": code in pits})
    if (len(roster) != len(drivers) or {row["code"] for row in roster} != set(ids)
            or not complete_classification(roster)):
        raise ValueError("Target GP qualifying classification must cover the entire modeled field")
    by_round = defaultdict(list)
    for row in results:
        by_round[_integer(row.get("round"), "result round")].append(row)
    prior = []
    for previous, rows in sorted(by_round.items()):
        scheduled = calendar.get(previous)
        if (scheduled is None or _instant(scheduled, "earlier GP race") >= qualifying
                or _instant(scheduled, "earlier GP race").year != year):
            raise ValueError("Every historical race needs an earlier dated GP calendar entry")
        qmap = {}
        previous_qualifying = [row for row in quali
                              if _integer(row.get("round"), "qualifying round") == previous]
        for row in previous_qualifying:
            provider, _, _ = _provider(row)
            if provider in qmap:
                raise ValueError("Earlier qualifying contains duplicate provider identities")
            qmap[provider] = row
        history_rows = []
        for row in rows:
            provider, _, team = _provider(row)
            qrow = qmap.get(provider)
            if qrow is not None and _provider(qrow)[2] != team:
                raise ValueError("Earlier race and qualifying constructor identities disagree")
            history_rows.append({"driver": provider, "team": team,
                "position": _integer(row.get("position"), "race position"),
                "qualifying_position": None if qrow is None else
                (CurrentSeasonDataLoader._row_position(qrow)
                 if QUALIFYING_CLASSIFICATION in qrow else
                 _integer(qrow.get("position"), "earlier qualifying position", optional=True)),
                **({QUALIFYING_CLASSIFICATION: deepcopy(qrow[QUALIFYING_CLASSIFICATION]),
                    "code": _provider(qrow)[1],
                    **{stage.lower(): _parse_time_seconds(qrow.get(stage))
                       for stage in ("Q1", "Q2", "Q3")}}
                   if qrow is not None and QUALIFYING_CLASSIFICATION in qrow else {}),
                "status": row.get("status")})
        if set(qmap) - {row["driver"] for row in history_rows}:
            raise ValueError("Earlier race results omit a qualifying entrant")
        prior.append({"year": year, "round": previous,
                      "rows": sorted(history_rows, key=lambda row: row["driver"])})
    predictors = {"year": year, "round": number,
                  "roster": sorted(roster, key=lambda row: row["driver"])}
    snapshot = build_feature_snapshot(predictors, prior)
    snapshot.update({"as_of": cutoff.isoformat(), "context": {
        "as_of": cutoff.isoformat(), "event": event,
        "qualifying_starts_at": qualifying.isoformat(), "race_starts_at": race.isoformat(),
        "provider_fetched_at": fetched.isoformat(), "source_provenance": provenance,
        "modeled_seats": {driver.id: driver.team_id for driver in drivers},
        "prior_race_starts_at": {str(previous): _instant(
            calendar[previous], "earlier GP race",
        ).isoformat() for previous in by_round},
        "target_predictors": predictors, "completed_history": prior,
        "target_predictors_sha256": canonical_digest(predictors),
        "completed_history_sha256": canonical_digest(prior),
        "target_race_performance_used": False,
        "history_scope": "strictly earlier completed same-season GP results; all entrants retained",
    }})
    snapshot["content_sha256"] = canonical_digest(snapshot)
    return validate_confirmed_grid_feature_snapshot(snapshot)
