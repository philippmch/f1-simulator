"""Fixed retrospective team-Q1 history for qualifying pace diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median
from typing import Any, Mapping, Sequence

from f1sim.data.current import CurrentSeasonDataError, _parse_time_seconds

HISTORY_WINDOW_EVENTS = 3


def _integer(value: Any) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _q1_time(row: Mapping[str, Any]) -> float | None:
    value = row.get("Q1")
    parsed = None if isinstance(value, bool) else _parse_time_seconds(value)
    return parsed if parsed is not None and math.isfinite(parsed) and parsed > 0 else None


def _unique_rows(loader, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate exact rows while rejecting conflicting driver evidence."""
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        identity = loader._strong_driver_identity(row)
        if identity is None:
            raise CurrentSeasonDataError("Historical Q1 evidence lacks driver identity")
        previous = unique.get(identity)
        if previous is not None and dict(previous) != dict(row):
            raise CurrentSeasonDataError("Conflicting historical Q1 driver records")
        unique[identity] = row
    return [dict(row) for row in unique.values()]


def _roster(loader, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    roster = []
    for row in rows:
        driver = loader._extract_driver(row)
        code = driver.get("code") or driver.get("driverId")
        name = loader._driver_name(driver)
        team = loader._row_team_id(row)
        if not isinstance(code, str) or not code.strip() or not name or team is None:
            raise CurrentSeasonDataError("Historical Q1 evidence has incomplete identity")
        roster.append({
            "id": code,
            "name": name,
            "team_name": team,
            "driverId": driver.get("driverId"),
            "code": driver.get("code"),
            "permanentNumber": driver.get("permanentNumber"),
        })
    return roster


def _event_history(loader, event: Mapping[str, Any], result_rows, qualifying_rows) -> dict:
    """Build one event summary using that event's own identities and coverage."""
    round_number = _integer(event.get("round"))
    if round_number is None:
        raise CurrentSeasonDataError("Historical Q1 event lacks a round")
    if not qualifying_rows:
        return {
            "round": round_number,
            "race": event.get("race"),
            "status": "missing_qualifying",
            "usable_q1_count": 0,
        }
    q_rows = _unique_rows(loader, qualifying_rows)
    roster = _roster(loader, q_rows)
    active, aliases = loader._build_active_driver_map(roster, {})
    if not active:
        return {
            "round": round_number,
            "race": event.get("race"),
            "status": "incomplete_identity",
            "usable_q1_count": 0,
        }
    r_rows = _unique_rows(loader, result_rows) if result_rows else []
    matched_results = {
        loader._resolve_row_driver(row, aliases)
        for row in r_rows
    }
    matched_results.discard(None)
    expected = max(len(active), len(r_rows))
    result_complete = loader._near_complete(len(matched_results), expected)
    q1_rows = []
    q1_ids = set()
    for row in q_rows:
        driver_id = loader._resolve_row_driver(row, aliases)
        value = _q1_time(row)
        team_id = loader._row_team_id(row)
        if driver_id is not None and value is not None and team_id is not None:
            if driver_id in q1_ids:
                raise CurrentSeasonDataError("Conflicting historical Q1 driver aliases")
            q1_rows.append({
                "driver_id": driver_id,
                "team_id": team_id,
                "q1_seconds": value,
            })
            q1_ids.add(driver_id)
    q1_complete = loader._near_complete(len(q1_ids), len(active))
    base = {
        "round": round_number,
        "race": event.get("race"),
        "qualifying_entrants": len(active),
        "result_entrants": len(r_rows),
        "matched_result_entrants": len(matched_results),
        "usable_q1_count": len(q1_rows),
        "result_coverage": result_complete,
        "q1_coverage": q1_complete,
    }
    if len(q1_rows) < 2:
        return {**base, "status": "insufficient_q1_times"}
    if not result_complete:
        return {**base, "status": "incomplete_result_coverage"}
    field_median = median(row["q1_seconds"] for row in q1_rows)
    grouped: dict[str, list[float]] = defaultdict(list)
    driver_counts: dict[str, int] = defaultdict(int)
    for row in q1_rows:
        grouped[row["team_id"]].append(row["q1_seconds"])
        driver_counts[row["team_id"]] += 1
    teams = {
        team: {
            "q1_count": driver_counts[team],
            "q1_median": median(values),
            "residual": median(values) / field_median - 1,
        }
        for team, values in sorted(grouped.items())
    }
    return {
        **base,
        "status": "scored",
        "field_q1_median": field_median,
        "teams": teams,
    }


def build_historical_q1_events(
    loader,
    events: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    qualifying: Sequence[Mapping[str, Any]],
    *,
    before_round: int | None = None,
) -> list[dict]:
    """Summarize eligible prefix events without fetching or mutating loader data."""
    results_by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    qualifying_by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in results:
        round_number = _integer(row.get("round"))
        if round_number is not None:
            results_by_round[round_number].append(row)
    for row in qualifying:
        round_number = _integer(row.get("round"))
        if round_number is not None:
            qualifying_by_round[round_number].append(row)
    selected_events = [
        event for event in events
        if before_round is None or (_integer(event.get("round")) or 0) < before_round
    ]
    return [
        _event_history(
            loader,
            event,
            results_by_round.get(_integer(event.get("round")), []),
            qualifying_by_round.get(_integer(event.get("round")), []),
        )
        for event in sorted(selected_events, key=lambda item: _integer(item.get("round")) or 0)
    ]


def select_history(events: Sequence[Mapping[str, Any]], target_round: int) -> list[dict]:
    """Select the fixed three-event scored prefix before a target."""
    scored = [
        dict(event)
        for event in events
        if event.get("status") == "scored"
        and _integer(event.get("round")) is not None
        and _integer(event["round"]) < target_round
    ]
    return sorted(scored, key=lambda item: item["round"])[-HISTORY_WINDOW_EVENTS:]


def recent_team_q1_predictions(
    native_predictions: Sequence[Mapping[str, Any]],
    history_events: Sequence[Mapping[str, Any]],
    target_round: int,
) -> tuple[list[dict], dict]:
    """Transform all native target predictions using fixed historical residuals."""
    rows = [dict(row) for row in native_predictions]
    valid = [
        row
        for row in rows
        if isinstance(row.get("team_id"), str)
        and isinstance(row.get("predicted_seconds"), (int, float))
        and not isinstance(row["predicted_seconds"], bool)
        and math.isfinite(row["predicted_seconds"])
        and row["predicted_seconds"] > 0
    ]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in valid:
        grouped[row["team_id"]].append(row["predicted_seconds"])
    if not valid:
        return rows, {
            "training_rounds": [],
            "source_coverage": {},
            "fallback_teams": [],
            "candidate_fallback": "no_valid_native_predictions",
            "prediction_fields": {
                "predicted_seconds": "experimental transformation",
                "driver_id_team_id_and_native_ratings": "copied from full_model",
            },
        }
    field_median = median(row["predicted_seconds"] for row in valid)
    native_team_medians = {
        team: median(values) for team, values in sorted(grouped.items())
    }
    native_residuals = {
        team: team_median / field_median - 1
        for team, team_median in native_team_medians.items()
    }
    training = select_history(history_events, target_round)
    history_by_team: dict[str, list[dict]] = defaultdict(list)
    for event in training:
        for team, details in event.get("teams", {}).items():
            if team in native_team_medians:
                history_by_team[team].append({
                    "round": event["round"],
                    "race": event.get("race"),
                    "q1_count": details["q1_count"],
                    "residual": details["residual"],
                })
    source_coverage = {}
    team_residuals = {}
    fallback_teams = []
    for team in sorted(native_team_medians):
        evidence = history_by_team[team]
        if evidence:
            team_residuals[team] = median(item["residual"] for item in evidence)
            source_coverage[team] = {
                "source": "historical_team_q1",
                "rounds": [item["round"] for item in evidence],
                "events": len(evidence),
                "usable_q1_observations": sum(item["q1_count"] for item in evidence),
            }
        else:
            team_residuals[team] = native_residuals[team]
            fallback_teams.append(team)
            source_coverage[team] = {
                "source": "native_full_model_team_residual",
                "rounds": [],
                "events": 0,
                "usable_q1_observations": 0,
                "fallback_residual": native_residuals[team],
            }
    metadata = {
        "target_round": target_round,
        "training_rounds": [event["round"] for event in training],
        "training_events": training,
        "source_coverage": source_coverage,
        "fallback_teams": fallback_teams,
        "candidate_fallback": None,
        "history_window_events": HISTORY_WINDOW_EVENTS,
        "prediction_fields": {
            "predicted_seconds": "experimental transformation",
            "driver_id_team_id_and_native_ratings": "copied from full_model",
        },
    }
    # Preserve exact native values for cold starts.  This also avoids a
    # mathematically unnecessary floating-point perturbation in round one.
    if not training:
        return rows, metadata
    candidate = []
    for row in rows:
        if row not in valid:
            candidate.append(dict(row))
            continue
        within_team = row["predicted_seconds"] / native_team_medians[row["team_id"]] - 1
        value = field_median * (1 + team_residuals[row["team_id"]] + within_team)
        candidate.append({**row, "predicted_seconds": value})
    if not all(
        isinstance(row.get("predicted_seconds"), (int, float))
        and not isinstance(row["predicted_seconds"], bool)
        and math.isfinite(row["predicted_seconds"])
        and row["predicted_seconds"] > 0
        for row in candidate
    ):
        metadata["candidate_fallback"] = "invalid_candidate_predictions"
        return rows, metadata
    return candidate, metadata
