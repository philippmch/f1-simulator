"""Leakage-sensitive assembly shared by current-season holdout evaluators."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from f1sim.data.current import (
    CurrentSeasonDataError,
    CurrentSeasonDataLoader,
    DriverStats,
)
from f1sim.models import Car, Driver, Track


@dataclass(frozen=True, slots=True)
class HoldoutCoverage:
    """Identity coverage measured while validating one target event."""

    qualifying_entrants: int
    result_entrants: int
    matched_result_entrants: int
    expected_result_entrants: int


@dataclass(frozen=True, slots=True)
class HoldoutFoldMetadata:
    """Training cutoff and coverage details, separate from model inputs."""

    target_round: int
    target_race: Any
    training_cutoff_round: int
    eligible_form_rounds: tuple[int, ...]
    form_rounds: tuple[int, ...]
    standings_round: int | None
    baseline_round: int | None
    coverage: HoldoutCoverage


@dataclass(slots=True)
class HoldoutFoldInputs:
    """Models and whitelisted target identities assembled without target labels."""

    drivers: list[Driver]
    cars: dict[str, Car]
    track: Track
    roster: list[dict[str, Any]]
    stats: dict[str, DriverStats]
    aliases: dict[str, str]
    metadata: HoldoutFoldMetadata
    constructor_standings: list[Mapping[str, Any]]
    race_rows: list[Mapping[str, Any]]
    quali_rows: list[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class HoldoutObservations:
    """Qualifying rows for scoring, returned apart from model inputs."""

    target_qualifying_rows: tuple[Mapping[str, Any], ...]
    prior_qualifying_rows: tuple[Mapping[str, Any], ...]


class InsufficientTargetCoverage(CurrentSeasonDataError):
    """Valid target identities whose result overlap misses the holdout gate."""

    def __init__(
        self,
        *,
        target_round: int,
        roster: list[dict[str, Any]],
        target_qualifying_rows: Sequence[Mapping[str, Any]],
        coverage: HoldoutCoverage,
    ) -> None:
        self.target_round = target_round
        self.roster = roster
        self.target_qualifying_rows = tuple(deepcopy(row) for row in target_qualifying_rows)
        self.coverage = coverage
        super().__init__(f"Incomplete target entrant coverage for round {target_round}")


def _integer(value: Any) -> int | None:
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _unique_rows(
    loader: CurrentSeasonDataLoader, rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Reject conflicting per-round driver evidence, rather than choose by order."""
    unique = {}
    for row in rows:
        identity = loader._strong_driver_identity(row)
        key = (_integer(row.get("round")), identity)
        if identity is None or key[0] is None:
            raise CurrentSeasonDataError("Evaluation requires round and driver identities")
        if key in unique and unique[key] != row:
            raise CurrentSeasonDataError("Conflicting evaluation driver records")
        unique[key] = row
    return list(unique.values())


def _target_roster(
    loader: CurrentSeasonDataLoader, rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Whitelist entrant identity fields; never copy performance into the roster."""
    roster = []
    for row in rows:
        driver = loader._extract_driver(row)
        code = driver.get("code") or driver.get("driverId")
        name = loader._driver_name(driver)
        team = loader._row_team_id(row)
        if not isinstance(code, str) or not code.strip() or not name or team is None:
            raise CurrentSeasonDataError("Incomplete target qualifying entrant identity")
        roster.append({
            "id": code, "name": name, "team_name": team,
            "driverId": driver.get("driverId"), "code": driver.get("code"),
            "permanentNumber": driver.get("permanentNumber"),
        })
    return sorted(roster, key=lambda row: row["id"])


def _check_resolved_identities(
    loader: CurrentSeasonDataLoader,
    rows: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
    *,
    reject_ambiguous_aliases: bool = False,
) -> None:
    seen = set()
    for row in rows:
        matched_drivers = {
            aliases[alias]
            for alias in loader._driver_aliases(row)
            if alias in aliases
        }
        if reject_ambiguous_aliases and len(matched_drivers) > 1:
            raise CurrentSeasonDataError("Conflicting aliases in evaluation driver records")
        driver = next(iter(matched_drivers)) if len(matched_drivers) == 1 else None
        if driver is not None:
            identity = (_integer(row["round"]), driver)
            if identity in seen:
                raise CurrentSeasonDataError("Conflicting aliases in evaluation driver records")
            seen.add(identity)


def _constructor_standings(
    loader: CurrentSeasonDataLoader, year: int, round_number: int,
) -> list[Mapping[str, Any]]:
    if round_number == 0:
        return []
    url = loader._with_pagination(
        f"{loader.JOLPICA_BASE_URL}/{year}/{round_number}/constructorstandings.json", 0,
    )
    payload = loader._fetch_json(url)
    loader._validate_payload_season(payload, year, url)
    try:
        root = payload["MRData"]
        table = root["StandingsTable"]
        if (_integer(table.get("season")) != year
                or _integer(table.get("round")) != round_number):
            raise ValueError("Wrong standings table cutoff")
        lists = table["StandingsLists"]
        if (not isinstance(lists, list) or len(lists) != 1
                or not isinstance(lists[0], dict)
                or _integer(lists[0].get("season")) != year
                or _integer(lists[0].get("round")) != round_number):
            raise ValueError("Wrong standings cutoff")
        rows = lists[0]["ConstructorStandings"]
        total = _integer(root.get("total"))
        if (not isinstance(rows, list) or not rows or total != len(rows)
                or total > 100 or _integer(root.get("offset")) != 0):
            raise ValueError("Incomplete standings page")
        ids, teams, aliases = set(), set(), set()
        for row in rows:
            constructor = row["Constructor"]
            key, team = constructor["constructorId"], loader._row_team_id(row)
            points = row["points"]
            row_aliases = set(loader._constructor_map([row]))
            if (not isinstance(key, str) or not key or team is None or key in ids
                    or team in teams or aliases & row_aliases or isinstance(points, bool)
                    or not math.isfinite(float(points))):
                raise ValueError("Invalid constructor evidence")
            ids.add(key)
            teams.add(team)
            aliases.update(row_aliases)
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError) as exc:
        loader._mark_failed_url(url)
        raise CurrentSeasonDataError(
            f"Expected complete constructor standings for {year} round {round_number}"
        ) from exc
    return rows


def assemble_holdout_fold(
    loader: CurrentSeasonDataLoader,
    year: int,
    event: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    qualifying: Sequence[Mapping[str, Any]],
    *,
    form_races: int = 3,
) -> tuple[HoldoutFoldInputs, HoldoutObservations]:
    """Build a pre-target model from target identities and strictly prior evidence.

    The model-input result contains a whitelisted entrant roster and no target
    result or qualifying labels. Scoring rows are returned as a separate value
    so callers can use them only after model assembly.
    """
    if type(form_races) is not int or not 0 <= form_races <= 24:
        raise ValueError("form_races must be between 0 and 24")

    target = int(event["round"])
    calendar_rounds = {int(calendar_event["round"]) for calendar_event in events}
    target_rows = _unique_rows(
        loader, [row for row in qualifying if _integer(row.get("round")) == target],
    )
    if not target_rows:
        raise CurrentSeasonDataError(f"No target qualifying evidence for round {target}")
    roster = _target_roster(loader, target_rows)
    active, aliases = loader._build_active_driver_map(roster, {})
    _check_resolved_identities(
        loader, target_rows, aliases, reject_ambiguous_aliases=True,
    )

    target_results = _unique_rows(
        loader, [row for row in results if _integer(row.get("round")) == target],
    )
    _check_resolved_identities(
        loader, target_results, aliases, reject_ambiguous_aliases=True,
    )
    matched_results = {
        loader._resolve_row_driver(row, aliases) for row in target_results
    }
    matched_results.discard(None)
    expected = max(len(active), len(target_results))
    coverage = HoldoutCoverage(
        qualifying_entrants=len(active),
        result_entrants=len(target_results),
        matched_result_entrants=len(matched_results),
        expected_result_entrants=expected,
    )
    if not loader._near_complete(len(matched_results), expected):
        raise InsufficientTargetCoverage(
            target_round=target,
            roster=roster,
            target_qualifying_rows=target_rows,
            coverage=coverage,
        )

    past_results = _unique_rows(
        loader, [row for row in results
                 if _integer(row.get("round")) in calendar_rounds
                 and _integer(row.get("round")) < target],
    )
    past_qualifying = _unique_rows(
        loader, [row for row in qualifying
                 if _integer(row.get("round")) in calendar_rounds
                 and _integer(row.get("round")) < target],
    )
    _check_resolved_identities(loader, past_results, aliases)
    _check_resolved_identities(loader, past_qualifying, aliases)
    for row in past_qualifying:
        value = loader._row_qualifying_time(row)
        if value is not None and not math.isfinite(value):
            raise CurrentSeasonDataError("Non-finite historical qualifying pace")
    for row in past_results:
        _, value = loader._row_race_metric(row)
        if value is not None and not math.isfinite(value):
            raise CurrentSeasonDataError("Non-finite historical race pace")

    result_ids = loader._round_driver_ids(past_results, aliases, qualifying=False)
    qualifying_ids = loader._round_driver_ids(past_qualifying, aliases, qualifying=True)
    eligible = sorted(
        round_number for round_number in result_ids
        if loader._near_complete(len(result_ids[round_number]), len(active))
        and loader._near_complete(len(qualifying_ids.get(round_number, set())), len(active))
    )
    selected = eligible[-form_races:] if form_races else []
    race_rows = [row for row in past_results if _integer(row["round"]) in selected]
    quali_rows = [row for row in past_qualifying if _integer(row["round"]) in selected]

    constructor_standings = _constructor_standings(loader, year, target - 1)
    stats = loader._build_driver_stats(
        year=year, target_event=event, roster=roster, driver_standings=[],
        constructor_standings=constructor_standings,
        race_rows=race_rows, quali_rows=quali_rows,
        target_qualifying_rows=[], track_weight=0.0, form_weight=0.3, quali_weight=0.2,
    )
    track = loader.create_track_from_stats(loader._track_stats_from_event(year, event))
    drivers = loader.create_drivers_from_stats(stats)
    cars = loader.create_cars_from_stats(stats)

    metadata = HoldoutFoldMetadata(
        target_round=target,
        target_race=event["race"],
        training_cutoff_round=target - 1,
        eligible_form_rounds=tuple(eligible),
        form_rounds=tuple(selected),
        standings_round=target - 1 if target > 1 else None,
        baseline_round=eligible[-1] if eligible else None,
        coverage=coverage,
    )
    fold_inputs = HoldoutFoldInputs(
        drivers=drivers,
        cars=cars,
        track=track,
        roster=roster,
        stats=stats,
        aliases=aliases,
        metadata=metadata,
        constructor_standings=constructor_standings,
        race_rows=race_rows,
        quali_rows=quali_rows,
    )
    observations = HoldoutObservations(
        target_qualifying_rows=tuple(deepcopy(row) for row in target_rows),
        prior_qualifying_rows=tuple(deepcopy(row) for row in past_qualifying),
    )
    return fold_inputs, observations
