"""Current-season round holdouts for qualifying pace, using today's revised feeds.

Target entrant identities are known; target performance and later rounds never
enter predictions. This is not a reconstruction of historically available data.
"""

from __future__ import annotations

import math
from statistics import median

import numpy as np

from f1sim.data.current import CurrentSeasonDataError, CurrentSeasonDataLoader, _parse_time_seconds
from f1sim.models import Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator


def _integer(value):
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def _unique_rows(loader, rows):
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


def _target_roster(loader, rows):
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


def _check_resolved_identities(loader, rows, aliases):
    seen = set()
    for row in rows:
        driver = loader._resolve_row_driver(row, aliases)
        if driver is not None:
            identity = (_integer(row["round"]), driver)
            if identity in seen:
                raise CurrentSeasonDataError("Conflicting aliases in evaluation driver records")
            seen.add(identity)


def _constructor_standings(loader, year, round_number):
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


def _midranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            ranks[index] = (start + 1 + end) / 2
        start = end
    return ranks


def _q1_time(row):
    value = row.get("Q1")
    parsed = None if isinstance(value, bool) else _parse_time_seconds(value)
    return parsed if parsed is not None and math.isfinite(parsed) and parsed > 0 else None


def _metrics(predicted, observed):
    count = len(observed)
    if count < 2:
        return {"drivers": count, "rank_mae": None, "relative_pace_mae_pct": None,
                "comparable_pairs": 0, "pairwise_concordance": None}
    p_ranks, o_ranks = _midranks(predicted), _midranks(observed)
    p_center, o_center = median(predicted), median(observed)
    pairs, agreement = 0, 0.0
    for i in range(count):
        for j in range(i):
            if observed[i] == observed[j]:
                continue
            pairs += 1
            agreement += (0.5 if predicted[i] == predicted[j] else
                          float((predicted[i] < predicted[j]) == (observed[i] < observed[j])))
    return {
        "drivers": count,
        "rank_mae": sum(abs(p - o) for p, o in zip(p_ranks, o_ranks)) / count,
        "relative_pace_mae_pct": sum(
            abs(p / p_center - o / o_center) * 100 for p, o in zip(predicted, observed)
        ) / count,
        "comparable_pairs": pairs,
        "pairwise_concordance": agreement / pairs if pairs else None,
    }


def _aggregate(metrics):
    scored = [row for row in metrics if row["rank_mae"] is not None]
    count = sum(row["drivers"] for row in scored)
    pairs = sum(row["comparable_pairs"] for row in scored)
    return {
        "scored_folds": len(scored), "driver_observations": count,
        "rank_mae": (sum(row["rank_mae"] * row["drivers"] for row in scored) / count
                     if count else None),
        "relative_pace_mae_pct": (
            sum(row["relative_pace_mae_pct"] * row["drivers"] for row in scored) / count
            if count else None
        ),
        "comparable_pairs": pairs,
        "pairwise_concordance": (
            sum((row["pairwise_concordance"] or 0) * row["comparable_pairs"] for row in scored)
            / pairs if pairs else None
        ),
    }


def evaluate_qualifying_pace(
    loader: CurrentSeasonDataLoader, year: int, *, target_race: str | int | None = None,
    form_races: int = 3, weather: Weather | None = None,
) -> dict:
    """Score noise-free qualifying pace against target Q1, with previous-Q1 baseline.

    Uses target qualifying identities/teams, prefix form rows, preceding-round
    constructor standings (including sprint points), and static venue physics.
    Weather is an explicit fixed scenario, not observed target weather. All
    provider data is fetched today and may contain retrospective corrections.
    """
    loader._assert_current_year(year)
    if type(form_races) is not int or not 0 <= form_races <= 24:
        raise ValueError("form_races must be between 0 and 24")
    weather = (Weather() if weather is None else weather).model_copy(
        deep=True, update={"change_probability": 0.0},
    )
    events = loader.get_event_schedule(year)
    results, qualifying = loader._season_data(year)
    completed = {_integer(row.get("round")) for row in results}
    if target_race is None:
        targets = [event for event in events if int(event["round"]) in completed]
    else:
        targets = [loader._event_for_race(year, target_race)]
        if int(targets[0]["round"]) not in completed:
            raise ValueError("Evaluation requires a completed current-season target")
    calendar_rounds = {int(event["round"]) for event in events}
    folds = []
    for event in sorted(targets, key=lambda row: int(row["round"])):
        target = int(event["round"])
        target_rows = _unique_rows(loader, [row for row in qualifying
                                           if _integer(row.get("round")) == target])
        if not target_rows:
            raise CurrentSeasonDataError(f"No target qualifying evidence for round {target}")
        roster = _target_roster(loader, target_rows)
        active, aliases = loader._build_active_driver_map(roster, {})
        target_results = _unique_rows(loader, [row for row in results
                                              if _integer(row.get("round")) == target])
        matched_results = {loader._resolve_row_driver(row, aliases) for row in target_results}
        matched_results.discard(None)
        expected = max(len(active), len(target_results))
        if not loader._near_complete(len(matched_results), expected):
            raise CurrentSeasonDataError(f"Incomplete target entrant coverage for round {target}")
        past_results = _unique_rows(loader, [row for row in results
                                            if _integer(row.get("round")) in calendar_rounds
                                            and _integer(row.get("round")) < target])
        past_qualifying = _unique_rows(loader, [row for row in qualifying
                                               if _integer(row.get("round")) in calendar_rounds
                                               and _integer(row.get("round")) < target])
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
        eligible = sorted(round_number for round_number in result_ids
                          if loader._near_complete(len(result_ids[round_number]), len(active))
                          and loader._near_complete(len(qualifying_ids.get(round_number, set())),
                                                    len(active)))
        selected = eligible[-form_races:] if form_races else []
        stats = loader._build_driver_stats(
            year=year, target_event=event, roster=roster, driver_standings=[],
            constructor_standings=_constructor_standings(loader, year, target - 1),
            race_rows=[row for row in past_results if _integer(row["round"]) in selected],
            quali_rows=[row for row in past_qualifying if _integer(row["round"]) in selected],
            target_qualifying_rows=[], track_weight=0.0, form_weight=0.3, quali_weight=0.2,
        )
        track = loader.create_track_from_stats(loader._track_stats_from_event(year, event))
        drivers = loader.create_drivers_from_stats(stats)
        cars = loader.create_cars_from_stats(stats)
        simulator = LapSimulator(np.random.default_rng(0))
        labels = {loader._resolve_row_driver(row, aliases): _q1_time(row)
                  for row in target_rows}
        baseline_round = eligible[-1] if eligible else None
        previous = {loader._resolve_row_driver(row, aliases): _q1_time(row)
                    for row in past_qualifying if _integer(row["round"]) == baseline_round}
        predictions = []
        for driver in sorted(drivers, key=lambda row: row.id):
            prediction = min(simulator.calculate_qualifying_lap(
                driver, cars[driver.team_id], track, tire, weather, sample_variation=False,
            ) for tire in TIRE_COMPOUNDS.values())
            observed, baseline = labels.get(driver.id), previous.get(driver.id)
            predictions.append({
                "driver_id": driver.id, "team_id": driver.team_id,
                "predicted_seconds": prediction,
                "observed_q1_seconds": observed if observed and observed > 0 else None,
                "previous_q1_seconds": baseline if baseline and baseline > 0 else None,
                "skill_rating": driver.skill_rating,
                "car_pace": cars[driver.team_id].base_pace,
                "constructor_points": stats[driver.id].constructor_points,
            })
        scored = [row for row in predictions if row["observed_q1_seconds"] is not None]
        paired = [row for row in scored if row["previous_q1_seconds"] is not None]
        obs = [row["observed_q1_seconds"] for row in scored]
        paired_obs = [row["observed_q1_seconds"] for row in paired]
        folds.append({
            "round": target, "race": event["race"], "entrants": len(active),
            "status": "scored" if len(scored) >= 2 else "insufficient_q1_times",
            "result_entrants": len(target_results), "matched_result_entrants": len(matched_results),
            "form_rounds": selected, "standings_round": target - 1 or None,
            "baseline_round": baseline_round, "static_reference_lap_seconds": track.base_lap_time,
            "predictions": predictions,
            "model": _metrics([row["predicted_seconds"] for row in scored], obs),
            "paired_comparison": {
                "driver_ids": [row["driver_id"] for row in paired],
                "model": _metrics([row["predicted_seconds"] for row in paired], paired_obs),
                "previous_q1": _metrics([row["previous_q1_seconds"] for row in paired], paired_obs),
            },
        })
    provenance = loader.get_provenance()
    return {
        "year": year, "evaluation": "round_holdout_q1", "form_races": form_races,
        "weather_assumption": weather.model_dump(),
        "entrant_basis": "target_qualifying_identities",
        "data_revision": "current_provider_data_not_historical_availability",
        "fetched_at": provenance["fetched_at"], "source_urls": provenance["urls"],
        "folds": folds,
        "aggregate": {
            "model": _aggregate([fold["model"] for fold in folds]),
            "paired_model": _aggregate([fold["paired_comparison"]["model"] for fold in folds]),
            "paired_previous_q1": _aggregate([
                fold["paired_comparison"]["previous_q1"] for fold in folds
            ]),
        },
    }
