"""Current-season round holdouts for qualifying pace, using today's revised feeds.

Target entrant identities are known; target performance and later rounds never
enter predictions. This is not a reconstruction of historically available data.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import median

import numpy as np

from f1sim.analysis.qualifying_history import (
    build_historical_q1_events,
    recent_team_q1_predictions,
)
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


_NEUTRAL_DRIVER_ASSUMPTIONS = {
    "skill_rating": 0.92,
    "consistency": 0.968,
    "wet_skill_modifier": 0.9936,
    "overtaking_skill": 0.90,
    "tire_management": 0.968,
}


def _component_assumptions() -> dict:
    """Return the fixed assumptions used by the opt-in pace decomposition."""
    return {
        "neutral_driver": {
            **_NEUTRAL_DRIVER_ASSUMPTIONS,
            "source": "fixed diagnostic assumption; not fitted",
        },
        "variants": {
            "constructor_prior": {
                "driver": "neutral_driver",
                "stats_weights": {"track": 0.0, "form": 0.0, "qualifying": 0.0},
                "cars": "constructor_prior_stats",
            },
            "team_form": {
                "driver": "neutral_driver",
                "cars": "full_model_stats",
            },
            "full_model": {
                "driver": "full_model_stats",
                "cars": "full_model_stats",
                "stats_weights": {"track": 0.0, "form": 0.3, "qualifying": 0.2},
            },
            "recent_team_q1": {
                "source": "experimental transformation of full_model predictions",
                "history_window_events": 3,
                "history_selection": "three most recent earlier scored events",
                "scored_event_gate": (
                    "evaluator identity/result coverage plus at least two usable Q1 labels; "
                    "Q1 field coverage is reported, not required"
                ),
                "form_races_independent": True,
                "team_identity": "historical qualifying row identity as recorded",
                "team_residual": "median(team Q1 median / event field Q1 median - 1)",
                "within_team_residual": (
                    "native full_model prediction relative to target team median"
                ),
                "missing_team_fallback": "native full_model team contribution",
                "candidate_formula": (
                    "native full-field median * (1 + historical team residual + "
                    "native within-team residual)"
                ),
                "prediction_labels_used": False,
                "caveat": (
                    "Retrospective evidence selected after inspecting this dataset; "
                    "not prospective validation or a production rating."
                ),
            },
        },
        "qualifying_simulation": {
            "sample_variation": False,
            "tires": "all fresh compounds",
        },
    }


def _neutral_driver_stats(stats):
    """Copy stats while removing driver evidence for the component variants.

    These values deliberately use the zero-residual intercept and the loader's
    fixed no-evidence dispersion. They are diagnostic assumptions, not fitted
    estimates and do not alter the production loader.
    """
    return {
        driver_id: item.model_copy(update={
            "driver_skill_rating": _NEUTRAL_DRIVER_ASSUMPTIONS["skill_rating"],
            "consistency_rating": _NEUTRAL_DRIVER_ASSUMPTIONS["consistency"],
            "wet_skill_modifier": _NEUTRAL_DRIVER_ASSUMPTIONS["wet_skill_modifier"],
            "overtaking_skill": _NEUTRAL_DRIVER_ASSUMPTIONS["overtaking_skill"],
            "tire_management": _NEUTRAL_DRIVER_ASSUMPTIONS["tire_management"],
        })
        for driver_id, item in stats.items()
    }


def _qualifying_predictions(
    drivers, cars, stats, track, weather, simulator, labels, previous,
) -> list[dict]:
    """Predict a fold with supplied cars and drivers using the real lap model."""
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
    return predictions


def _team_median_metrics(rows) -> dict:
    """Score median predicted and observed pace for each observed team."""
    rows = [
        row for row in rows
        if row.get("predicted_seconds") is not None
        and row.get("observed_q1_seconds") is not None
    ]
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        team = row["team_id"]
        grouped.setdefault(team, {"predicted": [], "observed": []})
        grouped[team]["predicted"].append(row["predicted_seconds"])
        grouped[team]["observed"].append(row["observed_q1_seconds"])
    predicted = [median(values["predicted"]) for values in grouped.values()]
    observed = [median(values["observed"]) for values in grouped.values()]
    metrics = _metrics(predicted, observed)
    return {
        "teams": len(grouped),
        **{key: value for key, value in metrics.items() if key != "drivers"},
    }


def _teammate_gap_metrics(rows) -> dict:
    """Score signed normalized gaps for every observed same-team pair."""
    rows = [
        row for row in rows
        if row.get("predicted_seconds") is not None
        and row.get("observed_q1_seconds") is not None
    ]
    predicted_median = median(row["predicted_seconds"] for row in rows) if rows else None
    observed_median = median(row["observed_q1_seconds"] for row in rows) if rows else None
    errors = []
    predicted_abs = []
    observed_abs = []
    teammate_pairs = 0
    comparable_pairs = 0
    agreement = 0.0
    ordered = sorted(rows, key=lambda row: row["driver_id"])
    for first, second in combinations(ordered, 2):
        if first["team_id"] != second["team_id"]:
            continue
        predicted_gap = (
            (first["predicted_seconds"] - second["predicted_seconds"])
            / predicted_median * 100
        )
        observed_gap = (
            (first["observed_q1_seconds"] - second["observed_q1_seconds"])
            / observed_median * 100
        )
        teammate_pairs += 1
        errors.append(abs(predicted_gap - observed_gap))
        predicted_abs.append(abs(predicted_gap))
        observed_abs.append(abs(observed_gap))
        if first["observed_q1_seconds"] == second["observed_q1_seconds"]:
            continue
        comparable_pairs += 1
        agreement += (
            0.5 if first["predicted_seconds"] == second["predicted_seconds"]
            else float((first["predicted_seconds"] < second["predicted_seconds"])
                       == (first["observed_q1_seconds"] < second["observed_q1_seconds"]))
        )
    return {
        "gap_mae_pct": sum(errors) / teammate_pairs if teammate_pairs else None,
        "mean_abs_predicted_gap_pct": (
            sum(predicted_abs) / teammate_pairs if teammate_pairs else None
        ),
        "mean_abs_observed_gap_pct": (
            sum(observed_abs) / teammate_pairs if teammate_pairs else None
        ),
        "teammate_pairs": teammate_pairs,
        "comparable_pairs": comparable_pairs,
        "pairwise_concordance": agreement / comparable_pairs if comparable_pairs else None,
    }


def _component_variant_metrics(predictions, scored_ids) -> dict:
    """Build main and paired metrics for one component prediction set."""
    by_id = {row["driver_id"]: row for row in predictions}
    scored = [
        by_id[driver_id] for driver_id in scored_ids
        if by_id[driver_id].get("predicted_seconds") is not None
        and by_id[driver_id].get("observed_q1_seconds") is not None
    ]
    paired = [row for row in scored if row["previous_q1_seconds"] is not None]
    previous_predictions = [
        {**row, "predicted_seconds": row["previous_q1_seconds"]} for row in paired
    ]
    return {
        "predictions": predictions,
        "model": _metrics(
            [row["predicted_seconds"] for row in scored],
            [row["observed_q1_seconds"] for row in scored],
        ),
        "team_medians": _team_median_metrics(scored),
        "teammate_gaps": _teammate_gap_metrics(scored),
        "paired_comparison": {
            "driver_ids": [row["driver_id"] for row in paired],
            "model": _metrics(
                [row["predicted_seconds"] for row in paired],
                [row["observed_q1_seconds"] for row in paired],
            ),
            "previous_q1": _metrics(
                [row["previous_q1_seconds"] for row in paired],
                [row["observed_q1_seconds"] for row in paired],
            ),
            "paired_team_medians": _team_median_metrics(paired),
            "paired_teammate_gaps": _teammate_gap_metrics(paired),
            "previous_q1_team_medians": _team_median_metrics(previous_predictions),
            "previous_q1_teammate_gaps": _teammate_gap_metrics(previous_predictions),
        },
    }


def _aggregate_team_metrics(metrics) -> dict:
    """Aggregate team medians using scored team observations as weights."""
    scored = [row for row in metrics if row["rank_mae"] is not None]
    count = sum(row["teams"] for row in scored)
    pairs = sum(row["comparable_pairs"] for row in scored)
    return {
        "scored_folds": len(scored),
        "team_observations": count,
        "rank_mae": (sum(row["rank_mae"] * row["teams"] for row in scored) / count
                     if count else None),
        "relative_pace_mae_pct": (
            sum(row["relative_pace_mae_pct"] * row["teams"] for row in scored) / count
            if count else None
        ),
        "comparable_pairs": pairs,
        "pairwise_concordance": (
            sum((row["pairwise_concordance"] or 0) * row["comparable_pairs"]
                for row in scored) / pairs
            if pairs else None
        ),
    }


def _aggregate_teammate_gaps(metrics) -> dict:
    """Aggregate teammate metrics by all same-team pairs, not driver rows."""
    scored = [row for row in metrics if row["teammate_pairs"]]
    pairs = sum(row["teammate_pairs"] for row in scored)
    comparable = sum(row["comparable_pairs"] for row in scored)
    return {
        "scored_folds": len(scored),
        "teammate_pairs": pairs,
        "gap_mae_pct": (
            sum(row["gap_mae_pct"] * row["teammate_pairs"] for row in scored) / pairs
            if pairs else None
        ),
        "mean_abs_predicted_gap_pct": (
            sum(row["mean_abs_predicted_gap_pct"] * row["teammate_pairs"] for row in scored)
            / pairs if pairs else None
        ),
        "mean_abs_observed_gap_pct": (
            sum(row["mean_abs_observed_gap_pct"] * row["teammate_pairs"] for row in scored)
            / pairs if pairs else None
        ),
        "comparable_pairs": comparable,
        "pairwise_concordance": (
            sum((row["pairwise_concordance"] or 0) * row["comparable_pairs"]
                for row in scored) / comparable
            if comparable else None
        ),
    }


def _aggregate_component_variant(folds, variant) -> dict:
    """Aggregate one component variant while retaining separate denominators."""
    rows = [fold["components"][variant] for fold in folds]
    return {
        "model": _aggregate([row["model"] for row in rows]),
        "team_medians": _aggregate_team_metrics([row["team_medians"] for row in rows]),
        "teammate_gaps": _aggregate_teammate_gaps([row["teammate_gaps"] for row in rows]),
        "paired_comparison": {
            "model": _aggregate([row["paired_comparison"]["model"] for row in rows]),
            "previous_q1": _aggregate([
                row["paired_comparison"]["previous_q1"] for row in rows
            ]),
            "paired_team_medians": _aggregate_team_metrics([
                row["paired_comparison"]["paired_team_medians"] for row in rows
            ]),
            "paired_teammate_gaps": _aggregate_teammate_gaps([
                row["paired_comparison"]["paired_teammate_gaps"] for row in rows
            ]),
            "previous_q1_team_medians": _aggregate_team_metrics([
                row["paired_comparison"]["previous_q1_team_medians"] for row in rows
            ]),
            "previous_q1_teammate_gaps": _aggregate_teammate_gaps([
                row["paired_comparison"]["previous_q1_teammate_gaps"] for row in rows
            ]),
        },
    }


def evaluate_qualifying_pace(
    loader: CurrentSeasonDataLoader, year: int, *, target_race: str | int | None = None,
    form_races: int = 3, weather: Weather | None = None, include_components: bool = False,
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
    historical_q1 = (
        build_historical_q1_events(
            loader,
            events,
            results,
            qualifying,
            before_round=max(int(event["round"]) for event in targets),
        )
        if include_components and targets
        else []
    )
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
        simulator = LapSimulator(np.random.default_rng(0))
        labels = {loader._resolve_row_driver(row, aliases): _q1_time(row)
                  for row in target_rows}
        baseline_round = eligible[-1] if eligible else None
        previous = {loader._resolve_row_driver(row, aliases): _q1_time(row)
                    for row in past_qualifying if _integer(row["round"]) == baseline_round}
        predictions = _qualifying_predictions(
            drivers, cars, stats, track, weather, simulator, labels, previous,
        )
        recent_team_q1_candidate = None
        recent_team_q1_metadata = None
        if include_components:
            recent_team_q1_candidate, recent_team_q1_metadata = (
                recent_team_q1_predictions(
                    predictions, historical_q1, target,
                )
            )
        scored = [row for row in predictions if row["observed_q1_seconds"] is not None]
        paired = [row for row in scored if row["previous_q1_seconds"] is not None]
        obs = [row["observed_q1_seconds"] for row in scored]
        paired_obs = [row["observed_q1_seconds"] for row in paired]
        fold = {
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
        }
        if include_components:
            constructor_stats = loader._build_driver_stats(
                year=year, target_event=event, roster=roster, driver_standings=[],
                constructor_standings=constructor_standings,
                race_rows=race_rows, quali_rows=quali_rows,
                target_qualifying_rows=[], track_weight=0.0, form_weight=0.0, quali_weight=0.0,
            )
            constructor_neutral_stats = _neutral_driver_stats(constructor_stats)
            team_neutral_stats = _neutral_driver_stats(stats)
            variant_predictions = {
                "constructor_prior": _qualifying_predictions(
                    loader.create_drivers_from_stats(constructor_neutral_stats),
                    loader.create_cars_from_stats(constructor_stats),
                    constructor_neutral_stats, track, weather, simulator, labels, previous,
                ),
                "team_form": _qualifying_predictions(
                    loader.create_drivers_from_stats(team_neutral_stats), cars,
                    team_neutral_stats, track, weather, simulator, labels, previous,
                ),
                "full_model": predictions,
                "recent_team_q1": recent_team_q1_candidate,
            }
            scored_ids = [row["driver_id"] for row in scored]
            fold["components"] = {
                "assumptions": _component_assumptions(),
                **{
                    name: _component_variant_metrics(rows, scored_ids)
                    for name, rows in variant_predictions.items()
                },
            }
            fold["components"]["recent_team_q1"]["forecast"] = recent_team_q1_metadata
        folds.append(fold)
    provenance = loader.get_provenance()
    aggregate = {
        "model": _aggregate([fold["model"] for fold in folds]),
        "paired_model": _aggregate([fold["paired_comparison"]["model"] for fold in folds]),
        "paired_previous_q1": _aggregate([
            fold["paired_comparison"]["previous_q1"] for fold in folds
        ]),
    }
    if include_components:
        aggregate["components"] = {
            "assumptions": _component_assumptions(),
            **{
                name: _aggregate_component_variant(folds, name)
                for name in (
                    "constructor_prior", "team_form", "full_model", "recent_team_q1",
                )
            },
        }
    return {
        "year": year, "evaluation": "round_holdout_q1", "form_races": form_races,
        "weather_assumption": weather.model_dump(),
        "entrant_basis": "target_qualifying_identities",
        "data_revision": "current_provider_data_not_historical_availability",
        "fetched_at": provenance["fetched_at"], "source_urls": provenance["urls"],
        "folds": folds,
        "aggregate": aggregate,
    }
