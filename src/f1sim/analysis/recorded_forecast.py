"""Record pre-qualifying forecasts and score their immutable probabilities later."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from f1sim.analysis.holdout_folds import _integer
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.race_probability_evaluation import _observed_winner, _target_rows
from f1sim.analysis.race_probability_scores import score_winner_counts, summarize_winner_trials
from f1sim.analysis.replay import _load_saved_runner
from f1sim.analysis.winner_baselines import build_winner_baselines, score_saved_winner_baselines
from f1sim.cancellation import raise_if_cancelled
from f1sim.models import Weather


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Forecast timestamps must be strings with a timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Forecast timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _qualifying_start(event):
    sessions = event.get("sessions", {})
    if not isinstance(sessions, dict):
        raise ValueError("Recording needs a dated qualifying session with an explicit UTC time")
    session = next((value for key, value in sessions.items()
                    if isinstance(key, str) and key.lower() == "qualifying"), None)
    if not isinstance(session, dict) or not session.get("date") or not session.get("time"):
        raise ValueError("Recording needs a dated qualifying session with an explicit UTC time")
    return _timestamp(f'{session["date"]}T{session["time"]}')


def _digest(body):
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _clock_time(clock):
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("The forecast clock must return a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _qualifying_summary(trials, driver_ids):
    poles = dict.fromkeys(driver_ids, 0)
    positions = defaultdict(list)
    for trial in trials:
        if {row.driver_id for row in trial} != set(driver_ids) or len(trial) != len(driver_ids):
            raise ValueError("Qualifying forecast trials must contain the complete roster")
        if {row.position for row in trial} != set(range(1, len(driver_ids) + 1)):
            raise ValueError("Qualifying forecast positions must form a full permutation")
        for row in trial:
            poles[row.driver_id] += row.position == 1
            positions[row.driver_id].append(row.position)
    return {identity: {"pole_count": poles[identity],
                       "pole_probability": poles[identity] / len(trials),
                       "mean_position": sum(positions[identity]) / len(trials)}
            for identity in driver_ids}


def record_race_forecast(loader, year, target_race, *, trials=100, seed=42, weather=None,
                         parallel=False, max_workers=None, progress_callback=None,
                         cancel_requested=None, now=None):
    """Build and record a future-event forecast without target performance data.

    The local timestamp and content seal detect accidental alteration. They do
    not prove independent publication; archive the file before the event for
    that stronger claim. Finishing after qualifying starts rejects the record.
    """
    loader._assert_current_year(year)
    if type(trials) is not int or not 1 <= trials <= 10_000:
        raise ValueError("trials must be an integer from 1 to 10000")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")
    raise_if_cancelled(cancel_requested)
    clock = now if now is not None else lambda: datetime.now(timezone.utc)
    started = _clock_time(clock)
    event = loader._event_for_race(year, target_race)
    deadline = _qualifying_start(event)
    if started >= deadline:
        raise ValueError("A pre-event forecast must start before qualifying")
    target = int(event["round"])
    results, qualifying = loader._season_data(year)
    if any((_integer(row.get("round")) or 0) >= target for row in (*results, *qualifying)):
        raise ValueError("Target or later performance exists; a pre-event forecast is unavailable")
    stats = loader.get_weighted_driver_stats(year, target)
    drivers = sorted(loader.create_drivers_from_stats(stats), key=lambda driver: driver.id)
    cars = loader.create_cars_from_stats(stats)
    track = loader.create_track_from_stats(loader._track_stats_from_event(year, event))
    roster = [{"id": driver.id, "name": driver.name, "team_name": cars[driver.team_id].team_name}
              for driver in drivers]
    _, aliases = loader._build_active_driver_map(roster, {})
    ids = [driver.id for driver in drivers]
    race_rounds = sorted({number for row in results
                         if (number := _integer(row.get("round"))) is not None and number > 0})
    earlier_rounds = sorted({number for row in (*results, *qualifying)
                            if (number := _integer(row.get("round"))) is not None and number > 0})
    prior_outcomes = [{"round": number, **_observed_winner(
        loader, _target_rows(loader, results, number), aliases, set(ids),
    )} for number in race_rounds]
    baselines = build_winner_baselines(
        {driver.id: driver.team_id for driver in drivers},
        {item.team_id: item.constructor_points for item in stats.values()},
        prior_outcomes, cutoff_round=target - 1,
    )
    assumed = (Weather() if weather is None else weather).model_copy(
        deep=True, update={"change_probability": 0.0},
    )
    raise_if_cancelled(cancel_requested)
    simulation = MonteCarloRunner(drivers, cars, track, assumed, seed=seed).run(
        trials, parallel=parallel, max_workers=max_workers, progress_callback=progress_callback,
        cancel_requested=cancel_requested,
    )
    raise_if_cancelled(cancel_requested)
    recorded = _clock_time(clock)
    if recorded >= deadline:
        raise ValueError("Forecast finished after qualifying started; no pre-event record is valid")
    body = {
        "schema_version": 1, "kind": "pre_qualifying_race_forecast", "year": year,
        "event": {key: event.get(key) for key in ("round", "race", "circuit_id", "date")},
        "started_at": started.isoformat(), "recorded_at": recorded.isoformat(),
        "qualifying_starts_at": deadline.isoformat(),
        "timing_evidence": "local_clock; independent publication not verified",
        "training_cutoff_round": target - 1, "performance_rounds": earlier_rounds,
        "target_performance_used": False,
        "weather_assumption": "specified fixed rainfall; surface wetness evolves",
        "roster": roster, "provenance": loader.get_provenance(),
        "metadata": {"seed": simulation.seed, "num_simulations": trials,
                     "race_engine": simulation.race_engine},
        "simulation_inputs": simulation.input_snapshot,
        "winner_forecast": summarize_winner_trials(simulation.race_results, ids, trials),
        "qualifying_forecast": _qualifying_summary(simulation.qualifying_results, ids),
        "baselines": baselines,
    }
    return {**body, "content_sha256": _digest(body)}


def save_recorded_forecast(path, record):
    """Create a new sealed record, refusing to overwrite an existing forecast."""
    validate_recorded_forecast(record)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        json.dump(record, destination, sort_keys=True, indent=2, allow_nan=False)
        destination.write("\n")
    return path


def validate_recorded_forecast(record):
    """Verify the seal, time boundary, counts and recorded historical references."""
    try:
        return _validate_recorded_forecast(record)
    except (KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError("Recorded forecast contains missing or invalid fields") from error


def _validate_recorded_forecast(record):
    if not isinstance(record, dict):
        raise ValueError("Recorded forecast must be a JSON object")
    body = {key: value for key, value in record.items() if key != "content_sha256"}
    if record.get("content_sha256") != _digest(body):
        raise ValueError("Recorded forecast content seal does not match")
    if (type(record.get("schema_version")) is not int or record["schema_version"] != 1
            or record.get("kind") != "pre_qualifying_race_forecast"):
        raise ValueError("Unsupported recorded forecast format")
    if type(record.get("year")) is not int or not 1 <= record["year"] <= 9999:
        raise ValueError("Recorded forecast needs a valid year")
    started, recorded, deadline = (_timestamp(record[key]) for key in (
        "started_at", "recorded_at", "qualifying_starts_at"))
    if not started <= recorded < deadline:
        raise ValueError("Recorded forecast must precede qualifying")
    if record["year"] != deadline.year:
        raise ValueError("Recorded forecast year does not match the qualifying session")
    event = record["event"]
    if (not isinstance(event, dict)
            or any(not isinstance(event.get(key), str) or not event[key].strip()
                   for key in ("race", "circuit_id", "date"))):
        raise ValueError("Recorded forecast needs a dated event and circuit identity")
    target = record["event"]["round"]
    if (type(target) is not int or target < 1
            or type(record["training_cutoff_round"]) is not int
            or record["training_cutoff_round"] != target - 1
            or record.get("target_performance_used") is not False
            or not isinstance(record["performance_rounds"], list)
            or any(type(number) is not int or not 1 <= number < target
                   for number in record["performance_rounds"])):
        raise ValueError("Recorded forecast contains invalid training boundaries")
    trials = record["metadata"]["num_simulations"]
    if type(trials) is not int or not 1 <= trials <= 10_000:
        raise ValueError("Recorded forecast needs a trial count from 1 to 10000")
    ids = [row["id"] for row in record["roster"]]
    if (len(set(ids)) != len(ids) or not ids
            or any(not isinstance(identity, str) or not identity.strip() for identity in ids)):
        raise ValueError("Recorded forecast needs a unique roster")
    winner = record["winner_forecast"]
    if set(winner["drivers"]) != set(ids):
        raise ValueError("Winner forecast must match its roster")
    wins = {identity: row["wins"] for identity, row in winner["drivers"].items()}
    score = score_winner_counts(wins, winner["no_classified_winner"]["count"], ids[0])
    if sum(wins.values()) + winner["no_classified_winner"]["count"] != trials:
        raise ValueError("Winner counts do not match the trial count")
    for identity, row in winner["drivers"].items():
        if (type(row["probability"]) not in (int, float)
                or row["probability"] != row["wins"] / trials):
            raise ValueError(f"Winner probability does not match counts for {identity}")
    no_winner = winner["no_classified_winner"]
    if (type(no_winner["probability"]) not in (int, float)
            or no_winner["probability"] != no_winner["count"] / trials):
        raise ValueError("No-winner probability does not match counts")
    grid = record["qualifying_forecast"]
    if set(grid) != set(ids):
        raise ValueError("Qualifying forecast must match the roster")
    if sum(row["pole_count"] for row in grid.values()) != trials:
        raise ValueError("Pole counts do not match the trial count")
    for row in grid.values():
        if (type(row["pole_count"]) is not int or not 0 <= row["pole_count"] <= trials
                or type(row["pole_probability"]) not in (int, float)
                or row["pole_probability"] != row["pole_count"] / trials
                or type(row["mean_position"]) not in (int, float)
                or not math.isfinite(row["mean_position"])
                or not 1 <= row["mean_position"] <= len(ids)):
            raise ValueError("Qualifying forecast contains invalid counts or positions")
    if not math.isclose(sum(row["mean_position"] for row in grid.values()),
                        len(ids) * (len(ids) + 1) / 2, abs_tol=1e-9, rel_tol=0.):
        raise ValueError("Qualifying mean positions do not match complete trial permutations")
    score_saved_winner_baselines(record["baselines"], ids, target_round=target,
                                 observed_winner=None)
    return score


def load_recorded_forecast(path):
    """Read sealed probabilities and validate their saved simulation models."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_recorded_forecast(record)
    runner, _ = _load_saved_runner(path)
    if {driver.id for driver in runner.drivers} != {row["id"] for row in record["roster"]}:
        raise ValueError("Saved simulation inputs must match the forecast roster")
    return record


def score_recorded_forecast(record, loader, results, qualifying):
    """Score frozen winner and pole probabilities without running or refitting."""
    validate_recorded_forecast(record)
    snapshot = deepcopy(record)
    target = record["event"]["round"]
    event = loader._event_for_race(record["year"], target)
    if any(event.get(key) != record["event"][key] for key in ("circuit_id", "date")):
        raise ValueError("The scheduled date or circuit changed since this forecast was recorded")
    ids = [row["id"] for row in record["roster"]]
    _, aliases = loader._build_active_driver_map(record["roster"], {})
    rows = _target_rows(loader, results, target)
    observed = _observed_winner(loader, rows, aliases, set(ids))
    if observed["status"] != "observed":
        return {"status": "unscored", "reason": observed["reason"],
                "forecast_sha256": record["content_sha256"]}
    matched = {loader._resolve_row_driver(row, aliases) for row in rows}
    if len(matched & set(ids)) < .8 * max(len(ids), len(rows)):
        raise ValueError("Observed race results do not cover the forecast roster")
    winner = record["winner_forecast"]
    score = score_winner_counts(
        {identity: row["wins"] for identity, row in winner["drivers"].items()},
        winner["no_classified_winner"]["count"], observed["winner_id"],
    )
    q_rows = _target_rows(loader, qualifying, target)
    poles = [row for row in q_rows if loader._row_position(row) == 1]
    pole_id = loader._resolve_row_driver(poles[0], aliases) if len(poles) == 1 else None
    grid = record["qualifying_forecast"]
    pole_score = score_winner_counts(
        {identity: row["pole_count"] for identity, row in grid.items()}, 0, pole_id,
    ) if pole_id in grid else None
    observed_positions = {loader._resolve_row_driver(row, aliases): loader._row_position(row)
                          for row in q_rows}
    outside_positions = [loader._row_position(row) for row in q_rows
                         if loader._resolve_row_driver(row, aliases) not in grid
                         and loader._row_position(row) is not None]
    observed_positions = {identity: position - sum(other < position for other in outside_positions)
                          for identity, position in observed_positions.items()
                          if identity in grid and position is not None}
    errors = [abs(grid[identity]["mean_position"] - position)
              for identity, position in observed_positions.items()
              if identity in grid and position is not None]
    assert record == snapshot
    return {
        "status": "scored", "year": record["year"], "round": target,
        "forecast_sha256": record["content_sha256"], "recorded_at": record["recorded_at"],
        "timing_evidence": record["timing_evidence"], "observed_winner": observed["winner_id"],
        "winner_score": score, "observed_pole": pole_id, "pole_score": pole_score,
        "qualifying_position_mae": sum(errors) / len(errors) if errors else None,
        "qualifying_scored_drivers": len(errors),
        "qualifying_observed_drivers": len(q_rows), "forecast_drivers": len(ids),
        "qualifying_position_scope": "recorded roster; known outside entrants removed",
        "baselines": score_saved_winner_baselines(record["baselines"], ids,
            target_round=target, observed_winner=observed["winner_id"]),
        "observations_provenance": loader.get_provenance(),
    }
