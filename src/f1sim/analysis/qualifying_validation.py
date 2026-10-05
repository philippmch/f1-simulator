"""Compare actual calibrated qualifying inputs with native and earlier-Q1 pace."""

from __future__ import annotations

import numpy as np

from f1sim.analysis.holdout_folds import assemble_holdout_fold
from f1sim.analysis.pace_evaluation import (
    _aggregate,
    _aggregate_team_metrics,
    _aggregate_teammate_gaps,
    _metrics,
    _q1_time,
    _qualifying_predictions,
    _team_median_metrics,
    _teammate_gap_metrics,
    evaluate_qualifying_pace,
)
from f1sim.analysis.provenance import simulation_runtime
from f1sim.analysis.qualifying_calibration import QUALIFYING_PACE_POLICY
from f1sim.models import Weather
from f1sim.simulation.lap import LapSimulator


def evaluate_qualifying_calibration(loader, year, *, target_race=None, form_races=3,
                                   weather=None, qualifying_only=False):
    """Score Q1/Q2/Q3 on shared cohorts, without target times in forecasts.

    Q2 and Q3 are conditional on the entrants with usable times in that session
    and a previous-Q1 reference. These are correlated retrospective observations
    with assumed weather, not independent prospective races. The existing Q1
    diagnostic supplies the uncalibrated comparison and fixed history evidence.
    """
    weather = (Weather() if weather is None else weather).model_copy(
        deep=True, update={"change_probability": 0.0},
    )
    native = evaluate_qualifying_pace(
        loader, year, target_race=target_race, form_races=form_races, weather=weather,
        include_components=True, qualifying_only=qualifying_only,
    )
    events = loader.get_event_schedule(year)
    results, qualifying = loader._season_data(year)
    sessions = {name: {"folds": []} for name in ("Q1", "Q2", "Q3")}
    for fold in native["folds"]:
        event = next(row for row in events if int(row["round"]) == fold["round"])
        assembled, observations = assemble_holdout_fold(
            loader, year, event, events, results, qualifying, form_races=form_races,
            require_result_coverage=not qualifying_only, calibrate_qualifying=True,
        )
        calibrated = _qualifying_predictions(
            assembled.drivers, assembled.cars, assembled.stats, assembled.track,
            weather, LapSimulator(np.random.default_rng(0)), {}, {},
        )
        model_by_id = {row["driver_id"]: row for row in calibrated}
        for session, data in sessions.items():
            labels = {
                loader._resolve_row_driver(row, assembled.aliases): _q1_time({
                    "Q1": row.get(session),
                }) for row in observations.target_qualifying_rows
            }
            cohort = [row["driver_id"] for row in fold["predictions"]
                      if labels.get(row["driver_id"]) is not None
                      and row["previous_q1_seconds"] is not None]
            variants = {
                "model": [model_by_id[row["driver_id"]] for row in fold["predictions"]],
                "native": fold["predictions"],
                "previous_q1": [{**row, "predicted_seconds": row["previous_q1_seconds"]}
                                for row in fold["predictions"]],
            }
            metrics, predictions = {}, {}
            for name, rows in variants.items():
                scored = [{**row, "observed_q1_seconds": labels[row["driver_id"]]}
                          for row in rows if row["driver_id"] in cohort]
                predictions[name] = [{
                    "driver_id": row["driver_id"], "team_id": row["team_id"],
                    "predicted_seconds": row["predicted_seconds"],
                    "observed_seconds": row["observed_q1_seconds"],
                } for row in scored]
                metrics[name] = {
                    "drivers": _metrics([row["predicted_seconds"] for row in scored],
                                        [row["observed_q1_seconds"] for row in scored]),
                    "teams": _team_median_metrics(scored),
                    "teammates": _teammate_gap_metrics(scored),
                }
            data["folds"].append({
                "round": fold["round"], "race": fold["race"],
                "observed_session_entrants": sum(value is not None for value in labels.values()),
                "shared_driver_ids": cohort, "metrics": metrics, "predictions": predictions,
                "forecast": assembled.qualifying_calibration,
                "model_inputs": {
                    "drivers": [driver.model_dump(mode="json") for driver in assembled.drivers],
                    "cars": {key: car.model_dump(mode="json")
                             for key, car in assembled.cars.items()},
                    "track": assembled.track.model_dump(mode="json"),
                },
            })
    for data in sessions.values():
        data["aggregate"] = {
            name: {
                "drivers": _aggregate([fold["metrics"][name]["drivers"] for fold in data["folds"]]),
                "teams": _aggregate_team_metrics([
                    fold["metrics"][name]["teams"] for fold in data["folds"]]),
                "teammates": _aggregate_teammate_gaps([
                    fold["metrics"][name]["teammates"] for fold in data["folds"]]),
            } for name in ("model", "native", "previous_q1")
        }
    return {
        "evaluation": "round_holdout_qualifying_calibration", "year": year,
        "policy": QUALIFYING_PACE_POLICY, "prediction_scope": "qualifying_only",
        "cohort": "observed_session_entrants_with_previous_q1",
        "data_revision": native["data_revision"],
        "target_basis": native["target_basis"], "weather_assumption": weather.model_dump(),
        "runtime": simulation_runtime(), "provenance": loader.get_provenance(),
        "sessions": sessions,
    }
