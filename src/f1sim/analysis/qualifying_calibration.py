"""Apply a fixed earlier-team-Q1 calibration to qualifying inputs only."""

from __future__ import annotations

import math

import numpy as np

from f1sim.analysis.qualifying_history import recent_team_q1_predictions
from f1sim.models import Weather
from f1sim.models._native import native_physics
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import _NATIVE_QUALIFYING_LAP, LapSimulator

# Qualifying-only promotion: protocol, Q2/Q3 and paired winner evidence are in docs.
DEFAULT_CURRENT_QUALIFYING_CALIBRATION = True
QUALIFYING_PACE_POLICY = "earlier_team_q1_v1"


def calibrate_qualifying_drivers(drivers, cars, track, history_events, target_round):
    """Return isolated calibrated drivers and the fixed history evidence.

    Candidate coefficients never use target observations. The baseline is the
    native dry qualifying prediction with zero previous adjustment, so repeated
    application cannot compound a correction. Race skill and cars are retained.
    Custom physics and out-of-range observations retain the supplied inputs.
    """
    supplied = list(drivers)
    weather = Weather(change_probability=0.0)
    if (LapSimulator.calculate_qualifying_lap is not _NATIVE_QUALIFYING_LAP
            or not native_physics(*supplied, *cars.values(), track, weather)):
        return supplied, {"policy": QUALIFYING_PACE_POLICY,
                          "candidate_fallback": "custom_physics", "training_rounds": []}
    baseline = [driver.model_copy(update={"qualifying_pace_adjustment": 0.0})
                for driver in supplied]
    simulator = LapSimulator(np.random.default_rng(0))
    predictions = [{
        "driver_id": driver.id, "team_id": driver.team_id,
        "predicted_seconds": min(simulator.calculate_qualifying_lap(
            driver, cars[driver.team_id], track, tire, weather, sample_variation=False,
        ) for tire in TIRE_COMPOUNDS.values()),
    } for driver in sorted(baseline, key=lambda item: item.id)]
    candidate, evidence = recent_team_q1_predictions(predictions, history_events, target_round)
    evidence = {**evidence, "policy": QUALIFYING_PACE_POLICY,
                "scope": "qualifying_only", "reference_weather": weather.model_dump()}
    if evidence.get("candidate_fallback"):
        return supplied, evidence
    native = {row["driver_id"]: row["predicted_seconds"] for row in predictions}
    adjustments = {
        row["driver_id"]: (row["predicted_seconds"] - native[row["driver_id"]])
        / (track.base_lap_time * .98) for row in candidate
    }
    if any(not math.isfinite(value) or abs(value) > .1 for value in adjustments.values()):
        return supplied, {**evidence, "candidate_fallback": "adjustment_out_of_bounds"}
    return [driver.model_copy(update={
        "qualifying_pace_adjustment": adjustments[driver.id],
    }) for driver in baseline], evidence
