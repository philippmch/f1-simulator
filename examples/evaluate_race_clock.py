"""Evaluate the frozen dry clock on saved pre-target inputs and v1 lap evidence.

This conditions on observed compound and stint age; it is not a race forecast.
No coefficient is fitted and no provider request is made.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

from f1sim.analysis.race_clock import (
    DRY_RACE_CLOCK_POLICY,
    dry_race_pace_adjustment_for_event,
)
from f1sim.analysis.relative_tyre_wear import _validate_reports
from f1sim.analysis.saved_validation import validate_saved_model
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator


def evaluate_event(report, fold, year, *, age_shift=0):
    _validate_reports(report)
    if report["meeting"] != fold["race"]:
        raise ValueError("Timing archive and input event do not match")
    adjustment = dry_race_pace_adjustment_for_event(year, fold["round"])
    if not adjustment:
        raise ValueError("Use a 2026 event strictly after the Canadian training race")
    inputs = fold["simulation_inputs"]
    drivers = [validate_saved_model(Driver, row) for row in inputs["drivers"]]
    cars = {key: validate_saved_model(Car, row) for key, row in inputs["cars"].items()}
    track = validate_saved_model(Track, inputs["track"])
    native = track.model_copy(update={"dry_race_pace_adjustment": 0.})
    corrected = track.model_copy(update={"dry_race_pace_adjustment": adjustment})
    simulator = LapSimulator()
    callbacks = [[simulator.prepare_deterministic_lap_time(
        driver, cars[driver.team_id], variant, track.total_laps) for driver in drivers]
        for variant in (native, corrected)]
    if not drivers or not all(callback for group in callbacks for callback in group):
        raise ValueError("Complete native model inputs are required")
    starts = defaultdict(list)
    for lap in report["evidence"]["laps"]:
        if lap.get("stint") is not None:
            starts[(lap["driver_number"], lap["stint"])].append(lap["lap_number"])
    weather = Weather(change_probability=0.)
    errors = [[], []]
    max_implementation_difference = 0.
    for lap in report["evidence"]["laps"]:
        number = lap["lap_number"]
        if not lap["eligible"] or lap["prior_wear"] is None or not 1 <= number <= track.total_laps:
            continue
        age = max(0, lap["prior_wear"] + number
                  - min(starts[(lap["driver_number"], lap["stint"])]) + age_shift)
        tire = TIRE_COMPOUNDS[TireCompound(lap["compound"].lower())]
        predicted = [median(callback(tire, weather, number, age) for callback in group)
                     for group in callbacks]
        for index in (0, 1):
            errors[index].append(abs(lap["duration_seconds"] - predicted[index]))
        max_implementation_difference = max(max_implementation_difference, abs(
            predicted[1] - predicted[0] - adjustment * track.base_lap_time))
    if not errors[0]:
        raise ValueError("No eligible laps with known prior wear")
    return {"laps": len(errors[0]), "native_mae": mean(errors[0]),
            "corrected_mae": mean(errors[1]),
            "max_difference_from_locked_offset_seconds": max_implementation_difference}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Saved winner-evaluation JSON containing frozen per-fold inputs")
    parser.add_argument("--archive", type=Path, action="append", required=True,
                        help="Version-1 normalized timing archive; repeat for every event")
    args = parser.parse_args(argv)
    baseline_bytes = args.baseline.read_bytes()
    baseline = json.loads(baseline_bytes)
    events = []
    for path in args.archive:
        archive_bytes = path.read_bytes()
        report = json.loads(archive_bytes)
        folds = [fold for fold in baseline["folds"] if fold["race"] == report["meeting"]]
        if len(folds) != 1:
            raise ValueError("Archive needs exactly one corresponding saved event")
        fold = folds[0]
        if any(row["round"] == fold["round"] for row in events):
            raise ValueError("Duplicate event would distort equal-event scoring")
        events.append({"round": fold["round"], "race": fold["race"],
                       "archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
                       "age_sensitivity": {str(shift): evaluate_event(
                           report, fold, baseline["year"], age_shift=shift) for shift in (0, 1)}})
    means = {str(shift): {metric: mean(row["age_sensitivity"][str(shift)][metric]
                                     for row in events)
                         for metric in ("native_mae", "corrected_mae")} for shift in (0, 1)}
    print(json.dumps({"policy": DRY_RACE_CLOCK_POLICY,
                      "conditional_on_observed_age_and_compound": True,
                      "coefficient_refitted": False,
                      "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
                      "events": events, "equal_event_means": means}, indent=2))


if __name__ == "__main__":
    main()
