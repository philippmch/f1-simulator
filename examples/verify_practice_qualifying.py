#!/usr/bin/env python3
"""Reproduce sealed qualifying predictions and scores offline, using real laps."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1sim.analysis.practice_qualifying import calibrate_practice_qualifying_drivers
from f1sim.analysis.saved_validation import validate_saved_model
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def verify(path):
    body = json.loads(path.read_text(encoding="utf-8"))
    seal = body.pop("content_sha256")
    if digest(body) != seal:
        raise ValueError("Qualifying evidence content digest does not match")
    model_path = Path(__file__).resolve().parents[1] / (
        "src/f1sim/analysis/parameters/practice_qualifying_2026.json"
    )
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != body["model_sha256"]:
        raise ValueError("Qualifying evidence belongs to different fitted parameters")
    records = {(row["round"], row["session"]): row for row in body["folds"]}
    scored = {session: [] for session in ("Q1", "Q2", "Q3")}
    simulator = LapSimulator(np.random.default_rng(0))

    def laps(drivers, cars, track):
        return {d.id: min(simulator.calculate_qualifying_lap(
            d, cars[d.team_id], track, tire, Weather(change_probability=0),
            sample_variation=False,
        ) for tire in TIRE_COMPOUNDS.values()) for d in drivers}

    for snapshot in body["physical_inputs"]:
        number = snapshot["round"]
        drivers = [validate_saved_model(Driver, row) for row in snapshot["baseline_drivers"]]
        cars = {key: validate_saved_model(Car, row) for key, row in snapshot["cars"].items()}
        track = validate_saved_model(Track, snapshot["track"])
        # Freeze predictions from prefix history and practice before reading labels.
        history = [event for event in body["history"] if event["round"] < number]
        fitted, evidence = calibrate_practice_qualifying_drivers(
            drivers, cars, track, history, snapshot["practice"], year=body["year"],
            target_round=number, circuit=track.id,
        )
        if evidence.get("candidate_fallback"):
            raise ValueError(f"Round {number} no longer applies its frozen qualifying model")
        baseline, predicted = laps(drivers, cars, track), laps(fitted, cars, track)
        for session in scored:
            record = records[(number, session)]
            rows = record["predictions"]
            for row in rows:
                name = row["driver"]
                if (not np.isclose(baseline[name], row["baseline_seconds"], atol=1e-10, rtol=0)
                        or not np.isclose(predicted[name], row["candidate_seconds"],
                                          atol=1e-10, rtol=0)):
                    raise ValueError(f"Round {number} {name} physical prediction changed")
            truth = np.array([row["observed_seconds"] for row in rows])
            truth_relative = 100 * (truth / np.median(truth) - 1)
            actual_rank = np.argsort(np.argsort(truth))
            score = {}
            for policy, field in (("existing", "baseline_seconds"),
                                  ("candidate", "candidate_seconds"),
                                  ("practice_only", "practice_seconds")):
                values = np.array([row[field] for row in rows])
                score[policy] = {
                    "relative_mae_percent": float(np.mean(np.abs(
                        100 * (values / np.median(values) - 1) - truth_relative,
                    ))),
                    "rank_mae": float(np.mean(np.abs(
                        np.argsort(np.argsort(values)) - actual_rank,
                    ))),
                    "absolute_mae_seconds": float(np.mean(np.abs(values - truth))),
                }
            scored[session].append(score)
    aggregate = {}
    for session, folds in scored.items():
        aggregate[session] = {policy: {metric: float(np.mean([
            event[policy][metric] for event in folds
        ])) for metric in folds[0][policy]} for policy in folds[0]}
        for policy, metrics in aggregate[session].items():
            for metric, value in metrics.items():
                if not np.isclose(value, body["aggregate"][session][policy][metric],
                                  atol=1e-12, rtol=0):
                    raise ValueError(f"{session} {policy} {metric} score changed")
    return {"verified_events": len(body["physical_inputs"]),
            "independent_untouched_test": body["independent_untouched_test"],
            "aggregate": aggregate}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path,
                        default=Path("evidence/practice-qualifying-2026.json"))
    print(json.dumps(verify(parser.parse_args().path), indent=2))
