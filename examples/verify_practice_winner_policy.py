#!/usr/bin/env python3
"""Verify sealed reporting-policy evidence offline, without rerunning races."""

import argparse
import hashlib
import json
import math
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1sim.analysis.race_probability_scores import score_winner_counts, summarize_winner_counts
from f1sim.analysis.teammate_forecast import score_teammate_forecast
from f1sim.analysis.winner_policy import simulation_winner_allocation
from f1sim.data.current import CurrentSeasonDataLoader, DriverStats
from f1sim.data.practice import _timestamp
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
        raise ValueError("Winner evidence seal does not match")
    parameter = Path(__file__).resolve().parents[1] / (
        "src/f1sim/analysis/parameters/practice_qualifying_2026.json"
    )
    if hashlib.sha256(parameter.read_bytes()).hexdigest() != body["practice_parameter_sha256"]:
        raise ValueError("Evidence belongs to a different qualifying model")
    old, new = [], []
    for event in body["events"]:
        number = event["round"]
        drivers = [Driver.model_validate(row) for row in event["drivers"]]
        cars = {k: Car.model_validate(row) for k, row in event["cars"].items()}
        track = Track.model_validate(event["track"])
        weather = Weather.model_validate(event["weather"])
        receipt = event["native_run_receipt"]
        if (len(drivers) != 22 or len({d.id for d in drivers}) != 22
                or set(receipt["wins"]) != {d.id for d in drivers}
                or receipt["trials"] != 400
                or sum(receipt["wins"].values()) + receipt["empty"] != 400
                or not receipt["original_100_trial_prefix_verified"]
                or any(r >= number for r in event["training_rounds"])
                or event["earlier_point_allocation"]["cutoff_round"] != number-1):
            raise ValueError("Invalid full-field native run receipt or history cutoff")
        forecast = event["practice_forecast"]
        lap = LapSimulator(np.random.default_rng(0))
        for driver in drivers:
            predicted = min(lap.calculate_qualifying_lap(
                driver, cars[driver.team_id], track, tire, weather,
                sample_variation=False,
            ) for tire in TIRE_COMPOUNDS.values())
            if not math.isclose(predicted, forecast["predicted_seconds"][driver.id],
                                abs_tol=1e-10, rel_tol=0):
                raise ValueError("Native qualifying inputs no longer match their evidence")
        starts = [_timestamp(event["event"]["sessions"].get(name))
                  for name in ("Qualifying", "SprintQualifying", "SprintShootout")]
        cutoff = min(start for start in starts if start is not None)
        # This is an isolated offline information-boundary reconstruction. It
        # never installs historical observations into a live forecast loader.
        with patch("f1sim.data.current._utc_now", lambda: cutoff-timedelta(minutes=1)):
            loader = CurrentSeasonDataLoader(current_year=2026)
            loader._event_for_race = lambda *a: event["event"]
            loader._qualifying_forecast = forecast
            loader._driver_stats = {d.id: DriverStats(
                driver_id=d.id, driver_name=d.name, team_id=d.team_id, team_name=d.team_id,
                driver_skill_rating=d.skill_rating,
                qualifying_pace_adjustment=d.qualifying_pace_adjustment,
                qualifying_pace_source="current_practice",
            ) for d in drivers}
            loader.get_winner_allocation = lambda *a: event["earlier_point_allocation"]
            if simulation_winner_allocation(loader, 2026, number, drivers, weather) is not None:
                raise ValueError("Production selection failed to preserve native probabilities")
        native = summarize_winner_counts(receipt["wins"], receipt["empty"])
        reported = score_teammate_forecast(native, event["earlier_point_allocation"],
                                           event["observed_winner"])
        preferred = score_winner_counts(receipt["wins"], receipt["empty"], event["observed_winner"])
        for calculated, stored in ((reported, event["reported_score"]),
                                    (preferred, event["native_score"])):
            for metric in ("brier_score", "uniform_baseline_brier_score"):
                if calculated[metric] != stored[metric]:
                    raise ValueError("Saved winner score does not reproduce")
            if (calculated["mc_adjustment"]["adjusted_brier_score"]
                    != stored["mc_adjustment"]["adjusted_brier_score"]):
                raise ValueError("Finite-ensemble winner correction does not reproduce")
        old.append(reported["brier_score"])
        new.append(preferred["brier_score"])
    gain = np.array(old)-np.array(new)
    boot = np.random.default_rng(42).choice(gain, (10000,len(gain)), replace=True).mean(axis=1)
    report = {"events": len(gain), "trials": 400, "reported_brier": float(np.mean(old)),
              "native_brier": float(np.mean(new)),
              "relative_gain": float(1-np.mean(new)/np.mean(old)),
              "paired_bootstrap_95_gain": np.quantile(boot,[.025,.975]).tolist(),
              "gain_without_three_best": float(np.sort(gain)[:-3].mean()),
              "improved_events": int((gain>0).sum())}
    if any(report[key] != body["summary"][key] for key in report):
        raise ValueError("Aggregate winner evidence no longer reproduces")
    return {**report, "independent_untouched_test": False, "broader_winner_goal_achieved": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path,
                        default=Path("evidence/practice-native-winner-2026.json"))
    print(json.dumps(verify(parser.parse_args().path), indent=2))
