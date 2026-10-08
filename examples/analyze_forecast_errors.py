#!/usr/bin/env python3
"""Reproduce current-season winner errors and trace practice, qualifying and race pace."""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from f1sim.analysis.forecast_errors import compare_winner_errors
from f1sim.analysis.practice_winner_reference import practice_rank_probabilities
from f1sim.analysis.race_probability_scores import score_winner_counts
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator


def read_sealed(path):
    document = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    seal = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if seal != document.get("content_sha256"):
        raise ValueError(f"Evidence content seal does not match: {path}")
    return document


def verify_experiments(path, native):
    """Check saved controlled counts and boundaries without rerunning experiments."""
    evidence = read_sealed(path)
    if (
        evidence.get("year") != native["year"]
        or evidence.get("native_evidence_sha256") != native["content_sha256"]
        or evidence.get("qualifying_held_fixed") is not True
        or evidence.get("weather_held_fixed") is not True
        or evidence.get("race_engine_held_fixed") is not True
    ):
        raise ValueError("Controlled experiments must match the native evidence and boundaries")
    events = {event["round"]: event for event in native["events"]}
    expected = {
        "persistent_race_pace_uncertainty": (list(range(1, 8)), False),
        "observed_race_pace_sensitivity": ([7, 11], True),
        "pre_qualifying_longrun_pace": (list(range(1, 17)), False),
    }
    reports = {}
    for experiment in evidence["experiments"]:
        name = experiment["name"]
        if name not in expected or name in reports:
            raise ValueError("Unexpected or duplicate controlled experiment")
        rounds, target_used = expected[name]
        if (
            [row["round"] for row in experiment["records"]] != rounds
            or experiment.get("trials") != 100
            or experiment.get("production_changed") is not False
            or experiment.get("independent_untouched_test") is not False
            or experiment.get("target_race_observations_used") is not target_used
        ):
            raise ValueError("Controlled experiment cohort and information scope changed")
        paired = []
        for row in experiment["records"]:
            event = events[row["round"]]
            ids = {driver["id"] for driver in event["drivers"]}
            variants = row["variants"]
            if (
                row["observed_winner"] != event["observed_winner"]
                or row["seed"] != event["native_run_receipt"]["seed"]
                or row["native_input_sha256"] != event["native_run_receipt"]["input_sha256"]
                or set(variants) != {"off", "on"}
                or variants["off"]["qualifying_mean_positions"]
                != variants["on"]["qualifying_mean_positions"]
            ):
                raise ValueError("Paired experiment changed qualifying, inputs or target labels")
            scores = {}
            for mode in ("off", "on"):
                value = variants[mode]
                if (
                    set(value["wins"]) != ids
                    or set(value["qualifying_mean_positions"]) != ids
                    or sum(value["wins"].values()) + value["empty"] != 100
                    or any(
                        type(position) not in (int, float)
                        or not math.isfinite(position)
                        or not 1 <= position <= len(ids)
                        for position in value["qualifying_mean_positions"].values()
                    )
                    or not math.isclose(
                        sum(value["qualifying_mean_positions"].values()),
                        len(ids) * (len(ids) + 1) / 2,
                        abs_tol=1e-9,
                        rel_tol=0.0,
                    )
                ):
                    raise ValueError("Controlled trial counts do not cover the full field")
                scores[mode] = score_winner_counts(
                    value["wins"],
                    value["empty"],
                    row["observed_winner"],
                )["brier_score"]
            paired.append(
                {
                    "round": row["round"],
                    "baseline_brier": scores["off"],
                    "candidate_brier": scores["on"],
                    "baseline_winner_probability": variants["off"]["wins"][row["observed_winner"]]
                    / 100,
                    "candidate_winner_probability": variants["on"]["wins"][row["observed_winner"]]
                    / 100,
                }
            )
        gains = np.array([row["baseline_brier"] - row["candidate_brier"] for row in paired])
        reports[name] = {
            "events": len(paired),
            "baseline_brier": float(np.mean([row["baseline_brier"] for row in paired])),
            "candidate_brier": float(np.mean([row["candidate_brier"] for row in paired])),
            "mean_gain": float(gains.mean()),
            "gain_without_three_best": float(np.sort(gains)[:-3].mean())
            if len(gains) > 3
            else None,
            "target_race_observations_used": target_used,
            "production_changed": False,
            "independent_untouched_test": False,
            "events_detail": paired,
        }
    if set(reports) != set(expected):
        raise ValueError("Controlled experiment evidence is incomplete")
    return reports


def analyze(native_path, practice_path):
    native, practice = read_sealed(native_path), read_sealed(practice_path)
    if (
        type(native.get("year")) is not int
        or type(practice.get("year")) is not int
        or native["year"] != practice["year"]
    ):
        raise ValueError("Native and reference evidence must belong to the same season")
    observed = {event["round"]: event for event in practice["events"]}
    target_rounds = {event["round"] for event in native["events"]}
    if (
        len(observed) != len(practice["events"])
        or len(target_rounds) != len(native["events"])
        or set(observed) != target_rounds
    ):
        raise ValueError(
            "Native and reference evidence must contain identical unique event cohorts"
        )
    output = []
    for event in native["events"]:
        number, winner = event["round"], event["observed_winner"]
        current = observed[number]
        if (
            current["circuit"] != event["event"]["circuit_id"]
            or current["session_number"] != event["practice_forecast"]["practice_session_number"]
        ):
            raise ValueError("Reference must use the same target circuit and practice session")
        ids = [row["id"] for row in event["drivers"]]
        receipt = event["native_run_receipt"]
        if (
            set(receipt["wins"]) != set(ids)
            or receipt["trials"] != 400
            or any(type(count) is not int or count < 0 for count in receipt["wins"].values())
            or sum(receipt["wins"].values()) + receipt["empty"] != receipt["trials"]
        ):
            raise ValueError("Native receipt does not describe the complete 400-trial field")
        probabilities = {key: receipt["wins"][key] / receipt["trials"] for key in ids}
        reference = practice_rank_probabilities(ids, current["rows"])
        comparison = compare_winner_errors(
            probabilities,
            reference,
            winner,
            no_winner_probability=receipt["empty"] / 400,
        )
        if not math.isclose(
            comparison["model_brier"],
            event["native_score"]["brier_score"],
            abs_tol=1e-12,
            rel_tol=0.0,
        ):
            raise ValueError("Native evidence scores do not reproduce")
        drivers = [Driver.model_validate(row) for row in event["drivers"]]
        cars = {key: Car.model_validate(value) for key, value in event["cars"].items()}
        track, weather = (
            Track.model_validate(event["track"]),
            Weather.model_validate(event["weather"]),
        )
        lap = LapSimulator(np.random.default_rng(0))
        tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
        clean = {
            driver.id: lap.calculate_lap_time(
                driver,
                cars[driver.team_id],
                track,
                tire,
                weather,
                1,
                track.total_laps,
                sample_variation=False,
            )
            for driver in drivers
        }
        qualifying = event["practice_forecast"]["predicted_seconds"]
        ranks = {row["driver"]: row["position"] for row in current["rows"]}
        output.append(
            {
                "round": number,
                "race": event["event"]["race"],
                "observed_winner": winner,
                "winner_probability": probabilities[winner],
                "practice_reference_winner_probability": reference[winner],
                "winner_practice_rank": ranks.get(winner),
                "winner_modeled_qualifying_pace_rank": sorted(ids, key=qualifying.get).index(winner)
                + 1,
                "winner_modeled_clean_race_pace_rank": sorted(ids, key=clean.get).index(winner) + 1,
                "winner_modeled_clean_race_pace_gap_seconds": clean[winner] - min(clean.values()),
                "modeled_clean_race_pace_scope": (
                    "fresh medium, lap 1, no traffic, no sampled variation"
                ),
                "model_practice_probability_disagreement_squared": math.fsum(
                    (probabilities[key] - reference[key]) ** 2 for key in ids
                ),
                "comparison": comparison,
            }
        )
    model = np.array([row["comparison"]["model_brier"] for row in output])
    baseline = np.array([row["comparison"]["reference_brier"] for row in output])
    adjusted = np.array(
        [
            event["native_score"]["mc_adjustment"]["adjusted_brier_score"]
            for event in native["events"]
        ]
    )
    delta = model - baseline
    confidence = np.quantile(
        np.random.default_rng(42)
        .choice(
            delta,
            (10_000, len(delta)),
            replace=True,
        )
        .mean(axis=1),
        [0.025, 0.975],
    ).tolist()
    return {
        "kind": "practice_native_winner_error_analysis",
        "year": native["year"],
        "native_evidence_sha256": native["content_sha256"],
        "practice_evidence_sha256": practice["content_sha256"],
        "independent_untouched_test": False,
        "production_accuracy_improved": False,
        "summary": {
            "events": len(output),
            "native_brier": float(model.mean()),
            "native_brier_mc_adjusted": float(adjusted.mean()),
            "practice_reference_brier": float(baseline.mean()),
            "model_minus_reference_brier": float(delta.mean()),
            "paired_bootstrap_95_model_minus_reference": confidence,
            "model_better_events": int((delta < 0).sum()),
            "model_worse_events": int((delta > 0).sum()),
            "mean_winner_probability_loss_difference": float(
                np.mean([row["comparison"]["winner_probability_loss_difference"] for row in output])
            ),
            "mean_other_outcome_loss_difference": float(
                np.mean([row["comparison"]["other_outcome_loss_difference"] for row in output])
            ),
        },
        "limitations": [
            "Already inspected revised historical outcomes and known entrant fields",
            "All native trials assumed dry weather, including races with observed rain",
            "Score contributions and stage ranks identify hypotheses, not causal proof",
        ],
        "events": sorted(output, key=lambda row: -row["comparison"]["model_minus_reference_brier"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--native", type=Path, default=Path("evidence/practice-native-winner-2026.json")
    )
    parser.add_argument(
        "--practice", type=Path, default=Path("evidence/practice-winner-reference-2026.json")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--experiments", type=Path, default=Path("evidence/forecast-error-experiments-2026.json")
    )
    args = parser.parse_args()
    report = analyze(args.native, args.practice)
    report["controlled_experiments"] = verify_experiments(
        args.experiments, read_sealed(args.native)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "controlled_experiments": {
                    name: {key: value for key, value in result.items() if key != "events_detail"}
                    for name, result in report["controlled_experiments"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
