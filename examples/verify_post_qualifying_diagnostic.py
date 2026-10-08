"""Recompute the sealed post-qualifying winner diagnostic without new simulations."""

import hashlib
import json
import math
from pathlib import Path

from f1sim.analysis.race_probability_scores import (
    score_winner_counts,
    score_winner_probabilities,
    summarize_winner_counts,
)
from f1sim.analysis.teammate_forecast import score_teammate_forecast
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.execution import validate_pit_lane_starters, validate_starting_grid


def sealed(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    body = {k: v for k, v in record.items() if k != "content_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()
    if digest != record.get("content_sha256"):
        raise ValueError("Diagnostic evidence seal does not match")
    return record


def same(actual, saved):
    if not math.isclose(actual, saved, abs_tol=1e-12, rel_tol=0):
        raise ValueError("Saved diagnostic loss differs from reconstructed probabilities")


def verify(path, coverage_path):
    evidence, coverage = sealed(path), sealed(coverage_path)
    if (evidence["grid_coverage_sha256"] != coverage["content_sha256"]
            or evidence["prospectively_recorded"] is not False
            or evidence["independent_untouched_test"] is not False
            or evidence["race_pace_model_changed"] is not False
            or evidence["reporting_policy_changed"] is not False):
        raise ValueError("Post-qualifying diagnostic scope differs from its source evidence")
    contexts = {e["round"]: e for e in coverage["events"]}
    events = evidence["events"]
    if [e["round"] for e in events] != list(range(1, 17)):
        raise ValueError("Diagnostic must cover all 16 current-season races")
    native, point, references = [], [], {f"grid_{scale}": [] for scale in (6, 12, 18)}
    for event in events:
        drivers = [Driver.model_validate(d) for d in event["drivers"]]
        cars = {key: Car.model_validate(c) for key, c in event["cars"].items()}
        track = Track.model_validate(event["track"])
        weather = Weather.model_validate(event["weather"])
        ids = [d.id for d in drivers]
        context = contexts[event["round"]]
        grid = validate_starting_grid(event["starting_grid"], ids)
        pits = validate_pit_lane_starters(event["pit_lane_starters"], grid)
        if (len(ids) != 22 or set(event["native_wins"]) != set(ids)
                or grid != context["starting_grid"] or pits != context["pit_lane_starters"]
                or any(d.team_id not in cars for d in drivers)
                or event["trials"] != 100 or track.total_laps < 1
                or weather.rain_intensity != 0 or weather.track_wetness != 0
                or weather.change_probability != 0):
            raise ValueError("Diagnostic model inputs must match the complete fixed-dry grid")
        empty = event["trials"] - sum(event["native_wins"].values())
        raw = score_winner_counts(event["native_wins"], empty, event["observed_winner"])
        estimated = score_teammate_forecast(
            summarize_winner_counts(event["native_wins"], empty),
            event["allocation"], event["observed_winner"],
        )
        same(raw["brier_score"], event["native_score"]["brier_score"])
        same(estimated["brier_score"], event["point_score"]["brier_score"])
        native.append(raw["brier_score"])
        point.append(estimated["brier_score"])
        for scale in (6, 12, 18):
            weights = [math.exp(-scale * i / (len(ids) - 1)) for i in range(len(ids))]
            p = dict(zip(grid, (w / math.fsum(weights) for w in weights), strict=True))
            value = score_winner_probabilities(p, 0., event["observed_winner"])["brier_score"]
            same(value, event["references"][f"grid_{scale}"]["brier_score"])
            references[f"grid_{scale}"].append(value)
    def mean(values):
        return math.fsum(values) / len(values)

    report = {"events": len(events), "native_brier": mean(native), "point_brier": mean(point),
              "references": {name: mean(values) for name, values in references.items()}}
    for key in ("native_brier", "point_brier"):
        same(report[key], evidence["summary"][key])
    for key, value in report["references"].items():
        same(value, evidence["summary"]["references"][key])
    winners = {e["round"]: e["observed_winner"] for e in events}
    report["rejected_experiments"] = {}
    for experiment in evidence["rejected_experiments"]:
        if (experiment["production_changed"] is not False
                or experiment["target_race_observations_used"] is not False
                or experiment["qualifying_and_grid_held_fixed"] is not True
                or experiment["fit_year"] != 2023
                or [r["round"] for r in experiment["records"]] != list(range(1, 17))):
            raise ValueError("Controlled candidate scope changed")
        scores = []
        for row in experiment["records"]:
            ids = set(contexts[row["round"]]["starting_grid"])
            if set(row["wins"]) != ids:
                raise ValueError("Candidate trial counts must retain every entrant")
            value = score_winner_counts(row["wins"], 100 - sum(row["wins"].values()),
                                        winners[row["round"]])["brier_score"]
            same(value, row["score"]["brier_score"])
            scores.append(value)
        value = mean(scores)
        same(value, experiment["summary"]["candidate_brier"])
        report["rejected_experiments"][experiment["name"]] = value
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "evidence"
    print(json.dumps(verify(root / "post-qualifying-winner-diagnostic-2026.json",
                            root / "published-grid-coverage-2026.json"), indent=2))
