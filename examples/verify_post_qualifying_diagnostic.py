"""Recompute the sealed post-qualifying winner diagnostic without new simulations."""

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path

from f1sim.analysis.race_probability_scores import (
    score_winner_counts,
    score_winner_probabilities,
    summarize_winner_counts,
)
from f1sim.analysis.teammate_forecast import build_teammate_allocation, score_teammate_forecast
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


def expanded_events(evidence):
    """Expand stored literal defaults and shared history; never use live seeds."""
    events = deepcopy(evidence["events"])
    version = evidence.get("format_version", 1)
    if version == 1:
        return events
    if version != 2:
        raise ValueError("Unsupported diagnostic evidence format")
    defaults, history = evidence["model_defaults"], evidence["point_history"]
    if evidence["allocation_definition"] != {
        "policy": "teammate_race_points_v1", "prior_points_per_driver": 25.0,
        "minimum_entries_per_driver": 2,
    }:
        raise ValueError("Compact evidence requires the original fixed allocation definition")
    for event in events:
        event["drivers"] = [{**defaults["drivers"], **d} for d in event["drivers"]]
        event["cars"] = {key: {**defaults["cars"], **car} for key, car in event["cars"].items()}
        event["weather"] = {**defaults["weather"], **event["weather"]}
        indexes = event["point_history_indices"]
        if any(type(i) is not int or not 0 <= i < len(history) for i in indexes):
            raise ValueError("Compact point history has an invalid reference")
        allocation = build_teammate_allocation(
            {d["id"]: d["team_id"] for d in event["drivers"]},
            [history[i] for i in indexes], cutoff_round=event["round"] - 1,
        )
        digest = hashlib.sha256(json.dumps(allocation, sort_keys=True,
                                          separators=(",", ":")).encode()).hexdigest()
        if digest != event["allocation_sha256"]:
            raise ValueError("Expanded allocation differs from the original frozen allocation")
        event["allocation"] = allocation
    return events


def verify(path, coverage_path):
    evidence, coverage = sealed(path), sealed(coverage_path)
    if (evidence["grid_coverage_sha256"] != coverage["content_sha256"]
            or evidence["prospectively_recorded"] is not False
            or evidence["independent_untouched_test"] is not False
            or evidence["race_pace_model_changed"] is not False
            or evidence["reporting_policy_changed"] is not False):
        raise ValueError("Post-qualifying diagnostic scope differs from its source evidence")
    contexts = {e["round"]: e for e in coverage["events"]}
    events = expanded_events(evidence)
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


def verify_pit_tie(path, baseline_path):
    """Check the small numerical correction against its unchanged input evidence."""
    receipt, baseline = sealed(path), sealed(baseline_path)
    if (receipt["baseline_content_sha256"] != baseline["content_sha256"]
            or receipt["prospectively_recorded"] is not False
            or receipt["independent_untouched_test"] is not False
            or receipt["qualifying_grid_weather_seeds_held_fixed"] is not True
            or receipt["trials_per_event"] != 100):
        raise ValueError("Pit cost correction evidence scope changed")
    contexts = {e["round"]: e for e in baseline["events"]}
    records = receipt["records"]
    if [r["round"] for r in records] != list(contexts):
        raise ValueError("Pit cost correction must retain every baseline event")
    losses = []
    for row in records:
        event = contexts[row["round"]]
        if (row["source_input_sha256"] != event["source_input_sha256"]
                or set(row["wins"]) != set(event["native_wins"])):
            raise ValueError("Pit cost correction inputs differ from the baseline")
        loss = score_winner_counts(row["wins"], 100 - sum(row["wins"].values()),
                                   event["observed_winner"])["brier_score"]
        same(loss, row["brier_score"])
        losses.append(loss)
    corrected = math.fsum(losses) / len(losses)
    original = math.fsum(e["native_score"]["brier_score"] for e in contexts.values()) / len(losses)
    same(corrected, receipt["summary"]["corrected_brier"])
    same(original, receipt["summary"]["baseline_brier"])
    same(original - corrected, receipt["summary"]["mean_gain"])
    return {"events": len(losses), "baseline_brier": original, "corrected_brier": corrected}


def verify_reporting_decision(path, baseline_path, corrected_path):
    """Compare both reporting policies on the exact same corrected trial counts."""
    decision, baseline, corrected = map(sealed, (path, baseline_path, corrected_path))
    if (decision["baseline_content_sha256"] != baseline["content_sha256"]
            or decision["corrected_trials_content_sha256"] != corrected["content_sha256"]
            or decision["prospectively_recorded"] is not False
            or decision["independent_untouched_test"] is not False):
        raise ValueError("Reporting decision scope differs from its paired trial evidence")
    contexts = {e["round"]: e for e in expanded_events(baseline)}
    counts = {r["round"]: r["wins"] for r in corrected["records"]}
    if [r["round"] for r in decision["records"]] != list(contexts):
        raise ValueError("Reporting comparison must retain every baseline event")
    point, native = [], []
    for row in decision["records"]:
        event, wins = contexts[row["round"]], counts[row["round"]]
        empty = 100 - sum(wins.values())
        point.append(score_teammate_forecast(summarize_winner_counts(wins, empty),
                      event["allocation"], event["observed_winner"])["brier_score"])
        native.append(score_winner_counts(wins, empty, event["observed_winner"])["brier_score"])
        same(point[-1], row["point_brier"])
        same(native[-1], row["native_brier"])
    point_mean, native_mean = math.fsum(point) / len(point), math.fsum(native) / len(native)
    same(point_mean, decision["summary"]["existing_point_reporting_brier"])
    same(native_mean, decision["summary"]["native_reporting_brier"])
    same(1 - native_mean / point_mean, decision["summary"]["relative_reduction"])
    rejected = {}
    for experiment in decision["rejected_experiments"]:
        if (experiment["production_changed"] is not False
                or experiment["qualifying_and_grid_held_fixed"] is not True
                or [r["round"] for r in experiment["records"]] != list(contexts)):
            raise ValueError("Rejected experiment scope changed")
        losses = []
        for row in experiment["records"]:
            event = contexts[row["round"]]
            if (row["source_input_sha256"] != event["source_input_sha256"]
                    or set(row["wins"]) != set(event["native_wins"])):
                raise ValueError("Rejected experiment inputs differ from baseline")
            losses.append(score_winner_counts(row["wins"], 100 - sum(row["wins"].values()),
                          event["observed_winner"])["brier_score"])
            same(losses[-1], row["brier_score"])
        rejected[experiment["name"]] = math.fsum(losses) / len(losses)
        same(rejected[experiment["name"]], experiment["mean_brier"])
    return {"existing_point_reporting_brier": point_mean, "native_reporting_brier": native_mean,
            "rejected_experiments": rejected}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "evidence"
    print(json.dumps(verify(root / "post-qualifying-winner-diagnostic-2026.json",
                            root / "published-grid-coverage-2026.json"), indent=2))
    print(json.dumps(verify_pit_tie(
        root / "post-qualifying-pit-tie-2026.json",
        root / "post-qualifying-winner-diagnostic-2026.json",
    ), indent=2))
    print(json.dumps(verify_reporting_decision(
        root / "post-qualifying-reporting-decision-2026.json",
        root / "post-qualifying-winner-diagnostic-2026.json",
        root / "post-qualifying-pit-tie-2026.json",
    ), indent=2))
