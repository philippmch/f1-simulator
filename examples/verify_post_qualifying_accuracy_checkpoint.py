"""Recompute the saved 400-trial accuracy checkpoint without fetching or refitting."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from verify_post_qualifying_input_correction import expanded_correction_events, paired, sealed

from f1sim.analysis.race_probability_scores import score_winner_counts
from f1sim.data.current import CurrentSeasonDataLoader

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
CHECKPOINT = EVIDENCE / "post-qualifying-accuracy-checkpoint-2026-10-12.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def read_contexts(record, evidence_dir):
    sources = {name: sealed(evidence_dir / name) for name in record["source_evidence_seals"]}
    for name, content in sources.items():
        if content["content_sha256"] != record["source_evidence_seals"][name]:
            raise ValueError("Checkpoint source evidence seal changed")
    source = sources["post-qualifying-input-correction.json"]
    contexts = expanded_correction_events(
        source, sources["post-qualifying-winner-diagnostic-2026.json"],
        sources["post-qualifying-pit-tie-2026.json"],
    )
    return source, {(e["year"], e["round"]): e for e in contexts}


def metrics(stats, observed, trials, size):
    positions, podiums, dnfs = [], [], []
    if set(stats) != set(observed):
        raise ValueError("Checkpoint statistics must include the complete observed field")
    for driver, values in stats.items():
        if (not isinstance(values, list) or len(values) != 3
                or isinstance(values[0], bool) or not isinstance(values[0], (float, int))
                or not math.isfinite(values[0]) or not 1 <= values[0] <= size
                or any(type(v) is not int or not 0 <= v <= trials for v in values[1:])):
            raise ValueError("Checkpoint driver statistics must have valid positions and counts")
        actual = observed[driver]
        if (type(actual["position"]) is not int or actual["position"] < 1
                or type(actual["podium"]) is not bool or type(actual["dnf"]) is not bool
                or actual["dnf"] == CurrentSeasonDataLoader._classified(actual)
                or actual["podium"] != (actual["position"] <= 3 and not actual["dnf"])):
            raise ValueError("Checkpoint observed outcome flags disagree with status and position")
        positions.append(abs(values[0] - actual["position"]))
        podiums.append((values[1] / trials - actual["podium"]) ** 2)
        dnfs.append((values[2] / trials - actual["dnf"]) ** 2)
    return {"position_mae": float(np.mean(positions)),
            "podium_brier": float(np.mean(podiums)), "dnf_brier": float(np.mean(dnfs))}


def confidence_probabilities(row, grid, trials, candidate):
    counts = [row["current_wins"][driver] for driver in grid] + [0]
    posterior = [(count + .5) / (trials + .5 * len(counts)) for count in counts]
    temperature = candidate["temperature"]
    values = [p ** (1 / temperature) for p in posterior]
    values = [p / sum(values) for p in values]
    if candidate["kind"] == "fixed_native_confidence":
        weight = candidate["native_weight"]
    elif candidate["kind"] == "round_progress_native_confidence":
        progress = min(1., max(0., (row["round"] - 1) / 23))
        weight = ((1 - progress) * candidate["round_1_native_weight"]
                  + progress * candidate["round_24_native_weight"])
    else:
        raise ValueError("Unknown confidence candidate")
    reference = [math.exp(-18 * rank / (len(grid) - 1)) for rank in range(len(grid))]
    reference = [p / sum(reference) for p in reference] + [0.]
    return [weight * p + (1 - weight) * q for p, q in zip(values, reference, strict=True)]


def assess(record, *, evidence_dir=EVIDENCE):
    body = {key: value for key, value in record.items() if key != "content_sha256"}
    if record.get("content_sha256") != digest(body) or record.get("format_version") != 1:
        raise ValueError("Checkpoint seal or format is invalid")
    source, contexts = read_contexts(record, evidence_dir)
    trials = record["trials_per_model_event"]
    if type(trials) is not int or trials != 400:
        raise ValueError("Checkpoint requires matched 400-trial results")
    keys = [(r["year"], r["round"]) for r in record["events"]]
    if len(keys) != len(set(keys)) or set(keys) != set(contexts) or len(keys) != 40:
        raise ValueError("Checkpoint must contain all 40 unique contexts")
    rows = []
    for row in record["events"]:
        key = row["year"], row["round"]
        context = contexts[key]
        if row["winner"] != context["observed_winner"]:
            raise ValueError("Checkpoint winner disagrees with its source evidence")
        grid = context["starting_grid"]
        kind = row["current_input_kind"]
        if kind == "updated_inputs":
            drivers, cars = context["updated_drivers"], context["updated_cars"]
        elif kind == "unchanged_without_timed_qualifying" and key == (2025, 6):
            drivers, cars = context["drivers"], context["cars"]
        else:
            raise ValueError("Unknown production input selection")
        overrides = row["candidate_car_overrides"]
        if set(overrides) != set(cars) or any(
            set(values) != {"base_pace", "wet_performance"} for values in overrides.values()
        ):
            raise ValueError("Candidate may override only the same cars' pace and wet performance")
        changed_cars = {team: {**car, **overrides[team]} for team, car in cars.items()}
        scores, measured = {}, {}
        for model, model_cars in (("current", cars), ("candidate", changed_cars)):
            actual_input = {"drivers": drivers, "cars": model_cars, "track": context["track"],
                            "weather": source["weather"], "seed": context["seed"],
                            "starting_grid": grid,
                            "pit_lane_starters": context["pit_lane_starters"]}
            if digest(actual_input) != row[f"{model}_input_sha256"]:
                raise ValueError("Checkpoint model input digest disagrees with frozen context")
            wins = row[f"{model}_wins"]
            if set(wins) != set(grid) or sum(wins.values()) != trials:
                raise ValueError("Checkpoint win counts must match the field and trial count")
            scores[model] = score_winner_counts(wins, 0, row["winner"])
            probability = wins[row["winner"]] / trials
            scores[model]["log_loss"] = -math.log(probability) if probability else None
            measured[model] = metrics(row[f"{model}_driver_stats"], row["observed_drivers"],
                                      trials, len(grid))
        rank = grid.index(row["winner"])
        references = {}
        for scale in (6, 12, 18):
            p = np.exp(-scale * np.arange(len(grid)) / (len(grid) - 1))
            p /= p.sum()
            references[str(scale)] = {"brier": float(p @ p + 1 - 2 * p[rank]),
                                      "log_loss": float(-np.log(p[rank]))}
        confidence = {}
        for candidate in record["confidence_candidates"]:
            p = confidence_probabilities(row, grid, trials, candidate)
            confidence[candidate["kind"]] = {
                "brier": sum(value * value for value in p) + 1 - 2 * p[rank],
                "log_loss": -math.log(p[rank]),
            }
        rows.append({"year": key[0], "round": key[1], "scores": scores,
                     "metrics": measured, "references": references, "confidence": confidence,
                     "pole_choice_correct": grid[0] == row["winner"],
                     "top_choices_correct": {
                         model: max(row[f"{model}_wins"], key=row[f"{model}_wins"].get)
                         == row["winner"] for model in ("current", "candidate")
                     }})
    summaries = {}
    for name, group in (("all", rows), ("2025", [r for r in rows if r["year"] == 2025]),
                        ("2026", [r for r in rows if r["year"] == 2026])):
        references = {s: float(np.mean([r["references"][s]["brier"] for r in group]))
                      for s in ("6", "12", "18")}
        best = min(references, key=references.get)
        summary = {"events": len(group), "reference_brier": references,
                   "strongest_reference_scale": int(best), "models": {}, "confidence": {},
                   "pole_choices_correct": sum(r["pole_choice_correct"] for r in group)}
        for model in ("current", "candidate"):
            summary["models"][model] = {
                "winner_brier": float(np.mean([r["scores"][model]["brier_score"] for r in group])),
                "zero_count_observed_winners": sum(
                    r["scores"][model]["log_loss"] is None for r in group),
                "top_choices_correct": sum(r["top_choices_correct"][model] for r in group),
                **{metric: float(np.mean([r["metrics"][model][metric] for r in group]))
                   for metric in ("position_mae", "podium_brier", "dnf_brier")},
            }
        summary["candidate_against_current"] = paired([
            r["scores"]["current"]["brier_score"] - r["scores"]["candidate"]["brier_score"]
            for r in group])
        summary["candidate_against_reference"] = paired([
            r["references"][best]["brier"] - r["scores"]["candidate"]["brier_score"]
            for r in group])
        summary["other_metric_gains"] = {
            metric: paired([r["metrics"]["current"][metric] - r["metrics"]["candidate"][metric]
                            for r in group])
            for metric in ("position_mae", "podium_brier", "dnf_brier")
        }
        for candidate in record["confidence_candidates"]:
            kind = candidate["kind"]
            summary["confidence"][kind] = {
                "winner_brier": float(np.mean([r["confidence"][kind]["brier"] for r in group])),
                "log_loss": float(np.mean([r["confidence"][kind]["log_loss"] for r in group])),
            }
        summaries[name] = summary
    return {"trials_per_event": trials, "summaries": summaries,
            "scope": record["scope"], "candidate_deployed": False}


if __name__ == "__main__":
    print(json.dumps(assess(sealed(CHECKPOINT)), indent=2, allow_nan=False))
