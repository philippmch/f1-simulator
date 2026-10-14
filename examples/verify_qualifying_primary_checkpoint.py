"""Recompute the rejected qualifying-primary screen offline from compact counts.

This checks saved data, scores and interpretation. It does not authenticate
omitted full race arrays, input archives or external sources, and runs no races.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from f1sim.analysis.grid_winner_reference import GRID_REFERENCE_SCALES
from f1sim.analysis.race_probability_scores import score_winner_counts, score_winner_probabilities
from f1sim.simulation.execution import validate_pit_lane_starters, validate_starting_grid

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/qualifying-primary-checkpoint-2026-10-14.json"
ASSESSMENT_SHA = "5580c08548142145f5ee53fbfd87a9474c3572187eeede55bed0aaf206271bfa"
POLICY_SHA = "d4d27544ed6180c8968f7207481322f2198221ad72693d6a4d9d38b0aec7d5c0"
INPUTS_SHA = "f1929e77460e2c6faa548c8a6d3c892c9a419d0b5d1e9a16031a0eb640db89dc"
RUNTIME_SHA = "6ae1dd5d88648ddb8e99d5944c51e72e1a8e9356cdcb7e2d2d6b61bf2434930c"


def digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_seal(value, *, expected=None):
    actual = digest({k: v for k, v in value.items() if k != "content_sha256"})
    require(value.get("content_sha256") == actual, "Evidence content seal is invalid")
    if expected is not None:
        require(actual == expected, "Original fixed evidence identity changed")


def close(saved, rebuilt):
    if isinstance(rebuilt, dict):
        require(isinstance(saved, dict) and saved.keys() == rebuilt.keys(), "Score keys differ")
        for key, value in rebuilt.items():
            close(saved[key], value)
    elif isinstance(rebuilt, list):
        require(isinstance(saved, list) and len(saved) == len(rebuilt), "Score lengths differ")
        for a, b in zip(saved, rebuilt, strict=True):
            close(a, b)
    elif isinstance(rebuilt, float):
        require(
            type(saved) in (int, float)
            and math.isfinite(saved)
            and math.isclose(saved, rebuilt, rel_tol=1e-11, abs_tol=1e-12),
            "Saved score disagrees with reconstructed probabilities",
        )
    else:
        require(type(saved) is type(rebuilt) and saved == rebuilt, "Saved value changed")


def winner_score(wins, no_winner, winner):
    score = score_winner_counts(wins, no_winner, winner)
    probability = wins[winner] / 32
    return {
        "brier": score["brier_score"],
        "log_loss": -math.log(probability) if probability > 0 else None,
        "no_classified_winner_probability": no_winner / 32,
    }


def references(grid, winner):
    result = {}
    for scale in GRID_REFERENCE_SCALES:
        weights = [math.exp(-scale * n / (len(grid) - 1)) for n in range(len(grid))]
        probabilities = dict(zip(grid, (w / math.fsum(weights) for w in weights), strict=True))
        result[str(scale)] = {
            "brier": score_winner_probabilities(probabilities, 0.0, winner)["brier_score"],
            "log_loss": -math.log(probabilities[winner]),
        }
    return result


def paired(gains):
    values = np.asarray(gains, dtype=float)
    rng = np.random.default_rng(20261009)
    resamples = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "gain": float(values.mean()),
        "paired_bootstrap_95": np.quantile(resamples, [0.025, 0.975]).tolist(),
        "gain_after_removing_three_largest": float(np.sort(values)[:-3].mean()),
    }


def summaries(scores):
    best = min(
        ("6", "12", "18"),
        key=lambda scale: np.mean([r["grid_references"][scale]["brier"] for r in scores]),
    )
    results = {}
    for label, cohort in [
        ("all", scores),
        ("2025", [r for r in scores if r["year"] == 2025]),
        ("2026", [r for r in scores if r["year"] == 2026]),
    ]:
        summary = {
            "events": len(cohort),
            "winner_metrics": {},
            "grid_references": {},
            "strongest_reference_selected_on_full_cohort": int(best),
        }
        for name in ("native", "candidate"):
            zero = sum(r[name]["log_loss"] is None for r in cohort)
            summary["winner_metrics"][name] = {
                "brier": float(np.mean([r[name]["brier"] for r in cohort])),
                "log_loss": None if zero else float(np.mean([r[name]["log_loss"] for r in cohort])),
                "zero_probability_observed_winners": zero,
                "log_loss_infinite": bool(zero),
            }
        for scale in ("6", "12", "18"):
            summary["grid_references"][scale] = {
                "brier": float(np.mean([r["grid_references"][scale]["brier"] for r in cohort])),
                "log_loss": float(
                    np.mean([r["grid_references"][scale]["log_loss"] for r in cohort])
                ),
                "candidate_gain": paired(
                    [r["grid_references"][scale]["brier"] - r["candidate"]["brier"] for r in cohort]
                ),
            }
        summary["candidate_gain_against_native"] = paired(
            [r["native"]["brier"] - r["candidate"]["brier"] for r in cohort]
        )
        results[label] = summary
    return results


def assess(evidence):
    verify_seal(evidence)
    require(
        evidence.get("format_version") == 1
        and evidence.get("kind") == "rejected_qualifying_primary_development_checkpoint",
        "Unknown checkpoint format",
    )
    require(
        evidence["evaluation_stage"] == "retrospective_post_qualifying_development_screen",
        "This checkpoint cannot claim a prospective forecast stage",
    )
    for name in (
        "production_promoted",
        "goal_achieved",
        "prospective_forecasts_recorded",
        "qualifying_weather_certified",
        "authentic_historical_inputs",
    ):
        require(evidence[name] is False, "Checkpoint cannot claim promotion or historical accuracy")
    require(evidence["trials_per_model_event"] == 32, "Fixed screen trial budget changed")
    verify_seal(evidence["policy_lock"], expected=POLICY_SHA)
    require(
        evidence["policy_lock"]["weights"]
        == {"observed_current_qualifying": 0.7, "strict_recent3GP_team_fastest_lap_form": 0.3},
        "Fixed qualifying/form weights changed",
    )
    require(
        evidence["candidate_inputs_sha256"] == INPUTS_SHA
        and evidence["runtime_sha256"] == RUNTIME_SHA,
        "Frozen input/runtime identity changed",
    )
    original = evidence["original_assessment"]
    verify_seal(original, expected=ASSESSMENT_SHA)
    require(
        original["decision"] == "reject_accuracy_candidate_do_not_deploy",
        "Original rejection must remain unchanged",
    )
    events = evidence["events"]
    require(
        [(r["year"], r["round"]) for r in events]
        == [(2025, n) for n in range(1, 25)] + [(2026, n) for n in range(1, 17)],
        "Checkpoint must retain all40 original events in order",
    )
    scores = []
    for row in events:
        grid = validate_starting_grid(row["starting_grid"])
        require(len(grid) >= 2 and row["observed_winner"] in grid, "Observed field is incomplete")
        validate_pit_lane_starters(row["pit_lane_starters"], grid)
        require(type(row["seed"]) is int, "Event seed must be an integer")
        scores_row = {"year": row["year"], "round": row["round"]}
        for name in ("native", "candidate"):
            model = row[name]
            require(set(model["wins"]) == set(grid), "Winner counts must cover the full grid")
            count = sum(model["wins"].values()) + model["no_classified_winner_count"]
            require(count == 32, "Winner counts must contain exactly32 trials")
            scores_row[name] = winner_score(
                model["wins"], model["no_classified_winner_count"], row["observed_winner"]
            )
        trials = row["candidate_trial_winners"]
        require(len(trials) == 32, "Candidate winner sequence must contain32 entries")
        counts = dict.fromkeys(grid, 0)
        no_winner = 0
        for index, trial in enumerate(trials):
            require(trial["seed"] == row["seed"] + index, "Candidate trial seeds changed")
            if trial["winner"] is None:
                no_winner += 1
            else:
                require(trial["winner"] in counts, "Trial winner is absent from the grid")
                counts[trial["winner"]] += 1
        require(
            counts == row["candidate"]["wins"]
            and no_winner == row["candidate"]["no_classified_winner_count"],
            "Candidate counts disagree with saved trial winner sequence",
        )
        scores_row["grid_references"] = references(grid, row["observed_winner"])
        close(row["winner_scores"], {k: scores_row[k] for k in ("native", "candidate")})
        close(row["grid_reference_scores"], scores_row["grid_references"])
        saved = next(
            r
            for r in original["event_scores"]
            if (r["year"], r["round"]) == (row["year"], row["round"])
        )
        close(
            {k: saved[k] for k in ("native", "candidate", "grid_references")},
            {k: scores_row[k] for k in ("native", "candidate", "grid_references")},
        )
        scores.append(scores_row)
    rebuilt = summaries(scores)
    for label, values in rebuilt.items():
        close(evidence["winner_summaries"][label], values)
        saved = original["summaries"][label]
        close(
            {
                **values,
                "winner_metrics": {k: saved["winner_metrics"][k] for k in ("native", "candidate")},
            },
            values,
        )
        for name in (
            "events",
            "grid_references",
            "strongest_reference_selected_on_full_cohort",
            "candidate_gain_against_native",
        ):
            close(saved[name], values[name])
    overall = rebuilt["all"]
    reference = overall["grid_references"][
        str(overall["strongest_reference_selected_on_full_cohort"])
    ]
    candidate = overall["winner_metrics"]["candidate"]
    native_gain = overall["candidate_gain_against_native"]
    ref_gain = reference["candidate_gain"]
    gates = {
        "complete40_events_two_seasons": True,
        "winner_brier_10_percent_below_strongest_reference": candidate["brier"]
        <= 0.9 * reference["brier"],
        "winner_log_loss_5_percent_below_reference": candidate["log_loss"] is not None
        and candidate["log_loss"] <= 0.95 * reference["log_loss"],
        "positive_paired_gain_over_native": native_gain["paired_bootstrap_95"][0] > 0,
        "positive_paired_gain_over_reference": ref_gain["paired_bootstrap_95"][0] > 0,
        "native_gain_positive_without_three_largest": native_gain[
            "gain_after_removing_three_largest"
        ]
        > 0,
        "reference_gain_positive_without_three_largest": ref_gain[
            "gain_after_removing_three_largest"
        ]
        > 0,
        "matched_native_auxiliary_comparison_completed": False,
        "400_trial_production_validation_completed": False,
        "prospective_before_race_forecasts_completed": False,
        "qualifying_weather_certified_for_live_policy": False,
    }
    close(original["gates"], gates)
    close(
        evidence["bootstrap"],
        {
            "resamples": 10000,
            "seed": 20261009,
            "unit": "event",
            "interval": [0.025, 0.975],
            "remove_largest_gains": 3,
        },
    )
    faults = evidence["historical_distance_faults"]
    require(
        [
            (
                r["year"],
                r["round"],
                r["circuit"],
                r["literal_simulation_laps"],
                r["scheduled_laps"],
                r["pdf_page_one_based"],
            )
            for r in faults
        ]
        == [(2025, 5, "jeddah", 57, 50, 67), (2025, 7, "imola", 57, 63, 69)],
        "Known historical distance faults must remain explicit",
    )
    for fault in faults:
        require(
            fault["source_url"]
            == "https://www.fia.com/sites/default/files/pdf_saudi_media_kit-2025-english.pdf"
            and fault["source_sha256"]
            == "936ff789ae9d332baeb42922947d85f0d29ce4bb0cfd95cb693187847f7a768a",
            "Scheduled-distance source identity changed",
        )
        event = next(
            r for r in events if (r["year"], r["round"]) == (fault["year"], fault["round"])
        )
        require(
            event["track"]["total_laps"] == fault["literal_simulation_laps"],
            "Checkpoint must preserve original incorrect simulation distance",
        )
    require(evidence["decision"] == original["decision"], "Checkpoint cannot change rejection")
    return {
        "events": 40,
        "trials_per_event": 32,
        "summaries": rebuilt,
        "gates": gates,
        "decision": evidence["decision"],
        "production_promoted": False,
        "authentic_historical_inputs": False,
        "historical_distance_faults": 2,
        "offline_scope": (
            "Scores, counts, winner sequences and seals; omitted full race arrays "
            "and external sources are not reauthenticated."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    result = assess(json.loads(args.evidence.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
