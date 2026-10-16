"""Recompute the qualifying-led pace sensitivity offline from sealed marginals.

No simulations, fitting or network. This proves the saved development scores,
not authentic historical race rules, source availability, or prospective skill.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/historical-qualifying-led-2026-10-16.json"
POLICY_SHA = "409747677764250bb76070c1ebc4c2a5432796f3490ecf011b1878e0c194f0a6"
ORIGINAL_DECISION_SHA = "01fbd079f396b9c11c1ed97c2af1304b2401b6e20635bb8b5bc8d51074378ae5"
RESULTS_SHA = "88b90c5ae6500e3fa170b4d9ead11540b2838f84ed154ed217c89153d2a5e272"
INPUT_IDENTITIES_SHA = "d789409c0cdc95591bfd43918cb41b253338dcb5e29f7ad6503d1afc8ccaff03"


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def seal(value, expected=None):
    actual = digest({k: v for k, v in value.items() if k != "content_sha256"})
    require(actual == value.get("content_sha256"), "Invalid evidence seal")
    require(expected is None or actual == expected, "Original evidence identity changed")


def close(saved, rebuilt):
    if isinstance(rebuilt, dict):
        require(isinstance(saved, dict) and saved.keys() == rebuilt.keys(), "Score keys differ")
        for key in rebuilt:
            close(saved[key], rebuilt[key])
    elif isinstance(rebuilt, list):
        require(isinstance(saved, list) and len(saved) == len(rebuilt), "Score lengths differ")
        for a, b in zip(saved, rebuilt, strict=True):
            close(a, b)
    elif isinstance(rebuilt, float):
        require(
            type(saved) in (float, int)
            and math.isfinite(saved)
            and math.isclose(saved, rebuilt, rel_tol=1e-10, abs_tol=1e-12),
            "Saved score differs from reconstructed counts",
        )
    else:
        require(type(saved) is type(rebuilt) and saved == rebuilt, "Saved interpretation changed")


def paired(gains):
    values = np.asarray(gains, dtype=float)
    rng = np.random.default_rng(20261009)
    sample = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {
        "gain": float(values.mean()),
        "gain_95": np.quantile(sample, [0.025, 0.975]).tolist(),
        "gain_removing_three_largest": float(np.sort(values)[:-3].mean()),
    }


def event_scores(row):
    grid = row["starting_grid"]
    require(
        isinstance(grid, list)
        and len(grid) >= 2
        and len(set(grid)) == len(grid)
        and all(isinstance(d, str) and d for d in grid),
        "Incomplete or duplicate grid",
    )
    require(
        set(row["pit_lane_starters"]) <= set(grid)
        and len(set(row["pit_lane_starters"])) == len(row["pit_lane_starters"]),
        "Invalid pit starter identities",
    )
    observed = row["observed_drivers"]
    require(set(observed) == set(grid), "Observed field must equal supplied grid")
    require(
        sorted(o["position"] for o in observed.values()) == list(range(1, len(grid) + 1)),
        "Observed positions must form a full permutation",
    )
    winner = row["observed_winner"]
    require(winner in grid and observed[winner]["position"] == 1, "Observed winner changed")
    for label in observed.values():
        require(
            type(label["dnf"]) is bool and type(label["podium"]) is bool,
            "Observed finish labels must be booleans",
        )
        require(
            label["podium"] == (label["position"] <= 3 and not label["dnf"]),
            "Podium label contradicts position/nonfinish",
        )
    result = {"year": row["year"], "round": row["round"], "grid_references": {}}
    for model in ("native", "qualifying_led"):
        source = row[model]
        require(
            set(source["wins"]) == set(source["marginals"]) == set(grid),
            "All model marginals must retain the complete field",
        )
        require(
            all(type(v) is int and 0 <= v <= 400 for v in source["wins"].values())
            and type(source["no_classified_winner_count"]) is int
            and source["no_classified_winner_count"] >= 0
            and sum(source["wins"].values()) + source["no_classified_winner_count"] == 400,
            "Winner counts must sum to exactly 400 trials",
        )
        require(
            isinstance(source["input_sha256"], str)
            and len(source["input_sha256"]) == 64
            and all(c in "0123456789abcdef" for c in source["input_sha256"]),
            "Frozen input identity missing",
        )
        prob = source["wins"][winner] / 400
        scores = {
            "winner_brier": sum((v / 400 - (d == winner)) ** 2 for d, v in source["wins"].items())
            + (source["no_classified_winner_count"] / 400) ** 2,
            "winner_log_loss": -math.log(prob) if prob else None,
            "zero_probability_observed_winner": prob == 0,
        }
        position, podium, nonfinish = [], [], []
        for driver, value in source["marginals"].items():
            require(
                set(value) == {"position_sum", "podium_count", "nonfinish_count"},
                "Unknown marginal layout",
            )
            require(
                type(value["position_sum"]) is int
                and 400 <= value["position_sum"] <= 400 * len(grid)
                and all(
                    type(value[k]) is int and 0 <= value[k] <= 400
                    for k in ("podium_count", "nonfinish_count")
                ),
                "Invalid trial marginal",
            )
            require(
                value["podium_count"] >= source["wins"][driver],
                "A winner must also be a podium finisher",
            )
            position.append(abs(value["position_sum"] / 400 - observed[driver]["position"]))
            podium.append((value["podium_count"] / 400 - observed[driver]["podium"]) ** 2)
            nonfinish.append((value["nonfinish_count"] / 400 - observed[driver]["dnf"]) ** 2)
        require(
            sum(v["position_sum"] for v in source["marginals"].values())
            == 400 * len(grid) * (len(grid) + 1) // 2,
            "Position marginals must preserve full trial permutations",
        )
        require(
            sum(v["podium_count"] for v in source["marginals"].values()) <= 1200,
            "At most three podium finishers are possible in each trial",
        )
        scores.update(
            {
                "position_mae": float(np.mean(position)),
                "podium_brier": float(np.mean(podium)),
                "nonfinish_brier": float(np.mean(nonfinish)),
            }
        )
        result[model] = scores
    rank = grid.index(winner)
    for scale in (6, 12, 18):
        p = np.exp(-scale * np.arange(len(grid)) / (len(grid) - 1))
        p /= p.sum()
        result["grid_references"][str(scale)] = {
            "winner_brier": float(p @ p + 1 - 2 * p[rank]),
            "winner_log_loss": float(-np.log(p[rank])),
        }
    return result


def summarize(scores):
    summaries = {}
    best = min(
        ("6", "12", "18"),
        key=lambda r: np.mean([e["grid_references"][r]["winner_brier"] for e in scores]),
    )
    for label, rows in (
        ("all", scores),
        ("2025", [r for r in scores if r["year"] == 2025]),
        ("2026", [r for r in scores if r["year"] == 2026]),
    ):
        summary = {
            "events": len(rows),
            "models": {},
            "grid_references": {},
            "strongest_reference_selected_on_full_cohort": int(best),
        }
        for model in ("native", "qualifying_led"):
            zero = sum(r[model]["zero_probability_observed_winner"] for r in rows)
            summary["models"][model] = {
                k: float(np.mean([r[model][k] for r in rows]))
                for k in ("winner_brier", "position_mae", "podium_brier", "nonfinish_brier")
            }
            summary["models"][model].update(
                {
                    "winner_log_loss": None
                    if zero
                    else float(np.mean([r[model]["winner_log_loss"] for r in rows])),
                    "winner_log_loss_infinite": bool(zero),
                    "zero_probability_observed_winners": zero,
                }
            )
        for ref in ("6", "12", "18"):
            summary["grid_references"][ref] = {
                k: float(np.mean([r["grid_references"][ref][k] for r in rows]))
                for k in ("winner_brier", "winner_log_loss")
            }
            summary["grid_references"][ref]["qualifying_led_gain"] = paired(
                [
                    r["grid_references"][ref]["winner_brier"] - r["qualifying_led"]["winner_brier"]
                    for r in rows
                ]
            )
        summary["qualifying_led_gain_against_native"] = {
            k: paired([r["native"][k] - r["qualifying_led"][k] for r in rows])
            for k in ("winner_brier", "position_mae", "podium_brier", "nonfinish_brier")
        }
        q, native = summary["models"]["qualifying_led"], summary["models"]["native"]
        summary["winner_brier_reduction_percent_against_native"] = 100 * (
            1 - q["winner_brier"] / native["winner_brier"]
        )
        summary["winner_brier_reduction_percent_against_best_reference"] = 100 * (
            1 - q["winner_brier"] / summary["grid_references"][best]["winner_brier"]
        )
        summaries[label] = summary
    return summaries


def gates(summary):
    all_rows = summary["all"]
    q, n = all_rows["models"]["qualifying_led"], all_rows["models"]["native"]
    ref = all_rows["grid_references"][str(all_rows["strongest_reference_selected_on_full_cohort"])]
    against_native = all_rows["qualifying_led_gain_against_native"]["winner_brier"]
    return {
        "complete_40_events_two_seasons": True,
        "winner_brier_10_percent_below_best_reference": q["winner_brier"]
        <= 0.9 * ref["winner_brier"],
        "raw_log_loss_5_percent_below_best_reference": q["winner_log_loss"] is not None
        and q["winner_log_loss"] <= 0.95 * ref["winner_log_loss"],
        "positive_paired_winner_gain_over_native": against_native["gain_95"][0] > 0,
        "positive_paired_winner_gain_over_reference": ref["qualifying_led_gain"]["gain_95"][0] > 0,
        "native_gain_positive_without_three_largest": against_native["gain_removing_three_largest"]
        > 0,
        "reference_gain_positive_without_three_largest": ref["qualifying_led_gain"][
            "gain_removing_three_largest"
        ]
        > 0,
        "position_no_regression": q["position_mae"] <= n["position_mae"],
        "podium_no_regression": q["podium_brier"] <= n["podium_brier"],
        "nonfinish_no_regression": q["nonfinish_brier"] <= n["nonfinish_brier"],
    }


def assess(evidence):
    seal(evidence)
    require(
        evidence.get("kind") == "historical_qualifying_led_pace_sensitivity"
        and evidence.get("format_version") == 1
        and evidence.get("evaluation_stage") == "retrospective_post_qualifying_development",
        "Unknown historical evidence stage",
    )
    for key in (
        "independent_untouched_test",
        "prospective_skill_established",
        "robust_gate_passed",
        "legacy_rules_reconstructed",
        "authenticated_historical_availability",
    ):
        require(evidence[key] is False, "Historical evidence cannot claim comprehensive validation")
    require(
        evidence["fixed_dry"] is True and evidence["trials_per_model_event"] == 400,
        "Original simulation assumptions changed",
    )
    seal(evidence["policy_lock"], POLICY_SHA)
    original = evidence["original_rejected_decision"]
    seal(original, ORIGINAL_DECISION_SHA)
    require(
        original["decision"] == "rejected_for_deployment"
        and original["results_file_sha256"] == RESULTS_SHA,
        "Original robust-gate rejection must remain unchanged",
    )
    require(
        [(r["year"], r["round"]) for r in evidence["events"]]
        == [(2025, n) for n in range(1, 25)] + [(2026, n) for n in range(1, 17)],
        "All forty original events must remain paired and ordered",
    )
    require(
        digest(
            [
                {
                    "year": r["year"],
                    "round": r["round"],
                    "native": r["native"]["input_sha256"],
                    "qualifying_led": r["qualifying_led"]["input_sha256"],
                }
                for r in evidence["events"]
            ]
        )
        == INPUT_IDENTITIES_SHA,
        "Original paired input identities changed",
    )
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
    scores = [event_scores(r) for r in evidence["events"]]
    summary = summarize(scores)
    close(evidence["summaries"], summary)
    actual_gates = gates(summary)
    close(evidence["robust_gate_checks"], actual_gates)
    require(not all(actual_gates.values()), "Original robust gate did not pass")
    overall = summary["all"]["models"]
    close(original["current_brier"], overall["native"]["winner_brier"])
    close(original["candidate_brier"], overall["qualifying_led"]["winner_brier"])
    for name, oldname in (
        ("position_mae", "position_mae"),
        ("podium_brier", "podium_brier"),
        ("nonfinish_brier", "dnf_brier"),
    ):
        close(original["other_metrics"]["all"][oldname]["current"], overall["native"][name])
        close(
            original["other_metrics"]["all"][oldname]["candidate"], overall["qualifying_led"][name]
        )
    require(
        evidence["historical_distance_faults"]
        == [
            {
                "year": 2025,
                "round": 5,
                "circuit": "jeddah",
                "literal_simulation_laps": 57,
                "scheduled_laps": 50,
            },
            {
                "year": 2025,
                "round": 7,
                "circuit": "imola",
                "literal_simulation_laps": 57,
                "scheduled_laps": 63,
            },
        ],
        "Known historical-distance limitations cannot be removed",
    )
    return {
        "events": 40,
        "trials_per_model_event": 400,
        "summaries": summary,
        "robust_gate_checks": actual_gates,
        "robust_gate_passed": False,
        "original_decision": original["decision"],
        "prospective_skill_established": False,
        "verification_scope": (
            "Saved marginal counts and scores; omitted full joint arrays and external source "
            "availability are not authenticated"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", nargs="?", type=Path, default=EVIDENCE)
    args = parser.parse_args()
    print(
        json.dumps(
            assess(json.loads(args.evidence.read_text(encoding="utf-8"))), indent=2, allow_nan=False
        )
    )


if __name__ == "__main__":
    main()
