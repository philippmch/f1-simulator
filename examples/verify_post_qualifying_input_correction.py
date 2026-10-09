"""Verify frozen paired input-correction counts and grid references offline."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from verify_post_qualifying_diagnostic import expanded_events, same, sealed

from f1sim.analysis.race_probability_scores import score_winner_counts
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.execution import validate_pit_lane_starters, validate_starting_grid

ROOT = Path(__file__).resolve().parents[1]


def expanded_correction_events(record, baseline, pit):
    previous = {e["round"]: e for e in expanded_events(baseline)}
    counts = {r["round"]: r["wins"] for r in pit["records"]}
    defaults = record["literal_model_defaults_2025"]
    rows = []
    for saved in record["events"]:
        row = deepcopy(saved)
        if row["year"] == 2026:
            original = previous[row["round"]]
            if row["source_input_sha256"] != original["source_input_sha256"]:
                raise ValueError("Current-season input identity changed")
            for key in ("drivers", "cars", "track", "seed", "starting_grid",
                        "pit_lane_starters", "observed_winner"):
                row[key] = deepcopy(original[key])
            row["baseline_wins"] = counts[row["round"]]
        elif row["year"] == 2025:
            row["drivers"] = [{**defaults["drivers"], **d} for d in row["drivers"]]
            row["cars"] = {k: {**defaults["cars"], **c} for k, c in row["cars"].items()}
            row["track"] = {**defaults["tracks"], **row["track"]}
        else:
            raise ValueError("Unexpected development season")
        row["updated_drivers"] = [
            {**d, **row["driver_changes"][d["id"]]} for d in row["drivers"]
        ]
        row["updated_cars"] = {k: {**c, **row["car_changes"][k]}
                               for k, c in row["cars"].items()}
        rows.append(row)
    return rows


def paired(gains):
    values = np.asarray(gains)
    rng = np.random.default_rng(20261009)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return {"gain": float(values.mean()),
            "gain_95": np.quantile(draws, [.025, .975]).tolist(),
            "gain_removing_three_largest": float(np.sort(values)[:-3].mean())}


def verify(path=ROOT / "evidence/post-qualifying-input-correction.json"):
    record = sealed(path)
    baseline = sealed(ROOT / "evidence/post-qualifying-winner-diagnostic-2026.json")
    pit = sealed(ROOT / "evidence/post-qualifying-pit-tie-2026.json")
    if (record["kind"] != "post_qualifying_input_signal_correction_development"
            or record["format_version"] != 1
            or record["reference_scales"] != [6, 12, 18]
            or record["baseline_diagnostic_sha256"] != baseline["content_sha256"]
            or record["baseline_pit_correction_sha256"] != pit["content_sha256"]
            or record["prospectively_recorded"] is not False
            or record["independent_untouched_test"] is not False
            or record["legacy_race_rules_reconstructed"] is not False
            or record["historical_runtime_seeds_used"] is not False
            or record["trials_per_event"] != 100):
        raise ValueError("Input-correction evidence scope differs from its declared protocol")
    events = expanded_correction_events(record, baseline, pit)
    identities = {(e["year"], e["round"]) for e in events}
    if identities != ({(2025, n) for n in range(1, 25)} | {(2026, n) for n in range(1, 17)}):
        raise ValueError("The correction must cover all 40 declared events")
    if len(events) != len(identities):
        raise ValueError("Duplicated correction event")
    Weather.model_validate(record["weather"])
    if any(record["weather"][key] != 0.0
           for key in ("rain_intensity", "track_wetness", "change_probability")):
        raise ValueError("Correction evidence requires its fixed dry assumption")
    old, new, references = [], [], []
    for e in sorted(events, key=lambda e: (e["year"], e["round"])):
        drivers = [Driver.model_validate(d) for d in e["drivers"]]
        updated = [Driver.model_validate(d) for d in e["updated_drivers"]]
        field = {d.id for d in drivers}
        if {d.id for d in updated} != field:
            raise ValueError("Updated model changes the driver field")
        for key in ("cars", "updated_cars"):
            cars = {k: Car.model_validate(c) for k, c in e[key].items()}
            if any(d.team_id not in cars for d in drivers):
                raise ValueError("Correction driver lacks its car")
        Track.model_validate(e["track"])
        validate_starting_grid(e["starting_grid"], field)
        validate_pit_lane_starters(e["pit_lane_starters"], e["starting_grid"])
        for key, losses in (("baseline_wins", old), ("updated_wins", new)):
            wins = e[key]
            if set(wins) != field or sum(wins.values()) > 100:
                raise ValueError("Correction counts do not cover the complete field")
            losses.append(score_winner_counts(wins, 100 - sum(wins.values()),
                                              e["observed_winner"])["brier_score"])
        rank = e["starting_grid"].index(e["observed_winner"])
        weights = np.exp(-18 * np.arange(len(field)) / (len(field) - 1))
        weights /= weights.sum()
        references.append(float(weights @ weights + 1 - 2 * weights[rank]))
    report = {"events": 40, "baseline_brier": float(np.mean(old)),
              "candidate_brier": float(np.mean(new)), "grid18_brier": float(np.mean(references)),
              "against_native": paired(np.array(old) - new),
              "against_grid18": paired(np.array(references) - new)}
    saved = record["paired_assessment"]
    for key in ("baseline_brier", "candidate_brier", "grid18_brier"):
        same(report[key], saved[key])
    for key in ("against_native", "against_grid18"):
        for metric in ("gain", "gain_removing_three_largest"):
            same(report[key][metric], saved[key][metric])
        for value, expected in zip(report[key]["gain_95"], saved[key]["gain_95"], strict=True):
            same(value, expected)
    return report


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
