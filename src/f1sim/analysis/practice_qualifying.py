"""Current-season qualifying pace forecasts from earlier Q1 and current practice.

This model reports relative pace, not simulated outcome counts. Historical
training produces fixed coefficients; live inference uses the requested UTC
season only, with no archived-driver or constructor seed.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from importlib.resources import files
from statistics import median

import numpy as np

PRACTICE_QUALIFYING_POLICY = "current_season_practice_q1_v1"
WINDOWS = (3, 6, 12)
MAX_HISTORY_ENTRIES = 10_000


def _finite(value, name, *, minimum=None, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be a finite number in its allowed range")
    return float(value)


def _identity(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 160:
        raise ValueError("Qualifying evidence requires non-empty driver and team identities")
    return value


def _recent(records, key, prior, window):
    selected = [row for row in records if row.get(key) is not None][-window:]
    weights = [.72 ** (len(selected) - i - 1) for i in range(len(selected))]
    return (math.fsum(w * row[key] for w, row in zip(weights, selected)) + prior) / (
        math.fsum(weights) + 1
    )


def _validate_event(event, target_round):
    if not isinstance(event, dict):
        raise ValueError("Qualifying history events must be objects")
    number = event.get("round")
    if type(number) is not int or not 1 <= number < target_round:
        raise ValueError("Qualifying history must precede the target round")
    circuit = _identity(event.get("circuit"))
    rows = event.get("rows")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 30:
        raise ValueError("Qualifying history requires a bounded event roster")
    normalized, identities = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Qualifying history rows must be objects")
        driver, team = _identity(row.get("driver")), _identity(row.get("team"))
        if driver in identities:
            raise ValueError("Conflicting or duplicate qualifying history identities")
        identities.add(driver)
        position, time_value, points = row.get("qualifying_position"), row.get("q1"), row.get(
            "points",
        )
        if position is not None and (type(position) is not int or not 1 <= position <= 30):
            raise ValueError("Invalid earlier qualifying position")
        if time_value is not None:
            time_value = _finite(time_value, "Q1 seconds", minimum=30, maximum=240)
        if points is not None:
            points = _finite(points, "Earlier race points", minimum=0, maximum=100)
        normalized.append({"driver": driver, "team": team, "qualifying_position": position,
                           "q1": time_value, "points": points})
    return {"round": number, "circuit": circuit, "rows": normalized}


def _practice_features(roster, practice, previous_points):
    if (not isinstance(practice, dict) or type(practice.get("session_number")) is not int
            or practice["session_number"] not in (0, 1, 2, 3)):
        raise ValueError("A qualifying forecast requires identified practice evidence")
    rows = practice.get("rows")
    if not isinstance(rows, list) or len(rows) > 30:
        raise ValueError("Invalid practice roster")
    observations = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Practice rows must be objects")
        driver = _identity(row.get("driver"))
        if driver in observations:
            raise ValueError("Duplicate practice driver")
        position = row.get("position")
        laps = row.get("laps")
        if (type(position) is not int or not 1 <= position <= 30
                or type(laps) is not int or not 0 <= laps <= 200):
            raise ValueError("Invalid practice position or lap count")
        lap = _finite(row.get("lap_seconds"), "Practice lap seconds", minimum=30, maximum=240)
        observations[driver] = {"position": position, "lap_seconds": lap, "laps": laps}
    if len(observations) < max(2, math.ceil(.5 * len(roster))):
        raise ValueError("Insufficient current-practice coverage")
    if not set(observations) <= set(roster):
        raise ValueError("Practice drivers must match the modeled roster")
    center = median(row["lap_seconds"] for row in observations.values())
    relative = {driver: float(np.clip(100 * (center - row["lap_seconds"]) / center, -5, 5))
                for driver, row in observations.items()}
    strength = {driver: 1 - (row["position"] - 1) / (len(roster) - 1)
                for driver, row in observations.items()}
    spread = float(np.std(list(relative.values())))
    output = {}
    for driver, team in roster.items():
        same_team = [name for name, item in roster.items()
                     if item == team and name in observations]
        team_strength = (math.fsum(strength[name] for name in same_team) / len(same_team)
                         if same_team else .5)
        team_pace = (math.fsum(relative[name] for name in same_team) / len(same_team)
                     if same_team else 0.)
        ds, dp = strength.get(driver, .5), relative.get(driver, 0.)
        output[driver] = {
            "practice_strength": ds, "practice_relative_pace": dp,
            "practice_team_strength": team_strength, "practice_team_pace": team_pace,
            "practice_teammate_delta": ds - team_strength,
            "practice_teammate_pace_delta": dp - team_pace,
            "practice_log_laps": (math.log1p(observations[driver]["laps"]) / 4
                                  if driver in observations else 0.),
            "practice_available": float(driver in observations),
            "practice_session_one": float(practice["session_number"] == 1),
            "practice_field_spread": spread,
            "practice_pace_x_season_points": dp * previous_points[driver],
            "practice_coverage": len(observations) / len(roster),
        }
    return output


def practice_qualifying_features(roster, history, practice, *, target_round, circuit):
    """Build named causal features. Target/later history is rejected, not truncated."""
    if type(target_round) is not int or not 1 <= target_round <= 100:
        raise ValueError("target_round must be an integer from 1 to 100")
    circuit = _identity(circuit)
    if (not isinstance(practice, dict) or type(practice.get("round")) is not int
            or practice.get("round") != target_round):
        raise ValueError("Practice must belong to the target round")
    if not isinstance(roster, dict) or not 2 <= len(roster) <= 30:
        raise ValueError("A qualifying forecast requires a bounded modeled roster")
    roster = dict(sorted((_identity(driver), _identity(team)) for driver, team in roster.items()))
    if not isinstance(history, list) or len(history) > 100:
        raise ValueError("Invalid qualifying event history")
    events = [_validate_event(event, target_round) for event in history]
    if sum(len(event["rows"]) for event in events) > MAX_HISTORY_ENTRIES:
        raise ValueError("Too many qualifying history entries")
    if len({event["round"] for event in events}) != len(events):
        raise ValueError("Duplicate qualifying history round")
    teams, drivers = defaultdict(list), defaultdict(list)
    for event in sorted(events, key=lambda row: row["round"]):
        rows, n = event["rows"], len(event["rows"])
        times = [row["q1"] for row in rows if row["q1"] is not None]
        center = median(times) if times else 90.
        strengths = {row["driver"]: (1 - (row["qualifying_position"] - 1) / (n - 1)
                                      if row["qualifying_position"] is not None else None)
                     for row in rows}
        gaps = {row["driver"]: (float(np.clip(100 * (center - row["q1"]) / center, -5, 5))
                                if row["q1"] is not None else None) for row in rows}
        previous = {
            row["driver"]: _recent(teams[row["team"]], "q", .5, 3)
            + _recent(drivers[row["driver"]], "qdelta", 0., 3) for row in rows
        }
        groups = defaultdict(list)
        for row in rows:
            groups[row["team"]].append(row)
        point_values = [row["points"] for row in rows if row["points"] is not None]
        max_points = max(point_values, default=25.) or 25.
        for team, entries in groups.items():
            qs = [strengths[row["driver"]] for row in entries
                  if strengths[row["driver"]] is not None]
            gs = [gaps[row["driver"]] for row in entries if gaps[row["driver"]] is not None]
            team_q = math.fsum(qs) / len(qs) if qs else .5
            team_gap = math.fsum(gs) / len(gs) if gs else None
            teams[team].append({"q": team_q, "qgap": team_gap})
            for row in entries:
                name, dq, dg = row["driver"], strengths[row["driver"]], gaps[row["driver"]]
                drivers[name].append({
                    "team": team, "circuit": event["circuit"],
                    "qdelta": dq - team_q if dq is not None else None,
                    "qgapdelta": dg - team_gap if dg is not None and team_gap is not None else None,
                    "points": row["points"] / max_points if row["points"] is not None else None,
                    "circuit_q_residual": dq - previous[name] if dq is not None else None,
                })
    features, previous_points = {}, {}
    for driver, team in roster.items():
        th, dh = teams[team], drivers[driver]
        row = {}
        for window in WINDOWS:
            row.update({
                f"team_q_{window}": _recent(th, "q", .5, window),
                f"team_qgap_{window}": _recent(th, "qgap", -.8, window),
                f"driver_qdelta_{window}": _recent(dh, "qdelta", 0., window),
                f"driver_qgapdelta_{window}": _recent(dh, "qgapdelta", 0., window),
            })
        confidence = (target_round - 1) / (target_round + 3)
        tq, tg, dq = row["team_q_6"], row["team_qgap_6"], row["driver_qdelta_6"]
        row.update({
            "current_confidence_q": confidence * (row["team_q_3"] - .5),
            "circuit_q_residual": _recent([r for r in dh if r["circuit"] == circuit],
                                          "circuit_q_residual", 0., 3),
            "team_q_trend": row["team_q_3"] - row["team_q_12"],
            "driver_q_trend": row["driver_qdelta_3"] - row["driver_qdelta_12"],
            "team_q_squared": tq * tq, "team_qgap_squared": math.copysign(tg * tg, tg),
            "driver_qdelta_x_team_q": dq * tq,
        })
        previous_points[driver] = _recent([r for r in dh if r["team"] == team], "points", .12, 6)
        features[driver] = row
    observed = _practice_features(roster, practice, previous_points)
    return {driver: {**features[driver], **observed[driver]} for driver in roster}


def load_practice_qualifying_model(year):
    """Load a fixed, year-scoped parameter asset, never archived live feed data."""
    if type(year) is not int or year != 2026:
        raise ValueError("Practice qualifying parameters are currently validated for 2026")
    path = files("f1sim.analysis").joinpath("parameters", "practice_qualifying_2026.json")
    model = json.loads(path.read_text(encoding="utf-8"))
    if (model.get("policy") != PRACTICE_QUALIFYING_POLICY or model.get("year") != year
            or model.get("fit_through_year") != year - 1):
        raise ValueError("Invalid practice qualifying parameter asset")
    return model


def predict_practice_qualifying(roster, history, practice, *, year, target_round, circuit,
                               model=None):
    """Predict Q1 relative time in percent, centered over the complete modeled roster."""
    model = load_practice_qualifying_model(year) if model is None else model
    if (type(year) is not int or not isinstance(practice, dict)
            or type(practice.get("year")) is not int or practice.get("year") != year):
        raise ValueError("Practice must belong to the target season")
    if (not isinstance(model, dict) or model.get("policy") != PRACTICE_QUALIFYING_POLICY
            or model.get("year") != year
            or model.get("fit_through_year") != year - 1):
        raise ValueError("Qualifying model must be fitted strictly before the target season")
    names, weights = model.get("feature_names"), model.get("coefficients")
    if (not isinstance(names, list) or not isinstance(weights, list) or not names
            or len(names) > 100 or not all(isinstance(name, str) for name in names)
            or len(names) != len(weights) or len(set(names)) != len(names)):
        raise ValueError("Invalid practice qualifying coefficients")
    weights = [_finite(weight, "Qualifying coefficient", minimum=-100, maximum=100)
               for weight in weights]
    features = practice_qualifying_features(roster, history, practice,
                                           target_round=target_round, circuit=circuit)
    result = {}
    for driver, row in features.items():
        if any(name not in row for name in names):
            raise ValueError("Unknown qualifying model feature")
        result[driver] = math.fsum(row[name] * weight for name, weight in zip(names, weights))
    center = math.fsum(result.values()) / len(result)
    result = {driver: value - center for driver, value in result.items()}
    if any(not math.isfinite(value) or abs(value) > 10 for value in result.values()):
        raise ValueError("Predicted qualifying relative pace is out of range")
    return {
        "policy": PRACTICE_QUALIFYING_POLICY, "year": year, "target_round": target_round,
        "scope": "qualifying_only", "training_cutoff_round": target_round - 1,
        "history_rounds": sorted(event["round"] for event in history),
        "practice_session_number": practice["session_number"],
        "practice_median_seconds": median(row["lap_seconds"] for row in practice["rows"]),
        "clock_offset_percent": (_finite(
            model["practice_clock_offsets_percent"][str(practice["session_number"])],
            "Practice clock offset", minimum=-5, maximum=5,
        ) if str(practice["session_number"]) in model.get("practice_clock_offsets_percent", {})
            else None),
        "relative_q1_time_percent": result,
    }


def calibrate_practice_qualifying_drivers(drivers, cars, track, history, practice, *, year,
                                         target_round, circuit, model=None):
    """Apply relative Q1 forecasts through the existing qualifying-only input.

    Use the observed practice clock with a fixed earlier-year session correction.
    Every race-pace input is preserved.
    The adjustment is rebuilt from zero, so applying it twice cannot compound.
    """
    from f1sim.models import Weather
    from f1sim.models._native import native_physics
    from f1sim.models.tire import TIRE_COMPOUNDS
    from f1sim.simulation.lap import _NATIVE_QUALIFYING_LAP, LapSimulator

    supplied = list(drivers)
    weather = Weather(change_probability=0.0)
    if (LapSimulator.calculate_qualifying_lap is not _NATIVE_QUALIFYING_LAP
            or not native_physics(*supplied, *cars.values(), track, weather)):
        return supplied, {"policy": PRACTICE_QUALIFYING_POLICY,
                          "candidate_fallback": "custom_physics"}
    roster = {driver.id: driver.team_id for driver in supplied}
    prediction = predict_practice_qualifying(roster, history, practice, year=year,
                                            target_round=target_round, circuit=circuit,
                                            model=model)
    simulator = LapSimulator(np.random.default_rng(0))

    def lap(driver):
        return min(simulator.calculate_qualifying_lap(
            driver, cars[driver.team_id], track, tire, weather, sample_variation=False,
        ) for tire in TIRE_COMPOUNDS.values())

    supplied_times = {driver.id: lap(driver) for driver in supplied}
    native_clock = median(supplied_times.values())
    offset = prediction["clock_offset_percent"]
    clock = (prediction["practice_median_seconds"] * (1 + offset / 100)
             if offset is not None else native_clock)
    baseline = [driver.model_copy(update={"qualifying_pace_adjustment": 0.0})
                for driver in supplied]
    native = {driver.id: lap(driver) for driver in baseline}
    relative = prediction["relative_q1_time_percent"]
    relative_center = median(relative.values())
    desired = {name: clock * (1 + (value - relative_center) / 100)
               for name, value in relative.items()}
    adjustments = {name: (desired[name] - value) / (track.base_lap_time * .98)
                   for name, value in native.items()}
    if any(not math.isfinite(value) or abs(value) > .1 for value in adjustments.values()):
        return supplied, {**prediction, "candidate_fallback": "adjustment_out_of_bounds"}
    calibrated = [driver.model_copy(update={"qualifying_pace_adjustment": adjustments[driver.id]})
                  for driver in baseline]
    # The physical lap floor must not silently flatten predicted driver gaps.
    achieved = {driver.id: lap(driver) for driver in calibrated}
    if any(not math.isclose(achieved[name], value, rel_tol=0., abs_tol=1e-8)
           for name, value in desired.items()):
        return supplied, {**prediction, "candidate_fallback": "qualifying_physics_floor"}
    return calibrated, {**prediction, "reference_weather": weather.model_dump(),
                        "native_median_seconds": native_clock,
                        "predicted_median_seconds": clock,
                        "predicted_seconds": achieved}
