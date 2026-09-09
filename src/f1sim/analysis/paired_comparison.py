"""Descriptive within-seed outcome differences for compatible saved simulations."""

from math import isfinite, sqrt
from numbers import Integral, Real
from statistics import mean, stdev

from pydantic import ValidationError

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.execution import validate_starting_tire_ages
from f1sim.simulation.race_points import POINTS_SYSTEM, points_for_result
from f1sim.simulation.randomness import RNG_POLICIES


def _integer(value, minimum=0):
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def _snapshot(result):
    snapshot = result.input_snapshot
    if not isinstance(snapshot, dict):
        return None
    version = snapshot.get("schema_version")
    if not _integer(version, 1) or version not in (1, 2, 3):
        return None
    policy = snapshot.get("rng_policy", "shared_v1" if version == 1 else None)
    if policy not in RNG_POLICIES:
        return None
    runtime = snapshot.get("runtime")
    runtime_fields = ("f1sim", "python", "numpy", "pydantic", "simulation_source_sha256")
    if not isinstance(runtime, dict) or not all(
        isinstance(runtime.get(key), str) and runtime[key].strip() for key in runtime_fields
    ):
        return None
    digest = runtime["simulation_source_sha256"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        return None
    drivers, cars = snapshot.get("drivers"), snapshot.get("cars")
    if not isinstance(drivers, list) or not drivers or not isinstance(cars, dict) or not cars:
        return None
    try:
        if not all(isinstance(row, dict) and row for row in drivers):
            return None
        roster = [Driver.model_validate(row) for row in drivers]
        if not all(isinstance(row, dict) and row for row in cars.values()):
            return None
        for row in cars.values():
            Car.model_validate(row)
        for key, model in (("track", Track), ("weather", Weather)):
            if not isinstance(snapshot.get(key), dict) or not snapshot[key]:
                return None
            model.model_validate(snapshot[key])
        ids = [driver.id for driver in roster]
        if version == 3 and not isinstance(snapshot.get("starting_tire_ages"), dict):
            return None
        if version < 3 and snapshot.get("starting_tire_ages"):
            return None
        validate_starting_tire_ages(snapshot.get("starting_tire_ages"),
                                    snapshot.get("starting_tires"), ids)
        if len(set(ids)) != len(ids) or any(not key for key in ids):
            return None
    except (ValidationError, TypeError, ValueError):
        return None
    return ({key: snapshot[key] for key in ("drivers", "cars", "track", "weather", "runtime")}
            | {"rng_policy": policy}, ids,
            [driver.id for driver in roster if driver.team_id in cars])


def _qualifying_valid(rows, roster, runnable):
    if not rows or not runnable:
        return False
    ids = [getattr(row, "driver_id", None) for row in rows]
    if any(not isinstance(driver, str) for driver in ids):
        return False
    if len(set(ids)) != len(ids) or not set(runnable) <= set(ids) <= set(roster):
        return False
    positions = [getattr(row, "position", None) for row in rows]
    if any(not _integer(position, 1) for position in positions):
        return False
    if len(set(positions)) != len(positions) or positions != sorted(positions):
        return False

    def valid_time(value):
        try:
            return isinstance(value, Real) and not isinstance(value, bool) and (
                isfinite(value) and value > 0
            )
        except OverflowError:
            return False

    for row in rows:
        if row.driver_id not in runnable:
            continue
        if not all(hasattr(row, field) for field in (
            "driver_name", "best_time", "q1_time", "q2_time", "q3_time", "eliminated_in",
        )):
            return False
        if not isinstance(row.driver_name, str) or not row.driver_name:
            return False
        if row.eliminated_in not in (None, "Q1", "Q2"):
            return False
        if not valid_time(row.best_time) or not valid_time(row.q1_time):
            return False
        if any(value is not None and not valid_time(value) for value in (row.q2_time, row.q3_time)):
            return False
        if row.eliminated_in == "Q1" and (row.q2_time is not None or row.q3_time is not None):
            return False
        if row.eliminated_in == "Q2" and (row.q2_time is None or row.q3_time is not None):
            return False
        if row.eliminated_in is None and (row.q2_time is None or row.q3_time is None):
            return False
    return True


def _ordered_trials(result):
    return (
        _integer(result.seed) and _integer(result.num_simulations, 1)
        and isinstance(result.race_results, (list, tuple))
        and len(result.race_results) <= result.num_simulations
        and isinstance(result.qualifying_results, (list, tuple))
        and len(result.qualifying_results) == len(result.race_results)
        and all(isinstance(race, (list, tuple)) for race in result.race_results)
        and all(isinstance(race, (list, tuple)) for race in result.qualifying_results)
    )


def _observation(race, driver):
    rows = [row for row in race if getattr(row, "driver_id", None) == driver]
    if len(rows) != 1:
        return None
    row = rows[0]
    status = getattr(row, "status", None)
    status = getattr(status, "value", status)
    if status not in ("finished", "dnf") or not _integer(getattr(row, "position", None), 1):
        return None
    classified = getattr(row, "classified", None)
    if classified is not None and not isinstance(classified, bool):
        return None
    award = getattr(row, "points_awarded", None)
    if award is not None and (not _integer(award) or award > max(POINTS_SYSTEM.values())):
        return None
    return int(points_for_result(row)), int(status == "dnf")


def _driver_statistics(observations, available):
    count = len(observations)
    differences = [variant[0] - reference[0] for reference, variant in observations]
    reference_dnfs = sum(reference[1] for reference, _ in observations)
    variant_dnfs = sum(variant[1] for _, variant in observations)
    return {
        "paired_races": count, "excluded_pairs": available - count,
        "reference_mean_points": float(mean(ref[0] for ref, _ in observations)) if count else None,
        "variant_mean_points": float(mean(var[0] for _, var in observations)) if count else None,
        "mean_points_difference": float(mean(differences)) if count else None,
        "points_difference_standard_error": stdev(differences) / sqrt(count) if count > 1 else None,
        "more_points_races": sum(value > 0 for value in differences),
        "equal_points_races": sum(value == 0 for value in differences),
        "fewer_points_races": sum(value < 0 for value in differences),
        "reference_dnfs": reference_dnfs, "variant_dnfs": variant_dnfs,
        "dnf_rate_difference_percentage_points": (
            (variant_dnfs - reference_dnfs) * 100 / count if count else None
        ),
    }


def paired_comparison_statistics(
    scenario_results: dict[str, SimulationResults], reference_scenario: str,
) -> dict:
    """Pair compatible trials by effective seed, conditioning on equal qualifying.

    Differences are variant minus reference. Standard errors describe the sampled
    paired differences, not a causal effect or an empirical calibration guarantee.
    Missing driver observations are excluded only for the affected driver.
    """
    if not isinstance(reference_scenario, str) or reference_scenario not in scenario_results:
        raise ValueError("reference_scenario must name a supplied scenario")
    reference = scenario_results[reference_scenario]
    reference_inputs = _snapshot(reference)
    variants = {}
    for label, variant in scenario_results.items():
        if label == reference_scenario:
            continue
        summary = dict(status="unavailable", reason=None, available_seed_pairs=0,
                       qualifying_mismatches=0, seed_from=None, seed_to=None,
                       driver_statistics={})
        variants[label] = summary
        variant_inputs = _snapshot(variant)
        if reference_inputs is None or variant_inputs is None:
            summary["reason"] = "Complete saved inputs and runtime provenance are required."
            continue
        if reference_inputs[0] != variant_inputs[0]:
            summary["reason"] = "Saved models, weather, runtime or random-stream policies differ."
            continue
        if not _ordered_trials(reference) or not _ordered_trials(variant):
            summary["reason"] = "Valid seeds and ordered race/qualifying records are required."
            continue
        lower = max(int(reference.seed), int(variant.seed))
        upper = min(int(reference.seed) + len(reference.race_results),
                    int(variant.seed) + len(variant.race_results))
        available = max(upper - lower, 0)
        summary["available_seed_pairs"] = available
        if not available:
            summary["reason"] = "No recorded trial seeds overlap."
            continue
        summary.update(status="paired", seed_from=lower, seed_to=upper - 1)
        observations = {driver: [] for driver in reference_inputs[1]}
        for seed in range(lower, upper):
            ri, vi = seed - int(reference.seed), seed - int(variant.seed)
            rq, vq = reference.qualifying_results[ri], variant.qualifying_results[vi]
            if not _qualifying_valid(rq, reference_inputs[1], reference_inputs[2]) or (
                not _qualifying_valid(vq, variant_inputs[1], variant_inputs[2]) or rq != vq
            ):
                summary["qualifying_mismatches"] += 1
                continue
            for driver, samples in observations.items():
                left = _observation(reference.race_results[ri], driver)
                right = _observation(variant.race_results[vi], driver)
                if left is not None and right is not None:
                    samples.append((left, right))
        summary["driver_statistics"] = {
            driver: _driver_statistics(samples, available)
            for driver, samples in observations.items()
        }
    return {"reference_scenario": reference_scenario, "variants": variants}
