"""Seed pairing uses observed awards and the variance of within-trial differences."""

import json
from copy import deepcopy
from math import sqrt
from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.models import Car, Driver, Track, Weather


def models():
    return ([Driver(id="A", name="A", team_id="A")],
            {"A": Car(team_id="A", team_name="A")},
            Track(id="t", name="T", country="T", total_laps=4, base_lap_time=90),
            Weather())


def row(points=None, **kwargs):
    return SimpleNamespace(**(dict(driver_id="A", position=1, status="finished",
                                  classified=None, points_awarded=points) | kwargs))


def result(awards, seed=10):
    drivers, cars, track, weather = models()
    return SimulationResults(
        num_simulations=max(len(awards), 1), track_name="T", driver_stats={},
        race_results=[[row(award)] for award in awards],
        qualifying_results=[[SimpleNamespace(
            driver_id="A", driver_name="A", position=1, best_time=90.,
            q1_time=92., q2_time=91., q3_time=90., eliminated_in=None,
        )] for _ in awards],
        seed=seed, input_snapshot=dict(
            schema_version=2, rng_policy="isolated_weather_v1", starting_tires={},
            drivers=[driver.model_dump() for driver in drivers],
            cars={key: car.model_dump() for key, car in cars.items()},
            track=track.model_dump(), weather=weather.model_dump(),
            runtime=dict(f1sim="1", python="3", numpy="2", pydantic="2",
                         simulation_source_sha256="a" * 64),
        ),
    )


def compare(reference, variant):
    return paired_comparison_statistics({"reference": reference, "variant": variant},
                                        "reference")["variants"]["variant"]


def test_paired_variance_and_operational_dnf_are_independent_of_awards():
    reference, variant = result([6, 10, 18]), result([6, 13, 12])
    variant.race_results[1][0].status = "dnf"
    variant.race_results[1][0].classified = True
    before = deepcopy((reference, variant))
    stats = compare(reference, variant)
    driver = stats["driver_statistics"]["A"]
    assert stats["status"] == "paired"
    assert driver == dict(
        paired_races=3, excluded_pairs=0, reference_mean_points=34 / 3,
        variant_mean_points=31 / 3, mean_points_difference=-1.,
        points_difference_standard_error=sqrt(7), more_points_races=1,
        equal_points_races=1, fewer_points_races=1, reference_dnfs=0,
        variant_dnfs=1, dnf_rate_difference_percentage_points=100 / 3,
    )
    assert (reference, variant) == before
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("awards,se", [([], None), ([12], None), ([12, 12], 0.)])
def test_zero_one_and_constant_pairs(awards, se):
    reference, variant = result(awards), result(awards)
    actual = compare(reference, variant)
    if awards:
        assert actual["driver_statistics"]["A"]["points_difference_standard_error"] == se
    else:
        assert actual["status"] == "unavailable"
        assert actual["available_seed_pairs"] == 0


def test_overlapping_offset_seeds_use_recorded_counts_and_not_declared_denominator():
    reference, variant = result([25, 18, 15]), result([12, 10], seed=11)
    reference.num_simulations = 100
    stats = compare(reference, variant)
    assert (stats["seed_from"], stats["seed_to"], stats["available_seed_pairs"]) == (11, 12, 2)
    assert stats["driver_statistics"]["A"]["mean_points_difference"] == -5.5
    variant.seed = 13
    assert compare(reference, variant)["status"] == "unavailable"


@pytest.mark.parametrize("change", [
    lambda x: x.input_snapshot.pop("runtime"),
    lambda x: x.input_snapshot["runtime"].pop("simulation_source_sha256"),
    lambda x: x.input_snapshot["runtime"].update(simulation_source_sha256="broken"),
    lambda x: x.input_snapshot["runtime"].update(python="different"),
    lambda x: x.input_snapshot.update(drivers=[]),
    lambda x: x.input_snapshot.update(cars={}),
    lambda x: x.input_snapshot.update(weather={}),
    lambda x: x.input_snapshot["weather"].update(rain_intensity=.3),
    lambda x: x.input_snapshot["track"].update(base_lap_time=100),
    lambda x: x.input_snapshot.update(rng_policy="shared_v1"),
    lambda x: x.input_snapshot.pop("rng_policy"),
    lambda x: x.input_snapshot.update(schema_version=True),
    lambda x: setattr(x, "seed", True),
    lambda x: setattr(x, "seed", -1),
    lambda x: setattr(x, "seed", None),
    lambda x: setattr(x, "num_simulations", 0),
    lambda x: setattr(x, "num_simulations", True),
    lambda x: setattr(x, "race_results", x.race_results * 2),
    lambda x: setattr(x, "qualifying_results", []),
])
def test_ineligible_inputs_and_ordering_are_not_paired(change):
    reference, variant = result([25]), result([25])
    change(variant)
    assert compare(reference, variant)["status"] == "unavailable"


def test_absent_inputs_are_not_evidence_of_equality():
    reference, variant = result([25]), result([25])
    reference.input_snapshot = variant.input_snapshot = None
    assert compare(reference, variant)["status"] == "unavailable"


def test_legacy_policy_normalization_and_permitted_variants():
    reference, variant = result([25]), result([25])
    reference.input_snapshot.update(schema_version=1)
    reference.input_snapshot.pop("rng_policy")
    variant.input_snapshot.update(rng_policy="shared_v1", starting_tires={"A": "hard"})
    variant.race_engine = "chronological"
    assert compare(reference, variant)["status"] == "paired"


@pytest.mark.parametrize("bad", [
    row(position=True), row(position=0), row(position=1.5), row(status="racing"),
    row(classified=1), row(True), row(-1), row(26), row(2.5), row(float("nan")),
])
def test_invalid_driver_observations_are_excluded(bad):
    reference, variant = result([25]), result([25])
    variant.race_results[0] = [bad]
    actual = compare(reference, variant)["driver_statistics"]["A"]
    assert actual["paired_races"] == 0
    assert actual["excluded_pairs"] == 1
    assert actual["mean_points_difference"] is None
    assert actual["dnf_rate_difference_percentage_points"] is None


def test_qualifying_missing_duplicate_and_missing_driver_exclusions():
    reference, variant = result([25] * 4), result([25] * 4)
    variant.qualifying_results[0] = []
    variant.race_results[1] = []
    variant.race_results[2] *= 2
    stats = compare(reference, variant)
    assert stats["qualifying_mismatches"] == 1
    assert stats["driver_statistics"]["A"]["paired_races"] == 1
    assert stats["driver_statistics"]["A"]["excluded_pairs"] == 3
    variant.qualifying_results[3][0].position = 2
    assert compare(reference, variant)["qualifying_mismatches"] == 2


def test_legacy_points_fallback_honors_classified_dnf():
    reference, variant = result([None]), result([None])
    variant.race_results[0][0].status = "dnf"
    variant.race_results[0][0].classified = True
    assert compare(reference, variant)["driver_statistics"]["A"]["mean_points_difference"] == 0
    variant.race_results[0][0].classified = False
    assert compare(reference, variant)["driver_statistics"]["A"]["mean_points_difference"] == -25


@pytest.mark.parametrize("reference", [None, 0, "missing"])
def test_invalid_reference_is_rejected(reference):
    with pytest.raises(ValueError, match="reference_scenario"):
        paired_comparison_statistics({"reference": result([25])}, reference)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_actual_saved_tyre_variants(engine):
    drivers, cars, track, weather = models()
    reference = MonteCarloRunner(
        drivers, cars, track, weather, seed=4, race_engine=engine,
        starting_tires={"A": "soft"},
    ).run(2, parallel=False)
    variant = MonteCarloRunner(
        drivers, cars, track, weather, seed=4, race_engine=engine,
        starting_tires={"A": "hard"},
    ).run(2, parallel=False)
    stats = compare(reference, variant)
    assert stats["status"] == "paired"
    assert stats["qualifying_mismatches"] == 0
    assert stats["driver_statistics"]["A"]["paired_races"] == 2


@pytest.mark.parametrize("change", [
    lambda rows: rows.extend(deepcopy(rows)),
    lambda rows: setattr(rows[0], "best_time", float("inf")),
    lambda rows: setattr(rows[0], "q1_time", None),
    lambda rows: setattr(rows[0], "q2_time", -1),
    lambda rows: setattr(rows[0], "q3_time", True),
    lambda rows: setattr(rows[0], "position", True),
    lambda rows: setattr(rows[0], "eliminated_in", "unknown"),
    lambda rows: setattr(rows[0], "eliminated_in", "Q1"),
    lambda rows: setattr(rows[0], "driver_id", "unknown"),
    lambda rows: delattr(rows[0], "q2_time"),
])
def test_identical_but_malformed_qualifying_is_excluded(change):
    reference, variant = result([25]), result([25])
    change(reference.qualifying_results[0])
    variant.qualifying_results = deepcopy(reference.qualifying_results)
    actual = compare(reference, variant)
    assert actual["qualifying_mismatches"] == 1
    assert actual["driver_statistics"]["A"]["paired_races"] == 0


def test_driver_exclusion_does_not_discard_valid_teammate_pair():
    reference, variant = result([25]), result([18])
    for sample in (reference, variant):
        driver = Driver(id="B", name="B", team_id="A")
        sample.input_snapshot["drivers"].append(driver.model_dump())
        qualifying = deepcopy(sample.qualifying_results[0][0])
        qualifying.driver_id = qualifying.driver_name = "B"
        qualifying.position = 2
        sample.qualifying_results[0].append(qualifying)
        sample.race_results[0].append(row(10, driver_id="B", position=2))
    variant.race_results[0].append(row(10, driver_id="B", position=2))
    actual = compare(reference, variant)["driver_statistics"]
    assert actual["A"]["paired_races"] == 1
    assert actual["B"]["paired_races"] == 0


def test_scenario_order_and_result_are_independent():
    sample = result([25])
    actual = paired_comparison_statistics({"z": sample, "reference": sample, "a": sample},
                                         "reference")
    assert list(actual["variants"]) == ["z", "a"]
    actual["variants"]["z"]["driver_statistics"]["A"]["paired_races"] = 999
    assert compare(sample, sample)["driver_statistics"]["A"]["paired_races"] == 1
