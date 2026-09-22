"""Seed pairing uses observed awards and the variance of within-trial differences."""

import json
from copy import deepcopy
from math import sqrt
from types import SimpleNamespace

import numpy as np
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


def pit_stop(lane, service, queue):
    return {
        "lane_loss": lane,
        "service_time": service,
        "queue_time": queue,
        "total_loss": lane + service + queue,
    }


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
        variant_dnfs=1, both_finished_races=2, both_dnf_races=0,
        reference_only_dnf_races=0, variant_only_dnf_races=1,
        dnf_rate_difference_percentage_points=100 / 3,
        dnf_rate_difference_standard_error_percentage_points=100 / 3,
        completed_distance=dict(
            paired_races=0, excluded_pairs=3, reference_mean_laps=None,
            variant_mean_laps=None, mean_laps_difference=None,
            laps_difference_standard_error=None, more_laps_races=0,
            equal_laps_races=0, fewer_laps_races=0,
        ),
        paid_stop_costs=dict(
            paired_races=0, excluded_pairs=3,
            reference_mean_paid_stops=None, variant_mean_paid_stops=None,
            mean_paid_stops_difference=None, paid_stops_difference_standard_error=None,
            reference_mean_total_loss_seconds=None, variant_mean_total_loss_seconds=None,
            mean_total_loss_seconds_difference=None,
            total_loss_seconds_difference_standard_error=None,
            reference_mean_lane_loss_seconds=None, variant_mean_lane_loss_seconds=None,
            mean_lane_loss_seconds_difference=None,
            lane_loss_seconds_difference_standard_error=None,
            reference_mean_service_time_seconds=None, variant_mean_service_time_seconds=None,
            mean_service_time_seconds_difference=None,
            service_time_seconds_difference_standard_error=None,
            reference_mean_queue_time_seconds=None, variant_mean_queue_time_seconds=None,
            mean_queue_time_seconds_difference=None,
            queue_time_seconds_difference_standard_error=None,
        ),
    )
    assert (reference, variant) == before
    json.dumps(stats, allow_nan=False)


@pytest.mark.parametrize("awards,se", [([], None), ([12], None), ([12, 12], 0.)])
def test_zero_one_and_constant_pairs(awards, se):
    reference, variant = result(awards), result(awards)
    actual = compare(reference, variant)
    if awards:
        assert actual["driver_statistics"]["A"]["points_difference_standard_error"] == se
        assert actual["driver_statistics"]["A"][
            "dnf_rate_difference_standard_error_percentage_points"
        ] == se
    else:
        assert actual["status"] == "unavailable"
        assert actual["available_seed_pairs"] == 0


def test_completed_distance_can_explain_equal_points_and_retirements():
    reference, variant = result([10] * 4), result([10] * 4)
    reference_laps = [4, 3, 0, 2]
    variant_laps = [4, 4, 1, 1]
    for index, (reference_lap, variant_lap) in enumerate(zip(reference_laps, variant_laps)):
        reference.race_results[index][0].laps_completed = reference_lap
        variant.race_results[index][0].laps_completed = variant_lap
        reference.race_results[index][0].status = variant.race_results[index][0].status = "dnf"

    stats = compare(reference, variant)["driver_statistics"]["A"]
    distance = stats["completed_distance"]
    assert stats["paired_races"] == 4
    assert stats["mean_points_difference"] == 0
    assert stats["both_dnf_races"] == 4
    assert distance == dict(
        paired_races=4, excluded_pairs=0,
        reference_mean_laps=pytest.approx(2.25), variant_mean_laps=pytest.approx(2.5),
        mean_laps_difference=pytest.approx(.25),
        laps_difference_standard_error=pytest.approx(sqrt(2.75 / 3) / 2),
        more_laps_races=2, equal_laps_races=1, fewer_laps_races=1,
    )


@pytest.mark.parametrize("bad_laps", [None, -1, 1.5, True, 5])
def test_invalid_completed_distance_does_not_drop_core_pair(bad_laps):
    reference, variant = result([12]), result([12])
    reference.race_results[0][0].laps_completed = 2
    variant.race_results[0][0].laps_completed = bad_laps

    stats = compare(reference, variant)["driver_statistics"]["A"]
    distance = stats["completed_distance"]
    assert stats["paired_races"] == 1
    assert stats["excluded_pairs"] == 0
    assert stats["mean_points_difference"] == 0
    assert distance["paired_races"] == 0
    assert distance["excluded_pairs"] == 1
    assert distance["reference_mean_laps"] is None
    assert distance["variant_mean_laps"] is None
    assert distance["mean_laps_difference"] is None
    assert distance["laps_difference_standard_error"] is None
    assert all(distance[key] == 0 for key in (
        "more_laps_races", "equal_laps_races", "fewer_laps_races",
    ))


def test_completed_distance_has_its_own_denominator_and_accepts_zero():
    reference, variant = result([12] * 3), result([12] * 3)
    for index, (reference_lap, variant_lap) in enumerate(((0, 0), (2, None), (3, 4))):
        reference.race_results[index][0].laps_completed = reference_lap
        variant.race_results[index][0].laps_completed = variant_lap

    stats = compare(reference, variant)["driver_statistics"]["A"]
    distance = stats["completed_distance"]
    assert stats["paired_races"] == 3
    assert stats["excluded_pairs"] == 0
    assert distance["paired_races"] == 2
    assert distance["excluded_pairs"] == 1
    assert distance["reference_mean_laps"] == pytest.approx(1.5)
    assert distance["variant_mean_laps"] == pytest.approx(2)
    assert distance["mean_laps_difference"] == pytest.approx(.5)
    assert distance["laps_difference_standard_error"] == pytest.approx(.5)
    assert (distance["more_laps_races"], distance["equal_laps_races"],
            distance["fewer_laps_races"]) == (1, 1, 0)


def test_completed_distance_normalizes_integral_types_for_statistics_and_json():
    reference, variant = result([12] * 3), result([12] * 3)
    for index, (reference_lap, variant_lap) in enumerate(((1, 2), (2, 3), (3, 3))):
        reference.race_results[index][0].laps_completed = np.int64(reference_lap)
        variant.race_results[index][0].laps_completed = np.int64(variant_lap)

    distance = compare(reference, variant)["driver_statistics"]["A"]["completed_distance"]
    assert distance["paired_races"] == 3
    assert distance["reference_mean_laps"] == pytest.approx(2)
    assert distance["variant_mean_laps"] == pytest.approx(8 / 3)
    assert distance["mean_laps_difference"] == pytest.approx(2 / 3)
    assert distance["laps_difference_standard_error"] == pytest.approx(1 / 3)
    json.dumps(distance, allow_nan=False)


def test_paid_stop_cost_subset_pairs_complete_histories_without_changing_core():
    reference, variant = result([12] * 3), result([12] * 3)
    reference_rows = [race[0] for race in reference.race_results]
    variant_rows = [race[0] for race in variant.race_results]
    for left, right in zip(reference_rows, variant_rows):
        left.laps_completed = right.laps_completed = 2
    reference_rows[0].pit_stops = np.int64(1)
    reference_rows[0].pit_stop_details = [
        {
            "lane_loss": np.float64(20), "service_time": np.float64(3),
            "queue_time": np.float64(0), "total_loss": np.float64(23),
        }
    ]
    variant_rows[0].pit_stops = np.int64(2)
    variant_rows[0].pit_stop_details = [
        pit_stop(np.float64(40), np.float64(6), np.float64(4)),
        pit_stop(0, 0, 0),
    ]
    reference_rows[1].pit_stops = 0
    reference_rows[1].pit_stop_details = []
    variant_rows[1].pit_stops = 1
    variant_rows[1].pit_stop_details = [pit_stop(20, 3, 5)]
    reference_rows[1].status = variant_rows[1].status = "dnf"
    reference_rows[2].pit_stops = variant_rows[2].pit_stops = 1
    reference_rows[2].pit_stop_details = [pit_stop(20, 3, 0)]
    variant_rows[2].pit_stop_details = [dict(pit_stop(20, 3, 0), total_loss=99)]

    stats = compare(reference, variant)["driver_statistics"]["A"]
    costs = stats["paid_stop_costs"]
    assert stats["paired_races"] == 3
    assert stats["completed_distance"]["paired_races"] == 3
    assert costs == dict(
        paired_races=2, excluded_pairs=1,
        reference_mean_paid_stops=pytest.approx(.5),
        variant_mean_paid_stops=pytest.approx(1.5),
        mean_paid_stops_difference=pytest.approx(1),
        paid_stops_difference_standard_error=0,
        reference_mean_total_loss_seconds=pytest.approx(11.5),
        variant_mean_total_loss_seconds=pytest.approx(39),
        mean_total_loss_seconds_difference=pytest.approx(27.5),
        total_loss_seconds_difference_standard_error=pytest.approx(.5),
        reference_mean_lane_loss_seconds=pytest.approx(10),
        variant_mean_lane_loss_seconds=pytest.approx(30),
        mean_lane_loss_seconds_difference=pytest.approx(20),
        lane_loss_seconds_difference_standard_error=0,
        reference_mean_service_time_seconds=pytest.approx(1.5),
        variant_mean_service_time_seconds=pytest.approx(4.5),
        mean_service_time_seconds_difference=pytest.approx(3),
        service_time_seconds_difference_standard_error=0,
        reference_mean_queue_time_seconds=pytest.approx(0),
        variant_mean_queue_time_seconds=pytest.approx(4.5),
        mean_queue_time_seconds_difference=pytest.approx(4.5),
        queue_time_seconds_difference_standard_error=pytest.approx(.5),
    )
    assert all(isinstance(value, (int, float, type(None))) for value in costs.values())
    json.dumps(costs, allow_nan=False)


def test_paid_stop_cost_extreme_finite_values_keep_representable_se():
    reference, variant = result([25, 25]), result([25, 25])
    for sample, losses in ((reference, (0.0, 1.7e308)), (variant, (1.7e308, 0.0))):
        for race, loss in zip(sample.race_results, losses):
            race[0].pit_stops = int(loss > 0)
            race[0].pit_stop_details = [pit_stop(loss, 0.0, 0.0)] if loss else []
    stats = compare(reference, variant)["driver_statistics"]["A"]
    costs = stats["paid_stop_costs"]
    assert stats["paired_races"] == costs["paired_races"] == 2
    assert costs["mean_total_loss_seconds_difference"] == 0
    assert costs["total_loss_seconds_difference_standard_error"] == pytest.approx(1.7e308)
    assert costs["lane_loss_seconds_difference_standard_error"] == pytest.approx(1.7e308)
    json.dumps(stats, allow_nan=False)


def test_paid_stop_cost_single_pair_has_no_estimable_se():
    reference, variant = result([12]), result([12])
    reference_row = reference.race_results[0][0]
    variant_row = variant.race_results[0][0]
    reference_row.pit_stops = 0
    reference_row.pit_stop_details = []
    variant_row.pit_stops = 1
    variant_row.pit_stop_details = [pit_stop(20, 3, 4)]

    costs = compare(reference, variant)["driver_statistics"]["A"]["paid_stop_costs"]
    assert costs["paired_races"] == 1
    assert costs["excluded_pairs"] == 0
    assert costs["reference_mean_paid_stops"] == 0
    assert costs["variant_mean_paid_stops"] == 1
    assert costs["mean_total_loss_seconds_difference"] == 27
    assert all(value is None for key, value in costs.items()
               if key.endswith("_standard_error"))


@pytest.mark.parametrize("stops,details", [
    (None, []), (True, []), (-1, []), (1.0, [pit_stop(1, 1, 1)]),
    (1, None), (1, [pit_stop(1, 1, 1) | {"queue_time": -1}]),
])
def test_invalid_paid_stop_costs_exclude_only_cost_subset(stops, details):
    reference, variant = result([12]), result([12])
    for sample in (reference, variant):
        sample.race_results[0][0].laps_completed = 2
        sample.race_results[0][0].pit_stops = stops
        sample.race_results[0][0].pit_stop_details = details

    stats = compare(reference, variant)["driver_statistics"]["A"]
    assert stats["paired_races"] == 1
    assert stats["completed_distance"]["paired_races"] == 1
    assert stats["paid_stop_costs"]["paired_races"] == 0
    assert stats["paid_stop_costs"]["excluded_pairs"] == 1


def test_invalid_core_observation_excludes_distance_pair_too():
    reference, variant = result([12]), result([12])
    reference.race_results[0][0].laps_completed = 3
    variant.race_results[0][0].laps_completed = 4
    variant.race_results[0][0].position = 0

    stats = compare(reference, variant)["driver_statistics"]["A"]
    assert stats["paired_races"] == 0
    assert stats["excluded_pairs"] == 1
    assert stats["completed_distance"] == dict(
        paired_races=0, excluded_pairs=1, reference_mean_laps=None,
        variant_mean_laps=None, mean_laps_difference=None,
        laps_difference_standard_error=None, more_laps_races=0,
        equal_laps_races=0, fewer_laps_races=0,
    )


def test_joint_retirement_counts_and_rate_se_retain_paired_statuses():
    aligned_reference, aligned_variant = result([12] * 4), result([12] * 4)
    swapped_reference, swapped_variant = result([12] * 4), result([12] * 4)
    for sample in (aligned_reference, aligned_variant, swapped_reference):
        for index in (0, 1):
            sample.race_results[index][0].status = "dnf"
    for index in (2, 3):
        swapped_variant.race_results[index][0].status = "dnf"

    aligned = compare(aligned_reference, aligned_variant)["driver_statistics"]["A"]
    swapped = compare(swapped_reference, swapped_variant)["driver_statistics"]["A"]

    assert (aligned["reference_dnfs"], aligned["variant_dnfs"]) == (2, 2)
    assert (swapped["reference_dnfs"], swapped["variant_dnfs"]) == (2, 2)
    assert aligned["dnf_rate_difference_percentage_points"] == 0
    assert swapped["dnf_rate_difference_percentage_points"] == 0
    assert aligned["dnf_rate_difference_standard_error_percentage_points"] == 0
    assert swapped["dnf_rate_difference_standard_error_percentage_points"] == pytest.approx(
        100 / sqrt(3)
    )
    assert (
        aligned["both_finished_races"], aligned["both_dnf_races"],
        aligned["reference_only_dnf_races"], aligned["variant_only_dnf_races"],
    ) == (2, 2, 0, 0)
    assert (
        swapped["both_finished_races"], swapped["both_dnf_races"],
        swapped["reference_only_dnf_races"], swapped["variant_only_dnf_races"],
    ) == (0, 0, 2, 2)
    for stats in (aligned, swapped):
        assert sum(stats[key] for key in (
            "both_finished_races", "both_dnf_races",
            "reference_only_dnf_races", "variant_only_dnf_races",
        )) == stats["paired_races"] == 4


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


@pytest.mark.parametrize(("path", "value"), [
    (("weather", "rain_intensity"), False),
    (("weather", "rain_intensity"), "0.0"),
    (("track", "total_laps"), True),
    (("track", "total_laps"), 4.0),
    (("track", "total_laps"), "4"),
    (("track", "sectors", 0, "base_time"), "30.0"),
    (("track", "active_aero_zones", 0, "zone_id"), 1.0),
])
@pytest.mark.parametrize("side", ["reference", "variant", "both"])
def test_coercible_saved_model_values_are_not_paired(path, value, side):
    reference, variant = result([25]), result([25])
    for sample in (reference, variant):
        sample.input_snapshot["track"]["sectors"] = [{
            "number": 1, "base_time": 30.0, "is_high_speed": False,
            "overtake_opportunity": 0.2,
        }]
        sample.input_snapshot["track"]["active_aero_zones"] = [{
            "zone_id": 1, "sector": 1, "time_gain": 0.3,
            "activation_point_pct": 0.0,
        }]
    targets = ((reference,) if side == "reference" else (variant,)
               if side == "variant" else (reference, variant))
    for target in targets:
        value_target = target.input_snapshot
        for key in path[:-1]:
            value_target = value_target[key]
        value_target[path[-1]] = value

    stats = compare(reference, variant)
    assert stats["status"] == "unavailable"
    assert stats["reason"] == (
        "Valid saved model inputs and complete runtime provenance are required."
    )


@pytest.mark.parametrize("version", [1, 2, 3, 4])
def test_native_model_dump_snapshots_remain_pairable_across_schemas(version):
    reference, variant = result([25]), result([25])
    for sample in (reference, variant):
        snapshot = sample.input_snapshot
        snapshot["schema_version"] = version
        if version == 1:
            snapshot.pop("rng_policy", None)
        if version >= 3:
            snapshot["starting_tire_ages"] = {}
        if version == 4:
            snapshot["starting_tires"] = {"A": "soft"}
            snapshot["starting_tire_ages"] = {"A": 5}
            snapshot["tire_inventory"] = {"A": [
                {"id": "s", "compound": "soft", "age": 5},
                {"id": "m", "compound": "medium", "age": 0},
            ]}

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
    assert actual["dnf_rate_difference_standard_error_percentage_points"] is None
    assert all(actual[key] == 0 for key in (
        "both_finished_races", "both_dnf_races",
        "reference_only_dnf_races", "variant_only_dnf_races",
    ))


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


@pytest.mark.parametrize("side", ["reference", "variant", "both"])
def test_non_runnable_driver_cannot_supply_paired_observations(side):
    reference, variant = result([25]), result([18])
    for label, sample in (("reference", reference), ("variant", variant)):
        sample.input_snapshot["drivers"].append(
            Driver(id="B", name="B", team_id="missing").model_dump()
        )
        if side in (label, "both"):
            sample.race_results[0].append(row(10, driver_id="B", position=2))
    actual = compare(reference, variant)
    assert actual["qualifying_mismatches"] == 0
    assert actual["driver_statistics"]["A"]["paired_races"] == 1
    excluded = actual["driver_statistics"]["B"]
    assert excluded["paired_races"] == 0
    assert excluded["excluded_pairs"] == 1
    assert excluded["mean_points_difference"] is None


def test_scenario_order_and_result_are_independent():
    sample = result([25])
    actual = paired_comparison_statistics({"z": sample, "reference": sample, "a": sample},
                                         "reference")
    assert list(actual["variants"]) == ["z", "a"]
    actual["variants"]["z"]["driver_statistics"]["A"]["paired_races"] = 999
    assert compare(sample, sample)["driver_statistics"]["A"]["paired_races"] == 1
