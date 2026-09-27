"""Constructor points aggregate only complete within-seed team pairs."""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.models import Car, Driver, Track, Weather


def result(team_by_driver, events, *, seed=10, car_teams=None, team_names=None):
    """Build saved inputs and aligned results for concise paired-statistics cases."""
    team_names = team_names or {}
    driver_models = [
        Driver(id=driver_id, name=driver_id, team_id=team_id)
        for driver_id, team_id in team_by_driver.items()
    ]
    if car_teams is None:
        car_teams = set(team_by_driver.values())
    cars = {
        team_id: Car(team_id=team_id, team_name=team_names.get(team_id, team_id))
        for team_id in car_teams
    }
    track = Track(id="test", name="Test", country="Test", total_laps=4, base_lap_time=90)
    weather = Weather()

    race_results = []
    qualifying_results = []
    for event in events:
        race = []
        for fallback_position, driver_id in enumerate(team_by_driver, start=1):
            if driver_id not in event:
                continue
            outcome = event[driver_id]
            race.append(SimpleNamespace(
                driver_id=driver_id,
                position=outcome.get("position", fallback_position),
                status=outcome.get("status", "finished"),
                classified=outcome.get("classified"),
                points_awarded=outcome.get("points"),
            ))
        race_results.append(race)
        qualifying_results.append([
            SimpleNamespace(
                driver_id=driver_id, driver_name=driver_id, position=position,
                best_time=90.0, q1_time=92.0, q2_time=91.0, q3_time=90.0,
                eliminated_in=None,
            )
            for position, driver_id in enumerate(team_by_driver, start=1)
        ])

    return SimulationResults(
        num_simulations=max(len(events), 1), track_name="Test", driver_stats={},
        race_results=race_results, qualifying_results=qualifying_results, seed=seed,
        input_snapshot={
            "schema_version": 2,
            "rng_policy": "isolated_weather_v1",
            "starting_tires": {},
            "drivers": [driver.model_dump() for driver in driver_models],
            "cars": {team_id: car.model_dump() for team_id, car in cars.items()},
            "track": track.model_dump(),
            "weather": weather.model_dump(),
            "runtime": {
                "f1sim": "1", "python": "3", "numpy": "2", "pydantic": "2",
                "simulation_source_sha256": "a" * 64,
            },
        },
    )


def events_for(team_by_driver, awards):
    points_position = {25: 1, 18: 2, 15: 3, 12: 4, 10: 5,
                       8: 6, 6: 7, 4: 8, 2: 9, 1: 10, 0: 11}
    return [
        {
            driver_id: {"points": point, "position": points_position[point]}
            for driver_id, point in event.items()
        }
        for event in awards
    ]


def compare(reference, variant):
    return paired_comparison_statistics(
        {"reference": reference, "variant": variant}, "reference",
    )["variants"]["variant"]


def test_teammate_point_changes_are_summed_per_race_before_averaging():
    teams = {"A": "alpha", "B": "alpha"}
    reference = result(teams, events_for(teams, [{"A": 4, "B": 15}]))
    variant = result(teams, events_for(teams, [{"A": 8, "B": 10}]))

    stats = compare(reference, variant)
    team = stats["constructor_statistics"]["alpha"]

    assert team == {
        "team_name": "alpha", "driver_ids": ["A", "B"],
        "paired_races": 1, "excluded_pairs": 0,
        "reference_mean_points": 19.0, "variant_mean_points": 18.0,
        "mean_points_difference": -1.0,
        "points_difference_standard_error": None,
        "more_points_races": 0, "equal_points_races": 0, "fewer_points_races": 1,
    }
    assert stats["driver_statistics"]["A"]["mean_points_difference"] == 4.0
    assert stats["driver_statistics"]["B"]["mean_points_difference"] == -5.0
    assert "finished_race_time" not in team
    json.dumps(stats, allow_nan=False)


def test_opposing_teammate_changes_cancel_with_zero_constructor_paired_se():
    teams = {"A": "alpha", "B": "alpha"}
    reference = result(teams, events_for(teams, [
        {"A": 8, "B": 12}, {"A": 12, "B": 8},
    ]))
    variant = result(teams, events_for(teams, [
        {"A": 12, "B": 8}, {"A": 8, "B": 12},
    ]))

    stats = compare(reference, variant)
    team = stats["constructor_statistics"]["alpha"]

    assert team["paired_races"] == 2
    assert team["mean_points_difference"] == 0.0
    assert team["points_difference_standard_error"] == 0.0
    assert team["equal_points_races"] == 2
    assert stats["driver_statistics"]["A"]["points_difference_standard_error"] == 4.0
    assert stats["driver_statistics"]["B"]["points_difference_standard_error"] == 4.0


def test_bad_teammate_excludes_only_that_constructor_seed_and_preserves_other_stats():
    teams = {"A1": "alpha", "A2": "alpha", "B1": "beta", "B2": "beta"}
    awards = [{"A1": 25, "A2": 18, "B1": 15, "B2": 12} for _ in range(3)]
    reference = result(teams, events_for(teams, awards))
    variant = result(teams, events_for(teams, awards))

    # One missing, one duplicate, and one invalid teammate observation.
    variant.race_results[0] = [row for row in variant.race_results[0] if row.driver_id != "A2"]
    reference.race_results[1].append(deepcopy(
        next(row for row in reference.race_results[1] if row.driver_id == "A2")
    ))
    next(row for row in variant.race_results[2] if row.driver_id == "A2").position = True

    stats = compare(reference, variant)
    alpha = stats["constructor_statistics"]["alpha"]
    beta = stats["constructor_statistics"]["beta"]

    assert alpha["paired_races"] == 0
    assert alpha["excluded_pairs"] == 3
    assert alpha["reference_mean_points"] is None
    assert alpha["variant_mean_points"] is None
    assert alpha["mean_points_difference"] is None
    assert alpha["points_difference_standard_error"] is None
    assert beta["paired_races"] == 3
    assert beta["excluded_pairs"] == 0
    assert stats["driver_statistics"]["A1"]["paired_races"] == 3
    assert stats["driver_statistics"]["A2"]["paired_races"] == 0
    assert stats["driver_statistics"]["B1"]["paired_races"] == 3


def test_qualifying_mismatch_is_in_constructor_excluded_pair_count():
    teams = {"A": "alpha"}
    awards = [{"A": 12}, {"A": 10}]
    reference = result(teams, events_for(teams, awards))
    variant = result(teams, events_for(teams, awards))
    variant.qualifying_results[1][0].position = 2

    stats = compare(reference, variant)
    team = stats["constructor_statistics"]["alpha"]

    assert stats["available_seed_pairs"] == 2
    assert stats["qualifying_mismatches"] == 1
    assert team["paired_races"] == 1
    assert team["excluded_pairs"] == 1


def test_paired_comparison_with_no_valid_constructor_observations_keeps_null_means():
    teams = {"A": "alpha"}
    events = events_for(teams, [{"A": 12}])
    reference = result(teams, events)
    variant = result(teams, events)
    variant.qualifying_results[0][0].position = 2

    stats = compare(reference, variant)
    team = stats["constructor_statistics"]["alpha"]

    assert stats["status"] == "paired"
    assert stats["qualifying_mismatches"] == 1
    assert team["paired_races"] == 0
    assert team["excluded_pairs"] == 1
    assert team["reference_mean_points"] is None
    assert team["variant_mean_points"] is None
    assert team["mean_points_difference"] is None
    assert team["points_difference_standard_error"] is None


def test_classified_dnf_points_are_included_and_single_driver_membership_is_explicit():
    teams = {"A": "alpha"}
    reference = result(teams, events_for(teams, [{"A": 8}]))
    variant = result(teams, events_for(teams, [{"A": 0}]))
    for sample, classified in ((reference, True), (variant, False)):
        sample.race_results[0][0].status = "dnf"
        sample.race_results[0][0].classified = classified

    team = compare(reference, variant)["constructor_statistics"]["alpha"]

    assert team["driver_ids"] == ["A"]
    assert team["paired_races"] == 1
    assert team["reference_mean_points"] == 8.0
    assert team["variant_mean_points"] == 0.0
    assert team["mean_points_difference"] == -8.0


def test_missing_car_team_is_not_runnable_or_listed_as_a_constructor():
    teams = {"A": "alpha", "B": "absent"}
    awards = [{"A": 25, "B": 18}]
    reference = result(teams, events_for(teams, awards), car_teams={"alpha"})
    variant = result(teams, events_for(teams, awards), car_teams={"alpha"})

    stats = compare(reference, variant)

    assert list(stats["constructor_statistics"]) == ["alpha"]
    assert stats["constructor_statistics"]["alpha"]["driver_ids"] == ["A"]
    assert stats["driver_statistics"]["B"]["paired_races"] == 0


def test_team_name_falls_back_to_exact_team_id_when_snapshot_name_is_empty():
    teams = {"A": "alpha"}
    events = events_for(teams, [{"A": 25}])
    reference = result(teams, events, team_names={"alpha": ""})
    variant = result(teams, events, team_names={"alpha": ""})

    assert compare(reference, variant)["constructor_statistics"]["alpha"]["team_name"] == "alpha"


def test_constructor_output_field_exists_for_unavailable_legacy_inputs():
    teams = {"A": "alpha"}
    events = events_for(teams, [{"A": 25}])
    reference = result(teams, events)
    variant = result(teams, events)
    variant.input_snapshot = None

    stats = compare(reference, variant)

    assert stats["status"] == "unavailable"
    assert stats["constructor_statistics"] == {}


def test_mismatched_models_do_not_produce_constructor_pairs():
    teams = {"A": "alpha"}
    events = events_for(teams, [{"A": 25}])
    reference = result(teams, events)
    variant = result(teams, events)
    variant.input_snapshot["cars"]["alpha"]["base_pace"] = 0.7

    stats = compare(reference, variant)

    assert stats["status"] == "unavailable"
    assert stats["constructor_statistics"] == {}


def test_no_overlapping_seeds_leave_unavailable_constructor_statistics_empty():
    teams = {"A": "alpha"}
    events = events_for(teams, [{"A": 25}])
    reference = result(teams, events, seed=10)
    variant = result(teams, events, seed=11)

    stats = compare(reference, variant)

    assert stats["status"] == "unavailable"
    assert stats["available_seed_pairs"] == 0
    assert stats["constructor_statistics"] == {}


def test_constructor_seed_overlap_uses_only_recorded_common_trials():
    teams = {"A": "alpha"}
    reference = result(teams, events_for(teams, [{"A": 25}, {"A": 18}, {"A": 15}]))
    variant = result(teams, events_for(teams, [{"A": 10}, {"A": 25}]), seed=11)

    stats = compare(reference, variant)
    team = stats["constructor_statistics"]["alpha"]

    assert (stats["seed_from"], stats["seed_to"], stats["available_seed_pairs"]) == (11, 12, 2)
    assert team["paired_races"] == 2
    assert team["excluded_pairs"] == 0
    assert team["reference_mean_points"] == 16.5
    assert team["variant_mean_points"] == 17.5
    assert team["mean_points_difference"] == 1.0
    assert team["points_difference_standard_error"] == pytest.approx(9.0)
