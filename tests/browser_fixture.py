"""Generate SYNTHETIC offline browser inputs using the real simulation/serializer.

This helper never loads season data and writes no persisted fixtures.
"""

import contextlib
import copy
import json
import sys

from f1sim.analysis import MonteCarloRunner, scenario_weather_from_label
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.comparison import render_comparison_report
from f1sim.web.server import (
    _serialize_ratings_snapshot,
    _serialize_track,
    _summarize_scenario_results,
    build_dashboard_html,
)


def build_fixture() -> dict:
    drivers = [
        Driver(id=f"S{i:02}", name=f"Synthetic Driver {i}", team_id=f"team{i // 2}")
        for i in range(22)
    ]
    cars = {
        driver.team_id: Car(team_id=driver.team_id, team_name=f"Synthetic {driver.team_id}")
        for driver in drivers
    }
    track = Track(
        id="synthetic", name="SYNTHETIC Test Circuit", country="Test",
        total_laps=12, base_lap_time=90, pit_lane_delta=20,
    )
    results, weather = {}, {}
    inventory = {"S00": [
        {"id": "set-1", "compound": "hard", "age": 5},
        {"id": "set-2", "compound": "soft", "age": 0},
        {"id": "set-3", "compound": "intermediate", "age": 0},
        {"id": "set-4", "compound": "wet", "age": 0},
    ]}
    for index, label in enumerate(("dry", "light_rain", "heavy_rain")):
        scenario = scenario_weather_from_label(Weather(change_probability=0), label)
        weather[label] = scenario.weather
        results[label] = MonteCarloRunner(
            drivers, cars, track, scenario.weather, seed=42 + index * 1000,
            race_engine="chronological",
            starting_tires={"S00": "hard", "S01": "soft"},
            starting_tire_ages={"S00": 5},
            tire_inventory=inventory,
        ).run(num_simulations=10, parallel=False)
        # Exercise the reliability table with observed simulated shares.  The
        # dashboard must render these observations without a reference split.
        results[label].event_stats.mechanical_failure_breakdown = {
            "engine": 2,
            "gearbox": 1,
        }
    payload = _summarize_scenario_results(results, scenario_weather=weather)
    payload["comparison_report_html"] = render_comparison_report(results)
    ratings = _serialize_ratings_snapshot(drivers, cars, {})
    ratings["source"] = "SYNTHETIC offline test"
    payload.update(
        track=track.name, track_details=_serialize_track(track), year=2026,
        race=track.name, ratings=ratings, provenance={"source": "SYNTHETIC offline test"},
        request={"race_engine": "chronological", "simulations": 10, "seed": 42, "parallel": False,
                 "weather_mode": "fixed_rainfall",
                 "starting_tires": {"S00": "hard", "S01": "soft"},
                 "starting_tire_ages": {"S00": 5},
                 "tire_inventory": inventory,
                 "scenarios": "dry,light_rain,heavy_rain", "qualifying_mode": "simulated"},
    )
    comparison_payload = copy.deepcopy(payload)
    comparison_payload["request"] = {
        **comparison_payload["request"],
        "pit_plans": {"S00": [{"lap": 4, "compound": "hard"}], "S01": []},
        "compare_automatic": True,
    }
    automatic_reference = copy.deepcopy(payload)
    automatic_reference.pop("comparison_report_html", None)
    automatic_reference["request"] = {
        **automatic_reference["request"],
        "pit_plans": {},
        "compare_automatic": False,
    }
    comparison_payload["automatic_reference"] = automatic_reference
    comparison_payload["strategy_comparisons"] = {
        label: {
            "reference_scenario": "automatic",
            "variants": {"custom": {
                "status": "paired",
                "available_seed_pairs": 10,
                "qualifying_mismatches": 0,
                "driver_statistics": {
                    "S00": {
                        "paired_races": 3, "excluded_pairs": 7,
                        "mean_points_difference": 1.25,
                        "points_difference_standard_error": 0.25,
                        "dnf_rate_difference_percentage_points": -3.333,
                        "dnf_rate_difference_standard_error_percentage_points": 0.5,
                        "completed_distance": {
                            "paired_races": 2, "excluded_pairs": 8,
                            "mean_laps_difference": 0.5,
                            "laps_difference_standard_error": None,
                        },
                        "paid_stop_costs": {
                            "paired_races": 1, "excluded_pairs": 9,
                            "mean_paid_stops_difference": 0,
                            "paid_stops_difference_standard_error": None,
                            "mean_total_loss_seconds_difference": -1.5,
                            "total_loss_seconds_difference_standard_error": None,
                            "mean_lane_loss_seconds_difference": -0.75,
                            "lane_loss_seconds_difference_standard_error": None,
                            "mean_service_time_seconds_difference": -0.5,
                            "service_time_seconds_difference_standard_error": None,
                            "mean_queue_time_seconds_difference": -0.25,
                            "queue_time_seconds_difference_standard_error": None,
                        },
                    },
                },
            }},
        }
        for label in results
    }
    comparison_payload["strategy_comparison_reports"] = {}
    for label, automatic_result in results.items():
        custom_result = copy.deepcopy(automatic_result)
        custom_result.input_snapshot = {
            **(custom_result.input_snapshot or {}),
            "schema_version": 5,
            "pit_plans": comparison_payload["request"]["pit_plans"],
        }
        comparison_payload["strategy_comparison_reports"][label] = render_comparison_report(
            {"automatic": automatic_result, "custom": custom_result},
            reference_scenario="automatic",
        )
    return {
        "html": build_dashboard_html(), "payload": payload,
        "comparison_payload": comparison_payload,
        "calendar": {"events": [{"round": 1, "race": track.name, "location": "Test"}]},
    }


if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stderr):
        fixture = build_fixture()
    print(json.dumps(fixture, allow_nan=False))
