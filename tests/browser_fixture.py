"""Generate SYNTHETIC offline browser inputs using the real simulation/serializer.

This helper never loads season data and writes no persisted fixtures.
"""

import contextlib
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
    for index, label in enumerate(("dry", "light_rain", "heavy_rain")):
        scenario = scenario_weather_from_label(Weather(change_probability=0), label)
        weather[label] = scenario.weather
        results[label] = MonteCarloRunner(
            drivers, cars, track, scenario.weather, seed=42 + index * 1000,
            race_engine="chronological",
            starting_tires={"S00": "hard", "S01": "soft"},
        ).run(num_simulations=10, parallel=False)
    payload = _summarize_scenario_results(results, scenario_weather=weather)
    payload["comparison_report_html"] = render_comparison_report(results)
    ratings = _serialize_ratings_snapshot(drivers, cars, {})
    ratings["source"] = "SYNTHETIC offline test"
    payload.update(
        track=track.name, track_details=_serialize_track(track), year=2026,
        race=track.name, ratings=ratings, provenance={"source": "SYNTHETIC offline test"},
        request={"race_engine": "chronological", "simulations": 10, "seed": 42, "parallel": False,
                 "starting_tires": {"S00": "hard", "S01": "soft"},
                 "scenarios": "dry,light_rain,heavy_rain", "qualifying_mode": "simulated"},
    )
    return {
        "html": build_dashboard_html(), "payload": payload,
        "calendar": {"events": [{"round": 1, "race": track.name, "location": "Test"}]},
    }


if __name__ == "__main__":
    with contextlib.redirect_stdout(sys.stderr):
        fixture = build_fixture()
    print(json.dumps(fixture, allow_nan=False))
