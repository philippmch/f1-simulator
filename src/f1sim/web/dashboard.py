"""Minimal web dashboard for running and comparing simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from f1sim.analysis import MonteCarloRunner, parse_scenario_labels, scenario_weather_from_label
from f1sim.data import HistoricalDataLoader
from f1sim.models import Weather, WeatherCondition


@dataclass
class DashboardRunRequest:
    """Input payload for dashboard simulation run."""

    year: int = 2025
    race: str = "Bahrain"
    simulations: int = 200
    scenarios: str = "dry,light_rain"
    seed: int = 42
    parallel: bool = True
    max_workers: int | None = None


def _summarize_scenario_results(results_by_name: dict[str, Any]) -> dict[str, Any]:
    """Build compact summary payload for UI responses."""
    summary: dict[str, Any] = {"scenarios": {}}
    for scenario_name, results in results_by_name.items():
        win_probs = results.get_win_probabilities()
        top_3 = list(win_probs.items())[:3]
        summary["scenarios"][scenario_name] = {
            "num_simulations": results.num_simulations,
            "seed": results.seed,
            "top3_win_probabilities": top_3,
            "event_rates": results.get_event_rates(),
            "team_projection": results.get_team_championship_projection(),
        }

    return summary


def run_dashboard_simulation(request: DashboardRunRequest) -> dict[str, Any]:
    """Execute one dashboard simulation bundle and return summary."""
    loader = HistoricalDataLoader(cache_dir="data/cache")

    track_stats = loader.get_track_stats(request.year, request.race)
    driver_stats = loader.get_weighted_driver_stats(
        year=request.year,
        target_race=request.race,
        form_races=3,
        track_weight=0.5,
        form_weight=0.3,
        quali_weight=0.2,
    )

    drivers = loader.create_drivers_from_stats(driver_stats)
    cars = loader.create_cars_from_stats(driver_stats)
    track = loader.create_track_from_stats(track_stats)
    historical_grid = loader.get_historical_grid(request.year, request.race)

    base_weather = Weather(
        condition=WeatherCondition.DRY,
        track_temperature=35.0,
        air_temperature=25.0,
        change_probability=track.weather_variability,
    )

    labels = parse_scenario_labels(request.scenarios)
    scenario_results = {}
    for idx, label in enumerate(labels):
        scenario = scenario_weather_from_label(base_weather, label)
        runner = MonteCarloRunner(
            drivers=drivers,
            cars=cars,
            track=track,
            weather=scenario.weather,
            seed=request.seed + idx * 1000,
            historical_grid=historical_grid,
        )
        result = runner.run(
            num_simulations=request.simulations,
            parallel=request.parallel,
            max_workers=request.max_workers,
        )
        scenario_results[scenario.name] = result

    payload = _summarize_scenario_results(scenario_results)
    payload["track"] = track.name
    payload["year"] = request.year
    payload["race"] = request.race
    return payload


def build_fastapi_app() -> Any:
    """Build FastAPI dashboard app.

    FastAPI import is lazy so core package remains usable without web deps.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover
        msg = "FastAPI is not installed. Install with: pip install -e '.[web]'"
        raise RuntimeError(msg) from exc

    app = FastAPI(title="F1Sim Dashboard", version="0.1")

    @app.get("/")
    def home() -> dict[str, Any]:
        return {
            "name": "f1sim-dashboard",
            "status": "ok",
            "run_endpoint": "/api/run",
            "example": {
                "year": 2025,
                "race": "Bahrain",
                "simulations": 200,
                "scenarios": "dry,light_rain,heavy_rain",
                "seed": 42,
            },
        }

    @app.post("/api/run")
    def run(payload: DashboardRunRequest) -> dict[str, Any]:
        try:
            return run_dashboard_simulation(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main() -> int:
    """Run dashboard with uvicorn."""
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        msg = "uvicorn is not installed. Install with: pip install -e '.[web]'"
        raise RuntimeError(msg) from exc

    app = build_fastapi_app()
    uvicorn.run(app, host="127.0.0.1", port=8080)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
