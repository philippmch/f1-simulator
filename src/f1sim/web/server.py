"""F1 Simulator web server – API and dashboard UI."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f1sim.analysis import MonteCarloRunner, parse_scenario_labels, scenario_weather_from_label
from f1sim.data import HistoricalDataLoader
from f1sim.models import Weather, WeatherCondition
from f1sim.output import Exporter

_EXPECTED_COMPONENT_RATES: dict[str, float] = {
    "engine": 0.34,
    "gearbox": 0.22,
    "electrical": 0.18,
    "cooling": 0.14,
    "brakes": 0.12,
}


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


def _safe_call(results: Any, method_name: str, *args: Any, default: Any = None) -> Any:
    """Call a result helper when available, otherwise return a default."""
    method = getattr(results, method_name, None)
    if callable(method):
        return method(*args)
    return default


def _serialize_track(track: Any) -> dict[str, Any]:
    """Serialize track data for frontend rendering."""
    return {
        "id": track.id,
        "name": track.name,
        "country": track.country,
        "total_laps": track.total_laps,
        "base_lap_time": track.base_lap_time,
        "pit_lane_delta": track.pit_lane_delta,
        "overtake_difficulty": track.overtake_difficulty,
        "tire_stress": track.tire_stress,
        "safety_car_probability": track.safety_car_probability,
        "weather_variability": track.weather_variability,
        "drs_zones": len(track.drs_zones),
    }


def _serialize_weather(weather: Weather) -> dict[str, Any]:
    """Serialize scenario weather metadata for the UI."""
    return {
        "condition": weather.condition.value,
        "track_temperature": weather.track_temperature,
        "air_temperature": weather.air_temperature,
        "humidity": weather.humidity,
        "rain_intensity": weather.rain_intensity,
        "track_wetness": weather.track_wetness,
        "change_probability": weather.change_probability,
    }


def _serialize_race_result(result: Any) -> dict[str, Any]:
    """Serialize one race result row."""
    return {
        "driver_id": result.driver_id,
        "driver_name": result.driver_name,
        "team": result.team,
        "team_key": _normalize_team_id(result.team),
        "position": result.position,
        "total_time": result.total_time,
        "gap_to_leader": result.gap_to_leader,
        "pit_stops": result.pit_stops,
        "fastest_lap": result.fastest_lap,
        "status": result.status.value,
        "dnf_reason": result.dnf_reason,
        "strategy": list(result.strategy),
    }


def _serialize_quali_result(result: Any) -> dict[str, Any]:
    """Serialize one qualifying/grid result row."""
    return {
        "driver_id": result.driver_id,
        "driver_name": result.driver_name,
        "position": result.position,
        "best_time": result.best_time,
        "q1_time": result.q1_time,
        "q2_time": result.q2_time,
        "q3_time": result.q3_time,
        "eliminated_in": result.eliminated_in,
    }


def _serialize_sample_race(results: Any) -> list[dict[str, Any]]:
    """Serialize the first simulated race, if present."""
    races = getattr(results, "race_results", [])
    if not races:
        return []
    return [_serialize_race_result(row) for row in races[0]]


def _serialize_sample_qualifying(results: Any) -> list[dict[str, Any]]:
    """Serialize the first qualifying/grid result, if present."""
    qualifying = getattr(results, "qualifying_results", [])
    if not qualifying:
        return []
    driver_stats = getattr(results, "driver_stats", {})
    serialized = []
    for row in qualifying[0]:
        payload = _serialize_quali_result(row)
        team = getattr(driver_stats.get(row.driver_id), "team", "Unknown")
        payload["team"] = team
        payload["team_key"] = _normalize_team_id(team)
        serialized.append(payload)
    return serialized


def _serialize_driver_statistics(results: Any) -> dict[str, Any]:
    """Serialize per-driver backend statistics for the frontend."""
    driver_stats = getattr(results, "driver_stats", {})
    if not isinstance(driver_stats, dict):
        return {}

    top_5 = _safe_call(results, "get_top_n_finish_probabilities", 5, default={}) or {}
    top_10 = _safe_call(results, "get_top_n_finish_probabilities", 10, default={}) or {}
    ordered_ids = list(
        (_safe_call(results, "get_win_probabilities", default={}) or driver_stats).keys()
    )

    serialized: dict[str, Any] = {}
    for driver_id in ordered_ids:
        stats = driver_stats.get(driver_id)
        if stats is None:
            continue

        serialized[driver_id] = {
            "driver_name": stats.driver_name,
            "team": stats.team,
            "team_key": _normalize_team_id(stats.team),
            "wins": stats.wins,
            "win_rate": stats.win_rate,
            "podiums": stats.podiums,
            "podium_rate": stats.podium_rate,
            "points_finishes": stats.points_finishes,
            "dnfs": stats.dnfs,
            "dnf_rate": stats.dnf_rate,
            "total_points": stats.total_points,
            "avg_position": stats.avg_position,
            "avg_qualifying": stats.avg_qualifying,
            "best_position": stats.best_position,
            "worst_position": stats.worst_position,
            "position_distribution": (
                _safe_call(results, "get_position_distribution", driver_id, default={}) or {}
            ),
            "position_percentiles": (
                _safe_call(results, "get_position_percentiles", driver_id, default={}) or {}
            ),
            "top_5_finish_probability": top_5.get(driver_id, 0.0),
            "top_10_finish_probability": top_10.get(driver_id, 0.0),
        }

    return serialized


def _serialize_ratings_snapshot(
    drivers: list[Any],
    cars: dict[str, Any],
    driver_stats: dict[str, Any],
) -> dict[str, Any]:
    """Serialize FastF1-derived driver and car ratings used for a run."""
    sample_sizes = {driver_id: stats.sample_size for driver_id, stats in driver_stats.items()}

    drivers_out = []
    for driver in drivers:
        car = cars.get(driver.team_id)
        drivers_out.append(
            {
                "id": driver.id,
                "name": driver.name,
                "team": car.team_name if car else driver.team_id,
                "team_key": _normalize_team_id(driver.team_id),
                "skill": round(driver.skill_rating, 4),
                "consistency": round(driver.consistency, 4),
                "wet_skill": round(driver.wet_skill_modifier, 4),
                "overtaking": round(driver.overtaking_skill, 4),
                "tire_management": round(driver.tire_management, 4),
                "sample_size": sample_sizes.get(driver.id),
            }
        )

    cars_out = []
    for car in cars.values():
        cars_out.append(
            {
                "team": car.team_name,
                "team_key": _normalize_team_id(car.team_id),
                "base_pace": round(car.base_pace, 4),
                "reliability": round(car.reliability, 4),
                "engine_reliability": round(car.engine_reliability, 4),
                "gearbox_reliability": round(car.gearbox_reliability, 4),
                "brakes_reliability": round(car.brakes_reliability, 4),
                "electrical_reliability": round(car.electrical_reliability, 4),
                "cooling_reliability": round(car.cooling_reliability, 4),
                "wet_performance": round(car.wet_performance, 4),
                "tire_degradation_factor": round(car.tire_degradation_factor, 4),
                "pit_stop_avg": round(car.pit_stop_avg, 4),
            }
        )

    drivers_out.sort(key=lambda row: row["skill"], reverse=True)
    cars_out.sort(key=lambda row: row["base_pace"], reverse=True)

    return {
        "source": "fastf1",
        "drivers": drivers_out,
        "cars": cars_out,
        "sample_sizes": sample_sizes,
    }


def _summarize_scenario_results(
    results_by_name: dict[str, Any],
    scenario_meta: dict[str, dict[str, float]] | None = None,
    scenario_weather: dict[str, Weather] | None = None,
    artifacts_by_name: dict[str, dict[str, str]] | None = None,
    historical_grid_used: bool = False,
) -> dict[str, Any]:
    """Build compact summary payload for UI responses."""
    summary: dict[str, Any] = {"scenarios": {}}
    scenario_meta = scenario_meta or {}
    scenario_weather = scenario_weather or {}
    artifacts_by_name = artifacts_by_name or {}
    for scenario_name, results in results_by_name.items():
        win_probs = _safe_call(results, "get_win_probabilities", default={}) or {}
        top_3 = list(win_probs.items())[:3]
        meta = scenario_meta.get(scenario_name, {})
        summary["scenarios"][scenario_name] = {
            "num_simulations": results.num_simulations,
            "seed": results.seed,
            "top3_win_probabilities": top_3,
            "win_probabilities": list(win_probs.items()),
            "top_5_finish_probabilities": list(
                (
                    _safe_call(results, "get_top_n_finish_probabilities", 5, default={}) or {}
                ).items()
            ),
            "top_10_finish_probabilities": list(
                (
                    _safe_call(results, "get_top_n_finish_probabilities", 10, default={}) or {}
                ).items()
            ),
            "championship_projection": list(
                (_safe_call(results, "get_championship_projection", default={}) or {}).items()
            ),
            "event_rates": _safe_call(results, "get_event_rates", default={}) or {},
            "team_projection": _safe_call(
                results,
                "get_team_championship_projection",
                default={},
            )
            or {},
            "mechanical_failure_breakdown": (
                getattr(getattr(results, "event_stats", None), "mechanical_failure_breakdown", {})
                or {}
            ),
            "mechanical_failure_component_rates": _safe_call(
                results,
                "get_mechanical_failure_component_rates",
                default={},
            )
            or {},
            "mechanical_tuning_suggestions": _safe_call(
                results,
                "get_mechanical_tuning_suggestions",
                _EXPECTED_COMPONENT_RATES,
                default={},
            )
            or {},
            "reliability_adjustment_recommendations": _safe_call(
                results,
                "get_reliability_adjustment_recommendations",
                _EXPECTED_COMPONENT_RATES,
                default={},
            )
            or {},
            "runtime_seconds": meta.get("runtime_seconds"),
            "simulations_per_second": meta.get("simulations_per_second"),
            "weather": _serialize_weather(scenario_weather[scenario_name])
            if scenario_name in scenario_weather
            else None,
            "sample_race": _serialize_sample_race(results),
            "sample_qualifying": _serialize_sample_qualifying(results),
            "qualifying_mode": "historical_grid" if historical_grid_used else "simulated",
            "driver_statistics": _serialize_driver_statistics(results),
            "artifacts": artifacts_by_name.get(scenario_name, {}),
        }

    return summary


def _read_run_history(output_dir: str | Path = "output") -> list[dict[str, Any]]:
    """Read exporter run history file if present."""
    history_path = Path(output_dir) / ".run_history.json"
    if not history_path.exists():
        return []

    data = json.loads(history_path.read_text())
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def run_dashboard_simulation(request: DashboardRunRequest) -> dict[str, Any]:
    """Execute one dashboard simulation bundle and return summary."""
    loader = _get_loader()

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
    scenario_meta: dict[str, dict[str, float]] = {}
    scenario_weather: dict[str, Weather] = {}
    artifacts_by_name: dict[str, dict[str, str]] = {}
    exporter = Exporter(output_dir="output")

    for idx, label in enumerate(labels):
        scenario = scenario_weather_from_label(base_weather, label)
        scenario_weather[scenario.name] = scenario.weather
        runner = MonteCarloRunner(
            drivers=drivers,
            cars=cars,
            track=track,
            weather=scenario.weather,
            seed=request.seed + idx * 1000,
            historical_grid=historical_grid,
        )
        t0 = time.perf_counter()
        result = runner.run(
            num_simulations=request.simulations,
            parallel=request.parallel,
            max_workers=request.max_workers,
        )
        runtime = max(time.perf_counter() - t0, 1e-9)
        scenario_results[scenario.name] = result
        scenario_meta[scenario.name] = {
            "runtime_seconds": float(runtime),
            "simulations_per_second": float(request.simulations / runtime),
        }
        files = exporter.export_all(result, prefix=f"{request.year}_{track.id}_{scenario.name}")
        artifacts_by_name[scenario.name] = {key: path.name for key, path in files.items()}

    payload = _summarize_scenario_results(
        scenario_results,
        scenario_meta=scenario_meta,
        scenario_weather=scenario_weather,
        artifacts_by_name=artifacts_by_name,
        historical_grid_used=bool(historical_grid),
    )
    payload["track"] = track.name
    payload["track_details"] = _serialize_track(track)
    payload["year"] = request.year
    payload["race"] = request.race
    payload["historical_grid_used"] = bool(historical_grid)
    payload["request"] = {
        "year": request.year,
        "race": request.race,
        "simulations": request.simulations,
        "scenarios": request.scenarios,
        "seed": request.seed,
        "parallel": request.parallel,
        "max_workers": request.max_workers,
    }
    payload["ratings"] = _serialize_ratings_snapshot(drivers, cars, driver_stats)
    return payload


def build_dashboard_html() -> str:
    """Return the shared polished frontend HTML."""
    frontend_path = Path(__file__).resolve().parents[3] / "frontend" / "index.html"
    return frontend_path.read_text(encoding="utf-8")


_TEAM_ID_NORMALIZE: dict[str, str] = {
    "red_bull_racing": "red_bull",
    "red_bull": "red_bull",
    "scuderia_ferrari": "ferrari",
    "ferrari": "ferrari",
    "mclaren": "mclaren",
    "mercedes": "mercedes",
    "aston_martin_aramco": "aston_martin",
    "aston_martin": "aston_martin",
    "alpine": "alpine",
    "bwt_alpine_f1_team": "alpine",
    "williams": "williams",
    "williams_racing": "williams",
    "rb": "rb",
    "racing_bulls": "rb",
    "visa_cash_app_rb": "rb",
    "sauber": "sauber",
    "kick_sauber": "sauber",
    "stake_f1_team_kick_sauber": "sauber",
    "haas": "haas",
    "haas_f1_team": "haas",
    "moneygram_haas_f1_team": "haas",
}


def _normalize_team_id(raw_team_id: str) -> str:
    """Map FastF1 team IDs to frontend-friendly keys."""
    cleaned = raw_team_id.lower().replace(" ", "_").replace("-", "_")
    if cleaned in _TEAM_ID_NORMALIZE:
        return _TEAM_ID_NORMALIZE[cleaned]
    for key, val in _TEAM_ID_NORMALIZE.items():
        if key in cleaned or cleaned in key:
            return val
    return cleaned


_shared_loader: HistoricalDataLoader | None = None


def _get_loader() -> HistoricalDataLoader:
    """Get a shared HistoricalDataLoader instance for cross-request caching."""
    global _shared_loader
    if _shared_loader is None:
        _shared_loader = HistoricalDataLoader(cache_dir="data/cache")
    return _shared_loader


def build_fastapi_app() -> Any:
    """Build FastAPI dashboard app.

    FastAPI import is lazy so core package remains usable without web deps.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles
    except Exception as exc:  # pragma: no cover
        msg = "FastAPI is not installed. Install with: pip install -e '.[web]'"
        raise RuntimeError(msg) from exc

    app = FastAPI(title="F1Sim Dashboard", version="0.1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    Path("output").mkdir(parents=True, exist_ok=True)
    app.mount("/output", StaticFiles(directory="output"), name="output")

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return build_dashboard_html()

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/runs")
    def runs() -> dict[str, Any]:
        return {"runs": _read_run_history()[:20]}

    @app.post("/api/run")
    def run(payload: DashboardRunRequest) -> dict[str, Any]:
        try:
            return run_dashboard_simulation(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/ratings")
    def ratings(
        year: int = Query(default=2025, ge=2020, le=2030),
        race: str = Query(..., description="Race name, e.g. 'Bahrain', 'Monaco'"),
    ) -> dict[str, Any]:
        """Return driver skill ratings and car pace derived from real FastF1 data."""
        try:
            import numpy as np

            loader = _get_loader()
            driver_stats = loader.get_weighted_driver_stats(
                year=year,
                target_race=race,
                form_races=3,
                track_weight=0.5,
                form_weight=0.3,
                quali_weight=0.2,
            )

            if not driver_stats:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data available for {year} {race}",
                )

            # Use a fixed seed so wet_skill_modifier is deterministic per request
            rng_state = np.random.get_state()
            np.random.seed(hash((year, race)) % 2**31)
            drivers = loader.create_drivers_from_stats(driver_stats)
            cars = loader.create_cars_from_stats(driver_stats)
            np.random.set_state(rng_state)

            drivers_out = []
            for d in drivers:
                drivers_out.append({
                    "id": d.id,
                    "name": d.name,
                    "team": _normalize_team_id(d.team_id),
                    "skill": round(d.skill_rating, 4),
                    "consistency": round(d.consistency, 4),
                    "wet_skill": round(d.wet_skill_modifier, 4),
                    "overtaking": round(d.overtaking_skill, 4),
                    "tire_management": round(d.tire_management, 4),
                })

            car_pace: dict[str, float] = {}
            for car in cars.values():
                team_key = _normalize_team_id(car.team_id)
                car_pace[team_key] = round(car.base_pace, 4)

            sample_sizes = {
                sid: s.sample_size for sid, s in driver_stats.items()
            }

            return {
                "year": year,
                "race": race,
                "drivers": drivers_out,
                "car_pace": car_pace,
                "sample_sizes": sample_sizes,
                "source": "fastf1",
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

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
