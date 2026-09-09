"""Current-season F1 simulator API and dashboard UI."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.resources import files
from typing import Any

from pydantic import StrictInt

from f1sim.analysis import MonteCarloRunner, parse_scenario_labels, scenario_weather_from_label
from f1sim.analysis.scenarios import validate_weather_mode
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.models import Weather, WeatherCondition
from f1sim.output.comparison import render_comparison_report
from f1sim.output.timing import finite_time
from f1sim.simulation.execution import (
    validate_race_engine,
    validate_starting_tire_ages,
    validate_starting_tires,
)
from f1sim.simulation.race import result_is_classified
from f1sim.simulation.race_points import points_for_result
from f1sim.web.capacity import RunCapacity

_LOGGER = logging.getLogger(__name__)


def _current_season() -> int:
    """Resolve the UTC season at request time, including across New Year."""

    return datetime.now(timezone.utc).year

_EXPECTED_COMPONENT_RATES: dict[str, float] = {
    "engine": 0.34,
    "gearbox": 0.22,
    "electrical": 0.18,
    "cooling": 0.14,
    "brakes": 0.12,
}

_MIN_DASHBOARD_SIMULATIONS = 10
_MAX_DASHBOARD_SIMULATIONS = 1000
_MAX_DASHBOARD_WORKERS = 16
_DEFAULT_DASHBOARD_WORKERS = min(8, os.cpu_count() or 1)
_MAX_SEED = 2**32 - 1


@dataclass
class DashboardRunRequest:
    """Input payload for dashboard simulation run."""

    year: int = field(default_factory=_current_season)
    race: str = "1"
    simulations: int = 200
    scenarios: str = "dry,light_rain"
    seed: int = 42
    qualifying_mode: str = "simulated"
    parallel: bool = True
    max_workers: int | None = None
    race_engine: str = "standard"
    starting_tires: dict[str, str] | None = None
    weather_mode: str = "evolving"
    starting_tire_ages: dict[str, StrictInt] | None = None


def _validate_dashboard_request(request: DashboardRunRequest) -> list[str]:
    """Validate resource bounds and return all scenarios before live I/O."""

    validate_weather_mode(request.weather_mode)
    validate_race_engine(request.race_engine)
    validate_starting_tires(request.starting_tires)
    validate_starting_tire_ages(request.starting_tire_ages, request.starting_tires)
    current_season = _current_season()
    if isinstance(request.year, bool) or not isinstance(request.year, int):
        raise ValueError(f"Only the live {current_season} F1 season is available.")
    if request.year != current_season:
        raise ValueError(
            f"Only the live {current_season} F1 season is available; received {request.year}."
        )
    if not isinstance(request.race, str) or not request.race.strip():
        raise ValueError("race must be a non-empty current-season race name or round")
    if len(request.race) > 160:
        raise ValueError("race must be at most 160 characters")
    if isinstance(request.simulations, bool) or not isinstance(request.simulations, int):
        raise ValueError("simulations must be an integer")
    if not _MIN_DASHBOARD_SIMULATIONS <= request.simulations <= _MAX_DASHBOARD_SIMULATIONS:
        raise ValueError(
            "simulations must be between "
            f"{_MIN_DASHBOARD_SIMULATIONS} and {_MAX_DASHBOARD_SIMULATIONS}"
        )
    if isinstance(request.seed, bool) or not isinstance(request.seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= request.seed <= _MAX_SEED:
        raise ValueError(f"seed must be between 0 and {_MAX_SEED}")
    if not isinstance(request.parallel, bool):
        raise ValueError("parallel must be a boolean")
    if request.max_workers is not None:
        if isinstance(request.max_workers, bool) or not isinstance(request.max_workers, int):
            raise ValueError("max_workers must be an integer or null")
        if not 1 <= request.max_workers <= _MAX_DASHBOARD_WORKERS:
            raise ValueError(f"max_workers must be between 1 and {_MAX_DASHBOARD_WORKERS}")
    if str(request.qualifying_mode or "simulated").strip().lower() != "simulated":
        raise ValueError("Only freshly simulated qualifying is available.")
    if not isinstance(request.scenarios, str) or len(request.scenarios) > 100:
        raise ValueError("scenarios must be a comma-separated string of at most 100 characters")

    return parse_scenario_labels(request.scenarios)


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
        "active_aero_zones": track.active_aero_zone_count,
        "active_aero_time_gain": track.total_active_aero_gain,
        "overtake_mode_detection_gap": track.overtake_mode_detection_gap,
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
        "pit_laps": (list(result.pit_laps)
                     if getattr(result, "pit_laps", None) is not None else None),
        "pit_stop_details": ([dict(stop) for stop in result.pit_stop_details]
                             if getattr(result, "pit_stop_details", None) is not None else None),
        "fastest_lap": result.fastest_lap,
        "status": result.status.value,
        "laps_completed": getattr(result, "laps_completed", None),
        "classified": result_is_classified(result),
        "race_time_limited": getattr(result, "race_time_limited", False),
        "points_awarded": points_for_result(result),
        "dnf_reason": result.dnf_reason,
        "strategy": list(result.strategy),
    }


def _serialize_quali_result(result: Any) -> dict[str, Any]:
    """Serialize one qualifying/grid result row."""
    return {
        "driver_id": result.driver_id,
        "driver_name": result.driver_name,
        "position": result.position,
        "best_time": finite_time(result.best_time),
        "q1_time": finite_time(result.q1_time),
        "q2_time": finite_time(result.q2_time),
        "q3_time": finite_time(result.q3_time),
        "eliminated_in": result.eliminated_in,
    }


def _representative_sample_index(results: Any) -> int:
    """Choose the simulated race closest to aggregate finishing positions."""

    races = getattr(results, "race_results", [])
    if len(races) <= 1:
        return 0
    driver_stats = getattr(results, "driver_stats", {})
    expected_positions = {
        driver_id: float(stats.avg_position)
        for driver_id, stats in driver_stats.items()
        if getattr(stats, "avg_position", 0) > 0
    }
    if not expected_positions:
        return 0

    def score(race: list[Any]) -> float:
        deviations = [
            abs(float(row.position) - expected_positions[row.driver_id])
            for row in race
            if row.driver_id in expected_positions
        ]
        return sum(deviations) / len(deviations) if deviations else float("inf")

    return min(range(len(races)), key=lambda index: score(races[index]))


def _serialize_sample_race(results: Any, index: int = 0) -> list[dict[str, Any]]:
    """Serialize one representative simulated race, if present."""

    races = getattr(results, "race_results", [])
    if not races:
        return []
    selected = min(max(index, 0), len(races) - 1)
    return [_serialize_race_result(row) for row in races[selected]]


def _serialize_sample_qualifying(results: Any, index: int = 0) -> list[dict[str, Any]]:
    """Serialize qualifying paired with the representative race."""

    qualifying = getattr(results, "qualifying_results", [])
    if not qualifying:
        return []
    selected = min(max(index, 0), len(qualifying) - 1)
    driver_stats = getattr(results, "driver_stats", {})
    serialized = []
    for row in qualifying[selected]:
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
    intervals = _safe_call(results, "get_probability_intervals", default={}) or {}
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
            "probability_intervals": intervals.get(driver_id),
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
    """Serialize live current-season driver and car ratings used for a run."""
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
        team_stats = [stats for stats in driver_stats.values()
                      if getattr(stats, "team_id", None) == car.team_id]
        uses_prior = bool(team_stats) and all(
            getattr(stats, "team_reliability_source", None) == "model_prior"
            and abs(getattr(stats, "team_reliability", -1) - car.reliability) < 1e-12
            for stats in team_stats
        ) and all(abs(value - car.reliability) < 1e-12
                  for value in car.component_reliability_map().values())
        cars_out.append(
            {
                "team": car.team_name,
                "team_key": _normalize_team_id(car.team_id),
                "base_pace": round(car.base_pace, 4),
                "reliability": round(car.reliability, 4),
                "reliability_source": "model_prior" if uses_prior else "provided",
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
        "source": "Jolpica + Formula1.com",
        "drivers": drivers_out,
        "cars": cars_out,
        "sample_sizes": sample_sizes,
    }


def _summarize_scenario_results(
    results_by_name: dict[str, Any],
    scenario_meta: dict[str, dict[str, float]] | None = None,
    scenario_weather: dict[str, Weather] | None = None,
) -> dict[str, Any]:
    """Build compact summary payload for UI responses."""
    summary: dict[str, Any] = {"scenarios": {}}
    scenario_meta = scenario_meta or {}
    scenario_weather = scenario_weather or {}
    for scenario_name, results in results_by_name.items():
        win_probs = _safe_call(results, "get_win_probabilities", default={}) or {}
        top_3 = list(win_probs.items())[:3]
        meta = scenario_meta.get(scenario_name, {})
        sample_index = _representative_sample_index(results)
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
            "pit_stop_statistics": _safe_call(results, "get_pit_stop_statistics", default={}) or {},
            "pit_loss_statistics": _safe_call(results, "get_pit_loss_statistics", default={}) or {},
            "strategy_statistics": _safe_call(results, "get_strategy_statistics", default={}) or {},
            "race_distance_statistics": _safe_call(
                results, "get_race_distance_statistics", default={},
            ) or {},
            "simulation_inputs": getattr(results, "input_snapshot", None),
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
            "event_rate_trials": _safe_call(results, "get_event_rate_trials", default=None),
            "simulations_per_second": meta.get("simulations_per_second"),
            "weather": _serialize_weather(scenario_weather[scenario_name])
            if scenario_name in scenario_weather
            else None,
            "sample_index": sample_index,
            "sample_race": _serialize_sample_race(results, sample_index),
            "sample_weather_history": (
                results.weather_histories[sample_index]
                if sample_index < len(getattr(results, "weather_histories", [])) else []
            ),
            "sample_qualifying": _serialize_sample_qualifying(results, sample_index),
            "race_engine": getattr(results, "race_engine", "standard"),
            "qualifying_mode": "simulated",
            "driver_statistics": _serialize_driver_statistics(results),
        }

    return summary


def run_dashboard_simulation(request: DashboardRunRequest) -> dict[str, Any]:
    """Execute one dashboard simulation bundle and return summary."""
    labels = _validate_dashboard_request(request)
    loader = _get_loader()
    round_number = loader.resolve_race_identifier(request.year, request.race)
    events = loader.list_available_events(request.year)
    canonical_race = next(
        (event["race"] for event in events if event["round"] == round_number),
        request.race,
    )

    driver_stats = loader.get_weighted_driver_stats(
        year=request.year,
        target_race=request.race,
        form_races=3,
        track_weight=0.5,
        form_weight=0.3,
        quali_weight=0.2,
    )
    # Loading the result feed first lets the calendar mark completed rounds
    # from authoritative rows before track fastest-lap calibration runs.
    track_stats = loader.get_track_stats(request.year, request.race)

    drivers = loader.create_drivers_from_stats(driver_stats)
    starting_tires = validate_starting_tires(request.starting_tires, (d.id for d in drivers))
    starting_tire_ages = validate_starting_tire_ages(
        request.starting_tire_ages, starting_tires, (d.id for d in drivers),
    )
    cars = loader.create_cars_from_stats(driver_stats)
    track = loader.create_track_from_stats(track_stats)
    base_weather = Weather(
        condition=WeatherCondition.DRY,
        track_temperature=35.0,
        air_temperature=25.0,
        change_probability=track.weather_variability,
    )

    scenario_results = {}
    scenario_meta: dict[str, dict[str, float]] = {}
    scenario_weather: dict[str, Weather] = {}
    effective_max_workers = (
        min(
            request.max_workers or _DEFAULT_DASHBOARD_WORKERS,
            request.simulations,
        )
        if request.parallel
        else None
    )

    for idx, label in enumerate(labels):
        scenario = scenario_weather_from_label(
            base_weather, label, weather_mode=request.weather_mode
        )
        scenario_weather[scenario.name] = scenario.weather
        runner = MonteCarloRunner(
            drivers=drivers,
            cars=cars,
            track=track,
            weather=scenario.weather,
            seed=request.seed + idx * 1000,
            race_engine=request.race_engine,
            **({"starting_tires": starting_tires} if starting_tires else {}),
            **({"starting_tire_ages": starting_tire_ages} if starting_tire_ages else {}),
        )
        t0 = time.perf_counter()
        result = runner.run(
            num_simulations=request.simulations,
            parallel=request.parallel,
            max_workers=effective_max_workers,
        )
        runtime = max(time.perf_counter() - t0, 1e-9)
        scenario_results[scenario.name] = result
        scenario_meta[scenario.name] = {
            "runtime_seconds": float(runtime),
            "simulations_per_second": float(request.simulations / runtime),
        }

    payload = _summarize_scenario_results(
        scenario_results,
        scenario_meta=scenario_meta,
        scenario_weather=scenario_weather,
    )
    payload["track"] = track.name
    payload["track_details"] = _serialize_track(track)
    payload["year"] = request.year
    payload["race"] = canonical_race
    payload["request"] = {
        "year": request.year,
        "race": canonical_race,
        "simulations": request.simulations,
        "scenarios": request.scenarios,
        "seed": request.seed,
        "race_engine": request.race_engine,
        "starting_tires": starting_tires,
        "starting_tire_ages": starting_tire_ages,
        "weather_mode": request.weather_mode,
        "qualifying_mode": "simulated",
        "parallel": request.parallel,
        "max_workers": effective_max_workers,
        "requested_max_workers": request.max_workers,
    }
    payload["ratings"] = _serialize_ratings_snapshot(drivers, cars, driver_stats)
    payload["provenance"] = loader.get_provenance()
    payload["comparison_report_html"] = render_comparison_report(scenario_results)
    return payload


def build_dashboard_html() -> str:
    """Return the shared polished frontend HTML."""
    frontend_path = files("f1sim.web").joinpath("static", "index.html")
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
    "rb_f1_team": "rb",
    "visa_cash_app_rb": "rb",
    "audi": "audi",
    "audi_f1_team": "audi",
    "cadillac": "cadillac",
    "cadillac_f1_team": "cadillac",
    "haas": "haas",
    "haas_f1_team": "haas",
    "moneygram_haas_f1_team": "haas",
}


def _normalize_team_id(raw_team_id: str) -> str:
    """Map current-season team names to frontend-friendly keys."""
    cleaned = raw_team_id.lower().replace(" ", "_").replace("-", "_")
    if cleaned in _TEAM_ID_NORMALIZE:
        return _TEAM_ID_NORMALIZE[cleaned]
    for key, val in _TEAM_ID_NORMALIZE.items():
        if key in cleaned or cleaned in key:
            return val
    return cleaned


def _get_loader() -> CurrentSeasonDataLoader:
    """Create an isolated live snapshot loader for one API request."""
    return CurrentSeasonDataLoader()


def build_fastapi_app() -> Any:
    """Build FastAPI dashboard app.

    FastAPI import is lazy so core package remains usable without web deps.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.responses import HTMLResponse
    except Exception as exc:  # pragma: no cover
        msg = "FastAPI is not installed. Install with: pip install -e '.[web]'"
        raise RuntimeError(msg) from exc

    app = FastAPI(title="F1Sim Dashboard", version="0.2")
    run_capacity = RunCapacity.from_environment()

    @app.middleware("http")
    async def disable_api_caching(request: Any, call_next: Any) -> Any:
        """Apply local-dashboard security and live-data cache controls."""

        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return build_dashboard_html()

    @app.get("/api/health")
    def health() -> dict[str, str | int]:
        return {
            "status": "ok",
            "season": _current_season(),
            "data_policy": "live-current-season-only",
        }

    @app.get("/api/calendar")
    def calendar(
        year: int | None = Query(default=None),
    ) -> dict[str, Any]:
        requested_year = _current_season() if year is None else year
        try:
            loader = _get_loader()
            return {
                "year": requested_year,
                "events": loader.list_available_events(requested_year),
                "provenance": loader.get_provenance(),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (CurrentSeasonDataError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            _LOGGER.exception("Unexpected calendar endpoint failure")
            raise HTTPException(status_code=500, detail="Unexpected server error") from exc

    @app.post("/api/run")
    def run(payload: DashboardRunRequest) -> dict[str, Any]:
        try:
            _validate_dashboard_request(payload)
            with run_capacity.acquire() as admitted:
                if not admitted:
                    raise HTTPException(
                        status_code=429,
                        detail="Simulation capacity is busy. Please retry shortly.",
                        headers={"Retry-After": "5"},
                    )
                return run_dashboard_simulation(payload)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (CurrentSeasonDataError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            _LOGGER.exception("Unexpected simulation endpoint failure")
            raise HTTPException(status_code=500, detail="Unexpected server error") from exc

    @app.get("/api/ratings")
    def ratings(
        year: int | None = Query(default=None),
        race: str = Query(
            ...,
            min_length=1,
            max_length=160,
            description="Current-season race name or round, e.g. 'Monaco' or '6'",
        ),
    ) -> dict[str, Any]:
        """Return ratings derived only from fresh current-season data."""
        requested_year = _current_season() if year is None else year
        try:
            loader = _get_loader()
            round_number = loader.resolve_race_identifier(requested_year, race)
            events = loader.list_available_events(requested_year)
            canonical_race = next(
                (event["race"] for event in events if event["round"] == round_number),
                race,
            )
            driver_stats = loader.get_weighted_driver_stats(
                year=requested_year,
                target_race=race,
                form_races=3,
                track_weight=0.5,
                form_weight=0.3,
                quali_weight=0.2,
            )

            if not driver_stats:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data available for {requested_year} {race}",
                )

            drivers = loader.create_drivers_from_stats(driver_stats)
            cars = loader.create_cars_from_stats(driver_stats)

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
                "year": requested_year,
                "race": canonical_race,
                "drivers": drivers_out,
                "car_pace": car_pace,
                "sample_sizes": sample_sizes,
                "source": "Jolpica + Formula1.com",
                "provenance": loader.get_provenance(),
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (CurrentSeasonDataError, OSError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            _LOGGER.exception("Unexpected ratings endpoint failure")
            raise HTTPException(status_code=500, detail="Unexpected server error") from exc

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
