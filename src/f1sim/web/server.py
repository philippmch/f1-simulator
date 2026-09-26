"""Current-season F1 simulator API and dashboard UI."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.resources import files
from threading import Event
from typing import Any

from pydantic import StrictBool, StrictInt, StrictStr

from f1sim.analysis import MonteCarloRunner, parse_scenario_labels, scenario_weather_from_label
from f1sim.analysis.cancellation import SimulationCancelled
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.scenarios import validate_weather_mode
from f1sim.data import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.models import Weather, WeatherCondition
from f1sim.output.comparison import render_comparison_report
from f1sim.output.timing import finite_time, suspension_statistics
from f1sim.simulation.execution import (
    validate_race_engine,
    validate_starting_tire_ages,
    validate_starting_tires,
)
from f1sim.simulation.race import get_race_suspension_seconds, result_is_classified
from f1sim.simulation.race_points import points_for_result
from f1sim.simulation.randomness import DEFAULT_RNG_POLICY, validate_rng_policy
from f1sim.simulation.tire_inventory import validate_tire_inventory
from f1sim.simulation.warmup import validate_tire_warmup
from f1sim.web.capacity import RunCapacity

_LOGGER = logging.getLogger(__name__)


def _current_season() -> int:
    """Resolve the UTC season at request time, including across New Year."""

    return datetime.now(timezone.utc).year

_MIN_DASHBOARD_SIMULATIONS = 10
_MAX_DASHBOARD_SIMULATIONS = 1000
_MAX_DASHBOARD_COMPARISON_SIMULATIONS = 500
_MAX_DASHBOARD_WORKERS = 16
_DEFAULT_DASHBOARD_WORKERS = min(8, os.cpu_count() or 1)
_MAX_SEED = 2**32 - 1


class _RunCapacityBusy(RuntimeError):
    """Signal that a dashboard request could not acquire a run slot."""


_CLIENT_DISCONNECTED_STATUS = 499
_CLIENT_DISCONNECTED_DETAIL = "Client disconnected before the simulation completed."


@dataclass
class DashboardRunRequest:
    """Input payload for dashboard simulation run."""

    year: StrictInt = field(default_factory=_current_season)
    race: str = "1"
    simulations: StrictInt = 200
    scenarios: str = "dry,light_rain"
    seed: StrictInt = 42
    qualifying_mode: str = "simulated"
    parallel: StrictBool = True
    max_workers: StrictInt | None = None
    race_engine: str = "standard"
    starting_tires: dict[str, str] | None = None
    weather_mode: str = "evolving"
    starting_tire_ages: dict[str, StrictInt] | None = None
    tire_inventory: Any = None
    pit_plans: Any = None
    compare_automatic: StrictBool = False
    rng_policy: StrictStr = DEFAULT_RNG_POLICY
    tire_warmup: Any = None


def _validate_dashboard_request(request: DashboardRunRequest) -> list[str]:
    """Validate resource bounds and return all scenarios before live I/O."""

    validate_weather_mode(request.weather_mode)
    validate_race_engine(request.race_engine)
    validate_rng_policy(request.rng_policy)
    validate_tire_warmup(request.tire_warmup)
    validate_starting_tires(request.starting_tires)
    validate_starting_tire_ages(request.starting_tire_ages, request.starting_tires)
    validate_tire_inventory(request.tire_inventory, request.starting_tires,
                            request.starting_tire_ages)
    from f1sim.simulation.pit_plans import validate_pit_plans

    # Validate shape and canonical compounds before any live data request.  The
    # roster, race distance and finite pool are checked again once loaded below.
    validate_pit_plans(request.pit_plans)
    if not isinstance(request.compare_automatic, bool):
        raise ValueError("compare_automatic must be a boolean")
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
    if request.compare_automatic:
        if not isinstance(request.pit_plans, dict) or not request.pit_plans:
            raise ValueError(
                "compare_automatic requires a nonempty pit_plans mapping"
            )
        if request.simulations > _MAX_DASHBOARD_COMPARISON_SIMULATIONS:
            raise ValueError(
                "compare_automatic simulations must be at most "
                f"{_MAX_DASHBOARD_COMPARISON_SIMULATIONS}"
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
        "race_suspension_seconds": get_race_suspension_seconds([result]),
        "gap_to_leader": result.gap_to_leader,
        "pit_stops": result.pit_stops,
        "pit_laps": (list(result.pit_laps)
                     if getattr(result, "pit_laps", None) is not None else None),
        "pit_stop_details": ([dict(stop) for stop in result.pit_stop_details]
                             if getattr(result, "pit_stop_details", None) is not None else None),
        "tire_set_history": ([dict(row) for row in result.tire_set_history]
                             if getattr(result, "tire_set_history", None) is not None else None),
        "tire_inventory": ([dict(row) for row in result.tire_inventory]
                           if getattr(result, "tire_inventory", None) is not None else None),
        "fastest_lap": result.fastest_lap,
        "status": result.status.value,
        "laps_completed": getattr(result, "laps_completed", None),
        "classified": result_is_classified(result),
        "race_time_limited": getattr(result, "race_time_limited", False),
        "points_awarded": points_for_result(result),
        "dnf_reason": result.dnf_reason,
        "strategy": list(result.strategy),
        "pit_plan_history": (
            [dict(record) for record in result.pit_plan_history]
            if getattr(result, "pit_plan_history", None) is not None else None
        ),
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
            "pit_decision_statistics": _safe_call(
                results, "get_pit_decision_statistics", default={},
            ) or {},
            "strategy_statistics": _safe_call(results, "get_strategy_statistics", default={}) or {},
            "pit_plan_statistics": _safe_call(
                results, "get_pit_plan_statistics", default={}
            ) or {},
            "race_distance_statistics": _safe_call(
                results, "get_race_distance_statistics", default={},
            ) or {},
            "suspension_statistics": suspension_statistics(results),
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
            # Keep compatibility fields in the dashboard payload, but do not
            # infer a target split or recommend changes from simulated outcomes.
            "mechanical_tuning_suggestions": {},
            "reliability_adjustment_recommendations": {},
            "runtime_seconds": meta.get("runtime_seconds"),
            "event_rate_trials": _safe_call(results, "get_event_rate_trials", default=None),
            "simulations_per_second": meta.get("simulations_per_second"),
            "weather": _serialize_weather(scenario_weather[scenario_name])
            if scenario_name in scenario_weather
            else None,
            "sample_index": sample_index,
            "sample_race": _serialize_sample_race(results, sample_index),
            "sample_race_suspension_seconds": get_race_suspension_seconds(
                getattr(results, "race_results", [])[min(
                    max(sample_index, 0), len(getattr(results, "race_results", [])) - 1
                )]
                if getattr(results, "race_results", []) else []
            ),
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


def _check_dashboard_cancellation(
    cancel_requested: Callable[[], bool] | None,
) -> None:
    """Stop dashboard work at a cooperative boundary when its client is gone."""

    if cancel_requested is None or not cancel_requested():
        return
    raise SimulationCancelled("Dashboard client disconnected during simulation.")


def _dashboard_runner(
    *,
    drivers: list[Any],
    cars: dict[str, Any],
    track: Any,
    weather: Weather,
    seed: int,
    request: DashboardRunRequest,
    tire_inventory: dict[str, list[dict]] | None,
    starting_tires: dict[str, str],
    starting_tire_ages: dict[str, int],
    pit_plans: dict[str, list[dict]] | None,
    copy_inputs: bool,
) -> MonteCarloRunner:
    """Create one isolated dashboard runner for a custom or reference run.

    Comparison alternatives execute against the same loaded provider snapshot.
    When requested, copying the mutable model/configuration inputs at this
    boundary keeps a runner or test double from leaking changes from one
    alternative into the other while preserving identical seeds and settings.
    Ordinary dashboard runs retain their existing object flow.
    """

    copy_value = deepcopy if copy_inputs else lambda value: value
    kwargs: dict[str, Any] = {
        "drivers": copy_value(drivers),
        "cars": copy_value(cars),
        "track": copy_value(track),
        "weather": copy_value(weather),
        "seed": seed,
        "race_engine": request.race_engine,
        "rng_policy": request.rng_policy,
    }
    tire_warmup = validate_tire_warmup(request.tire_warmup)
    if tire_warmup:
        kwargs["tire_warmup"] = tire_warmup
    if tire_inventory:
        kwargs["tire_inventory"] = copy_value(tire_inventory)
    if starting_tires:
        kwargs["starting_tires"] = copy_value(starting_tires)
    if starting_tire_ages:
        kwargs["starting_tire_ages"] = copy_value(starting_tire_ages)
    if pit_plans:
        kwargs["pit_plans"] = copy_value(pit_plans)
    return MonteCarloRunner(**kwargs)


def _dashboard_request_metadata(
    request: DashboardRunRequest,
    *,
    canonical_race: str,
    tire_inventory: dict[str, list[dict]] | None,
    starting_tires: dict[str, str],
    starting_tire_ages: dict[str, int],
    pit_plans: dict[str, list[dict]],
    effective_max_workers: int | None,
    compare_automatic: bool,
) -> dict[str, Any]:
    """Serialize the replayable request settings for one dashboard variant."""

    return {
        "year": request.year,
        "race": canonical_race,
        "simulations": request.simulations,
        "scenarios": request.scenarios,
        "seed": request.seed,
        "race_engine": request.race_engine,
        "rng_policy": request.rng_policy,
        **({"tire_warmup": warmup, "tire_warmup_policy": "post_fit_first_lap_v1"}
           if (warmup := validate_tire_warmup(request.tire_warmup)) else {}),
        "tire_inventory": deepcopy(tire_inventory) if tire_inventory else {},
        "starting_tires": deepcopy(starting_tires),
        "starting_tire_ages": deepcopy(starting_tire_ages),
        "pit_plans": deepcopy(pit_plans),
        "weather_mode": request.weather_mode,
        "qualifying_mode": "simulated",
        "parallel": request.parallel,
        "max_workers": effective_max_workers,
        "requested_max_workers": request.max_workers,
        "compare_automatic": compare_automatic,
    }


def run_dashboard_simulation(
    request: DashboardRunRequest,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute one dashboard simulation bundle and return summary.

    Cancellation is cooperative: the callback is checked between live-data
    phases, scenarios, and summary sections. The Monte Carlo runner receives
    it only when a caller supplies one, preserving the direct-call contract
    for integrations and tests that provide a runner with the original
    ``run`` signature.
    """
    labels = _validate_dashboard_request(request)
    _check_dashboard_cancellation(cancel_requested)
    loader = _get_loader()
    _check_dashboard_cancellation(cancel_requested)
    round_number = loader.resolve_race_identifier(request.year, request.race)
    _check_dashboard_cancellation(cancel_requested)
    events = loader.list_available_events(request.year)
    _check_dashboard_cancellation(cancel_requested)
    canonical_race = next(
        (event["race"] for event in events if event["round"] == round_number),
        request.race,
    )

    _check_dashboard_cancellation(cancel_requested)
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
    _check_dashboard_cancellation(cancel_requested)
    track_stats = loader.get_track_stats(request.year, request.race)

    _check_dashboard_cancellation(cancel_requested)
    drivers = loader.create_drivers_from_stats(driver_stats)
    tire_inventory = validate_tire_inventory(
        request.tire_inventory, request.starting_tires, request.starting_tire_ages,
        (d.id for d in drivers),
    )
    starting_tires = validate_starting_tires(request.starting_tires, (d.id for d in drivers))
    starting_tire_ages = validate_starting_tire_ages(
        request.starting_tire_ages, starting_tires, (d.id for d in drivers),
    )
    cars = loader.create_cars_from_stats(driver_stats)
    _check_dashboard_cancellation(cancel_requested)
    track = loader.create_track_from_stats(track_stats)
    from f1sim.simulation.pit_plans import validate_pit_plans

    pit_plans = validate_pit_plans(
        request.pit_plans,
        driver_ids=(driver.id for driver in drivers),
        total_laps=track.total_laps,
        tire_inventory=tire_inventory,
    )
    base_weather = Weather(
        condition=WeatherCondition.DRY,
        track_temperature=35.0,
        air_temperature=25.0,
        change_probability=track.weather_variability,
    )

    scenario_results: dict[str, Any] = {}
    scenario_meta: dict[str, dict[str, float]] = {}
    scenario_weather: dict[str, Weather] = {}
    automatic_results: dict[str, Any] = {}
    automatic_meta: dict[str, dict[str, float]] = {}
    effective_max_workers = (
        min(
            request.max_workers or _DEFAULT_DASHBOARD_WORKERS,
            request.simulations,
        )
        if request.parallel
        else None
    )

    for idx, label in enumerate(labels):
        _check_dashboard_cancellation(cancel_requested)
        scenario = scenario_weather_from_label(
            base_weather, label, weather_mode=request.weather_mode
        )
        scenario_weather[scenario.name] = scenario.weather
        scenario_seed = request.seed + idx * 1000
        runner = _dashboard_runner(
            drivers=drivers,
            cars=cars,
            track=track,
            weather=scenario.weather,
            seed=scenario_seed,
            request=request,
            tire_inventory=tire_inventory,
            starting_tires=starting_tires,
            starting_tire_ages=starting_tire_ages,
            pit_plans=pit_plans,
            copy_inputs=request.compare_automatic,
        )
        t0 = time.perf_counter()
        run_kwargs: dict[str, Any] = {
            "num_simulations": request.simulations,
            "parallel": request.parallel,
            "max_workers": effective_max_workers,
        }
        if cancel_requested is not None:
            run_kwargs["cancel_requested"] = cancel_requested
        result = runner.run(**run_kwargs)
        runtime = max(time.perf_counter() - t0, 1e-9)
        scenario_results[scenario.name] = result
        scenario_meta[scenario.name] = {
            "runtime_seconds": float(runtime),
            "simulations_per_second": float(request.simulations / runtime),
        }
        _check_dashboard_cancellation(cancel_requested)

        if request.compare_automatic:
            # Keep the reference fully automatic, including for drivers whose
            # submitted plan is an explicit empty list.  It receives the same
            # loaded model/weather snapshot and seed as the custom run.
            reference_runner = _dashboard_runner(
                drivers=drivers,
                cars=cars,
                track=track,
                weather=scenario.weather,
                seed=scenario_seed,
                request=request,
                tire_inventory=tire_inventory,
                starting_tires=starting_tires,
                starting_tire_ages=starting_tire_ages,
                pit_plans=None,
                copy_inputs=True,
            )
            _check_dashboard_cancellation(cancel_requested)
            reference_t0 = time.perf_counter()
            reference_result = reference_runner.run(**run_kwargs)
            reference_runtime = max(time.perf_counter() - reference_t0, 1e-9)
            _check_dashboard_cancellation(cancel_requested)
            automatic_results[scenario.name] = reference_result
            automatic_meta[scenario.name] = {
                "runtime_seconds": float(reference_runtime),
                "simulations_per_second": float(request.simulations / reference_runtime),
            }

    _check_dashboard_cancellation(cancel_requested)
    payload = _summarize_scenario_results(
        scenario_results,
        scenario_meta=scenario_meta,
        scenario_weather=scenario_weather,
    )
    _check_dashboard_cancellation(cancel_requested)
    payload["track"] = track.name
    payload["track_details"] = _serialize_track(track)
    payload["year"] = request.year
    payload["race"] = canonical_race
    payload["request"] = _dashboard_request_metadata(
        request,
        canonical_race=canonical_race,
        tire_inventory=tire_inventory,
        starting_tires=starting_tires,
        starting_tire_ages=starting_tire_ages,
        pit_plans=pit_plans,
        effective_max_workers=effective_max_workers,
        compare_automatic=request.compare_automatic,
    )
    _check_dashboard_cancellation(cancel_requested)
    ratings = _serialize_ratings_snapshot(drivers, cars, driver_stats)
    payload["ratings"] = ratings
    _check_dashboard_cancellation(cancel_requested)
    provenance = loader.get_provenance()
    payload["provenance"] = provenance
    _check_dashboard_cancellation(cancel_requested)
    payload["comparison_report_html"] = render_comparison_report(scenario_results)
    _check_dashboard_cancellation(cancel_requested)

    if request.compare_automatic:
        _check_dashboard_cancellation(cancel_requested)
        reference_payload = _summarize_scenario_results(
            automatic_results,
            scenario_meta=automatic_meta,
            scenario_weather=scenario_weather,
        )
        _check_dashboard_cancellation(cancel_requested)
        reference_payload["track"] = track.name
        reference_payload["track_details"] = _serialize_track(track)
        reference_payload["year"] = request.year
        reference_payload["race"] = canonical_race
        reference_payload["request"] = _dashboard_request_metadata(
            request,
            canonical_race=canonical_race,
            tire_inventory=tire_inventory,
            starting_tires=starting_tires,
            starting_tire_ages=starting_tire_ages,
            pit_plans={},
            effective_max_workers=effective_max_workers,
            compare_automatic=False,
        )
        reference_payload["ratings"] = deepcopy(ratings)
        reference_payload["provenance"] = deepcopy(provenance)
        payload["automatic_reference"] = reference_payload

        comparisons: dict[str, Any] = {}
        for scenario_name in scenario_results:
            _check_dashboard_cancellation(cancel_requested)
            comparisons[scenario_name] = paired_comparison_statistics(
                {
                    "automatic": automatic_results[scenario_name],
                    "custom": scenario_results[scenario_name],
                },
                "automatic",
            )
            _check_dashboard_cancellation(cancel_requested)
        payload["strategy_comparisons"] = comparisons

        strategy_reports: dict[str, str] = {}
        for scenario_name in scenario_results:
            _check_dashboard_cancellation(cancel_requested)
            strategy_reports[scenario_name] = render_comparison_report(
                {
                    "automatic": automatic_results[scenario_name],
                    "custom": scenario_results[scenario_name],
                },
                reference_scenario="automatic",
            )
            _check_dashboard_cancellation(cancel_requested)
        payload["strategy_comparison_reports"] = strategy_reports

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


def _run_dashboard_worker(
    run_capacity: RunCapacity,
    payload: DashboardRunRequest,
    cancel_event: Event,
) -> dict[str, Any]:
    """Run one complete dashboard request while holding its capacity slot."""

    with run_capacity.acquire() as admitted:
        if not admitted:
            raise _RunCapacityBusy
        return run_dashboard_simulation(payload, cancel_requested=cancel_event.is_set)


async def _watch_dashboard_disconnect(request: Any, cancel_event: Event) -> bool:
    """Set cooperative cancellation when the ASGI client disconnects."""

    while not cancel_event.is_set():
        try:
            if await request.is_disconnected():
                cancel_event.set()
                return True
        except asyncio.CancelledError:
            return False
        except Exception:  # pragma: no cover - defensive against custom ASGI receive
            _LOGGER.debug("Unable to poll dashboard request disconnect", exc_info=True)
        await asyncio.sleep(0.1)
    return False


async def _drain_dashboard_worker(worker_task: asyncio.Task[Any]) -> None:
    """Wait for a worker despite repeated endpoint task cancellation."""

    while not worker_task.done():
        try:
            await asyncio.shield(worker_task)
        except asyncio.CancelledError:
            # A second transport cancellation must not abandon the thread.
            continue
        except (SimulationCancelled, _RunCapacityBusy):
            return
        except Exception:
            _LOGGER.debug(
                "Dashboard worker failed while the client was disconnected",
                exc_info=True,
            )
            return

    try:
        worker_task.result()
    except (SimulationCancelled, _RunCapacityBusy, asyncio.CancelledError):
        pass
    except Exception:
        _LOGGER.debug(
            "Dashboard worker failed while the client was disconnected",
            exc_info=True,
        )


async def _stop_dashboard_disconnect_watcher(
    disconnect_task: asyncio.Task[Any],
    cancel_event: Event,
) -> None:
    """Stop and observe the disconnect watcher without leaking its task."""

    cancel_event.set()
    disconnect_task.cancel()
    while not disconnect_task.done():
        try:
            await asyncio.shield(disconnect_task)
        except asyncio.CancelledError:
            continue
        except Exception:
            _LOGGER.debug("Dashboard disconnect watcher failed", exc_info=True)
            return
    try:
        disconnect_task.result()
    except asyncio.CancelledError:
        pass
    except Exception:
        _LOGGER.debug("Dashboard disconnect watcher failed", exc_info=True)


def _client_disconnected_response(response_type: Any) -> Any:
    """Build the private cancellation response without exposing worker errors."""

    return response_type(
        status_code=_CLIENT_DISCONNECTED_STATUS,
        content={"detail": _CLIENT_DISCONNECTED_DETAIL},
    )


def build_fastapi_app() -> Any:
    """Build FastAPI dashboard app.

    FastAPI import is lazy so core package remains usable without web deps.
    """
    try:
        from fastapi import FastAPI, HTTPException, Query, Request
        from fastapi.responses import HTMLResponse, JSONResponse
        from starlette.concurrency import run_in_threadpool
        from starlette.datastructures import MutableHeaders
    except Exception as exc:  # pragma: no cover
        msg = "FastAPI is not installed. Install with: pip install -e '.[web]'"
        raise RuntimeError(msg) from exc

    app = FastAPI(title="F1Sim Dashboard", version="0.2")
    run_capacity = RunCapacity.from_environment()

    class _DashboardHeadersMiddleware:
        """Add response headers without buffering request disconnect messages."""

        def __init__(self, application: Any) -> None:
            self.application = application

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            if scope["type"] != "http":
                await self.application(scope, receive, send)
                return

            async def send_with_headers(message: dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    headers = MutableHeaders(scope=message)
                    headers["X-Content-Type-Options"] = "nosniff"
                    headers["X-Frame-Options"] = "DENY"
                    headers["Referrer-Policy"] = "no-referrer"
                    if scope["path"].startswith("/api/"):
                        headers["Cache-Control"] = "no-store, max-age=0"
                        headers["Pragma"] = "no-cache"
                await send(message)

            await self.application(scope, receive, send_with_headers)

    app.add_middleware(_DashboardHeadersMiddleware)

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

    async def run(request: Request, payload: DashboardRunRequest) -> Any:
        """Run a dashboard request without abandoning its synchronous worker."""

        try:
            # Keep all deterministic validation ahead of capacity admission.
            _validate_dashboard_request(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        cancel_event = Event()
        worker_task = asyncio.create_task(
            run_in_threadpool(_run_dashboard_worker, run_capacity, payload, cancel_event),
        )
        disconnect_task = asyncio.create_task(
            _watch_dashboard_disconnect(request, cancel_event),
        )
        disconnected = False
        try:
            done, _ = await asyncio.wait(
                {worker_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done:
                disconnected = bool(disconnect_task.result())
            if worker_task in done:
                result = worker_task.result()
            else:
                try:
                    # The worker owns capacity cleanup. Shielding makes sure a
                    # disconnected endpoint still drains it before returning.
                    result = await asyncio.shield(worker_task)
                except Exception:
                    if disconnected or cancel_event.is_set():
                        return _client_disconnected_response(JSONResponse)
                    raise
            if disconnected or cancel_event.is_set():
                return _client_disconnected_response(JSONResponse)
            return result
        except asyncio.CancelledError:
            # ASGI servers may cancel the endpoint task when the transport
            # closes. Keep the worker alive, request cooperative cancellation,
            # and drain it before allowing the capacity context to release.
            cancel_event.set()
            await _drain_dashboard_worker(worker_task)
            return _client_disconnected_response(JSONResponse)
        except HTTPException:
            if cancel_event.is_set():
                return _client_disconnected_response(JSONResponse)
            raise
        except SimulationCancelled:
            return _client_disconnected_response(JSONResponse)
        except ValueError as exc:
            if cancel_event.is_set():
                return _client_disconnected_response(JSONResponse)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (CurrentSeasonDataError, OSError) as exc:
            if cancel_event.is_set():
                return _client_disconnected_response(JSONResponse)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except _RunCapacityBusy as exc:
            if cancel_event.is_set():
                return _client_disconnected_response(JSONResponse)
            raise HTTPException(
                status_code=429,
                detail="Simulation capacity is busy. Please retry shortly.",
                headers={"Retry-After": "5"},
            ) from exc
        except Exception as exc:
            if cancel_event.is_set():
                return _client_disconnected_response(JSONResponse)
            _LOGGER.exception("Unexpected simulation endpoint failure")
            raise HTTPException(status_code=500, detail="Unexpected server error") from exc
        finally:
            await _stop_dashboard_disconnect_watcher(disconnect_task, cancel_event)

    # ``from __future__ import annotations`` keeps the nested function's
    # annotations as strings; install the lazy FastAPI Request type before
    # registering the route so the module remains importable without FastAPI.
    run.__annotations__["request"] = Request
    run.__annotations__["payload"] = DashboardRunRequest
    app.post("/api/run")(run)

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
