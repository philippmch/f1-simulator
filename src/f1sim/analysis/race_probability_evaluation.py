"""Held-out current-season race-winner probability evaluation.

Targets use today's revised provider feeds. Target qualifying contributes only
the entrant identities and teams; model evidence is restricted to earlier
events and constructor standings through the preceding round.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum
from typing import Any

import numpy as np

from f1sim.analysis.holdout_folds import (
    HoldoutCoverage,
    InsufficientTargetCoverage,
    assemble_holdout_fold,
)
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.race_probability_scores import (
    score_winner_probabilities,
    summarize_winner_trials,
)
from f1sim.analysis.scenarios import scenario_weather_from_label
from f1sim.data.current import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.models import Weather
from f1sim.simulation.execution import validate_race_engine
from f1sim.simulation.randomness import DEFAULT_RNG_POLICY, validate_rng_policy

_MAX_TRIALS = 10_000
_MAX_TOTAL_TRIALS = 10_000
_MAX_SEED = 2**32 - 1
_WEATHER_SCENARIOS = ("dry", "light_rain", "heavy_rain")


def _validate_integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def _round_number(value: Any) -> int | None:
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdecimal():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def derive_event_seed(base_seed: int, year: int, round_number: int) -> int:
    """Derive a stable 32-bit seed independent of event selection and trial count."""
    _validate_integer(base_seed, "seed", 0, _MAX_SEED)
    _validate_integer(year, "year", 1, 9999)
    _validate_integer(round_number, "round_number", 1, 10_000)
    state = np.random.SeedSequence([base_seed, year, round_number]).generate_state(
        1, dtype=np.uint32,
    )
    return int(state[0])


def _selected_events(
    loader: CurrentSeasonDataLoader,
    year: int,
    events: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    *,
    target_race: str | int | None,
    all_targets: bool,
) -> list[Mapping[str, Any]]:
    result_rounds = {
        round_number
        for row in results
        if isinstance(row, Mapping)
        and (round_number := _round_number(row.get("round"))) is not None
    }
    if all_targets:
        targets = [
            event for event in events
            if _round_number(event.get("round")) in result_rounds
        ]
        if not targets:
            raise ValueError("Evaluation requires at least one completed current-season target")
        return sorted(targets, key=lambda event: _round_number(event.get("round")) or 0)

    if target_race is None:
        raise ValueError("Specify target_race or set all_targets=True")
    if isinstance(target_race, bool) or not isinstance(target_race, (str, int)):
        raise ValueError("target_race must be a race name or round number")
    if isinstance(target_race, str) and not target_race.strip():
        raise ValueError("target_race must be a race name or round number")
    event = loader._event_for_race(year, target_race)
    round_number = _round_number(event.get("round"))
    if round_number is None or round_number not in result_rounds:
        raise ValueError("Evaluation requires a completed current-season target")
    return [event]


def _target_rows(
    loader: CurrentSeasonDataLoader,
    rows: Sequence[Mapping[str, Any]],
    round_number: int,
) -> list[Mapping[str, Any]]:
    """Match shared fold deduplication so repeated feed rows do not fake two winners."""
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or _round_number(row.get("round")) != round_number:
            continue
        identity = loader._strong_driver_identity(row)
        if identity is None:
            raise CurrentSeasonDataError("Evaluation requires round and driver identities")
        if identity in unique and unique[identity] != row:
            raise CurrentSeasonDataError("Conflicting evaluation driver records")
        unique[identity] = row
    return list(unique.values())


def _coverage_dict(coverage: HoldoutCoverage) -> dict[str, int]:
    return {
        "qualifying_entrants": coverage.qualifying_entrants,
        "result_entrants": coverage.result_entrants,
        "matched_result_entrants": coverage.matched_result_entrants,
        "expected_result_entrants": coverage.expected_result_entrants,
    }


def _observed_winner(
    loader: CurrentSeasonDataLoader,
    target_results: Sequence[Mapping[str, Any]],
    aliases: Mapping[str, str],
    driver_ids: set[str],
) -> dict[str, Any]:
    position_one = [row for row in target_results if loader._row_position(row) == 1]
    if not position_one:
        return {"status": "excluded", "reason": "missing_position_one_result"}
    if len(position_one) != 1:
        return {"status": "excluded", "reason": "ambiguous_position_one_result"}

    row = position_one[0]
    if not loader._classified(row):
        return {"status": "excluded", "reason": "position_one_result_not_classified"}
    winner_id = loader._resolve_row_driver(row, aliases)
    if winner_id is None:
        strong_identity = loader._strong_driver_identity(row) or ""
        if strong_identity.startswith(("driverId:", "id:", "code:", "abbreviation:")):
            return {"status": "excluded", "reason": "winner_outside_entrant_roster"}
        return {"status": "excluded", "reason": "unresolved_position_one_identity"}
    if winner_id not in driver_ids:
        return {"status": "excluded", "reason": "winner_outside_entrant_roster"}
    return {"status": "observed", "winner_id": winner_id}


def _aggregate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scored = [fold["score"] for fold in folds if fold["status"] == "scored"]
    count = len(scored)
    return {
        "selected_events": len(folds),
        "scored_events": count,
        "excluded_events": len(folds) - count,
        "mean_brier_score": (
            fsum(score["brier_score"] for score in scored) / count if count else None
        ),
        "mean_uniform_baseline_brier_score": (
            fsum(score["uniform_baseline_brier_score"] for score in scored) / count
            if count else None
        ),
        "mean_delta_from_uniform_baseline": (
            fsum(score["delta_from_uniform_baseline"] for score in scored) / count
            if count else None
        ),
        "event_weighting": "equal_weight_per_scored_event",
    }


def evaluate_race_probabilities(
    loader: CurrentSeasonDataLoader,
    year: int,
    *,
    target_race: str | int | None = None,
    all_targets: bool = False,
    trials: int = 100,
    seed: int = 0,
    form_races: int = 3,
    scenario: str = "dry",
    race_engine: str = "standard",
    rng_policy: str = DEFAULT_RNG_POLICY,
) -> dict[str, Any]:
    """Evaluate Monte Carlo winner probabilities against completed races.

    Exactly one target must be selected with ``target_race`` or the complete
    set of available completed targets selected with ``all_targets=True``.
    The target results are used only for result-coverage validation and the
    observed outcome; they never enter the forecast model.

    The same event has the same seed whether selected alone or with other
    events, and increasing ``trials`` preserves its existing trial prefix.
    The default execution is serial and uses the simulator's default RNG
    stream policy.
    """
    loader._assert_current_year(year)
    if type(all_targets) is not bool:
        raise ValueError("all_targets must be a boolean")
    if (target_race is None) == (not all_targets):
        raise ValueError("Specify exactly one target_race or all_targets=True")
    trials = _validate_integer(trials, "trials", 1, _MAX_TRIALS)
    seed = _validate_integer(seed, "seed", 0, _MAX_SEED)
    form_races = _validate_integer(form_races, "form_races", 0, 24)
    if not isinstance(scenario, str) or scenario not in _WEATHER_SCENARIOS:
        raise ValueError(f"scenario must be one of: {', '.join(_WEATHER_SCENARIOS)}")
    race_engine = validate_race_engine(race_engine)
    rng_policy = validate_rng_policy(rng_policy)

    events = loader.get_event_schedule(year)
    if not events:
        raise CurrentSeasonDataError("Current-season calendar contains no race events")
    results, qualifying = loader._season_data(year)
    targets = _selected_events(
        loader, year, events, results,
        target_race=target_race,
        all_targets=all_targets,
    )
    total_trials = len(targets) * trials
    if total_trials > _MAX_TOTAL_TRIALS:
        raise ValueError(
            f"Selected event trial budget exceeds {_MAX_TOTAL_TRIALS} total trials"
        )

    weather = scenario_weather_from_label(
        Weather(), scenario, weather_mode="fixed_rainfall",
    ).weather
    folds: list[dict[str, Any]] = []
    for event in targets:
        round_number = _round_number(event.get("round"))
        if round_number is None:
            raise CurrentSeasonDataError("Calendar event has an invalid round number")
        target_results = _target_rows(loader, results, round_number)
        per_event_seed = derive_event_seed(seed, year, round_number)
        try:
            assembled, _observations = assemble_holdout_fold(
                loader, year, event, events, results, qualifying, form_races=form_races,
            )
        except InsufficientTargetCoverage as exc:
            active, aliases = loader._build_active_driver_map(exc.roster, {})
            driver_ids = {driver_id for driver_id in active}
            observed = _observed_winner(loader, target_results, aliases, driver_ids)
            folds.append({
                "round": round_number,
                "race": event["race"],
                "status": "excluded",
                "reason": "insufficient_target_coverage",
                "coverage": _coverage_dict(exc.coverage),
                "entrant_ids": sorted(driver_ids),
                "observed_outcome": observed,
                "forecast": None,
                "simulation_inputs": None,
                "score": None,
                "training": {
                    "status": "not_assembled",
                    "cutoff_round": round_number - 1,
                    "form_races_requested": form_races,
                    "form_rounds": None,
                    "standings_round": None,
                },
                "simulation": {
                    "trials": trials,
                    "event_seed": per_event_seed,
                    "race_engine": race_engine,
                    "rng_policy": rng_policy,
                },
            })
            continue

        # Do not let provider row order decide which entrant receives a random
        # stream. Car keys are sorted for the same reproducibility reason.
        drivers = sorted(assembled.drivers, key=lambda driver: driver.id)
        cars = {key: assembled.cars[key] for key in sorted(assembled.cars)}
        driver_ids = [driver.id for driver in drivers]
        runner = MonteCarloRunner(
            drivers=drivers,
            cars=cars,
            track=assembled.track,
            weather=weather.model_copy(deep=True),
            seed=per_event_seed,
            race_engine=race_engine,
            rng_policy=rng_policy,
        )
        simulation = runner.run(
            num_simulations=trials,
            parallel=False,
        )
        forecast = summarize_winner_trials(
            simulation.race_results,
            driver_ids,
            trials,
        )

        # Resolve the observation only after the forecast has been produced.
        observed = _observed_winner(
            loader,
            target_results,
            assembled.aliases,
            set(driver_ids),
        )
        score = None
        status = "excluded"
        reason = observed.get("reason")
        if observed["status"] == "observed":
            score = score_winner_probabilities(
                {
                    driver_id: item["probability"]
                    for driver_id, item in forecast["drivers"].items()
                },
                forecast["no_classified_winner"]["probability"],
                observed["winner_id"],
            )
            status = "scored"
            reason = None

        coverage = assembled.metadata.coverage
        folds.append({
            "round": round_number,
            "race": event["race"],
            "status": status,
            "reason": reason,
            "coverage": _coverage_dict(coverage),
            "entrant_ids": driver_ids,
            "observed_outcome": observed,
            "forecast": forecast,
            "simulation_inputs": simulation.input_snapshot,
            "score": score,
            "training": {
                "status": "assembled",
                "cutoff_round": assembled.metadata.training_cutoff_round,
                "eligible_form_rounds": list(assembled.metadata.eligible_form_rounds),
                "form_races_requested": form_races,
                "form_rounds": list(assembled.metadata.form_rounds),
                "standings_round": assembled.metadata.standings_round,
                "baseline_round": assembled.metadata.baseline_round,
                "model_weights": {"track": 0.0, "form": 0.3, "qualifying": 0.2},
                "target_qualifying_performance_used": False,
            },
            "simulation": {
                "trials": trials,
                "event_seed": per_event_seed,
                "event_seed_policy": "seed_sequence_uint32_v1",
                "trial_seed_policy": "event_seed_plus_zero_based_trial_index",
                "race_engine": race_engine,
                "rng_policy": rng_policy,
                "parallel": False,
                "weather": weather.model_dump(),
                "weather_scenario": scenario,
                "weather_mode": "fixed_rainfall_surface_evolves",
                "qualifying": "simulated_per_trial",
                "opening_strategy": "automatic",
                "pit_strategy": "automatic",
                "race_tire_sets": "unlimited",
                "tire_warmup": "off",
            },
        })

    provenance = loader.get_provenance()
    return {
        "year": year,
        "evaluation": "round_holdout_race_winner_probabilities",
        "form_races": form_races,
        "trials_per_event": trials,
        "trial_budget": {
            "requested_total": total_trials,
            "executed_total": sum(
                fold["forecast"]["trials"] for fold in folds if fold["forecast"] is not None
            ),
            "max_total": _MAX_TOTAL_TRIALS,
        },
        "base_seed": seed,
        "race_engine": race_engine,
        "rng_policy": rng_policy,
        "weather_assumption": {
            "scenario": scenario,
            "mode": "fixed_rainfall_surface_evolves",
            **weather.model_dump(),
        },
        "entrant_basis": "target_qualifying_identities",
        "data_revision": "current_provider_data_not_historical_availability",
        "fetched_at": provenance["fetched_at"],
        "source_urls": provenance["urls"],
        "seed_derivation": "numpy_seed_sequence_base_year_round_uint32_v1",
        "trials_per_event_policy": "event_seed_plus_zero_based_trial_index_prefix_stable",
        "folds": folds,
        "aggregate": _aggregate(folds),
    }
