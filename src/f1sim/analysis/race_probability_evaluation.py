"""Held-out current-season race-winner probability evaluation.

Targets use today's revised provider feeds. Target qualifying contributes only
the entrant identities and teams; model evidence is restricted to earlier
events and constructor standings through the preceding round.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from math import fsum, isclose
from typing import Any

import numpy as np

from f1sim.analysis.holdout_folds import (
    HoldoutCoverage,
    InsufficientTargetCoverage,
    assemble_holdout_fold,
)
from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.race_probability_scores import (
    score_winner_counts,
    score_winner_probabilities,
    summarize_winner_trials,
)
from f1sim.analysis.scenarios import scenario_weather_from_label
from f1sim.analysis.winner_baselines import (
    build_winner_baselines,
    score_saved_winner_baselines,
    summarize_baseline_comparisons,
)
from f1sim.cancellation import raise_if_cancelled
from f1sim.data.current import CurrentSeasonDataError, CurrentSeasonDataLoader
from f1sim.models import Weather
from f1sim.simulation.execution import DEFAULT_RACE_ENGINE, validate_race_engine
from f1sim.simulation.randomness import DEFAULT_RNG_POLICY, validate_rng_policy

_MAX_TRIALS = 10_000
_MAX_TOTAL_TRIALS = 10_000
_MAX_SEED = 2**32 - 1
_MAX_WORKERS = 61
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
    adjusted = [score for score in scored if score["mc_adjustment"]["status"] == "available"]
    adjusted_count = len(adjusted)
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
        "adjusted_events": adjusted_count,
        "mean_adjusted_brier_score": (
            fsum(score["mc_adjustment"]["adjusted_brier_score"] for score in adjusted)
            / adjusted_count if adjusted_count else None
        ),
        "mean_estimated_finite_trial_bias": (
            fsum(score["mc_adjustment"]["estimated_empirical_score_bias"] for score in adjusted)
            / adjusted_count if adjusted_count else None
        ),
        "mean_adjusted_delta_from_uniform_baseline": (
            fsum(score["mc_adjustment"]["adjusted_delta_from_uniform_baseline"]
                 for score in adjusted) / adjusted_count if adjusted_count else None
        ),
        "adjusted_event_weighting": "equal_weight_per_event_with_at_least_two_trials",
    }


def _notify_progress(callback, state, phase, **updates) -> None:
    state.update(phase=phase, **updates)
    if callback is not None:
        callback(dict(state))


def rescore_saved_winner_evaluation(report: Mapping[str, Any]) -> dict[str, Any]:
    """Add finite-trial scores to an exported evaluation without running races.

    Recorded counts must match the declared trial count, roster and empirical
    probabilities. Original forecasts, inputs and provenance are preserved.
    This validates scoring evidence, not the source model or real-race labels.
    """
    if not isinstance(report, Mapping) or report.get("evaluation") != (
        "round_holdout_race_winner_probabilities"
    ):
        raise ValueError("report must be a saved race-winner probability evaluation")
    folds = report.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("report must contain a non-empty folds list")
    _validate_integer(report.get("year"), "report year", 1, 9999)
    seen_rounds = set()
    rescored = []
    for fold in folds:
        if not isinstance(fold, Mapping) or fold.get("status") not in ("scored", "excluded"):
            raise ValueError("every fold must have scored or excluded status")
        round_number = _validate_integer(fold.get("round"), "fold round", 1, 10_000)
        if round_number in seen_rounds:
            raise ValueError("report contains duplicate event rounds")
        seen_rounds.add(round_number)
        updated = deepcopy(dict(fold))
        forecast = fold.get("forecast")
        if forecast is None:
            if fold["status"] != "excluded" or fold.get("score") is not None:
                raise ValueError("a fold without a forecast must be excluded and unscored")
            updated["score"] = None
            rescored.append(updated)
            continue
        if not isinstance(forecast, Mapping):
            raise ValueError("fold forecast must be a mapping or null")
        trials = _validate_integer(forecast.get("trials"), "forecast trials", 1, _MAX_TRIALS)
        drivers = forecast.get("drivers")
        no_winner = forecast.get("no_classified_winner")
        if (not isinstance(drivers, Mapping) or not drivers
                or any(not isinstance(item, Mapping) for item in drivers.values())
                or not isinstance(no_winner, Mapping)):
            raise ValueError("forecast must contain driver and no-winner count records")
        roster = fold.get("entrant_ids")
        if (not isinstance(roster, list) or any(not isinstance(name, str) for name in roster)
                or len(roster) != len(drivers) or set(roster) != set(drivers)):
            raise ValueError("forecast drivers must match the recorded entrant roster")
        observed_id = None
        if fold["status"] == "scored":
            observed = fold.get("observed_outcome")
            if (not isinstance(observed, Mapping) or observed.get("status") != "observed"
                    or "winner_id" not in observed):
                raise ValueError("a scored fold requires an observed winner category")
            observed_id = observed["winner_id"]
        elif fold.get("score") is not None:
            raise ValueError("an excluded fold must not contain a score")
        score = score_winner_counts(
            {name: item.get("wins") for name, item in drivers.items()},
            no_winner.get("count"), observed_id,
        )
        if score["mc_adjustment"]["trials"] != trials:
            raise ValueError("forecast counts must sum to its declared trial count")
        probabilities = {name: item.get("probability") for name, item in drivers.items()}
        score_winner_probabilities(probabilities, no_winner.get("probability"), observed_id)
        count_records = [(item.get("probability"), item["wins"]) for item in drivers.values()]
        count_records.append((no_winner.get("probability"), no_winner["count"]))
        if any(not isclose(probability, count / trials, rel_tol=1e-12, abs_tol=1e-12)
               for probability, count in count_records):
            raise ValueError("forecast probabilities must match its winner counts")
        updated["score"] = score if fold["status"] == "scored" else None
        if "baselines" in fold:
            updated["baselines"] = score_saved_winner_baselines(
                fold["baselines"], roster, target_round=round_number,
                observed_winner=observed_id,
            )
        rescored.append(updated)
    updated_report = deepcopy(dict(report))
    updated_report["folds"] = rescored
    updated_report["aggregate"] = _aggregate(rescored)
    if any("baselines" in fold for fold in rescored):
        updated_report["baseline_comparisons"] = summarize_baseline_comparisons(rescored)
    updated_report["rescoring"] = {
        "method": "finite_ensemble_multiclass_brier_v1",
        "basis": "recorded_winner_counts_and_observed_categories",
        "live_data_fetched": False,
        "additional_simulation_trials": 0,
        "source_inputs_and_provenance": "preserved_from_saved_report",
    }
    return updated_report


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
    race_engine: str = DEFAULT_RACE_ENGINE,
    rng_policy: str = DEFAULT_RNG_POLICY,
    parallel: bool = False,
    max_workers: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate Monte Carlo winner probabilities against completed races.

    Exactly one target must be selected with ``target_race`` or the complete
    set of available completed targets selected with ``all_targets=True``.
    The target results are used only for result-coverage validation and the
    observed outcome; they never enter the forecast model.

    The same event has the same seed whether selected alone or with other
    events, and increasing ``trials`` preserves its existing trial prefix.
    Events run in order; optional parallel execution distributes their trials
    without changing seeds. Progress callbacks run only in the calling process.
    Cancellation raises before a complete report exists, including during races.
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
    if type(parallel) is not bool:
        raise ValueError("parallel must be a boolean")
    if max_workers is not None:
        max_workers = _validate_integer(max_workers, "max_workers", 1, _MAX_WORKERS)
        if not parallel:
            raise ValueError("max_workers requires parallel=True")
    if progress_callback is not None and not callable(progress_callback):
        raise TypeError("progress_callback must be callable or None")
    if cancel_requested is not None and not callable(cancel_requested):
        raise TypeError("cancel_requested must be callable or None")

    progress = {
        "year": year, "events_total": None, "events_completed": 0,
        "collection_events_completed": 0, "trials_requested": None,
        "trials_total": None, "trials_completed": 0,
    }
    raise_if_cancelled(cancel_requested)
    _notify_progress(progress_callback, progress, "loading")
    raise_if_cancelled(cancel_requested)

    events = loader.get_event_schedule(year)
    raise_if_cancelled(cancel_requested)
    if not events:
        raise CurrentSeasonDataError("Current-season calendar contains no race events")
    results, qualifying = loader._season_data(year)
    raise_if_cancelled(cancel_requested)
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
    progress.update(events_total=len(targets), trials_requested=total_trials)

    weather = scenario_weather_from_label(
        Weather(), scenario, weather_mode="fixed_rainfall",
    ).weather

    # Finish every network-dependent holdout assembly before spending any race
    # simulation work. The loader's fetch deadline is shared across the whole
    # invocation; lengthy Monte Carlo runs must not consume time needed by a
    # later fold's constructor standings request.
    prepared_folds: list[dict[str, Any]] = []
    for event_index, event in enumerate(targets, 1):
        raise_if_cancelled(cancel_requested)
        round_number = _round_number(event.get("round"))
        if round_number is None:
            raise CurrentSeasonDataError("Calendar event has an invalid round number")
        target_results = _target_rows(loader, results, round_number)
        per_event_seed = derive_event_seed(seed, year, round_number)
        _notify_progress(
            progress_callback, progress, "collecting", event_index=event_index,
            round=round_number, race=event["race"], collection_events_completed=event_index - 1,
        )
        raise_if_cancelled(cancel_requested)
        try:
            assembled, _observations = assemble_holdout_fold(
                loader, year, event, events, results, qualifying, form_races=form_races,
            )
        except InsufficientTargetCoverage as exc:
            prepared_folds.append({
                "event": event,
                "round": round_number,
                "target_results": target_results,
                "event_seed": per_event_seed,
                "coverage_error": exc,
            })
        else:
            # Freeze the references before simulation and target scoring. Their
            # evidence has the same cutoff as the model, with no target labels.
            prior_outcomes = [
                {"round": earlier, **_observed_winner(
                    loader, _target_rows(loader, results, earlier), assembled.aliases,
                    {driver.id for driver in assembled.drivers},
                )}
                for earlier in sorted({
                    _round_number(row.get("round")) for row in events
                    if _round_number(row.get("round")) is not None
                    and _round_number(row.get("round")) < round_number
                })
                if any(_round_number(row.get("round")) == earlier for row in results)
            ]
            baselines = build_winner_baselines(
                {driver.id: driver.team_id for driver in assembled.drivers},
                {loader._row_team_id(row): row["points"]
                 for row in assembled.constructor_standings},
                prior_outcomes, cutoff_round=round_number - 1,
            )
            prepared_folds.append({
                "event": event,
                "round": round_number,
                "target_results": target_results,
                "event_seed": per_event_seed,
                "assembled": assembled,
                "baselines": baselines,
            })
        raise_if_cancelled(cancel_requested)

    # Capture provenance after all required collection and model assembly, and
    # before simulation duration can affect the point-in-time metadata.
    provenance = deepcopy(loader.get_provenance())
    _notify_progress(
        progress_callback, progress, "collected", collection_events_completed=len(targets),
        trials_total=sum("assembled" in prepared for prepared in prepared_folds) * trials,
        event_index=None, round=None, race=None,
    )
    folds: list[dict[str, Any]] = []
    completed_trials = 0
    for event_index, prepared in enumerate(prepared_folds, 1):
        raise_if_cancelled(cancel_requested)
        event = prepared["event"]
        round_number = prepared["round"]
        target_results = prepared["target_results"]
        per_event_seed = prepared["event_seed"]
        progress.update(
            event_index=event_index, round=round_number, race=event["race"],
            event_trials_completed=0, event_trials_total=trials, status=None, reason=None,
        )
        coverage_error = prepared.get("coverage_error")
        if coverage_error is not None:
            exc = coverage_error
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
                    "parallel": parallel,
                    "max_workers": max_workers,
                },
            })
            _notify_progress(
                progress_callback, progress, "event_complete", events_completed=event_index,
                event_trials_total=0, status="excluded", reason="insufficient_target_coverage",
            )
            raise_if_cancelled(cancel_requested)
            continue

        assembled = prepared["assembled"]
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
        run_options = {"num_simulations": trials, "parallel": parallel}
        if max_workers is not None:
            run_options["max_workers"] = max_workers
        if cancel_requested is not None:
            run_options["cancel_requested"] = cancel_requested
        if progress_callback is not None:
            def trial_progress(completed: int, _total: int) -> None:
                _notify_progress(
                    progress_callback, progress, "simulating", event_trials_completed=completed,
                    trials_completed=completed_trials + completed,
                )

            run_options["progress_callback"] = trial_progress
        simulation = runner.run(**run_options)
        raise_if_cancelled(cancel_requested)
        completed_trials += trials
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
            score = score_winner_counts(
                {
                    driver_id: item["wins"]
                    for driver_id, item in forecast["drivers"].items()
                },
                forecast["no_classified_winner"]["count"],
                observed["winner_id"],
            )
            status = "scored"
            reason = None

        coverage = assembled.metadata.coverage
        baselines = score_saved_winner_baselines(
            prepared["baselines"], driver_ids, target_round=round_number,
            observed_winner=observed.get("winner_id") if status == "scored" else None,
        )
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
            "baselines": baselines,
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
                "parallel": parallel,
                "max_workers": max_workers,
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
        _notify_progress(
            progress_callback, progress, "event_complete", events_completed=event_index,
            event_trials_completed=trials, trials_completed=completed_trials,
            status=status, reason=reason,
        )
        raise_if_cancelled(cancel_requested)

    _notify_progress(
        progress_callback, progress, "complete", event_index=None, round=None, race=None,
        event_trials_completed=None, event_trials_total=None, status=None, reason=None,
    )
    raise_if_cancelled(cancel_requested)
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
        "execution": {"parallel": parallel, "max_workers": max_workers},
        "weather_assumption": {
            "scenario": scenario,
            "mode": "fixed_rainfall_surface_evolves",
            **weather.model_dump(),
        },
        "entrant_basis": "target_qualifying_identities",
        "data_revision": "current_provider_data_not_historical_availability",
        "fetched_at": provenance["fetched_at"],
        "source_urls": provenance["urls"],
        "provenance": provenance,
        "seed_derivation": "numpy_seed_sequence_base_year_round_uint32_v1",
        "trials_per_event_policy": "event_seed_plus_zero_based_trial_index_prefix_stable",
        "folds": folds,
        "aggregate": _aggregate(folds),
        "baseline_comparisons": summarize_baseline_comparisons(folds),
    }
