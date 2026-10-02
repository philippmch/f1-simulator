"""Conditional opening-policy comparisons under fixed rainfall, without race RNG."""

import json
from dataclasses import dataclass
from math import inf

import numpy as np

from f1sim.cancellation import cancellation_checkpoint, raise_if_cancelled
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import (
    forecast_decision,
    forecast_dump,
    native_forecast_cache,
    register_forecast_helpers,
    register_forecast_values,
    restore_model,
)
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.race_timing import RaceFinishClock, forecast_final_lap
from f1sim.simulation.warmup import validate_tire_warmup
from f1sim.simulation.weather_schedule import validate_forecast_context

REACTION_SEEDS = tuple(range(8))
OPENING_CANDIDATES = (TireCompound.INTERMEDIATE, TireCompound.SOFT,
                      TireCompound.MEDIUM, TireCompound.HARD)
SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)


@dataclass(frozen=True, order=True)
class OpeningPolicyScore:
    """Prefer greater mean race distance, then less elapsed time at that distance."""

    negative_mean_laps: float
    mean_time: float


def _policy_snapshots(driver, car, track, weather, tuning, profiles):
    """Keep physical/configuration inputs; remove identity and prior race state."""
    clean_driver = driver.model_copy(deep=True)
    clean_driver.reset_race_state()
    clean_driver.id = clean_driver.name = clean_driver.team_id = "projection"
    clean_car = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    return [forecast_dump(clean_driver), forecast_dump(clean_car), forecast_dump(track),
            forecast_dump(weather), tuning, profiles]


@forecast_decision
def dry_opening_policy_costs(driver, car, track, weather, strategy, tuning, profiles,
                             *, tire_warmup=None, forecast_context=None):
    """Score actual slick-opening policies, including the timed race finish.

    Rain-free, dry surfaces have deterministic pit decisions, so one private
    seed suffices. Transitional surfaces retain the wet policy's reaction sample.
    Identity and prior race state do not affect these isolated policy paths.
    """
    from f1sim.simulation import race_timing

    validate_forecast_context(forecast_context)
    tire_warmup = validate_tire_warmup(tire_warmup)
    snapshots = _policy_snapshots(driver, car, track, weather, tuning, profiles)
    snapshots.extend([tire_warmup,
                      {c.value: forecast_dump(tire) for c, tire in TIRE_COMPOUNDS.items()}])
    return _cached_dry_policy_costs(
        *(json.dumps(value, sort_keys=True) for value in snapshots), strategy.value,
        race_timing.RACING_TIME_LIMIT_SECONDS,
        **({"forecast_context": forecast_context} if forecast_context is not None else {}),
    )


@native_forecast_cache(maxsize=128)
def _cached_dry_policy_costs(driver_json, car_json, track_json, weather_json,
                             tuning_json, profiles_json, warmup_json, tire_config_json, strategy,
                             racing_time_limit, forecast_context=None):
    # Configuration and deadline values key each synchronous projection. The
    # shared policy runner reads that same current configuration on a cache miss.
    from f1sim.simulation.race import TeamStrategyArchetype

    driver = restore_model(Driver, driver_json)
    car = restore_model(Car, car_json)
    track = restore_model(Track, track_json)
    weather = restore_model(Weather, weather_json)
    tuning, profiles = json.loads(tuning_json), json.loads(profiles_json)
    tire_warmup = json.loads(warmup_json)
    seeds = ((0,) if weather.rain_intensity == 0 and weather.track_wetness < 0.08
             else REACTION_SEEDS)
    scores = []
    for compound in SLICKS:
        cancellation_checkpoint()
        outcomes = [_policy_path_outcome(
            driver, car, track, weather, TeamStrategyArchetype(strategy), tuning, profiles,
            compound, seed, **({"tire_warmup": tire_warmup} if tire_warmup else {}),
            **({"forecast_context": forecast_context} if forecast_context is not None else {}),
        ) for seed in seeds]
        mean_time = sum(time for _, time in outcomes) / len(outcomes)
        negative_mean_laps = (-sum(laps for laps, _ in outcomes) / len(outcomes)
                              if mean_time != inf else inf)
        scores.append((compound, OpeningPolicyScore(negative_mean_laps, mean_time)))
    return tuple(scores)


@forecast_decision
def opening_policy_costs(driver, car, track, weather, strategy, tuning, profiles,
                         *, tire_warmup=None, forecast_context=None):
    """Return immutable distance/time scores; normalize transient state for caching."""
    from f1sim.simulation import race_timing

    validate_forecast_context(forecast_context)
    tire_warmup = validate_tire_warmup(tire_warmup)
    snapshots = _policy_snapshots(driver, car, track, weather, tuning, profiles)
    snapshots.extend([tire_warmup,
                      {c.value: forecast_dump(tire) for c, tire in TIRE_COMPOUNDS.items()}])
    return _cached_policy_costs(
        *(json.dumps(value, sort_keys=True) for value in snapshots), strategy.value,
        race_timing.RACING_TIME_LIMIT_SECONDS,
        **({"forecast_context": forecast_context} if forecast_context is not None else {}),
    )


@native_forecast_cache(maxsize=128)
def _cached_policy_costs(driver_json, car_json, track_json, weather_json,
                         tuning_json, profiles_json, warmup_json, tire_config_json, strategy,
                         racing_time_limit, forecast_context=None):
    # Local import avoids a race-selector import cycle. Configuration and the
    # deadline key each synchronous projection, as in dry/finite opening scores.
    from f1sim.simulation.race import TeamStrategyArchetype

    driver = restore_model(Driver, driver_json)
    car = restore_model(Car, car_json)
    track = restore_model(Track, track_json)
    weather = restore_model(Weather, weather_json)
    tuning, profiles = json.loads(tuning_json), json.loads(profiles_json)
    tire_warmup = json.loads(warmup_json)
    scores = []
    candidates = (tuple(compound for compound in TireCompound
                        if weather.tire_mismatch(compound) != "critical")
                  if forecast_context is not None and forecast_context.schedule
                  else OPENING_CANDIDATES)
    for compound in candidates:
        cancellation_checkpoint()
        outcomes = [_policy_path_outcome(
            driver, car, track, weather, TeamStrategyArchetype(strategy), tuning, profiles,
            compound, seed, **({"tire_warmup": tire_warmup} if tire_warmup else {}),
            **({"forecast_context": forecast_context} if forecast_context is not None else {}),
        ) for seed in REACTION_SEEDS]
        mean_time = sum(time for _, time in outcomes) / len(outcomes)
        # A policy that cannot finish legally must not win by running farther.
        negative_mean_laps = (-sum(laps for laps, _ in outcomes) / len(outcomes)
                              if mean_time != inf else inf)
        scores.append((compound, OpeningPolicyScore(negative_mean_laps, mean_time)))
    return tuple(scores)


def _policy_path_cost(driver, car, track, weather, strategy, tuning, profiles, compound, seed,
                      *, tire_warmup=None, forecast_context=None):
    """Return elapsed time for direct comparisons with an isolated actual race."""
    return _policy_path_outcome(
        driver, car, track, weather, strategy, tuning, profiles, compound, seed,
        tire_warmup=tire_warmup,
        **({"forecast_context": forecast_context} if forecast_context is not None else {}),
    )[1]


@forecast_decision
def inventory_opening_policy_costs(driver, car, track, weather, strategy, tuning, profiles,
                                    records, *, tire_warmup=None, forecast_context=None):
    """Compare each physical opening set through the finite, timed race policy."""
    from f1sim.simulation import race_timing

    validate_forecast_context(forecast_context)
    tire_warmup = validate_tire_warmup(tire_warmup)
    snapshots = _policy_snapshots(driver, car, track, weather, tuning, profiles)
    snapshots.extend([tire_warmup, records,
                      {c.value: forecast_dump(tire) for c, tire in TIRE_COMPOUNDS.items()}])
    return _cached_inventory_policy_costs(
        *(json.dumps(value, sort_keys=True) for value in snapshots), strategy.value,
        race_timing.RACING_TIME_LIMIT_SECONDS,
        **({"forecast_context": forecast_context} if forecast_context is not None else {}),
    )


@native_forecast_cache(maxsize=128)
def _cached_inventory_policy_costs(driver_json, car_json, track_json, weather_json,
                                   tuning_json, profiles_json, warmup_json, records_json,
                                   tires_json,
                                   strategy, racing_time_limit, forecast_context=None):
    from f1sim.simulation.race import TeamStrategyArchetype

    driver = restore_model(Driver, driver_json)
    car = restore_model(Car, car_json)
    track = restore_model(Track, track_json)
    weather = restore_model(Weather, weather_json)
    records = json.loads(records_json)
    tire_warmup = json.loads(warmup_json)
    eligible = [item for item in records
                if weather.tire_mismatch(TireCompound(item["compound"])) != "critical"]
    scores = []
    equivalent_outcomes = {}
    # Finite-set policy uses deterministic costs throughout, so private lap
    # variation/events are unnecessary. The actual race RNG is untouched.
    for item in eligible or records:
        cancellation_checkpoint()
        key = (item["compound"], item.get("age", 0))
        if key not in equivalent_outcomes:
            # Only elapsed time and completed distance are returned. Fitting
            # an equivalent identity leaves the same anonymous future pool;
            # retain every physical record and the original score/tie order.
            equivalent_outcomes[key] = _policy_path_outcome(
                driver, car, track, weather, TeamStrategyArchetype(strategy),
                json.loads(tuning_json), json.loads(profiles_json),
                TireCompound(item["compound"]), 0, tire_warmup=tire_warmup,
                tire_inventory=records, opening_set_id=item["id"],
                **({"forecast_context": forecast_context} if forecast_context is not None else {}),
            )
        laps, time = equivalent_outcomes[key]
        scores.append((item["id"], OpeningPolicyScore(-laps if time != inf else inf, time)))
    return tuple(scores)


def _policy_path_outcome(driver, car, track, weather, strategy, tuning, profiles, compound, seed,
                         *, tire_inventory=None, opening_set_id=None, tire_warmup=None,
                         forecast_context=None):
    """Run one isolated existing pit policy with deterministic pace and mean service."""
    from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator

    simulator = RaceSimulator(np.random.default_rng(seed), tuning, profiles,
                              tire_warmup=tire_warmup)
    simulator.weather_forecast_context = forecast_context
    local_driver = driver.model_copy(deep=True)
    local_driver.reset_race_state()
    plans = simulator._plan_pit_lap_options(strategy, track)
    state = DriverRaceState(local_driver, car.model_copy(deep=True), 1,
                            current_tire=TIRE_COMPOUNDS[compound].model_copy(deep=True),
                            strategy_archetype=strategy, planned_pit_laps=plans[0],
                            pit_plan_options=plans)
    if tire_inventory is not None:
        from f1sim.simulation.tire_inventory import TireInventory

        inventory = TireInventory.from_sets(tire_inventory)
        simulator._initialize_inventory(state, inventory, inventory.sets[opening_set_id])
    projected = weather.model_copy(deep=True)
    finish_clock = RaceFinishClock(track.total_laps)
    final_lap = finish_clock.final_lap
    observed_running_pace = None
    for lap in range(1, track.total_laps + 1):
        raise_if_cancelled()
        planning_final_lap = forecast_final_lap(
            final_lap, lap - 1, state.total_time, observed_running_pace,
            finish_clock.time_limit_seconds,
        )
        planning_track = (track if planning_final_lap == track.total_laps else
                          track.model_copy(update={"total_laps": planning_final_lap}))
        loss = 0.0
        if simulator._should_pit(
            state, [state], planning_track, lap, False, projected,
            **({"physical_total_laps": track.total_laps}
               if planning_final_lap < track.total_laps else {}),
        ):
            loss = simulator._execute_pit_stop(
                state, planning_track, projected, lap, sample_service=False,
                **({"physical_total_laps": track.total_laps}
                   if planning_final_lap < track.total_laps else {}),
            )
            if state.status != DriverStatus.RACING:
                return state.laps_completed, inf
            state.total_time += loss
            state.pit_stops += 1
            state.pit_laps.append(lap)
            state.force_pit_next_lap = False
        running = simulator.lap_simulator.calculate_lap_time(
            state.driver, state.car, track, state.current_tire, projected, lap, track.total_laps,
            sample_variation=False,
        )
        # Forecast recurring pace from physics; a fitting cost delays this
        # crossing once, without repeating on every remaining lap.
        observed_running_pace = running
        if simulator.tire_warmup and state.fit_lap_pending:
            running += simulator._consume_tire_warmup(state)
        state.total_time += running
        state.last_lap_time = running + loss
        state.tire_laps += 1
        state.driver.current_tire_laps = state.tire_laps
        state.laps_completed += 1
        state.last_crossing_position = 1
        final_lap = finish_clock.observe_leader_crossing(lap, state.total_time)
        if lap >= final_lap:
            break
        if simulator.weather_forecast_context is None:
            projected = projected.project_surface()
        else:
            projected = simulator.weather_forecast_context.project_next(projected)
            simulator.weather_forecast_context = simulator.weather_forecast_context.advanced()
    if state.laps_completed > 1 and not (simulator._has_used_wet_compound(state)
                                    or len(simulator._used_slick_compounds(state)) >= 2):
        return state.laps_completed, inf
    return state.laps_completed, state.total_time


register_forecast_helpers(globals(), ('_policy_path_outcome',))

register_forecast_values(globals(), ("REACTION_SEEDS", "OPENING_CANDIDATES", "SLICKS"))
