"""Conditional opening-policy comparisons under fixed rainfall, without race RNG."""

import json
from dataclasses import dataclass
from functools import lru_cache
from math import inf

import numpy as np

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.race_timing import announced_final_lap

REACTION_SEEDS = tuple(range(8))
OPENING_CANDIDATES = (TireCompound.INTERMEDIATE, TireCompound.SOFT,
                      TireCompound.MEDIUM, TireCompound.HARD)


@dataclass(frozen=True, order=True)
class OpeningPolicyScore:
    """Prefer greater mean race distance, then less elapsed time at that distance."""

    negative_mean_laps: float
    mean_time: float


def opening_policy_costs(driver, car, track, weather, strategy, tuning, profiles):
    """Return immutable distance/time scores; normalize transient state for caching."""
    clean_driver = driver.model_copy(deep=True)
    clean_driver.reset_race_state()
    snapshots = [clean_driver.model_dump(), car.model_dump(), track.model_dump(),
                 weather.model_dump(), tuning, profiles,
                 {c.value: tire.model_dump() for c, tire in TIRE_COMPOUNDS.items()}]
    return _cached_policy_costs(*(json.dumps(value, sort_keys=True) for value in snapshots),
                                strategy.value)


@lru_cache(maxsize=128)
def _cached_policy_costs(driver_json, car_json, track_json, weather_json,
                         tuning_json, profiles_json, tire_config_json, strategy):
    # Local import avoids a race-selector import cycle. Tyre configuration is
    # included in the cache key; each synchronous miss uses that current config.
    from f1sim.simulation.race import TeamStrategyArchetype

    driver = Driver.model_validate_json(driver_json)
    car = Car.model_validate_json(car_json)
    track = Track.model_validate_json(track_json)
    weather = Weather.model_validate_json(weather_json)
    tuning, profiles = json.loads(tuning_json), json.loads(profiles_json)
    scores = []
    for compound in OPENING_CANDIDATES:
        outcomes = [_policy_path_outcome(
            driver, car, track, weather, TeamStrategyArchetype(strategy), tuning, profiles,
            compound, seed,
        ) for seed in REACTION_SEEDS]
        mean_time = sum(time for _, time in outcomes) / len(outcomes)
        # A policy that cannot finish legally must not win by running farther.
        negative_mean_laps = (-sum(laps for laps, _ in outcomes) / len(outcomes)
                              if mean_time != inf else inf)
        scores.append((compound, OpeningPolicyScore(negative_mean_laps, mean_time)))
    return tuple(scores)


def _policy_path_cost(driver, car, track, weather, strategy, tuning, profiles, compound, seed):
    """Return elapsed time for direct comparisons with an isolated actual race."""
    return _policy_path_outcome(
        driver, car, track, weather, strategy, tuning, profiles, compound, seed,
    )[1]


def _policy_path_outcome(driver, car, track, weather, strategy, tuning, profiles, compound, seed):
    """Run one isolated existing pit policy with deterministic pace and mean service."""
    from f1sim.simulation.race import DriverRaceState, RaceSimulator

    simulator = RaceSimulator(np.random.default_rng(seed), tuning, profiles)
    local_driver = driver.model_copy(deep=True)
    local_driver.reset_race_state()
    plans = simulator._plan_pit_lap_options(strategy, track)
    state = DriverRaceState(local_driver, car.model_copy(deep=True), 1,
                            current_tire=TIRE_COMPOUNDS[compound].model_copy(deep=True),
                            strategy_archetype=strategy, planned_pit_laps=plans[0],
                            pit_plan_options=plans)
    projected = weather.model_copy(deep=True)
    final_lap = track.total_laps
    for lap in range(1, track.total_laps + 1):
        planning_track = (track if final_lap == track.total_laps else
                          track.model_copy(update={"total_laps": final_lap}))
        loss = 0.0
        if simulator._should_pit(
            state, [state], planning_track, lap, False, projected,
            **({"physical_total_laps": track.total_laps} if final_lap < track.total_laps else {}),
        ):
            loss = simulator._execute_pit_stop(
                state, planning_track, projected, lap, sample_service=False,
            )
            state.total_time += loss
            state.pit_stops += 1
            state.pit_laps.append(lap)
            state.force_pit_next_lap = False
        running = simulator.lap_simulator.calculate_lap_time(
            state.driver, state.car, track, state.current_tire, projected, lap, track.total_laps,
            sample_variation=False,
        )
        state.total_time += running
        state.last_lap_time = running + loss
        state.tire_laps += 1
        state.driver.current_tire_laps = state.tire_laps
        state.laps_completed += 1
        state.last_crossing_position = 1
        final_lap = announced_final_lap(final_lap, lap, state.total_time)
        if lap >= final_lap:
            break
        projected = projected.project_surface()
    if state.laps_completed > 1 and not (simulator._has_used_wet_compound(state)
                                    or len(simulator._used_slick_compounds(state)) >= 2):
        return state.laps_completed, inf
    return state.laps_completed, state.total_time
