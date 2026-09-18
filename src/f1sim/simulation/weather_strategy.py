"""Optimistic paid weather-stop plans under a fixed rainfall surface projection."""

import json
from dataclasses import dataclass
from functools import lru_cache
from math import inf

import numpy as np

from f1sim.models import Car, Driver, Tire, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import (
    normalize_weather_intervals,
    projected_surfaces,
    suffix_weather_intervals,
)


@dataclass(frozen=True)
class WeatherStopCosts:
    pit_now_cost: float
    stay_cost: float


def _validate_weather_clock(weather_clock, horizon):
    if weather_clock is not None:
        if not isinstance(weather_clock, StrategyWeatherClock):
            raise ValueError("weather_clock must be a StrategyWeatherClock")
        weather_clock.validate_horizon(horizon)


def _clock_weather_stop_costs(
    driver, car, track, weather, current_tire, tire_age, current_lap,
    *, pit_lane_factor, additional_current_stop_cost, current_lap_time_modifier,
    active_aero_enabled, traffic_possible, physical_total_laps, weather_clock,
):
    """Evaluate weather-stop bounds while each paid stop delays the clock."""
    horizon = track.total_laps - current_lap + 1
    simulator = LapSimulator(np.random.default_rng(0))
    driver = driver.model_copy(deep=True)
    driver.reset_race_state()
    clean = car.model_copy(deep=True)
    service = expected_stationary_time(clean)
    projected = [weather]
    branch_surfaces = {}
    surface_json = {id(weather): weather.model_dump_json()}

    def surface(offset, paid_stops, stopped_first):
        branch = (offset, paid_stops, stopped_first)
        if branch in branch_surfaces:
            return branch_surfaces[branch]
        updates = weather_clock.updates(offset, paid_stops, stopped_first)
        while len(projected) <= updates:
            value = projected[-1].project_surface()
            projected.append(value)
            surface_json[id(value)] = value.model_dump_json()
        value = projected[updates]
        branch_surfaces[branch] = value
        return value

    @lru_cache(maxsize=None)
    def running(offset, tire_key, compound, age, gap_kind, surface_json):
        tire = current_tire if tire_key == "retained" else TIRE_COMPOUNDS[compound]
        driver.current_tire_laps = age
        # A proposed stop's out-lap is the optimistic clear-air bound.  The
        # retained branch keeps the observed traffic gap when that control is
        # enabled; future free-green laps are clear air in either branch.
        gap = (0.0 if traffic_possible else None) if gap_kind == 2 else None
        value = simulator.calculate_lap_time(
            driver, clean, track, tire, Weather.model_validate_json(surface_json),
            current_lap + offset, physical_total_laps,
            gap_to_car_ahead=gap,
            active_aero_enabled=(active_aero_enabled if offset == 0 else True),
            sample_variation=False,
        )
        return value * current_lap_time_modifier if offset == 0 else value

    def run(offset, tire_key, compound, age, branch_surface, first=False, retained=False):
        gap_kind = 1 if first else 2 if retained else 0
        return running(offset, tire_key, compound.value if isinstance(compound, TireCompound)
                       else compound, age, gap_kind, surface_json[id(branch_surface)])

    def evaluate(initial, cache, actions_for):
        """Evaluate an offset-increasing strategy DAG without recursion."""
        if initial in cache:
            return cache[initial]
        frames = [[initial, None, 0, inf]]
        while frames:
            state, actions, index, best = frames[-1]
            if state in cache:
                frames.pop()
                continue
            if actions is None:
                if state[0] == horizon:
                    # Retain the value in the frame so completion propagates
                    # the incoming edge to its parent.
                    frames[-1][1:] = [[], 0, 0.0]
                    continue
                actions = actions_for(state)
                frames[-1][1:] = [actions, 0, inf]
                continue
            if index < len(actions):
                child, edge = actions[index]
                frames[-1][2] += 1
                if child in cache:
                    frames[-1][3] = min(frames[-1][3], edge + cache[child])
                else:
                    frames.append([child, None, 0, inf])
                continue
            cache[state] = best
            frames.pop()
            if frames:
                parent = frames[-1]
                child, edge = parent[1][parent[2] - 1]
                parent[3] = min(parent[3], edge + cache[state])
        return cache[initial]

    future_cache = {}

    def future_actions(state):
        offset, tire_key, compound, age, paid_stops, stopped_first = state
        before = surface(offset, paid_stops, stopped_first)
        current = TireCompound(compound)
        actions = []
        if before.tire_mismatch(current) != "critical":
            actions.append((
                (offset + 1, tire_key, current.value, age + 1,
                 paid_stops, stopped_first),
                run(offset, tire_key, current, age, before,
                    retained=tire_key == "retained"),
            ))
        candidates = tuple(
            compound for compound in TireCompound
            if before.tire_mismatch(compound) != "critical"
        )
        for candidate in candidates:
            after_paid = paid_stops + 1
            after = surface(offset, after_paid, stopped_first)
            actions.append((
                (offset + 1, candidate.value, candidate.value, 1,
                 after_paid, stopped_first),
                track.pit_lane_delta + service
                + run(offset, candidate.value, candidate, 0, after),
            ))
        return actions

    def future(initial):
        return evaluate(initial, future_cache, future_actions)

    retained_cache = {}

    def retained_actions(state):
        offset, tire_key, compound, age, paid_stops, stopped_first = state
        before = surface(offset, paid_stops, stopped_first)
        current = TireCompound(compound)
        if before.tire_mismatch(current) != "critical":
            return [((offset + 1, tire_key, current.value, age + 1,
                      paid_stops, stopped_first),
                     run(offset, tire_key, current, age, before, retained=True))]
        required = before.fresh_rain_compound()
        candidates = ((required,) if required is not None else tuple(
            compound for compound in (
                TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
            ) if before.tire_mismatch(compound) != "critical"
        ))
        actions = []
        for candidate in candidates:
            if before.tire_mismatch(candidate) == "critical":
                continue
            after_paid = paid_stops + 1
            after = surface(offset, after_paid, stopped_first)
            actions.append((
                (offset + 1, candidate.value, candidate.value, 1,
                 after_paid, stopped_first),
                track.pit_lane_delta + service
                + run(offset, candidate.value, candidate, 0, after),
            ))
        return actions

    def retained(initial):
        return evaluate(initial, retained_cache, retained_actions)

    first = surface(0, 0, False)
    wait = retained((0, "retained", current_tire.compound.value, tire_age, 0, False))
    pit = inf
    required = first.fresh_rain_compound()
    candidates = ((required,) if required is not None else tuple(
        compound for compound in (
            TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
        ) if first.tire_mismatch(compound) != "critical"
    ))
    for candidate in candidates:
        after = surface(0, 1, True)
        cost = (track.pit_lane_delta * pit_lane_factor + service
                + additional_current_stop_cost
                + run(0, candidate.value, candidate, 0, after)
                + future((1, candidate.value, candidate.value, 1, 1, True)))
        pit = min(pit, cost)
    return WeatherStopCosts(pit, wait)


def _surface_path(weather: Weather, laps: int, intervals=None) -> tuple[Weather, ...]:
    return projected_surfaces(weather, laps, intervals)


def _running(simulator, driver, car, track, tire, weather, lap, age, *, gap=None, aero=True,
             physical_total_laps=None):
    driver.current_tire_laps = age  # Projection owns this isolated driver.
    total_laps = track.total_laps if physical_total_laps is None else physical_total_laps
    return simulator.calculate_lap_time(
        driver, car, track, tire, weather, lap, total_laps,
        gap_to_car_ahead=gap, active_aero_enabled=aero, sample_variation=False,
    )


@lru_cache(maxsize=4096)
def _fresh_plan_costs(driver_json: str, car_json: str, track_json: str,
                      weather_json: str, tires_json: str, current_lap: int,
                      physical_total_laps: int, intervals=None
                      ) -> tuple[tuple[TireCompound, float, float], ...]:
    """Immutable first-set costs and first-lap times; cached futures assume green."""
    driver = Driver.model_validate_json(driver_json)
    # Snapshots originate from an existing model, whose supported mutable
    # overrides (for example deterministic pit_stop_std=0) must survive here.
    car = Car.model_construct(**json.loads(car_json))
    track = Track.model_validate_json(track_json)
    weather = Weather.model_validate_json(weather_json)
    tires = {TireCompound(key): Tire.model_validate(value)
             for key, value in json.loads(tires_json).items()}
    simulator = LapSimulator(np.random.default_rng(0))
    horizon = track.total_laps - current_lap + 1
    surfaces = _surface_path(weather, horizon, intervals)
    service = track.pit_lane_delta + expected_stationary_time(car)
    # Each cached row represents one possible future stop. Reusing rows at
    # their actual lap/weather lets later decisions share the same suffixes.
    future = [0.0] * (horizon + 1)
    for offset in range(horizon - 1, 0, -1):
        suffix = _fresh_plan_costs(
            driver_json, car_json, track_json,
            json.dumps(surfaces[offset].model_dump(), sort_keys=True),
            tires_json, current_lap + offset, physical_total_laps,
            suffix_weather_intervals(intervals, offset, surfaces[offset]),
        )
        future[offset] = service + min(cost for _, cost, _ in suffix)
    first = []
    for compound, tire in tires.items():
        stint, candidate, first_lap = 0.0, inf, 0.0
        for offset, surface in enumerate(surfaces):
            if surface.tire_mismatch(compound) == "critical":
                break
            running = _running(simulator, driver, car, track, tire, surface,
                               current_lap + offset, offset,
                               physical_total_laps=physical_total_laps)
            if offset == 0:
                first_lap = running
            stint += running
            candidate = min(candidate, stint + future[offset + 1])
        first.append((compound, candidate, first_lap))
    return tuple(first)


@lru_cache(maxsize=4096)
def _retained_costs(driver_json, car_json, track_json, weather_json, tire_json,
                    tire_age, current_lap, traffic_possible, physical_total_laps, tires_json,
                    intervals=None):
    """Retain while safe, paying for required future changes before running.

    This is a feasible waiting policy, not an optimal delayed-stop plan. Its
    future stops receive no current queue or neutralization discount.
    """
    driver = Driver.model_validate_json(driver_json)
    car = Car.model_construct(**json.loads(car_json))
    track = Track.model_validate_json(track_json)
    weather = Weather.model_validate_json(weather_json)
    tire = Tire.model_validate_json(tire_json)
    simulator = LapSimulator(np.random.default_rng(0))
    total, first = 0.0, 0.0
    for offset, surface in enumerate(_surface_path(
        weather, track.total_laps - current_lap + 1, intervals,
    )):
        if surface.tire_mismatch(tire.compound) == "critical":
            # A currently critical set is handled before this projection.
            # Each recursive replacement is noncritical, so its next stop
            # must occur at a strictly later lap.
            required = surface.fresh_rain_compound()
            candidates = {required} if required is not None else {
                TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
            }
            fresh = {TireCompound(key): value for key, value in json.loads(tires_json).items()}
            suffix = min((
                _retained_costs(
                    driver_json, car_json, track_json,
                    json.dumps(surface.model_dump(), sort_keys=True),
                    json.dumps(fresh[compound], sort_keys=True), 0, current_lap + offset,
                    traffic_possible, physical_total_laps, tires_json,
                    suffix_weather_intervals(intervals, offset, surface),
                )[0] for compound in candidates
                if surface.tire_mismatch(compound) != "critical"
            ), default=inf)
            total += track.pit_lane_delta + expected_stationary_time(car) + suffix
            break
        running = _running(simulator, driver, car, track, tire, surface,
                           current_lap + offset, tire_age + offset,
                           gap=0.0 if traffic_possible else None,
                           physical_total_laps=physical_total_laps)
        if offset == 0:
            first = running
        total += running
    return total, first


def weather_stop_costs(
    driver: Driver, car: Car, track: Track, weather: Weather, current_tire: Tire,
    tire_age: int, current_lap: int, *, pit_lane_factor: float = 1.0,
    additional_current_stop_cost: float = 0.0, current_lap_time_modifier: float = 1.0,
    active_aero_enabled: bool = True, traffic_possible: bool = True,
    physical_total_laps: int | None = None,
    weather_intervals: tuple[int, ...] | None = None,
    weather_clock: StrategyWeatherClock | None = None,
) -> WeatherStopCosts:
    """Compare retaining while safe with an optimistic schedule of paid refits.

    Rainfall/condition stay fixed while the shared surface model evolves. Later
    refits may use any noncritical fresh set without budget or compound-rule
    constraints, but pay the full expected stop cost. The waiting alternative
    retains each set until it becomes critical, then pays for an appropriate
    fresh set before running that lap. Only a currently critical set bypasses
    the veto. This is a cost bound under projected weather, not a forecast.
    The track bounds the planning horizon; physical_total_laps preserves the
    original fuel schedule when a time limit shortens that horizon.
    weather_intervals optionally supplies cumulative surface-update counts for
    each remaining own lap, starting at zero; None uses one update per lap.
    """
    physical_total_laps = track.total_laps if physical_total_laps is None else physical_total_laps
    horizon = track.total_laps - current_lap + 1
    if horizon <= 0:
        raise ValueError("current_lap must not exceed the race distance")
    intervals = normalize_weather_intervals(horizon, weather_intervals, weather=weather)
    _validate_weather_clock(weather_clock, horizon)
    weather_json = json.dumps(weather.model_dump(), sort_keys=True)
    if weather.tire_mismatch(current_tire.compound) == "critical":
        return WeatherStopCosts(0.0, inf)
    if weather_clock is not None:
        return _clock_weather_stop_costs(
            driver, car, track, weather, current_tire, int(tire_age), int(current_lap),
            pit_lane_factor=float(pit_lane_factor),
            additional_current_stop_cost=float(additional_current_stop_cost),
            current_lap_time_modifier=float(current_lap_time_modifier),
            active_aero_enabled=active_aero_enabled, traffic_possible=traffic_possible,
            physical_total_laps=int(physical_total_laps), weather_clock=weather_clock,
        )
    clean = driver.model_copy(deep=True)
    clean.reset_race_state()
    # Names and identifiers do not enter lap or service physics. Normalize
    # only those fields so equivalent entrants can share immutable plans.
    clean.id = clean.name = clean.team_id = "projection"
    clean_car = car.model_copy(update={"team_id": "projection", "team_name": "projection"})
    snapshots = (clean.model_dump(), clean_car.model_dump(), track.model_dump(),
                 {compound.value: tire.model_dump() for compound, tire in TIRE_COMPOUNDS.items()})
    driver_json, car_json, track_json, tires_json = (
        json.dumps(value, sort_keys=True) for value in snapshots
    )
    fresh = _fresh_plan_costs(
        driver_json, car_json, track_json, weather_json,
        tires_json, current_lap, physical_total_laps, intervals,
    )
    simulator = LapSimulator(np.random.default_rng(0))
    stay, stay_first = _retained_costs(
        driver_json, car_json, track_json, weather_json,
        json.dumps(current_tire.model_dump(), sort_keys=True), tire_age, current_lap,
        traffic_possible, physical_total_laps, tires_json, intervals,
    )
    if current_lap_time_modifier != 1.0 or not active_aero_enabled:
        actual_stay_first = _running(
            simulator, clean, car, track, current_tire, weather, current_lap, tire_age,
            gap=0.0 if traffic_possible else None, aero=active_aero_enabled,
            physical_total_laps=physical_total_laps,
        )
        stay += actual_stay_first * current_lap_time_modifier - stay_first
    required = weather.fresh_rain_compound()
    candidates = {required} if required is not None else {
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    }
    pit = inf
    for compound, cost, baseline_first in fresh:
        if compound not in candidates:
            continue
        actual_first = baseline_first if active_aero_enabled else _running(
            simulator, clean, car, track, TIRE_COMPOUNDS[compound],
            weather, current_lap, 0, aero=False, physical_total_laps=physical_total_laps,
        )
        pit = min(pit, cost - baseline_first + actual_first * current_lap_time_modifier)
    pit += (track.pit_lane_delta * pit_lane_factor + expected_stationary_time(car)
            + additional_current_stop_cost)
    return WeatherStopCosts(pit, stay)
