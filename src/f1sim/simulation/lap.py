"""Lap time calculation engine."""

from collections.abc import Callable
from functools import lru_cache

import numpy as np

from f1sim.models import Car, Driver, Tire, Track, Weather
from f1sim.models.tire import TireCompound
from f1sim.simulation.surface_projection import projected_surfaces

# The execution floor is an absolute fraction of the circuit reference lap.
# Keep it in one place so deterministic forecasts use exactly the same lower
# bound as sampled race laps.
MIN_LAP_TIME_FRACTION = 0.95


def minimum_lap_time(track: Track) -> float:
    """Return the shared absolute lower bound for a green lap on ``track``."""
    return track.base_lap_time * MIN_LAP_TIME_FRACTION


@lru_cache(maxsize=128)
def _track_profile_from_values(
    sectors: tuple[tuple[float, bool, float], ...], overtake_difficulty: float,
) -> tuple[float, float]:
    """Cache bounded scalar profiles by values, never by mutable model identity."""
    if not sectors:
        return 0.0, float(np.clip(1.0 - overtake_difficulty, 0.0, 1.0))
    weights = np.asarray([max(base_time, 0.0) for base_time, _, _ in sectors])
    if float(weights.sum()) <= 0.0:
        weights = np.ones(len(sectors), dtype=float)
    weights /= weights.sum()
    high_speed_mix = float(np.clip(
        sum(weight for weight, (_, high_speed, _) in zip(weights, sectors) if high_speed),
        0.0, 1.0,
    ))
    opportunity_mix = float(np.clip(
        sum(weight * opportunity for weight, (_, _, opportunity) in zip(weights, sectors)),
        0.0, 1.0,
    ))
    return high_speed_mix, opportunity_mix


@lru_cache(maxsize=1024)
def _track_car_delta_from_values(
    high_speed_mix: float, opportunity_mix: float, downforce_level: float,
    straight_line_speed: float, reference_lap_time: float,
) -> float:
    """Reuse the car/track term while retaining every input in the cache key."""
    corner_mix = 1.0 - high_speed_mix
    straight_mix = float(np.clip(0.55 * high_speed_mix + 0.45 * opportunity_mix, 0.0, 1.0))
    neutral = 0.8
    downforce_delta = (
        (neutral - downforce_level) * corner_mix
        + (downforce_level - neutral) * high_speed_mix
    )
    straight_delta = (neutral - straight_line_speed) * straight_mix
    return reference_lap_time * (0.006 * downforce_delta + 0.008 * straight_delta)


@lru_cache(maxsize=1024)
def _weather_pace_multiplier_from_values(
    weather_multiplier: float,
    wet_severity: float,
    wet_skill_modifier: float,
    wet_performance: float,
) -> float:
    """Cache the scalar weather term without retaining mutable model objects."""
    # Wet skill develops with exposure rather than switching on the
    # separate boolean threshold used for race rules and tyre safety.
    wet_adjustment = 1.0 + (1.0 - wet_skill_modifier) * 0.02 * wet_severity
    weather_multiplier *= wet_adjustment

    # Wet-performance is a car-package property, distinct from the driver's
    # ability to find grip. Include rain intensity as a signal even before
    # wetness crosses the tyre-change threshold.
    if wet_severity > 0.0:
        car_wet_penalty = (1.0 - wet_performance) * wet_severity * 0.06
        weather_multiplier *= 1.0 + float(np.clip(car_wet_penalty, 0.0, 0.06))

    return weather_multiplier


class LapSimulator:
    """Calculates realistic lap times with all contributing factors."""

    # Fresh slick compounds are not interchangeable: Pirelli's softer tyre
    # normally gives a measurable one-lap advantage, while the harder tyre
    # trades that grip for a longer usable stint.  Express the delta as a
    # fraction of the circuit's reference lap so the same calibration scales
    # naturally from a 70-second street lap to a 110-second power circuit.
    # Degradation is still applied independently below, so the soft tyre can
    # lose its fresh-lap advantage as a stint ages.
    _COMPOUND_PACE_FACTORS = {
        TireCompound.SOFT: -0.0045,
        TireCompound.MEDIUM: 0.0,
        TireCompound.HARD: 0.0045,
        # Wet compounds are deliberately only a small baseline offset.  The
        # weather mismatch/crossover model remains the dominant signal when
        # the track is damp or flooded.
        TireCompound.INTERMEDIATE: 0.006,
        TireCompound.WET: 0.010,
    }

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize the lap simulator.

        Args:
            rng: Random number generator (creates new if None)
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    @staticmethod
    def traffic_pace_contribution(gap: float | None) -> float:
        """Dirty-air seconds for a frozen gap, bounded between zero and 0.5."""
        return 0.0 if gap is None else 0.5 * max(0.0, min(1.0, 1.0 - gap / 2.0))

    def calculate_lap_time(
        self,
        driver: Driver,
        car: Car,
        track: Track,
        tire: Tire,
        weather: Weather,
        lap_number: int,
        total_laps: int,
        gap_to_car_ahead: float | None = None,
        active_aero_enabled: bool = True,
        overtake_mode_active: bool = False,
        sample_variation: bool = True,
    ) -> float:
        """Calculate a single lap time with all factors.

        Args:
            driver: Driver performing the lap
            car: Car being driven
            track: Circuit being raced
            tire: Current tire set
            weather: Current weather conditions
            lap_number: Current lap (1-indexed)
            total_laps: Total race laps
            gap_to_car_ahead: Gap in seconds to car ahead (None if leading)
            active_aero_enabled: Whether Straight Mode is available this lap
            overtake_mode_active: Whether this car deployed Overtake Mode
            sample_variation: Sample driver variation, or project a noise-free lap

        Returns:
            Lap time in seconds
        """
        # Base lap time from track
        base_time = track.base_lap_time

        # Car performance delta.  ``base_pace`` describes the package as a
        # whole, while the aero/top-speed terms below let the circuit profile
        # decide where that pace is useful.  Keeping the specialised terms to
        # well under one percent of a lap avoids turning a small rating
        # difference into an implausible multi-second swing.
        car_delta = car.pace_delta_seconds(base_time)
        car_delta += self._track_car_delta(car, track, base_time)

        # Driver skill effect (top driver ~0.3-0.5s faster per lap than midfield)
        # Skill range is ~0.75-1.0, so delta ranges from 0 to ~0.75s per lap
        skill_delta = (1.0 - driver.skill_rating) * base_time * 0.03

        # Random variation based on driver consistency
        variation_std = driver.lap_time_variation_std(base_std=0.25)
        random_variation = self.rng.normal(0, variation_std) if sample_variation else 0.0

        tire_delta = self.tire_pace_contribution(
            driver, car, track, tire, driver.current_tire_laps
        )

        # Fuel effect (lighter = faster, ~0.03s per lap of fuel burned)
        fuel_remaining_pct = (total_laps - lap_number + 1) / total_laps
        fuel_delta = fuel_remaining_pct * base_time * 0.02  # ~2% slower at race start

        weather_multiplier = self.weather_pace_multiplier(driver, car, weather)

        # Tire/weather mismatch penalty (catastrophic if wrong tires)
        mismatch_penalty = self._tire_weather_mismatch(tire, weather)

        # Traffic/dirty air effect
        traffic_delta = self.traffic_pace_contribution(gap_to_car_ahead)

        # Active Aero Straight Mode is common to every green-running car on
        # the configured sections.  Unlike Overtake Mode, it is deliberately
        # not proximity-gated by the gap to the car ahead.
        active_aero_gain = 0.0
        if active_aero_enabled and track.total_active_aero_gain > 0.0:
            _, opportunity_mix = self._track_profile(track)
            active_aero_effectiveness = (0.65 + 0.35 * opportunity_mix) * (
                0.9 + 0.2 * car.straight_line_speed
            )
            active_aero_gain = track.total_active_aero_gain * 0.8 * active_aero_effectiveness

        # Overtake Mode is a separate, short-duration deployment.  Energy
        # accounting is owned by RaceSimulator; this term only converts an
        # eligible deployment into a bounded lap-time advantage.  Scale its
        # effect with actual straight-mode opportunities and package speed so
        # a zero-zone venue (e.g. Monaco) cannot accidentally gain a bonus.
        overtake_mode_gain = 0.0
        if overtake_mode_active and not weather.is_wet():
            active_aero_mix = float(
                np.clip(track.total_active_aero_gain / 1.0, 0.0, 1.0)
            )
            overtake_mode_gain = min(
                0.35,
                0.18
                * active_aero_mix
                * (0.85 + 0.3 * car.straight_line_speed),
            )

        return self._compose_lap_time(
            base_time, car_delta, skill_delta, random_variation,
            tire_delta, fuel_delta, traffic_delta, active_aero_gain,
            overtake_mode_gain, weather_multiplier, mismatch_penalty, track,
        )

    @staticmethod
    def _compose_lap_time(
        base_time: float,
        car_delta: float,
        skill_delta: float,
        random_variation: float,
        tire_delta: float,
        fuel_delta: float,
        traffic_delta: float,
        active_aero_gain: float,
        overtake_mode_gain: float,
        weather_multiplier: float,
        mismatch_penalty: float,
        track: Track,
    ) -> float:
        """Combine lap terms in the same order used by all evaluators."""
        # Keep this grouping stable: deterministic strategy choices are keyed
        # by these floats, so an algebraically equivalent rewrite can change
        # a tie at the last bit and select a different physical tyre set.
        lap_time = (
            base_time
            + car_delta
            + skill_delta
            + random_variation
        )
        lap_time += (
            tire_delta
            + fuel_delta
            + traffic_delta
            - active_aero_gain
            - overtake_mode_gain
        )
        lap_time *= weather_multiplier
        lap_time += mismatch_penalty  # Add after multiplier (flat penalty)

        # Ensure minimum realistic lap time
        return max(minimum_lap_time(track), lap_time)

    def prepare_deterministic_lap_time(
        self,
        driver: Driver,
        car: Car,
        track: Track,
        total_laps: int,
    ) -> Callable[..., float] | None:
        """Prepare a native, deterministic evaluator for repeated forecasts.

        The inventory strategy evaluates the same driver/car/track package for
        hundreds of thousands of candidate laps.  This prepares the invariant
        pace terms once while leaving tyre age, weather, fuel lap, gap, and
        active-aero state as explicit evaluator inputs.

        ``None`` means that a custom simulator implementation must be used.
        The guard is deliberately conservative: subclasses and monkeypatched
        physics methods retain the normal ``calculate_lap_time`` dispatch
        rather than being silently bypassed by this native shortcut.
        """
        if not self._native_deterministic_evaluator_available():
            return None
        if (type(driver) is not Driver or type(car) is not Car
                or type(track) is not Track):
            return None
        if type(total_laps) is not int or total_laps <= 0:
            raise ValueError("total_laps must be a positive integer")

        base_time = track.base_lap_time
        car_delta = car.pace_delta_seconds(base_time)
        car_delta += self._track_car_delta(car, track, base_time)
        skill_delta = (1.0 - driver.skill_rating) * base_time * 0.03

        stress = 0.75 + 0.5 * min(1.0, max(0.0, track.tire_stress))
        degradation_multiplier = min(
            1.75, max(0.5, car.tire_degradation_factor * stress)
        )

        total_active_aero_gain = track.total_active_aero_gain
        active_aero_gain = 0.0
        if total_active_aero_gain > 0.0:
            _, opportunity_mix = self._track_profile(track)
            active_aero_effectiveness = (0.65 + 0.35 * opportunity_mix) * (
                0.9 + 0.2 * car.straight_line_speed
            )
            active_aero_gain = (
                total_active_aero_gain * 0.8 * active_aero_effectiveness
            )

        # Capture the native bound method for the rare custom-model fallback.
        # It retains the same driver-state update as the ordinary planner path.
        calculate_lap_time = self.calculate_lap_time
        tire_pace = LapSimulator._tire_pace_from_multiplier
        compose = self._compose_lap_time
        wet_skill_modifier = driver.wet_skill_modifier
        wet_performance = car.wet_performance

        def evaluate(
            tire: Tire,
            weather: Weather,
            lap_number: int,
            tire_age: int,
            gap_to_car_ahead: float | None = None,
            active_aero_enabled: bool = True,
        ) -> float:
            # Model subclasses may override methods used by the native path.
            # Delegate those values through the original public method so a
            # prepared evaluator never changes extension semantics.
            if type(tire) is not Tire or type(weather) is not Weather:
                driver.current_tire_laps = tire_age
                return calculate_lap_time(
                    driver, car, track, tire, weather, lap_number, total_laps,
                    gap_to_car_ahead=gap_to_car_ahead,
                    active_aero_enabled=active_aero_enabled,
                    sample_variation=False,
                )

            tire_delta = tire_pace(
                driver,
                car,
                track,
                tire,
                tire_age,
                degradation_multiplier,
            )
            fuel_remaining_pct = (total_laps - lap_number + 1) / total_laps
            fuel_delta = fuel_remaining_pct * base_time * 0.02
            weather_multiplier = _weather_pace_multiplier_from_values(
                weather.lap_time_multiplier(),
                weather.wet_severity(),
                wet_skill_modifier,
                wet_performance,
            )
            mismatch_penalty = LapSimulator._tire_weather_mismatch(tire, weather)
            traffic_delta = LapSimulator.traffic_pace_contribution(
                gap_to_car_ahead
            )
            return compose(
                base_time,
                car_delta,
                skill_delta,
                0.0,
                tire_delta,
                fuel_delta,
                traffic_delta,
                active_aero_gain if active_aero_enabled else 0.0,
                0.0,
                weather_multiplier,
                mismatch_penalty,
                track,
            )

        return evaluate

    def _native_deterministic_evaluator_available(self) -> bool:
        """Whether native deterministic preparation can preserve dispatch."""
        # The method references are populated after class creation below.  A
        # small helper keeps the guard readable and also handles instance-level
        # monkeypatches of ``calculate_lap_time``.
        if type(self) is not LapSimulator:
            return False
        if any(name in self.__dict__ for name in _NATIVE_METHODS):
            return False
        if _method_function(LapSimulator.calculate_lap_time) is not _NATIVE_METHODS[
            "calculate_lap_time"
        ]:
            return False
        for name, native in _NATIVE_METHODS.items():
            if name == "calculate_lap_time":
                continue
            if _method_function(getattr(LapSimulator, name)) is not native:
                return False
        return True

    @staticmethod
    def weather_pace_multiplier(driver: Driver, car: Car, weather: Weather) -> float:
        """Shared weather scaling for actual laps and tyre-relative pace."""
        # Call model helpers on every invocation so mutable models and custom
        # weather implementations remain visible to the scalar cache.
        return _weather_pace_multiplier_from_values(
            weather.lap_time_multiplier(),
            weather.wet_severity(),
            driver.wet_skill_modifier,
            car.wet_performance,
        )

    @classmethod
    def tire_weather_pace_contribution(
        cls, driver: Driver, car: Car, track: Track, tire: Tire, tire_age: int, weather: Weather,
    ) -> float:
        """Tyre-only seconds: wear/compound scaled by weather plus flat mismatch."""
        return (cls.tire_pace_contribution(driver, car, track, tire, tire_age)
                * cls.weather_pace_multiplier(driver, car, weather)
                + cls._tire_weather_mismatch(tire, weather))

    @classmethod
    def tire_pace_contribution(
        cls, driver: Driver, car: Car, track: Track, tire: Tire, tire_age: int
    ) -> float:
        """Deterministic tyre seconds before weather, shared with stint planning."""
        stress = 0.75 + 0.5 * min(1.0, max(0.0, track.tire_stress))
        degradation_multiplier = min(
            1.75, max(0.5, car.tire_degradation_factor * stress)
        )
        return cls._tire_pace_from_multiplier(
            driver, car, track, tire, tire_age, degradation_multiplier,
        )

    @classmethod
    def _tire_pace_from_multiplier(
        cls,
        driver: Driver,
        car: Car,
        track: Track,
        tire: Tire,
        tire_age: int,
        degradation_multiplier: float,
    ) -> float:
        """Evaluate tyre pace after the invariant stress multiplier is known."""
        degradation = tire.time_penalty_per_lap(
            tire_age, track.base_lap_time, driver.tire_management
        )
        degradation *= degradation_multiplier
        return cls._compound_pace_delta(tire, track.base_lap_time, tire_age) + degradation

    @classmethod
    def projected_tire_stint_cost(
        cls, driver: Driver, car: Car, track: Track, tire: Tire, laps: int
    ) -> float:
        """Sum fresh-set tyre seconds over a dry stint without drawing randomness.

        Common fuel, car, traffic and driver pace terms cancel between sets.
        This projects the existing pace model, not future weather or stop timing.
        """
        return sum(cls.tire_pace_contribution(driver, car, track, tire, age)
                   for age in range(laps))

    def projected_stint_lap_cost(
        self, driver: Driver, car: Car, track: Track, tire: Tire, laps: int,
        current_lap: int, weather: Weather, *, physical_total_laps: int | None = None,
        current_lap_time_modifier: float = 1.0, active_aero_enabled: bool = True,
        weather_intervals: tuple[int, ...] | None = None,
    ) -> float:
        """Fresh-set running time, including the lap floor and projected surface.

        Fuel follows the physical race distance even when strategy plans to an
        earlier finish. Only the upcoming lap uses current race control; later
        laps assume green running. No future incidents or traffic are forecast.
        ``weather_intervals`` optionally supplies cumulative deterministic
        surface-update counts for each projected lap, starting at zero.
        """
        projected_driver = driver.model_copy(deep=True)
        surfaces = (projected_surfaces(weather, laps, weather_intervals)
                    if weather_intervals is not None else None)
        surface = weather.model_copy(deep=True)
        total = 0.0
        for age in range(laps):
            if surfaces is not None:
                surface = surfaces[age]
            projected_driver.current_tire_laps = age
            total += self.calculate_lap_time(
                projected_driver, car, track, tire, surface, current_lap + age,
                physical_total_laps if physical_total_laps is not None else track.total_laps,
                active_aero_enabled=active_aero_enabled if age == 0 else True,
                sample_variation=False,
            ) * (current_lap_time_modifier if age == 0 else 1.0)
            if surfaces is None:
                surface = surface.project_surface()
        return total

    @classmethod
    def _compound_pace_delta(
        cls,
        tire: Tire,
        reference_lap_time: float,
        tire_age: int = 0,
    ) -> float:
        """Return the fresh-compound pace offset in seconds.

        The offset is intentionally small compared with weather mismatch and
        tyre degradation.  This keeps wet/intermediate crossover behaviour
        governed by track conditions while ensuring equal-age dry compounds
        have a realistic order.
        """
        factor = cls._COMPOUND_PACE_FACTORS.get(tire.compound, 0.0)
        if tire.compound == TireCompound.SOFT:
            # Fresh soft grip fades into the normal degradation curve as the
            # stint ages.  This creates the expected crossover against a
            # medium tyre instead of preserving a permanent qualifying-like
            # bonus after the soft set has fallen off its cliff.
            factor *= max(0.0, 1.0 - max(tire_age, 0) / 35.0)
        return reference_lap_time * factor

    @staticmethod
    def _track_profile(track: Track) -> tuple[float, float]:
        """Return time-weighted high-speed and passing-opportunity mixes.

        Tracks built from sparse data may not have sectors.  In that case the
        opportunity value still gives the car model a useful, bounded signal
        without inventing a circuit layout.
        """
        return _track_profile_from_values(
            tuple((float(sector.base_time), sector.is_high_speed,
                   float(sector.overtake_opportunity)) for sector in track.sectors),
            track.overtake_difficulty,
        )

    @classmethod
    def _track_car_delta(cls, car: Car, track: Track, reference_lap_time: float) -> float:
        """Calculate bounded aero/top-speed delta for a car on a track.

        ``downforce_level`` is rewarded in corner-heavy sectors and incurs a
        small drag cost in high-speed sectors.  ``straight_line_speed`` is
        rewarded on high-speed and overtaking sections.  Ratings are centred
        on the model's neutral value (0.8), so existing default cars retain
        their previous pace.
        """
        high_speed_mix, opportunity_mix = cls._track_profile(track)
        # At most roughly 1.2% for a maximally specialised rating on a
        # representative lap, before the normal base-pace term is applied.
        return _track_car_delta_from_values(
            high_speed_mix, opportunity_mix, car.downforce_level,
            car.straight_line_speed, reference_lap_time,
        )

    def calculate_pit_stop_time(self, car: Car, tire_change: bool = True) -> float:
        """Calculate pit stop duration.

        Args:
            car: Car performing pit stop
            tire_change: Whether tires are being changed

        Returns:
            Pit stop time in seconds (stationary + pit lane delta)
        """
        if tire_change:
            # Sample from team's pit stop distribution
            stationary_time = self.rng.normal(car.pit_stop_avg, car.pit_stop_std)
            # Occasional slow stops
            if self.rng.random() < 0.05:
                stationary_time += self.rng.uniform(2, 8)
            stationary_time = max(1.8, stationary_time)  # Minimum possible
        else:
            # Drive-through or stop-go
            stationary_time = 0.0

        return stationary_time

    def calculate_qualifying_lap(
        self,
        driver: Driver,
        car: Car,
        track: Track,
        tire: Tire,
        weather: Weather,
        push_level: float = 1.0,
        sample_variation: bool = True,
    ) -> float:
        """Calculate a qualifying lap time.

        Args:
            driver: Driver performing the lap
            car: Car being driven
            track: Circuit being raced
            tire: Current tire set (usually soft)
            weather: Current weather conditions
            push_level: How hard the driver is pushing (0-1)
            sample_variation: Sample variation/mistakes, or return noise-free pace

        Returns:
            Lap time in seconds
        """
        # Base qualifying time (faster than race pace)
        base_time = track.base_lap_time * 0.98  # ~2% quicker

        # Car performance, including the circuit's aero/top-speed balance.
        car_delta = car.pace_delta_seconds(base_time)
        car_delta += self._track_car_delta(car, track, base_time)

        # Driver skill (more important in qualifying)
        skill_delta = (1.0 - driver.skill_rating) * base_time * 0.012

        # Push level variation (higher push = more risk of mistakes)
        risk_factor = push_level * 0.3
        variation_std = driver.lap_time_variation_std(base_std=0.15)
        random_variation = self.rng.normal(0, variation_std) if sample_variation else 0.0

        # Mistake chance increases with push
        if sample_variation and self.rng.random() < risk_factor * 0.1:
            # Small mistake
            random_variation += self.rng.uniform(0.2, 1.0)
        elif sample_variation and self.rng.random() < risk_factor * 0.02:
            # Big mistake (ruined lap)
            random_variation += self.rng.uniform(3.0, 10.0)

        # Fresh-set grip, including rain compounds when required.
        tire_bonus = (tire.initial_grip - 1.0) * 0.5  # Bonus from soft tire grip

        weather_multiplier = self.weather_pace_multiplier(driver, car, weather)

        lap_time = (
            base_time + car_delta + skill_delta + random_variation - tire_bonus
        ) * weather_multiplier
        lap_time += self._tire_weather_mismatch(tire, weather)

        return max(track.base_lap_time * 0.93, lap_time)

    @staticmethod
    def _tire_weather_mismatch(tire: Tire, weather: Weather) -> float:
        """Calculate penalty for wrong tire compound in current conditions.

        Args:
            tire: Current tire compound
            weather: Current weather conditions

        Returns:
            Time penalty in seconds (0 if tires are appropriate)
        """
        compound = tire.compound
        is_slick = compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
        is_inter = compound == TireCompound.INTERMEDIATE
        is_wet = compound == TireCompound.WET

        water = max(0.0, min(1.0, weather.track_wetness))
        # Join explicit model anchors continuously. Preserve dry/flooded
        # endpoints and the existing zero-penalty windows; tiny changes in
        # surface water must not introduce fixed multi-second pace jumps.
        # Tyre survivability remains a separate Weather.tire_mismatch rule.
        if is_slick:
            # (water, seconds): (0.2, 0), (0.5, 7.5), (1, 30).
            return 25.0 * max(0.0, water - 0.2) + 20.0 * max(0.0, water - 0.5)
        if is_inter:
            # (0, 7), (0.15, 0), (0.8, 0), (1, 10).
            return (7.0 * max(0.0, (0.15 - water) / 0.15)
                    + 10.0 * max(0.0, (water - 0.8) / 0.2))
        if is_wet:
            # (0, 14), (0.3, 0); no added mismatch above 0.3.
            return 14.0 * max(0.0, (0.3 - water) / 0.3)
        return 0.0


def _method_function(value):
    """Return the underlying function for a bound or class method."""
    return getattr(value, "__func__", value)


# Keep immutable references to the native dispatch points.  Runtime tests and
# integrations commonly monkeypatch ``calculate_lap_time``; comparing against
# these references makes preparation opt out instead of changing their model.
_NATIVE_METHODS = {
    name: _method_function(getattr(LapSimulator, name))
    for name in (
        "calculate_lap_time",
        "_compose_lap_time",
        "_track_car_delta",
        "_track_profile",
        "_tire_pace_from_multiplier",
        "_compound_pace_delta",
        "tire_pace_contribution",
        "weather_pace_multiplier",
        "_tire_weather_mismatch",
        "traffic_pace_contribution",
    )
}
