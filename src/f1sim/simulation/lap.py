"""Lap time calculation engine."""

import numpy as np

from f1sim.models import Car, Driver, Tire, Track, Weather
from f1sim.models.tire import TireCompound


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
        random_variation = self.rng.normal(0, variation_std)

        tire_delta = self.tire_pace_contribution(
            driver, car, track, tire, driver.current_tire_laps
        )

        # Fuel effect (lighter = faster, ~0.03s per lap of fuel burned)
        fuel_remaining_pct = (total_laps - lap_number + 1) / total_laps
        fuel_delta = fuel_remaining_pct * base_time * 0.02  # ~2% slower at race start

        # Weather effect
        weather_multiplier = weather.lap_time_multiplier()

        # Adjust for driver wet skill
        if weather.is_wet():
            wet_adjustment = 1.0 + (1.0 - driver.wet_skill_modifier) * 0.02
            weather_multiplier *= wet_adjustment

        # Wet-performance is a car-package property, distinct from the
        # driver's ability to find grip.  Include rain intensity as a signal
        # even before wetness crosses the tyre-change threshold.
        wet_severity = float(
            np.clip(max(weather.track_wetness, weather.rain_intensity * 0.7), 0.0, 1.0)
        )
        if wet_severity > 0.0:
            car_wet_penalty = (1.0 - car.wet_performance) * wet_severity * 0.06
            weather_multiplier *= 1.0 + float(np.clip(car_wet_penalty, 0.0, 0.06))

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

        # Calculate final lap time
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
        min_lap_time = track.base_lap_time * 0.95
        return max(min_lap_time, lap_time)

    @classmethod
    def tire_pace_contribution(
        cls, driver: Driver, car: Car, track: Track, tire: Tire, tire_age: int
    ) -> float:
        """Deterministic tyre seconds before weather, shared with stint planning."""
        degradation = tire.time_penalty_per_lap(
            tire_age, track.base_lap_time, driver.tire_management
        )
        stress = 0.75 + 0.5 * min(1.0, max(0.0, track.tire_stress))
        degradation *= min(1.75, max(0.5, car.tire_degradation_factor * stress))
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
        sectors = list(track.sectors)
        if not sectors:
            return 0.0, float(np.clip(1.0 - track.overtake_difficulty, 0.0, 1.0))

        weights = np.asarray([max(float(sector.base_time), 0.0) for sector in sectors])
        if float(weights.sum()) <= 0.0:
            weights = np.ones(len(sectors), dtype=float)
        weights /= weights.sum()

        high_speed_mix = float(
            np.clip(
                sum(weight for weight, sector in zip(weights, sectors) if sector.is_high_speed),
                0.0,
                1.0,
            )
        )
        opportunity_mix = float(
            np.clip(
                sum(
                    weight * float(sector.overtake_opportunity)
                    for weight, sector in zip(weights, sectors)
                ),
                0.0,
                1.0,
            )
        )
        return high_speed_mix, opportunity_mix

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
        corner_mix = 1.0 - high_speed_mix
        straight_mix = float(np.clip(0.55 * high_speed_mix + 0.45 * opportunity_mix, 0.0, 1.0))

        neutral = 0.8
        downforce_delta = (
            (neutral - car.downforce_level) * corner_mix
            + (car.downforce_level - neutral) * high_speed_mix
        )
        straight_delta = (neutral - car.straight_line_speed) * straight_mix

        # At most roughly 1.2% for a maximally specialised rating on a
        # representative lap, before the normal base-pace term is applied.
        return reference_lap_time * (0.006 * downforce_delta + 0.008 * straight_delta)

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
    ) -> float:
        """Calculate a qualifying lap time.

        Args:
            driver: Driver performing the lap
            car: Car being driven
            track: Circuit being raced
            tire: Current tire set (usually soft)
            weather: Current weather conditions
            push_level: How hard the driver is pushing (0-1)

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
        random_variation = self.rng.normal(0, variation_std)

        # Mistake chance increases with push
        if self.rng.random() < risk_factor * 0.1:
            # Small mistake
            random_variation += self.rng.uniform(0.2, 1.0)
        elif self.rng.random() < risk_factor * 0.02:
            # Big mistake (ruined lap)
            random_variation += self.rng.uniform(3.0, 10.0)

        # Tire grip (fresh soft tires in qualifying)
        tire_bonus = (tire.initial_grip - 1.0) * 0.5  # Bonus from soft tire grip

        # Weather effect.  Qualifying still benefits from a car's wet package
        # when the session is not fully dry.
        weather_multiplier = weather.lap_time_multiplier()
        wet_severity = float(
            np.clip(max(weather.track_wetness, weather.rain_intensity * 0.7), 0.0, 1.0)
        )
        if wet_severity > 0.0:
            weather_multiplier *= 1.0 + float(
                np.clip((1.0 - car.wet_performance) * wet_severity * 0.06, 0.0, 0.06)
            )

        lap_time = (
            base_time + car_delta + skill_delta + random_variation - tire_bonus
        ) * weather_multiplier

        return max(track.base_lap_time * 0.93, lap_time)

    def _tire_weather_mismatch(self, tire: Tire, weather: Weather) -> float:
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

        track_wetness = weather.track_wetness

        # Slicks on wet track = disaster (aquaplaning)
        if is_slick and track_wetness > 0.5:
            # 10-30 seconds slower per lap, plus high crash risk
            return 10.0 + (track_wetness - 0.5) * 40.0

        # Slicks on damp track = very slow, but survivable
        if is_slick and track_wetness > 0.2:
            return 3.0 + (track_wetness - 0.2) * 15.0

        # Inters on very wet track = too much water
        if is_inter and track_wetness > 0.8:
            return 5.0 + (track_wetness - 0.8) * 25.0

        # Wet tires on dry track = massive overheating, graining
        if is_wet and track_wetness < 0.3:
            return 8.0 + (0.3 - track_wetness) * 20.0

        # Inters on dry track = overheating but less severe
        if is_inter and track_wetness < 0.15:
            return 4.0 + (0.15 - track_wetness) * 20.0

        # Inters in optimal window (0.3-0.6 wetness) = good
        # Wets in optimal window (0.6+ wetness) = good
        # Slicks on dry (< 0.2 wetness) = good

        return 0.0
