"""Lap time calculation engine."""

import numpy as np

from f1sim.models import Car, Driver, Tire, Track, Weather
from f1sim.models.tire import TireCompound


class LapSimulator:
    """Calculates realistic lap times with all contributing factors."""

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize the lap simulator.

        Args:
            rng: Random number generator (creates new if None)
        """
        self.rng = rng if rng is not None else np.random.default_rng()

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
        is_drs_enabled: bool = True,
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
            is_drs_enabled: Whether DRS is active this lap

        Returns:
            Lap time in seconds
        """
        # Base lap time from track
        base_time = track.base_lap_time

        # Car performance delta
        car_delta = car.pace_delta_seconds(base_time)

        # Driver skill effect (top driver ~0.3-0.5s faster per lap than midfield)
        # Skill range is ~0.75-1.0, so delta ranges from 0 to ~0.75s per lap
        skill_delta = (1.0 - driver.skill_rating) * base_time * 0.03

        # Random variation based on driver consistency
        variation_std = driver.lap_time_variation_std(base_std=0.25)
        random_variation = self.rng.normal(0, variation_std)

        # Tire degradation effect
        tire_delta = tire.time_penalty_per_lap(
            driver.current_tire_laps,
            base_time,
            driver.tire_management,
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

        # Tire/weather mismatch penalty (catastrophic if wrong tires)
        mismatch_penalty = self._tire_weather_mismatch(tire, weather)

        # Traffic/dirty air effect
        traffic_delta = 0.0
        if gap_to_car_ahead is not None and gap_to_car_ahead < 2.0:
            # Dirty air effect (worse when closer, up to 0.5s loss)
            dirty_air_factor = max(0, 1.0 - gap_to_car_ahead / 2.0)
            traffic_delta = dirty_air_factor * 0.5

        # DRS effect (if enabled and within 1s of car ahead)
        drs_gain = 0.0
        if (
            is_drs_enabled
            and gap_to_car_ahead is not None
            and gap_to_car_ahead <= 1.0
            and not weather.is_wet()
        ):
            drs_gain = track.total_drs_gain * 0.8  # 80% of theoretical gain

        # Calculate final lap time
        lap_time = base_time + car_delta + skill_delta + random_variation
        lap_time += tire_delta + fuel_delta + traffic_delta - drs_gain
        lap_time *= weather_multiplier
        lap_time += mismatch_penalty  # Add after multiplier (flat penalty)

        # Ensure minimum realistic lap time
        min_lap_time = track.base_lap_time * 0.95
        return max(min_lap_time, lap_time)

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

        # Car performance
        car_delta = car.pace_delta_seconds(base_time)

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

        # Weather effect
        weather_multiplier = weather.lap_time_multiplier()

        lap_time = (base_time + car_delta + skill_delta + random_variation - tire_bonus) * weather_multiplier

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
