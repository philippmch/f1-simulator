"""Weather model and conditions."""

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field


class WeatherCondition(str, Enum):
    """Weather condition types."""

    DRY = "dry"
    CLOUDY = "cloudy"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"


class Weather(BaseModel):
    """Represents current weather conditions."""

    condition: WeatherCondition = Field(
        default=WeatherCondition.DRY,
        description="Current weather condition",
    )
    track_temperature: float = Field(
        default=35.0,
        ge=10.0,
        le=60.0,
        description="Track surface temperature in Celsius",
    )
    air_temperature: float = Field(
        default=25.0,
        ge=5.0,
        le=45.0,
        description="Air temperature in Celsius",
    )
    humidity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Relative humidity (0-1)",
    )
    rain_intensity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Rain intensity (0 = none, 1 = heavy)",
    )
    track_wetness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Track wetness level (0 = dry, 1 = flooded)",
    )
    change_probability: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Probability of weather change per lap",
    )

    def lap_time_multiplier(self) -> float:
        """Calculate lap time multiplier based on conditions.

        Returns:
            Multiplier > 1.0 for slower conditions
        """
        if self.condition == WeatherCondition.DRY:
            return 1.0
        elif self.condition == WeatherCondition.CLOUDY:
            return 1.01  # Slightly cooler track
        elif self.condition == WeatherCondition.LIGHT_RAIN:
            return 1.05 + (self.track_wetness * 0.05)
        else:  # HEAVY_RAIN
            return 1.10 + (self.track_wetness * 0.10)

    def is_wet(self) -> bool:
        """Check if conditions require wet/intermediate tires."""
        return self.track_wetness > 0.3

    def requires_wet_tires(self) -> bool:
        """Check if full wet tires are needed."""
        return self.track_wetness > 0.7

    def evolve(self, rng: np.random.Generator) -> "Weather":
        """Generate next lap's weather based on current conditions.

        Args:
            rng: Random number generator

        Returns:
            New Weather instance for next lap
        """
        new_weather = self.model_copy(deep=True)

        # Track wetness changes even without condition change
        if new_weather.rain_intensity > 0:
            # Rain adds wetness quickly
            wetness_gain = 0.08 + 0.12 * new_weather.rain_intensity
            new_weather.track_wetness = min(1.0, self.track_wetness + wetness_gain)
        else:
            # Track dries slowly (takes ~20 laps to fully dry)
            new_weather.track_wetness = max(0.0, self.track_wetness - 0.03)

        if rng.random() > self.change_probability:
            # No condition change, but wetness already updated
            return new_weather

        # Weather condition can change
        if self.condition == WeatherCondition.DRY:
            if rng.random() < 0.5:
                new_weather.condition = WeatherCondition.CLOUDY
                new_weather.humidity = min(1.0, self.humidity + 0.2)
        elif self.condition == WeatherCondition.CLOUDY:
            roll = rng.random()
            if roll < 0.25:
                new_weather.condition = WeatherCondition.DRY
            elif roll < 0.6:
                # Rain starts
                new_weather.condition = WeatherCondition.LIGHT_RAIN
                new_weather.rain_intensity = rng.uniform(0.2, 0.4)
        elif self.condition == WeatherCondition.LIGHT_RAIN:
            roll = rng.random()
            if roll < 0.25:
                # Rain stops but track still wet
                new_weather.condition = WeatherCondition.CLOUDY
                new_weather.rain_intensity = 0.0
            elif roll < 0.5:
                # Rain intensifies
                new_weather.condition = WeatherCondition.HEAVY_RAIN
                new_weather.rain_intensity = rng.uniform(0.7, 1.0)
            # else: stays light rain
        else:  # HEAVY_RAIN
            roll = rng.random()
            if roll < 0.35:
                new_weather.condition = WeatherCondition.LIGHT_RAIN
                new_weather.rain_intensity = rng.uniform(0.2, 0.4)
            # Heavy rain tends to persist

        return new_weather
