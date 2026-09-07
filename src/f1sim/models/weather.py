"""Weather model and conditions."""

from enum import Enum

import numpy as np
from pydantic import BaseModel, Field

from f1sim.models.tire import TireCompound

# Normalized surface response: drainage balances rainfall at the intensity target.
WETNESS_RESPONSE_PER_LAP = 0.2


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

    def fresh_rain_compound(self) -> TireCompound | None:
        """Choose a fresh rain set, or leave dry compound selection to strategy."""
        if self.requires_wet_tires():
            return TireCompound.WET
        if self.track_wetness > 0.2 or self.rain_intensity > 0.4:
            return TireCompound.INTERMEDIATE
        return None

    def project_surface(self) -> "Weather":
        """Advance one lap of drainage under unchanged rainfall, without randomness."""
        projected = self.model_copy(deep=True)
        if self.rain_intensity > 0:
            projected.track_wetness += WETNESS_RESPONSE_PER_LAP * (
                self.rain_intensity - self.track_wetness
            )
        else:
            projected.track_wetness = max(0.0, self.track_wetness - 0.03)
        return projected

    def tire_mismatch(self, compound: TireCompound) -> str:
        """Shared survivability thresholds for fitted tyres and strategy projections."""
        slick = compound in (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
        if slick and (self.track_wetness > 0.45
                      or (self.track_wetness > 0.35 and self.rain_intensity > 0.6)):
            return "critical"
        if slick and self.fresh_rain_compound() is not None:
            return "suboptimal"
        if compound == TireCompound.INTERMEDIATE and self.track_wetness > 0.72:
            return "suboptimal"
        if compound == TireCompound.WET and self.track_wetness < 0.2 and self.rain_intensity < 0.3:
            return "critical"
        if compound == TireCompound.WET and self.track_wetness < 0.42:
            return "suboptimal"
        if (compound == TireCompound.INTERMEDIATE
                and self.track_wetness < 0.08 and self.rain_intensity < 0.15):
            return "critical"
        return "ok"

    def evolve(self, rng: np.random.Generator) -> "Weather":
        """Generate next lap's weather based on current conditions.

        Args:
            rng: Random number generator

        Returns:
            New Weather instance for next lap
        """
        new_weather = self.project_surface()

        if rng.random() >= self.change_probability:
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
