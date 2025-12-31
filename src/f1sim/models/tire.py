"""Tire model with degradation curves."""

from enum import Enum

from pydantic import BaseModel, Field


class TireCompound(str, Enum):
    """Available tire compounds."""

    SOFT = "soft"
    MEDIUM = "medium"
    HARD = "hard"
    INTERMEDIATE = "intermediate"
    WET = "wet"


class Tire(BaseModel):
    """Represents a tire set with degradation characteristics."""

    compound: TireCompound = Field(..., description="Tire compound type")

    # Performance characteristics
    initial_grip: float = Field(
        default=1.0,
        ge=0.0,
        le=1.2,
        description="Initial grip level (soft > medium > hard)",
    )
    degradation_rate: float = Field(
        default=0.02,
        ge=0.0,
        le=0.1,
        description="Grip loss per lap (soft degrades faster)",
    )
    cliff_threshold: int = Field(
        default=30,
        gt=0,
        description="Lap count where severe degradation begins",
    )
    cliff_multiplier: float = Field(
        default=3.0,
        ge=1.0,
        le=10.0,
        description="How much faster degradation is past the cliff",
    )
    optimal_temp_range: tuple[float, float] = Field(
        default=(80.0, 100.0),
        description="Optimal operating temperature range in Celsius",
    )

    def grip_at_lap(self, lap: int, tire_management: float = 1.0) -> float:
        """Calculate grip level after N laps on this tire.

        Args:
            lap: Number of laps completed on this tire set
            tire_management: Driver's tire management skill (0-1)

        Returns:
            Grip level (0.0 to initial_grip)
        """
        # Apply tire management skill (reduces effective degradation)
        effective_deg_rate = self.degradation_rate * (2.0 - tire_management)

        if lap < self.cliff_threshold:
            # Normal degradation
            grip = self.initial_grip - (effective_deg_rate * lap)
        else:
            # Cliff degradation
            laps_past_cliff = lap - self.cliff_threshold
            normal_loss = effective_deg_rate * self.cliff_threshold
            cliff_loss = effective_deg_rate * self.cliff_multiplier * laps_past_cliff
            grip = self.initial_grip - normal_loss - cliff_loss

        return max(0.5, grip)  # Minimum 50% grip

    def time_penalty_per_lap(self, lap: int, base_lap_time: float, tire_management: float = 1.0) -> float:
        """Calculate time penalty from tire degradation.

        Args:
            lap: Laps on current tire set
            base_lap_time: Reference lap time
            tire_management: Driver skill

        Returns:
            Time penalty in seconds
        """
        grip = self.grip_at_lap(lap, tire_management)
        grip_loss = self.initial_grip - grip
        # Each 0.1 grip loss = ~0.3% slower
        return base_lap_time * grip_loss * 0.03


# Pre-configured tire compounds with typical characteristics
TIRE_COMPOUNDS = {
    TireCompound.SOFT: Tire(
        compound=TireCompound.SOFT,
        initial_grip=1.05,
        degradation_rate=0.025,
        cliff_threshold=20,
        cliff_multiplier=3.5,
    ),
    TireCompound.MEDIUM: Tire(
        compound=TireCompound.MEDIUM,
        initial_grip=1.0,
        degradation_rate=0.015,
        cliff_threshold=30,
        cliff_multiplier=3.0,
    ),
    TireCompound.HARD: Tire(
        compound=TireCompound.HARD,
        initial_grip=0.95,
        degradation_rate=0.008,
        cliff_threshold=45,
        cliff_multiplier=2.5,
    ),
    TireCompound.INTERMEDIATE: Tire(
        compound=TireCompound.INTERMEDIATE,
        initial_grip=0.9,  # On damp track
        degradation_rate=0.02,
        cliff_threshold=35,
        cliff_multiplier=2.0,
    ),
    TireCompound.WET: Tire(
        compound=TireCompound.WET,
        initial_grip=0.85,  # On wet track
        degradation_rate=0.015,
        cliff_threshold=40,
        cliff_multiplier=2.0,
    ),
}
