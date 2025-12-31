"""Driver model with skill attributes."""

from pydantic import BaseModel, Field


class Driver(BaseModel):
    """Represents an F1 driver with performance characteristics."""

    id: str = Field(..., description="Unique driver identifier (e.g., 'VER')")
    name: str = Field(..., description="Full name")
    team_id: str = Field(..., description="Team identifier")

    # Performance attributes (0.0 to 1.0 scale)
    skill_rating: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall skill level affecting base lap time",
    )
    consistency: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Higher = less lap time variation (std dev multiplier)",
    )
    wet_skill_modifier: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description="Multiplier for wet conditions (>1 = better in wet)",
    )
    overtaking_skill: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Ability to complete overtakes",
    )
    tire_management: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Ability to preserve tires (reduces degradation)",
    )

    # Race state (mutable during simulation)
    current_tire_laps: int = Field(default=0, description="Laps on current tire set")
    total_race_time: float = Field(default=0.0, description="Cumulative race time in seconds")
    position: int = Field(default=0, description="Current race position")
    pit_stops: int = Field(default=0, description="Number of pit stops made")
    dnf: bool = Field(default=False, description="Did not finish flag")
    dnf_reason: str | None = Field(default=None, description="Reason for DNF if applicable")

    def reset_race_state(self) -> None:
        """Reset mutable state for a new race."""
        self.current_tire_laps = 0
        self.total_race_time = 0.0
        self.position = 0
        self.pit_stops = 0
        self.dnf = False
        self.dnf_reason = None

    def lap_time_variation_std(self, base_std: float = 0.3) -> float:
        """Calculate lap time standard deviation based on consistency."""
        # Higher consistency = lower variation
        return base_std * (2.0 - self.consistency)
