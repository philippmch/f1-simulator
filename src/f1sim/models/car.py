"""Car performance model."""

from pydantic import BaseModel, Field


class Car(BaseModel):
    """Represents an F1 car with performance characteristics."""

    team_id: str = Field(..., description="Team identifier (e.g., 'red_bull')")
    team_name: str = Field(..., description="Full team name")

    # Performance attributes (relative to field, 0.0 = slowest, 1.0 = fastest)
    base_pace: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall car pace (affects base lap time)",
    )
    downforce_level: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Cornering performance (high = better in corners, worse on straights)",
    )
    straight_line_speed: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Top speed performance",
    )
    reliability: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Probability of finishing without mechanical failure",
    )
    tire_degradation_factor: float = Field(
        default=1.0,
        ge=0.5,
        le=1.5,
        description="Multiplier for tire degradation (lower = easier on tires)",
    )
    wet_performance: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Performance in wet conditions",
    )
    pit_stop_avg: float = Field(
        default=2.5,
        ge=1.5,
        le=5.0,
        description="Average pit stop time in seconds (stationary)",
    )
    pit_stop_std: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Pit stop time standard deviation",
    )

    def pace_delta_seconds(self, reference_lap_time: float) -> float:
        """Calculate pace advantage/disadvantage in seconds.

        A car with base_pace=1.0 is the fastest, base_pace=0.0 is ~3% slower.
        """
        # Max delta is ~3% of lap time (roughly 2.5s on a 90s lap)
        max_delta_pct = 0.03
        pace_deficit = 1.0 - self.base_pace
        return reference_lap_time * max_delta_pct * pace_deficit
