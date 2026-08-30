"""Track model with sectors and 2026 active-aero zones."""

from pydantic import BaseModel, Field


class Sector(BaseModel):
    """Represents a track sector."""

    number: int = Field(..., ge=1, le=3, description="Sector number (1-3)")
    base_time: float = Field(..., gt=0, description="Base sector time in seconds")
    is_high_speed: bool = Field(
        default=False,
        description="Whether this sector favors high downforce",
    )
    overtake_opportunity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How easy it is to overtake in this sector",
    )


class ActiveAeroZone(BaseModel):
    """Represents a configured straight-mode active-aero section."""

    zone_id: int = Field(..., ge=1, description="Active-aero zone identifier")
    sector: int = Field(..., ge=1, le=3, description="Which sector this zone is in")
    time_gain: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Straight-mode time gain in seconds",
    )
    activation_point_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Track percentage where straight mode begins",
    )


class Track(BaseModel):
    """Represents an F1 circuit."""

    id: str = Field(..., description="Track identifier (e.g., 'monza')")
    name: str = Field(..., description="Official track name")
    country: str = Field(..., description="Country")

    # Track characteristics
    total_laps: int = Field(..., gt=0, description="Number of laps in race")
    base_lap_time: float = Field(
        ...,
        gt=0,
        description="Reference lap time in seconds (for average car/driver)",
    )
    pit_lane_delta: float = Field(
        default=20.0,
        gt=0,
        description="Time lost entering/exiting pit lane in seconds",
    )

    # Track sections
    sectors: list[Sector] = Field(
        default_factory=list,
        description="Track sectors (should have 3)",
    )
    active_aero_zones: list[ActiveAeroZone] = Field(
        default_factory=list,
        description="Configured straight-mode active-aero sections",
    )
    overtake_mode_detection_gap: float = Field(
        default=1.0,
        gt=0.0,
        le=3.0,
        description="Detection gap in seconds for Overtake Mode",
    )

    # Track characteristics affecting racing
    overtake_difficulty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="0 = easy to overtake (Monza), 1 = very hard (Monaco)",
    )
    tire_stress: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How hard the track is on tires (affects degradation)",
    )
    safety_car_probability: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Base probability of safety car per race",
    )
    weather_variability: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Likelihood of weather changes during race",
    )

    def get_sector_time(self, sector_num: int) -> float:
        """Get base time for a specific sector."""
        for sector in self.sectors:
            if sector.number == sector_num:
                return sector.base_time
        # If sectors not defined, split base lap time evenly
        return self.base_lap_time / 3

    @property
    def total_active_aero_gain(self) -> float:
        """Maximum straight-mode time gain from configured sections."""
        return sum(zone.time_gain for zone in self.active_aero_zones)

    @property
    def active_aero_zone_count(self) -> int:
        """Number of configured straight-mode active-aero sections."""
        return len(self.active_aero_zones)
