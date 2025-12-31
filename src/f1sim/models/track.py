"""Track model with sectors and DRS zones."""

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


class DRSZone(BaseModel):
    """Represents a DRS zone on track."""

    zone_id: int = Field(..., ge=1, description="DRS zone identifier")
    sector: int = Field(..., ge=1, le=3, description="Which sector this zone is in")
    time_gain: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Time gain in seconds when DRS is used",
    )
    detection_point_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Track percentage where DRS detection occurs",
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
    drs_zones: list[DRSZone] = Field(
        default_factory=list,
        description="DRS zones on track",
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
    def total_drs_gain(self) -> float:
        """Maximum time gain from all DRS zones."""
        return sum(zone.time_gain for zone in self.drs_zones)
