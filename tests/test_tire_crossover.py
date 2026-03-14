import numpy as np

from f1sim.models import Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.race import RaceSimulator


def test_slicks_become_critical_in_heavy_rain_transition() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(11))
    slick = TIRE_COMPOUNDS[TireCompound.SOFT]
    weather = Weather(
        condition=WeatherCondition.HEAVY_RAIN,
        track_wetness=0.4,
        rain_intensity=0.8,
    )

    assert sim._check_tire_weather_mismatch(slick, weather) == "critical"


def test_inters_critical_on_mostly_dry_track() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(12))
    inter = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    weather = Weather(
        condition=WeatherCondition.CLOUDY,
        track_wetness=0.06,
        rain_intensity=0.0,
    )

    assert sim._check_tire_weather_mismatch(inter, weather) == "critical"
