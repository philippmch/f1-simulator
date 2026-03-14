import numpy as np

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.simulation.events import EventManager


def _track() -> Track:
    return Track(
        id="test",
        name="Test Circuit",
        country="Nowhere",
        total_laps=50,
        base_lap_time=90.0,
        overtake_difficulty=0.7,
        tire_stress=0.8,
        safety_car_probability=0.4,
    )


def test_mechanical_failure_probability_increases_with_lap_progress() -> None:
    manager = EventManager(rng=np.random.default_rng(1))
    car = Car(team_id="rb", team_name="Red Bull", reliability=0.92)
    track = _track()
    weather = Weather(condition=WeatherCondition.DRY, track_temperature=35.0)

    early = manager._mechanical_failure_probability(car, track, weather, lap=5)
    late = manager._mechanical_failure_probability(car, track, weather, lap=45)

    assert late > early


def test_mechanical_failure_probability_increases_with_temperature() -> None:
    manager = EventManager(rng=np.random.default_rng(2))
    car = Car(team_id="rb", team_name="Red Bull", reliability=0.92)
    track = _track()

    cool = Weather(condition=WeatherCondition.DRY, track_temperature=35.0)
    hot = Weather(condition=WeatherCondition.DRY, track_temperature=55.0)

    p_cool = manager._mechanical_failure_probability(car, track, cool, lap=25)
    p_hot = manager._mechanical_failure_probability(car, track, hot, lap=25)

    assert p_hot > p_cool


def test_mechanical_failure_probability_reflects_component_reliability() -> None:
    manager = EventManager(rng=np.random.default_rng(21))
    track = _track()
    weather = Weather(condition=WeatherCondition.DRY, track_temperature=35.0)

    robust = Car(
        team_id="rb",
        team_name="Robust",
        reliability=0.95,
        engine_reliability=0.97,
        gearbox_reliability=0.97,
        brakes_reliability=0.97,
        electrical_reliability=0.97,
        cooling_reliability=0.97,
    )
    fragile_engine = Car(
        team_id="rb",
        team_name="Fragile",
        reliability=0.95,
        engine_reliability=0.75,
        gearbox_reliability=0.97,
        brakes_reliability=0.97,
        electrical_reliability=0.97,
        cooling_reliability=0.97,
    )

    p_robust = manager._mechanical_failure_probability(robust, track, weather, lap=30)
    p_fragile = manager._mechanical_failure_probability(fragile_engine, track, weather, lap=30)

    assert p_fragile > p_robust


def test_incident_probability_increases_in_wet_conditions() -> None:
    manager = EventManager(rng=np.random.default_rng(3))
    drivers = [
        Driver(id="VER", name="Max", team_id="rb", consistency=0.95),
        Driver(id="NOR", name="Lando", team_id="mcl", consistency=0.95),
    ]
    track = _track()

    dry = Weather(condition=WeatherCondition.DRY, track_wetness=0.0, rain_intensity=0.0)
    wet = Weather(
        condition=WeatherCondition.HEAVY_RAIN,
        track_wetness=0.9,
        rain_intensity=0.9,
    )

    p_dry = manager._incident_probability(drivers, track, dry)
    p_wet = manager._incident_probability(drivers, track, wet)

    assert p_wet > p_dry


def test_safety_probs_scale_with_track_risk() -> None:
    manager = EventManager(rng=np.random.default_rng(5))
    low_risk = _track().model_copy(update={"safety_car_probability": 0.1})
    high_risk = _track().model_copy(update={"safety_car_probability": 0.8})
    weather = Weather(condition=WeatherCondition.DRY, track_wetness=0.0, rain_intensity=0.0)

    low_sc, low_vsc = manager._calibrated_safety_probs(
        low_risk,
        weather,
        incidents=1,
        lap=20,
    )
    high_sc, high_vsc = manager._calibrated_safety_probs(
        high_risk,
        weather,
        incidents=1,
        lap=20,
    )

    assert high_sc > low_sc
    assert high_vsc > low_vsc
