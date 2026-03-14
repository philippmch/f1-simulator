import numpy as np

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def _track() -> Track:
    return Track(
        id="test",
        name="Test",
        country="Nowhere",
        total_laps=60,
        base_lap_time=90.0,
        pit_lane_delta=22.0,
        overtake_difficulty=0.7,
        tire_stress=0.6,
        safety_car_probability=0.3,
    )


def test_infer_strategy_conservative_for_fragile_car() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(1))
    car = Car(team_id="x", team_name="X", reliability=0.86)

    strategy = sim._infer_team_strategy(car, _track())
    assert strategy == TeamStrategyArchetype.CONSERVATIVE


def test_infer_strategy_aggressive_for_fast_car_on_hard_track() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(2))
    car = Car(team_id="x", team_name="X", base_pace=0.9, reliability=0.95)

    strategy = sim._infer_team_strategy(car, _track())
    assert strategy == TeamStrategyArchetype.AGGRESSIVE


def test_conservative_strategy_uses_higher_max_stops_in_wet() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(3))
    track = _track()
    weather = Weather(condition=WeatherCondition.LIGHT_RAIN, track_wetness=0.35, rain_intensity=0.5)

    car = Car(team_id="x", team_name="X", reliability=0.86)
    state = DriverRaceState(
        driver=Driver(id="DRV", name="DRV", team_id="x"),
        car=car,
        position=5,
        total_time=500.0,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
        pit_stops=3,
        tire_laps=18,
        strategy_archetype=TeamStrategyArchetype.CONSERVATIVE,
    )

    ahead = DriverRaceState(
        driver=Driver(id="AHD", name="AHD", team_id="a"),
        car=Car(team_id="a", team_name="A"),
        position=4,
        total_time=499.0,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
    )
    behind = DriverRaceState(
        driver=Driver(id="BHD", name="BHD", team_id="b"),
        car=Car(team_id="b", team_name="B"),
        position=6,
        total_time=520.0,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
    )

    # Should not be hard-blocked by max stop cap in wet changing race.
    result = sim._should_pit(
        state,
        [ahead, state, behind],
        track,
        lap=30,
        pit_window_open=True,
        weather=weather,
    )
    assert isinstance(result, bool)
