import numpy as np

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.race import DriverRaceState, RaceSimulator


def _track() -> Track:
    return Track(
        id="test",
        name="Test Track",
        country="Nowhere",
        total_laps=60,
        base_lap_time=90.0,
        pit_lane_delta=22.0,
        overtake_difficulty=0.8,
        tire_stress=0.7,
        safety_car_probability=0.3,
    )


def _state(
    driver_id: str,
    position: int,
    total_time: float,
    tire: TireCompound = TireCompound.MEDIUM,
) -> DriverRaceState:
    driver = Driver(id=driver_id, name=driver_id, team_id=f"team_{driver_id}")
    car = Car(team_id=f"team_{driver_id}", team_name=f"Team {driver_id}")
    return DriverRaceState(
        driver=driver,
        car=car,
        position=position,
        total_time=total_time,
        current_tire=TIRE_COMPOUNDS[tire].model_copy(deep=True),
    )


def test_late_race_stop_on_hard_switches_to_soft() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(1))
    state = _state("VER", position=3, total_time=1000.0, tire=TireCompound.HARD)
    weather = Weather(condition=WeatherCondition.DRY, track_wetness=0.0)

    sim._execute_pit_stop(state, _track(), weather, current_lap=52)

    assert state.current_tire.compound == TireCompound.SOFT


def test_pit_window_with_free_gap_triggers_stop() -> None:
    sim = RaceSimulator(rng=np.random.default_rng(7))
    track = _track()
    weather = Weather(condition=WeatherCondition.DRY, track_wetness=0.0)

    state = _state("VER", position=2, total_time=1000.0, tire=TireCompound.MEDIUM)
    state.tire_laps = 16
    state.pit_stops = 0

    ahead = _state("NOR", position=1, total_time=999.2, tire=TireCompound.MEDIUM)
    # Big gap behind > ~pit delta => free stop under SC/VSC window
    behind = _state("HAM", position=3, total_time=1024.0, tire=TireCompound.MEDIUM)

    should_pit = sim._should_pit(
        state,
        [ahead, state, behind],
        track,
        lap=22,
        pit_window_open=True,
        weather=weather,
    )

    assert should_pit is True
