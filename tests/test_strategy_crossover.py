"""Weather stops must resolve the mismatch that caused the stop."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype


def make_state(compound=TireCompound.MEDIUM):
    return DriverRaceState(
        driver=Driver(id="A", name="A", team_id="A"),
        car=Car(team_id="A", team_name="A"),
        position=1,
        current_tire=TIRE_COMPOUNDS[compound].model_copy(deep=True),
    )


def make_track():
    return Track(id="test", name="Test", country="Test", total_laps=30, base_lap_time=90)


def test_stable_damp_race_does_not_repeat_mismatch_stops(monkeypatch):
    simulator = RaceSimulator(rng=np.random.default_rng(3))
    state = make_state()
    stops = []
    execute = simulator._execute_pit_stop

    def record_stop(state, track, weather, current_lap, **kwargs):
        loss = execute(state, track, weather, current_lap, **kwargs)
        stops.append((current_lap, state.current_tire.compound))
        assert simulator._check_tire_weather_mismatch(state.current_tire, weather) == "ok"
        return loss

    monkeypatch.setattr(simulator, "_execute_pit_stop", record_stop)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda **kwargs: [])
    results = simulator.simulate_race(
        [state.driver], {"A": state.car}, make_track(),
        Weather(condition=WeatherCondition.LIGHT_RAIN, track_wetness=0.25,
                rain_intensity=0.25, change_probability=0),
        ["A"], starting_tires={"A": TireCompound.MEDIUM},
    )

    assert stops == [(1, TireCompound.INTERMEDIATE)]
    assert results[0].pit_stops == 1
    assert results[0].strategy == ["intermediate"]  # Starting slicks never ran.


@pytest.mark.parametrize("wetness,rain,expected", [
    (0.25, 0.0, TireCompound.INTERMEDIATE),
    (0.03, 0.8, TireCompound.INTERMEDIATE),
    (0.73, 0.8, TireCompound.WET),
    (0.41, 0.0, TireCompound.INTERMEDIATE),
    (0.2001, 0.0, TireCompound.INTERMEDIATE),
])
def test_all_fresh_tyre_choices_fit_crossover_conditions(wetness, rain, expected):
    simulator = RaceSimulator(rng=np.random.default_rng(5))
    weather = Weather(track_wetness=wetness, rain_intensity=rain, change_probability=0)
    previous = TireCompound.WET if wetness == 0.41 else TireCompound.MEDIUM
    if wetness == 0.73:
        previous = TireCompound.INTERMEDIATE
    state = make_state(previous)
    simulator._execute_pit_stop(state, make_track(), weather, 10)
    assert state.current_tire.compound == expected
    assert simulator._check_tire_weather_mismatch(state.current_tire, weather) == "ok"
    assert simulator._choose_red_flag_tire(state, weather, make_track(), 10) == expected
    assert simulator._choose_starting_compound(
        TeamStrategyArchetype.BALANCED, make_track(), weather,
    ) == expected


@pytest.mark.parametrize("wetness,rain", [(0.2, 0.0), (0.079, 0.0), (0.0, 0.4)])
def test_drying_stop_can_fit_slicks_at_crossover(wetness, rain):
    simulator = RaceSimulator(rng=np.random.default_rng(6))
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    state = make_state(TireCompound.WET)
    simulator._execute_pit_stop(state, make_track(), weather, 10)
    assert state.current_tire.compound in {
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    }
    assert simulator._check_tire_weather_mismatch(state.current_tire, weather) == "ok"


def test_existing_rain_tyres_keep_drying_hysteresis_and_emergency_distinction():
    simulator = RaceSimulator()
    mismatch = simulator._check_tire_weather_mismatch
    assert mismatch(TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                    Weather(track_wetness=0.08)) == "ok"
    assert mismatch(TIRE_COMPOUNDS[TireCompound.INTERMEDIATE],
                    Weather(track_wetness=0.079)) == "critical"
    assert mismatch(TIRE_COMPOUNDS[TireCompound.WET], Weather(track_wetness=0.42)) == "ok"
    assert mismatch(TIRE_COMPOUNDS[TireCompound.WET],
                    Weather(track_wetness=0.41)) == "suboptimal"
    assert mismatch(TIRE_COMPOUNDS[TireCompound.WET],
                    Weather(track_wetness=0.19)) == "critical"
    assert mismatch(TIRE_COMPOUNDS[TireCompound.MEDIUM],
                    Weather(track_wetness=0.25)) == "suboptimal"
    assert mismatch(TIRE_COMPOUNDS[TireCompound.MEDIUM],
                    Weather(track_wetness=0.46)) == "critical"


def test_conservative_switch_persists_for_future_stint_choices(monkeypatch):
    simulator = RaceSimulator(rng=np.random.default_rng(8))
    state = make_state()
    state.position = 8
    state.total_time = 90.5
    state.strategy_archetype = TeamStrategyArchetype.CONSERVATIVE
    ahead = make_state()
    ahead.position = 7
    ahead.total_time = 90
    simulator._should_pit(state, [ahead, state], make_track(), 15, False, Weather())
    assert state.strategy_archetype == TeamStrategyArchetype.BALANCED

    # Even after escaping traffic, the next tyre choice uses the balanced profile.
    state.total_time = 95
    simulator._should_pit(state, [ahead, state], make_track(), 16, False, Weather())
    profiles_used = []

    class RecordingProfiles(dict):
        def __getitem__(self, key):
            profiles_used.append(key)
            return super().__getitem__(key)

    monkeypatch.setattr(
        simulator, "strategy_profiles", RecordingProfiles(simulator.strategy_profiles),
    )
    simulator._choose_compound_for_next_stint(state, make_track(), 16)
    assert profiles_used == [TeamStrategyArchetype.BALANCED]
