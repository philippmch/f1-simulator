"""Fallback fresh-set ranking uses complete race-lap physics."""

import copy

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.models.track import ActiveAeroZone
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import DriverRaceState, RaceSimulator

SLICKS = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]


def fixture(floor):
    driver = Driver(id="A", name="A", team_id="A", skill_rating=1, tire_management=0.8)
    car = Car(team_id="A", team_name="A", base_pace=0.8)
    track = Track(
        id="test", name="Test", country="Test", total_laps=100, base_lap_time=70,
        active_aero_zones=[ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1)
                           for i in range(6)] if floor else [],
    )
    return driver, car, track


def oracle(driver, car, track, tire, weather, *, physical=100, modifier=1, aero=True,
           first_lap=82, laps=19):
    running_driver = driver.model_copy(deep=True)
    surface = weather.model_copy(deep=True)
    sim = LapSimulator(np.random.default_rng(99))
    times = []
    for age in range(laps):
        running_driver.current_tire_laps = age
        times.append(sim.calculate_lap_time(
            running_driver, car, track, tire, surface, first_lap + age, physical,
            active_aero_enabled=aero if age == 0 else True, sample_variation=False,
        ) * (modifier if age == 0 else 1))
        surface = surface.project_surface()
    return sum(times)


@pytest.mark.parametrize("floor", [False, True])
@pytest.mark.parametrize("compound", SLICKS)
@pytest.mark.parametrize("modifier,aero", [(1, True), (1.4, False)])
def test_complete_projection_matches_actual_laps_without_mutation(floor, compound, modifier, aero):
    driver, car, track = fixture(floor)
    driver.current_tire_laps = 27
    tire = TIRE_COMPOUNDS[compound].model_copy(update={"initial_grip": 0.95}, deep=True)
    weather = Weather(track_wetness=0.18, rain_intensity=0.1)
    models = [driver, car, track, tire, weather]
    before = [model.model_dump() for model in models]
    sim = LapSimulator(np.random.default_rng(8))
    rng_before = copy.deepcopy(sim.rng.bit_generator.state)
    cost = sim.projected_stint_lap_cost(
        driver, car, track, tire, 19, 82, weather,
        current_lap_time_modifier=modifier, active_aero_enabled=aero,
    )
    assert cost == pytest.approx(oracle(
        driver, car, track, tire, weather, modifier=modifier, aero=aero,
    ), abs=1e-9)
    assert [model.model_dump() for model in models] == before
    assert sim.rng.bit_generator.state == rng_before


def test_shorter_planning_distance_keeps_original_fuel():
    driver, car, track = fixture(False)
    planning = track.model_copy(update={"total_laps": 90})
    tire = TIRE_COMPOUNDS[TireCompound.MEDIUM]
    weather = Weather(track_wetness=0.08)
    sim = LapSimulator()
    expected = oracle(driver, car, planning, tire, weather, laps=9)
    assert sim.projected_stint_lap_cost(
        driver, car, planning, tire, 9, 82, weather, physical_total_laps=100,
    ) == pytest.approx(expected)
    assert sim.projected_stint_lap_cost(
        driver, car, planning, tire, 9, 82, weather,
    ) != pytest.approx(expected)


@pytest.mark.parametrize("distinct", [False, True])
def test_damp_pit_choice_respects_actual_cost_bound_and_preserves_style(monkeypatch, distinct):
    driver, car, track = fixture(True)
    state = DriverRaceState(
        driver=driver, car=car, position=1,
        current_tire=TIRE_COMPOUNDS[TireCompound.HARD].model_copy(deep=True),
        tire_laps=10, tire_compound_history=["hard"] if distinct else ["hard", "medium"],
        planned_pit_laps=[],
    )
    weather = Weather(track_wetness=0.08)
    sim = RaceSimulator(np.random.default_rng(8))
    monkeypatch.setattr(sim, "_preferred_stint_compound", lambda *args: TireCompound.SOFT)
    costs = {compound: oracle(driver, car, track, TIRE_COMPOUNDS[compound], weather)
             for compound in SLICKS}
    sim._execute_pit_stop(state, track, weather, 82, sample_service=False)
    allowed = [TireCompound.SOFT, TireCompound.MEDIUM] if distinct else SLICKS
    assert costs[state.current_tire.compound] <= min(costs[c] for c in allowed) + 19 * 0.05
    # The small medium advantage is inside the documented style tolerance.
    assert state.current_tire.compound == TireCompound.SOFT


def test_floor_rejects_style_preference_that_exceeds_actual_cost_bound():
    driver, car, track = fixture(True)
    weather = Weather(track_wetness=0.08)
    sim = RaceSimulator(np.random.default_rng(4))
    state = DriverRaceState(
        driver=driver, car=car, position=1,
        current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True),
        tire_laps=20, tire_compound_history=["soft", "medium"], pit_stops=1,
    )
    relative = {c: sim.lap_simulator.projected_tire_stint_cost(
        driver, car, track, TIRE_COMPOUNDS[c], 20,
    ) for c in SLICKS}
    actual = {c: oracle(driver, car, track, TIRE_COMPOUNDS[c], weather,
                        first_lap=81, laps=20) for c in SLICKS}
    # Old tyre-only ranking allowed this soft preference, but full laps exceed
    # the one-second allowance. Use the real seeded style choice and pit path.
    assert relative[TireCompound.SOFT] <= min(relative.values()) + 1
    assert actual[TireCompound.SOFT] > min(actual.values()) + 1
    sim._execute_pit_stop(state, track, weather, 81, sample_service=False)
    assert state.current_tire.compound == TireCompound.MEDIUM

@pytest.mark.parametrize("distinct", [False, True])
def test_pit_fallback_forwards_physical_fuel_and_current_control(monkeypatch, distinct):
    driver, car, track = fixture(False)
    track.total_laps = 90
    state = DriverRaceState(
        driver=driver, car=car, position=1,
        current_tire=TIRE_COMPOUNDS[TireCompound.HARD].model_copy(deep=True),
        tire_laps=10, tire_compound_history=["hard"] if distinct else ["hard", "medium"],
        planned_pit_laps=[],
    )
    weather = Weather(track_wetness=0.08)
    sim = RaceSimulator(np.random.default_rng(8))
    sim.event_manager.safety_car_active = True
    observed = []
    original = sim.lap_simulator.projected_stint_lap_cost

    def record(*args, **kwargs):
        observed.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(sim.lap_simulator, "projected_stint_lap_cost", record)
    sim._execute_pit_stop(
        state, track, weather, 82, sample_service=False, physical_total_laps=100,
    )
    assert len(observed) == (2 if distinct else 3)
    for args, kwargs in observed:
        assert args[4:7] == (9, 82, weather)
        assert kwargs == {
            "physical_total_laps": 100,
            "current_lap_time_modifier": sim.event_manager.get_lap_time_modifier(),
            "active_aero_enabled": False,
        }
