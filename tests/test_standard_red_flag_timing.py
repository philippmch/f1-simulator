"""Absolute-clock red-flag timing for the standard race engine."""

import math

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator
from f1sim.simulation.race_timing import forecast_final_lap


def _controlled(monkeypatch, *, laps=3, paces=None, red=(1,), pause=600.0):
    paces = paces or {"A": 90.0, "B": 110.0}
    drivers = [Driver(id=driver_id, name=driver_id, team_id=driver_id)
               for driver_id in paces]
    cars = {driver_id: Car(team_id=driver_id, team_name=driver_id, reliability=1.0)
            for driver_id in paces}
    track = Track(id="t", name="Test", country="Test", total_laps=laps,
                  base_lap_time=90)
    simulator = RaceSimulator(np.random.default_rng(7), red_flag_pause_seconds=pause)
    simulator._should_pit = lambda *args, **kwargs: False
    simulator._process_overtakes = lambda *args, **kwargs: 0
    simulator.lap_simulator.calculate_lap_time = lambda **kwargs: paces[kwargs["driver"].id]

    def events(lap, **kwargs):
        return [RaceEvent(EventType.RED_FLAG, lap)] if lap in red else []

    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    return simulator, drivers, cars, track


@pytest.mark.parametrize("pause", [0.0, 600.0])
def test_first_lap_red_flag_preserves_crossings_and_uses_common_resume(monkeypatch, pause):
    simulator, drivers, cars, track = _controlled(monkeypatch, pause=pause)
    crossings = []
    update = simulator._update_positions

    def observe(states):
        update(states)
        crossings.extend(
            (state.driver.id, state.laps_completed + 1, state.total_time)
            for state in states if state.status == DriverStatus.RACING
        )

    monkeypatch.setattr(simulator, "_update_positions", observe)
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM},
    )

    resume = 110.0 + pause
    assert simulator.suspensions == [(90.0, resume, ("A", "B"))]
    assert [row.total_time for row in results] == [290.0 + pause, 330.0 + pause]
    assert [row.fastest_lap for row in results] == [90.0, 110.0]
    assert [(driver, lap) for driver, lap, _ in crossings] == [
        ("A", 1), ("B", 1), ("A", 2), ("B", 2), ("A", 3), ("B", 3),
    ]
    assert [time for driver, _, time in crossings if driver == "A"] == [
        90.0, resume + 90.0, resume + 180.0,
    ]
    assert [time for driver, _, time in crossings if driver == "B"] == [
        110.0, resume + 110.0, resume + 220.0,
    ]


def test_large_lag_is_collected_without_rewinding_completed_clocks(monkeypatch):
    simulator, drivers, cars, track = _controlled(
        monkeypatch, paces={"A": 90.0, "B": 300.0}, pause=0.0,
    )
    starts = []
    update = simulator._update_positions

    def observe(states):
        update(states)
        starts.append({state.driver.id: state.total_time for state in states
                       if state.status == DriverStatus.RACING})

    monkeypatch.setattr(simulator, "_update_positions", observe)
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM},
    )

    assert simulator.suspensions == [(90.0, 300.0, ("A", "B"))]
    assert starts[0] == {"A": 90.0, "B": 300.0}
    assert starts[1] == {"A": 390.0, "B": 600.0}
    assert starts[2] == {"A": 480.0, "B": 900.0}
    assert [row.total_time for row in results] == [480.0, 900.0]


def test_restart_retirement_rolls_back_to_pre_wait_completed_clock(monkeypatch):
    simulator, drivers, cars, track = _controlled(monkeypatch, pause=600.0)

    def events(lap, **kwargs):
        if lap == 1:
            return [RaceEvent(EventType.RED_FLAG, lap)]
        if lap == 2:
            driver = next(driver for driver in kwargs["drivers"] if driver.id == "A")
            driver.dnf = True
            driver.dnf_reason = "Restart failure"
            return [RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"])]
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM},
    )
    retired = next(row for row in results if row.driver_id == "A")
    survivor = next(row for row in results if row.driver_id == "B")
    assert retired.status == DriverStatus.DNF
    assert retired.laps_completed == 1 and retired.total_time == 90.0
    assert survivor.laps_completed == 3 and survivor.total_time == 930.0


def test_over_cap_restart_horizon_is_used_by_refit_and_pit_planning(monkeypatch):
    from f1sim.simulation import race_timing

    monkeypatch.setattr(race_timing, "RACING_TIME_LIMIT_SECONDS", 180.0)
    simulator, drivers, cars, track = _controlled(
        monkeypatch, laps=10, pause=4000.0,
    )
    refit_horizons = []
    original_refit = simulator._fit_red_flag_tires

    def capture_refit(states, weather, planning_track, current_lap, **kwargs):
        refit_horizons.append((current_lap, planning_track.total_laps))
        return original_refit(states, weather, planning_track, current_lap, **kwargs)

    monkeypatch.setattr(simulator, "_fit_red_flag_tires", capture_refit)
    pit_horizons = []

    def observe_pit_decision(state, all_states, planning_track, lap, *args, **kwargs):
        if lap == 2:
            pit_horizons.append((state.driver.id, planning_track.total_laps))
        return False

    monkeypatch.setattr(simulator, "_should_pit", observe_pit_decision)
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM},
    )

    assert simulator.suspensions == [(90.0, 4110.0, ("A", "B"))]
    assert [(row.driver_id, row.laps_completed, row.total_time) for row in results] == [
        ("A", 3, 4290.0), ("B", 3, 4330.0),
    ]
    assert refit_horizons == [(1, 3)]
    assert pit_horizons == [("A", 3), ("B", 3)]


def test_restart_paid_stops_share_common_arrival_and_box_queue(monkeypatch):
    drivers = [Driver(id=driver_id, name=driver_id, team_id="T")
               for driver_id in ("A", "B")]
    car = Car(team_id="T", team_name="T", reliability=1.0)
    cars = {"T": car}
    track = Track(id="t", name="Test", country="Test", total_laps=3,
                  base_lap_time=90, pit_lane_delta=20)
    simulator = RaceSimulator(np.random.default_rng(7), red_flag_pause_seconds=600.0)
    simulator._process_overtakes = lambda *args, **kwargs: 0
    simulator.lap_simulator.calculate_lap_time = lambda **kwargs: {
        "A": 90.0, "B": 110.0,
    }[kwargs["driver"].id]
    services = []
    monkeypatch.setattr(
        simulator.lap_simulator, "calculate_pit_stop_time",
        lambda service_car: services.append(service_car.team_id) or 3.0,
    )
    decisions = []

    def should_pit(state, all_states, planning_track, lap, *args, **kwargs):
        if lap == 2:
            decisions.append((
                state.driver.id,
                state.total_time,
                tuple(other.total_time for other in all_states),
            ))
        return (lap == 1 and state.driver.id == "B") or lap == 2

    monkeypatch.setattr(simulator, "_should_pit", should_pit)
    monkeypatch.setattr(
        simulator.event_manager,
        "process_lap",
        lambda lap, **kwargs: [RaceEvent(EventType.RED_FLAG, lap)] if lap == 1 else [],
    )
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM},
    )
    by_driver = {row.driver_id: row for row in results}

    assert simulator.suspensions == [(90.0, 733.0, ("A", "B"))]
    assert decisions == [("A", 733.0, (733.0, 733.0)),
                         ("B", 733.0, (733.0, 733.0))]
    assert services == ["T", "T", "T"]
    assert [(row.driver_id, row.laps_completed, row.total_time) for row in results] == [
        ("A", 3, 936.0), ("B", 3, 979.0),
    ]
    assert by_driver["A"].pit_stops == 1 and by_driver["A"].pit_laps == [2]
    assert by_driver["B"].pit_stops == 2 and by_driver["B"].pit_laps == [1, 2]
    assert by_driver["A"].pit_stop_details == [
        {"lap": 2, "from_compound": "soft", "to_compound": "soft",
         "tire_age": 0, "condition": "dry", "rain_intensity": 0.0,
         "track_wetness": 0.0, "control": "green", "lane_loss": 20.0,
         "service_time": 3.0, "queue_time": 0.0, "total_loss": 23.0},
    ]
    assert by_driver["B"].pit_stop_details == [
        {"lap": 1, "from_compound": "medium", "to_compound": "soft",
         "tire_age": 0, "condition": "dry", "rain_intensity": 0.0,
         "track_wetness": 0.0, "control": "green", "lane_loss": 20.0,
         "service_time": 3.0, "queue_time": 0.0, "total_loss": 23.0},
        {"lap": 2, "from_compound": "medium", "to_compound": "medium",
         "tire_age": 0, "condition": "dry", "rain_intensity": 0.0,
         "track_wetness": 0.0, "control": "green", "lane_loss": 20.0,
         "service_time": 3.0, "queue_time": 3.0, "total_loss": 26.0},
    ]


def test_finite_pool_wear_is_conserved_through_a_free_restart_fit(monkeypatch):
    simulator, drivers, cars, track = _controlled(
        monkeypatch, laps=3, paces={"A": 90.0}, pause=600.0,
    )
    records = [
        {"id": "M", "compound": "medium", "age": 5},
        {"id": "S", "compound": "soft", "age": 0},
    ]
    result, = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A"],
        starting_tires={"A": TireCompound.MEDIUM},
        starting_tire_ages={"A": 5},
        tire_inventory={"A": records},
    )

    assert simulator.suspensions == [(90.0, 690.0, ("A",))]
    assert result.laps_completed == 3 and result.total_time == 870.0
    assert sum(stint["laps_used"] for stint in result.tire_set_history) == 3
    ages = {item["id"]: item["age"] for item in records}
    for stint in result.tire_set_history:
        assert stint["age_at_fit"] == ages[stint["set_id"]]
        assert stint["age_at_end"] == stint["age_at_fit"] + stint["laps_used"]
        ages[stint["set_id"]] = stint["age_at_end"]
    assert {item["id"]: item["age"] for item in result.tire_inventory} == ages
    assert any(stint["kind"] == "red_flag" for stint in result.tire_set_history)


def test_suspension_ledger_resets_when_simulator_is_reused(monkeypatch):
    simulator, drivers, cars, track = _controlled(
        monkeypatch, laps=4, red=(1, 3), pause=600.0,
    )
    kwargs = dict(
        starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM},
    )
    first = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"], **kwargs,
    )
    first_ledger = list(simulator.suspensions)
    second = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"], **kwargs,
    )
    expected = [(90.0, 710.0, ("A", "B")), (890.0, 1530.0, ("A", "B"))]
    assert first_ledger == expected
    assert [row.total_time for row in second] == [row.total_time for row in first]
    assert simulator.suspensions == expected
    assert [row.total_time for row in second] == [1620.0, 1640.0]

    # A simulator instance reused for an empty grid must not expose the prior
    # race's suspension history.
    assert simulator.simulate_race([], {}, track, Weather(change_probability=0), []) == []
    assert simulator.suspensions == []


def test_final_red_flag_has_no_restart_fit_or_weather_evolution(monkeypatch):
    simulator, drivers, cars, track = _controlled(monkeypatch, laps=2, red=(2,))
    simulator.event_manager.set_forced_red_flag(2)
    # Use the real event manager for the forced final flag while suppressing
    # stochastic hazards that are unrelated to this boundary.
    monkeypatch.setattr(
        simulator.event_manager,
        "process_lap",
        simulator.event_manager.__class__.process_lap.__get__(simulator.event_manager),
    )
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(simulator.event_manager, "_deploy_safety_measure", lambda *a, **k: None)
    evolves = []
    fits = []
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: evolves.append(1)
                        or self.model_copy(deep=True))
    monkeypatch.setattr(simulator, "_fit_red_flag_tires",
                        lambda *args, **kwargs: fits.append(1))
    simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": TireCompound.MEDIUM, "B": TireCompound.MEDIUM},
    )
    assert not simulator.suspensions
    assert not fits and len(evolves) == 1
    assert simulator.event_manager.red_flag_active


@pytest.mark.parametrize("value", [True, -1, math.inf, math.nan, "600"])
def test_standard_red_flag_pause_setting_is_validated(value):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        RaceSimulator(red_flag_pause_seconds=value)


def test_restart_forecast_uses_actual_crossing_and_common_start():
    # The actual completed crossing is before the deadline, while a long pause
    # places the next lap after the one-hour suspension extension cap.
    assert forecast_final_lap(
        10, 1, 90.0, 90.0, 3780.0, next_lap_start_time=4110.0,
    ) == 3
