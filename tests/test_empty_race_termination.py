"""Do not simulate vacant laps after the final retirement or an empty grid."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator


def track():
    return Track(id="test", name="Test", country="Test", total_laps=10, base_lap_time=90)


@pytest.mark.parametrize("retirement_lap", [1, 2, 10])
def test_final_retirement_stops_weather_and_later_forced_events(monkeypatch, retirement_lap):
    simulator = RaceSimulator(np.random.default_rng(42))
    field = [Driver(id=name, name=name, team_id=name) for name in ("A", "B")]
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in field}
    processed, evolved = [], []
    original_process = simulator.event_manager.process_lap

    def failure(driver, car, race_track, lap, weather):
        if lap == retirement_lap:
            driver.dnf = True
            driver.dnf_reason = "engine failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    def process(**kwargs):
        processed.append(kwargs["lap"])
        return original_process(**kwargs)

    def evolve(weather, rng):
        evolved.append(True)
        return weather.model_copy(deep=True)

    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", failure)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.event_manager, "_deploy_safety_measure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "process_lap", process)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *a, **kw: 0)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lambda **kw: 90)
    monkeypatch.setattr(Weather, "evolve", evolve)
    simulator.event_manager.set_forced_safety_car(retirement_lap + 1)
    results = simulator.simulate_race(field, cars, track(), Weather(), [d.id for d in field])

    assert processed == list(range(1, retirement_lap + 1))
    assert len(evolved) == retirement_lap - 1
    assert simulator.event_manager.current_lap == retirement_lap
    assert simulator.event_manager.safety_car_deployments == 0
    assert [event.event_type for event in simulator.event_manager.events] == [
        EventType.MECHANICAL_FAILURE, EventType.MECHANICAL_FAILURE,
    ]
    assert all(r.status == DriverStatus.DNF and not r.classified for r in results)
    assert all(r.laps_completed == retirement_lap - 1 for r in results)


@pytest.mark.parametrize("missing_car", [False, True])
def test_empty_usable_grid_resets_events_without_simulating_laps(monkeypatch, missing_car):
    simulator = RaceSimulator()
    simulator.event_manager.deploy_red_flag(5)
    simulator.event_manager.events.append(RaceEvent(EventType.RED_FLAG, 5))
    field = [Driver(id="A", name="A", team_id="A")] if missing_car else []
    def unexpected(*a, **kw):
        pytest.fail("An empty usable grid must not simulate a lap")
    monkeypatch.setattr(simulator.event_manager, "process_lap", unexpected)
    monkeypatch.setattr(Weather, "evolve", unexpected)
    assert simulator.simulate_race(field, {}, track(), Weather(), [d.id for d in field]) == []
    assert simulator.event_manager.events == []
    assert simulator.event_manager.current_lap is None
    assert not simulator.event_manager.red_flag_active
