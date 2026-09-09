"""The last retirement ends control sampling without discarding the incident."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceSimulator


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
@pytest.mark.parametrize("retirement_lap", [1, 3])
@pytest.mark.parametrize("cause", ["mechanical", "crash"])
@pytest.mark.parametrize("signal", ["forced_sc", "forced_red", "background"])
def test_last_retirement_prevents_new_control_events(
    monkeypatch, engine_name, retirement_lap, cause, signal,
):
    simulator = RaceSimulator(np.random.default_rng(8))
    control = simulator.event_manager
    driver = Driver(id="A", name="A", team_id="T")
    cars = {"T": Car(team_id="T", team_name="T")}
    track = Track(id="t", name="T", country="T", total_laps=5, base_lap_time=90)
    event_type = (EventType.MECHANICAL_FAILURE if cause == "mechanical"
                  else EventType.COLLISION)
    hazards, evolves = [], []

    def retirement(lap):
        if lap != retirement_lap:
            return None
        driver.dnf = True
        driver.dnf_reason = cause
        return RaceEvent(event_type, lap, [driver.id])

    def mechanical(driver, car, track, lap, weather):
        return retirement(lap) if cause == "mechanical" else None

    def incident(drivers, track, weather, lap, **kwargs):
        return retirement(lap) if cause == "crash" else None

    def background(lap, *args):
        hazards.append(lap)
        if signal == "background" and lap == retirement_lap:
            control.safety_car_active = True
            control.safety_car_deployments += 1
            return RaceEvent(EventType.SAFETY_CAR, lap)
        return None

    monkeypatch.setattr(control, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(control, "_check_random_incident", incident)
    monkeypatch.setattr(control, "_deploy_safety_measure", background)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *a, **kw: 0)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lambda *a, **kw: 90)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng:
                        evolves.append(True) or self.model_copy(deep=True))
    if signal == "forced_sc":
        control.set_forced_safety_car(retirement_lap)
    elif signal == "forced_red":
        control.set_forced_red_flag(retirement_lap)
    run = (simulator.simulate_race if engine_name == "standard"
           else ChronologicalRace(simulator).run)
    (result,) = run([driver], cars, track, Weather(), [driver.id],
                   starting_tires={driver.id: TireCompound.HARD})

    assert [(event.event_type, event.lap, event.drivers_involved) for event in control.events] == [
        (event_type, retirement_lap, [driver.id]),
    ]
    assert hazards == list(range(1, retirement_lap))
    assert len(evolves) == retirement_lap - 1
    assert not control.safety_car_active and not control.red_flag_active
    assert control.safety_car_deployments == control.red_flag_deployments == 0
    assert result.status == DriverStatus.DNF and not result.classified
    assert result.laps_completed == retirement_lap - 1
    assert result.total_time == 90 * (retirement_lap - 1)
    assert result.points_awarded == 0


def test_surviving_car_still_receives_control_after_another_car_retires(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(8))
    control = simulator.event_manager
    drivers = [Driver(id=key, name=key, team_id="T") for key in ("A", "B")]
    track = Track(id="t", name="T", country="T", total_laps=5, base_lap_time=90)

    def mechanical(driver, car, track, lap, weather):
        if driver.id == "A":
            driver.dnf = True
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(control, "_check_mechanical_failure", mechanical)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **kw: None)
    control.set_forced_safety_car(1)
    events = control.process_lap(
        1, drivers, {"T": Car(team_id="T", team_name="T")}, track, Weather(),
    )
    assert [event.event_type for event in events] == [
        EventType.MECHANICAL_FAILURE, EventType.SAFETY_CAR,
    ]
    assert control.safety_car_active and not drivers[1].dnf
