"""Recorded pit laps include lane, service and queue losses exactly once."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import RaceSimulator


@pytest.mark.parametrize("neutralization,lane_factor,lap_factor", [
    (None, 1.0, 1.0), ("safety_car_active", 0.55, 1.4), ("vsc_active", 0.75, 1.2),
])
def test_fastest_pit_lap_records_actual_shared_box_losses(
    monkeypatch, neutralization, lane_factor, lap_factor,
):
    simulator = RaceSimulator(np.random.default_rng(4))
    field = [Driver(id=name, name=name, team_id="team") for name in ("A", "B")]
    car = Car(team_id="team", team_name="Team")
    track = Track(id="test", name="Test", country="Test", total_laps=2,
                  base_lap_time=90, pit_lane_delta=20)
    monkeypatch.setattr(simulator, "_should_pit", lambda s, states, t, lap, *a, **kw: lap == 2)
    monkeypatch.setattr(simulator, "_process_overtakes", lambda *a, **kw: 0)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda **kw: 200 if kw["lap_number"] == 1 else 80)

    def events(lap, **kwargs):
        if lap == 1 and neutralization:
            setattr(simulator.event_manager, neutralization, True)
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    results = simulator.simulate_race(field, {"team": car}, track, Weather(), ["A", "B"])
    expected = [80 * lap_factor + 20 * lane_factor + 3,
                80 * lap_factor + 20 * lane_factor + 6]
    assert [r.fastest_lap for r in results] == pytest.approx(expected)
    assert [r.total_time for r in results] == pytest.approx([200 + t for t in expected])
    assert [r.pit_stops for r in results] == [1, 1]


@pytest.mark.parametrize("retire", [False, True])
def test_fast_clean_pace_on_a_pit_lap_cannot_replace_previous_fastest(monkeypatch, retire):
    simulator = RaceSimulator(np.random.default_rng(5))
    driver = Driver(id="A", name="A", team_id="team")
    car = Car(team_id="team", team_name="Team")
    track = Track(id="test", name="Test", country="Test", total_laps=2,
                  base_lap_time=90, pit_lane_delta=20)
    monkeypatch.setattr(simulator, "_should_pit", lambda s, states, t, lap, *a, **kw: lap == 2)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 3)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda **kw: 90 if kw["lap_number"] == 1 else 80)

    def events(lap, drivers, **kwargs):
        if lap == 2 and retire:
            drivers[0].dnf = True
            drivers[0].dnf_reason = "engine failure"
            return [RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"])]
        return []

    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    result = simulator.simulate_race([driver], {"team": car}, track, Weather(), ["A"])[0]
    assert result.fastest_lap == 90
    assert result.total_time == (90 if retire else 193)
    assert result.laps_completed == (1 if retire else 2)
