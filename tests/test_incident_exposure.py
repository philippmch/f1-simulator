"""Random incidents follow active-car exposure, including the last survivor."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventManager, EventType


def drivers(count):
    return [Driver(id=str(i), name=str(i), team_id=str(i)) for i in range(count)]


def track():
    return Track(id="test", name="Test", country="Test", total_laps=60,
                 base_lap_time=90, safety_car_probability=0.5)


def probability(field):
    return EventManager()._incident_probability(field, track(), Weather())


def test_full_grid_preserves_previous_incident_prior():
    # Snapshot of the pre-exposure model at this exact synthetic configuration.
    assert probability(drivers(22)) == pytest.approx(0.023890837665976043)


def test_retirement_reduces_risk_without_making_last_survivor_immune():
    risks = [probability(drivers(n)) for n in (0, 1, 2, 11, 22)]
    assert risks[0] == 0
    assert all(a < b for a, b in zip(risks, risks[1:]))
    field = drivers(22)
    for driver in field[2:]:
        driver.dnf = True
    assert probability(field) == probability(field[:2])


def test_partitioning_equal_drivers_preserves_combined_survival_probability():
    full_survival = 1 - probability(drivers(22))
    assert (1 - probability(drivers(11))) ** 2 == pytest.approx(full_survival)
    assert (1 - probability(drivers(1))) ** 22 == pytest.approx(full_survival)


def test_oversized_synthetic_field_retains_defensive_probability_cap():
    assert probability(drivers(500)) == pytest.approx(0.08)


class IncidentRng:
    def __init__(self, severity):
        self.draws = iter([0.0, severity])

    def random(self):
        return next(self.draws)

    def choice(self, candidates, p):
        assert len(candidates) == 1
        assert p == pytest.approx([1.0])
        return candidates[0]

    def uniform(self, low, high):
        return (low + high) / 2


@pytest.mark.parametrize("severity,event_type,retired,forced_stop", [
    (0.1, EventType.SPIN, False, False),
    (0.4, EventType.PUNCTURE, False, True),
    (0.9, EventType.COLLISION, True, False),
])
def test_last_survivor_can_have_each_single_car_incident(
    monkeypatch, severity, event_type, retired, forced_stop,
):
    manager = EventManager(IncidentRng(severity))
    field = drivers(1)
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(manager, "_deploy_safety_measure", lambda *a: None)
    events = manager.process_lap(
        20, field, {"0": Car(team_id="0", team_name="0")}, track(), Weather(),
    )
    assert len(events) == 1
    assert events[0].event_type == event_type
    assert events[0].drivers_involved == ["0"]
    assert events[0].forces_pit_stop is forced_stop
    assert events[0].time_loss_seconds > 0
    assert field[0].dnf is retired


def test_empty_or_retired_field_does_not_sample_incidents():
    manager = EventManager(np.random.default_rng(1))
    before = manager.rng.bit_generator.state
    retired = drivers(1)
    retired[0].dnf = True
    for field in ([], retired):
        assert manager._check_random_incident(field, track(), Weather(), 20) is None
    assert manager.rng.bit_generator.state == before


@pytest.mark.parametrize("flag", ["safety_car_active", "vsc_active", "red_flag_active"])
def test_last_survivor_remains_protected_from_racing_incidents_under_neutralisation(
    monkeypatch, flag,
):
    manager = EventManager()
    setattr(manager, flag, True)
    manager.safety_car_laps_remaining = manager.vsc_laps_remaining = 2
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *a: None)
    def unexpected(*args):
        pytest.fail("Racing incidents must not be sampled during neutralisation")
    monkeypatch.setattr(manager, "_check_random_incident", unexpected)
    assert manager.process_lap(
        20, drivers(1), {"0": Car(team_id="0", team_name="0")}, track(), Weather(),
    ) == []
