"""Own-lap incident checks retain field-relative driver risk."""

import copy
import math

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventManager, EventType
from f1sim.simulation.race import DriverStatus, RaceSimulator


def field(count=2):
    return [Driver(id=str(i), name=str(i), team_id="T") for i in range(count)]


def circuit():
    return Track(id="t", name="T", country="T", total_laps=60,
                 base_lap_time=90, safety_car_probability=.5)


def wet():
    return Weather(track_wetness=.8, rain_intensity=.8)


def risk(selected, exposure, weather=None):
    return EventManager()._incident_subset_probability(
        selected, exposure, circuit(), weather or wet(),
    )


def test_heterogeneous_individual_survival_recovers_field_prior_without_mutation():
    drivers = field(22)
    for i, driver in enumerate(drivers):
        driver.consistency = .2 + i / 30
        driver.wet_skill_modifier = .5 + i / 25
    manager = EventManager(np.random.default_rng(4))
    before = copy.deepcopy((drivers, manager.rng.bit_generator.state))
    probabilities = [manager._incident_subset_probability([d], drivers, circuit(), wet())
                     for d in drivers]
    assert 1 - math.prod(1 - p for p in probabilities) == pytest.approx(
        manager._incident_probability(drivers, circuit(), wet()), abs=1e-14,
    )
    assert (drivers, manager.rng.bit_generator.state) == before
    assert risk(drivers[:7], drivers) == pytest.approx(
        1 - math.prod(1 - p for p in probabilities[:7]), abs=1e-14,
    )


def test_wet_skill_and_consistency_change_individual_risk():
    drivers = field()
    drivers[0].wet_skill_modifier = 1.5
    drivers[1].wet_skill_modifier = .5
    assert risk(drivers[:1], drivers) < risk(drivers[1:], drivers)
    assert risk(drivers[:1], drivers, Weather()) == risk(drivers[1:], drivers, Weather())
    drivers[0].wet_skill_modifier = drivers[1].wet_skill_modifier
    drivers[0].consistency = 1
    drivers[1].consistency = .2
    assert risk(drivers[:1], drivers) < risk(drivers[1:], drivers)


@pytest.mark.parametrize("count", [1, 2, 11, 22])
def test_uniform_field_preserves_previous_singleton_hazard(count):
    drivers = field(count)
    assert risk(drivers[:1], drivers) == pytest.approx(
        EventManager()._incident_probability(drivers[:1], circuit(), wet()),
    )


def test_retired_entries_reduce_exposure_and_last_survivor_keeps_risk():
    drivers = field(3)
    drivers[1].dnf = drivers[2].dnf = True
    assert risk(drivers, drivers) == pytest.approx(risk(drivers[:1], drivers[:1]))
    assert risk(drivers[:1], drivers) > 0
    assert risk([], drivers) == 0
    assert risk(drivers[1:], drivers) == 0
    assert risk([], []) == 0


@pytest.mark.parametrize("selected,exposure", [([0, 0], [0, 1]), ([0], [0, 0]),
                                                ([0], [1]), ([0], [])])
def test_invalid_active_ids_rejected_without_rng_draw(selected, exposure):
    drivers = field()
    manager = EventManager(np.random.default_rng(8))
    before = copy.deepcopy(manager.rng.bit_generator.state)
    with pytest.raises(ValueError):
        manager._check_random_incident([drivers[i] for i in selected], circuit(), wet(), 1,
                                      exposure_drivers=[drivers[i] for i in exposure])
    assert manager.rng.bit_generator.state == before


def test_large_field_cap_is_allocated_not_repeated_per_car():
    drivers = field(500)
    per_car = risk(drivers[:1], drivers)
    assert 1 - (1 - per_car) ** 500 == pytest.approx(.08)
    assert risk(drivers, drivers) == pytest.approx(.08)
    assert risk(drivers, drivers) == .08


class ThresholdRng:
    def __init__(self, threshold):
        self.threshold = threshold
        self.draws = 0

    def random(self):
        self.draws += 1
        return self.threshold if self.draws == 1 else .1

    def choice(self, candidates, p):
        assert len(candidates) == 1
        return candidates[0]

    def uniform(self, low, high):
        return 4.0


def test_between_threshold_draw_only_incidents_weaker_wet_driver():
    drivers = field()
    drivers[0].wet_skill_modifier = 1.5
    drivers[1].wet_skill_modifier = .5
    threshold = (risk(drivers[:1], drivers) + risk(drivers[1:], drivers)) / 2
    for driver, expected in zip(drivers, [False, True]):
        manager = EventManager(ThresholdRng(threshold))
        event = manager._check_random_incident([driver], circuit(), wet(), 2,
                                              exposure_drivers=drivers)
        assert (event is not None) is expected
        if event:
            assert event.event_type == EventType.SPIN
            assert event.drivers_involved == [driver.id]


def test_real_chronological_wet_lifecycle_context_and_delay_sampling(monkeypatch):
    drivers = field(4)
    simulator = RaceSimulator(np.random.default_rng(14))
    engine = ChronologicalRace(simulator)
    manager = simulator.event_manager
    monkeypatch.setattr(manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (True, False))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *a, **kw: 90 + int(driver.id) * 3)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    calls = []
    contexts = []
    original = manager._check_random_incident

    def check(selected, track, weather, lap, *, exposure_drivers):
        key = (selected[0].id, lap)
        assert key not in calls  # Spin-delayed crossing must not sample again.
        calls.append(key)
        ids = {d.id for d in exposure_drivers}
        assert ids == {d.driver.id for d in engine.states.values()
                       if d.status == DriverStatus.RACING and not d.driver.dnf}
        contexts.append(ids)
        if key == ("0", 1):
            # Retire another car through the actual lifecycle before later crossings.
            engine._retire("3", engine.pending["0"].ready, "test retirement")
        manager.rng = ThresholdRng(0 if key == ("1", 1) else 1)
        return original(selected, track, weather, lap, exposure_drivers=exposure_drivers)

    monkeypatch.setattr(manager, "_check_random_incident", check)
    track = circuit().model_copy(update={"total_laps": 3})
    results = engine.run(drivers, {"T": Car(team_id="T", team_name="T")}, track, wet(),
                         [d.id for d in drivers],
                         starting_tires={d.id: TireCompound.WET for d in drivers})
    assert len(results) == 4
    assert len(calls) == 9
    assert any(len(ids) == 1 for ids in contexts)
    assert all("3" not in ids for ids in contexts[1:])
    spins = [e for e in manager.events if e.event_type == EventType.SPIN]
    assert len(spins) == 1
    assert spins[0].drivers_involved == ["1"]


def test_chronological_exposure_includes_car_still_in_pit_lane(monkeypatch):
    drivers = field()
    simulator = RaceSimulator(np.random.default_rng(7))
    engine = ChronologicalRace(simulator)
    manager = simulator.event_manager
    monkeypatch.setattr(manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(simulator, "_should_pit",
                        lambda state, states, track, lap, *a, **kw:
                        state.driver.id == "1" and lap == 2)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda *a, **kw: 90)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 100)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (True, False))
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    observed = []

    def check(selected, track, weather, lap, *, exposure_drivers):
        if selected[0].id == "0" and lap == 2:
            assert not engine.pending["1"].on_track
            assert {d.id for d in exposure_drivers} == {"0", "1"}
            observed.append(True)
        return None

    monkeypatch.setattr(manager, "_check_random_incident", check)
    engine.run(drivers, {"T": Car(team_id="T", team_name="T")},
               circuit().model_copy(update={"total_laps": 4}), wet(), ["0", "1"],
               starting_tires={d.id: TireCompound.WET for d in drivers})
    assert observed == [True]
