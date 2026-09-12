"""Physical tyre conservation through both native race execution paths."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventManager, EventType, RaceEvent
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.tire_inventory import TireInventory


@pytest.fixture(params=["standard", "chronological"])
def race(request, monkeypatch):
    sim = RaceSimulator(np.random.default_rng(7))
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    services, running = [], []
    calculate = sim.lap_simulator.calculate_lap_time

    def pace(*args, **kwargs):
        current = args[0] if args else kwargs["driver"]
        running.append(current.current_tire_laps)
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", pace)
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time",
                        lambda car: services.append(car.team_id) or expected_stationary_time(car))
    monkeypatch.setattr(sim.event_manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(sim.event_manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(sim.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))

    def run(records, compound="soft", age=0, weather=None):
        execute = sim.simulate_race if request.param == "standard" else ChronologicalRace(sim).run
        return execute(
            [driver], {"A": car}, track, weather or Weather(change_probability=0), ["A"],
            starting_tires={"A": compound} if compound else None,
            starting_tire_ages={"A": age} if compound else None,
            tire_inventory={"A": records},
        )[0]

    return request.param, sim, track, run, services, running


def conserved(result, initial):
    ages = {item["id"]: item.get("age", 0) for item in initial}
    for stint in result.tire_set_history:
        assert stint["age_at_fit"] == ages[stint["set_id"]]
        assert stint["laps_used"] >= 0
        assert stint["age_at_end"] == stint["age_at_fit"] + stint["laps_used"]
        ages[stint["set_id"]] = stint["age_at_end"]
    assert {item["id"]: item["age"] for item in result.tire_inventory} == ages
    assert sum(item["current"] for item in result.tire_inventory) == 1
    assert all(not (item["available"] and (item["current"] or item["unavailable"]))
               for item in result.tire_inventory)


def test_used_set_returns_and_runs_again_without_fresh_wear(race, monkeypatch):
    _, sim, _, run, services, running = race
    records = [{"id": "S", "compound": "soft", "age": 5},
               {"id": "H", "compound": "hard"}]
    before = deepcopy(records)

    def policy(state, states, track, lap, *args, **kwargs):
        selected = {2: "H", 4: "S"}.get(lap)
        state.inventory_pit_proposal = (lap, selected)
        return selected is not None

    monkeypatch.setattr(sim, "_should_pit", policy)
    result = run(records, age=5)
    assert result.status == DriverStatus.FINISHED
    assert result.strategy == ["soft", "hard", "soft"]
    assert result.pit_laps == [2, 4]
    assert len(services) == result.pit_stops == 2
    assert running == [5, 0, 1, 6, 7, 8, 9, 10]
    assert [item["incoming_tire_age"] for item in result.pit_stop_details] == [0, 6]
    assert [(item["from_set_id"], item["to_set_id"]) for item in result.pit_stop_details] == [
        ("S", "H"), ("H", "S"),
    ]
    conserved(result, records)
    assert sum(stint["laps_used"] for stint in result.tire_set_history) == 8
    assert records == before


def test_missing_legal_compound_withdraws_before_final_service(race):
    _, _, _, run, services, running = race
    result = run([{"id": "S", "compound": "soft"}])
    assert result.status == DriverStatus.DNF
    assert result.dnf_reason == "No suitable replacement tyre set available"
    assert result.laps_completed == len(running) == 7
    assert result.pit_stops == 0
    assert services == result.pit_stop_details == []


def test_critical_opening_without_safe_replacement_never_runs(race):
    _, _, _, run, services, running = race
    result = run([{"id": "W", "compound": "wet"}], compound="wet")
    assert result.status == DriverStatus.DNF
    assert result.laps_completed == result.pit_stops == 0
    assert services == running == []
    assert result.tire_set_history[0]["laps_used"] == 0


def puncture(monkeypatch, engine, sim, lap=2):
    event = RaceEvent(EventType.PUNCTURE, lap, ["A"], forces_pit_stop=True)
    if engine == "standard":
        monkeypatch.setattr(sim.event_manager, "process_lap",
                            lambda **kw: [event] if kw["lap"] == lap else [])
    else:
        monkeypatch.setattr(sim.event_manager, "_check_random_incident",
                            lambda drivers, track, weather, current, **kw:
                            event if current == lap else None)


@pytest.mark.parametrize("replacement", [False, True])
def test_punctured_set_cannot_return_and_empty_pool_withdraws(race, monkeypatch, replacement):
    engine, sim, _, run, services, running = race
    puncture(monkeypatch, engine, sim)
    records = [{"id": "I1", "compound": "intermediate", "age": 3}]
    if replacement:
        records.append({"id": "I2", "compound": "intermediate", "age": 1})
    result = run(records, "intermediate", 3,
                 Weather(track_wetness=.5, rain_intensity=.3, change_probability=0))
    damaged = next(item for item in result.tire_inventory if item["id"] == "I1")
    assert damaged["unavailable"] and not damaged["available"]
    assert damaged["age"] == 5
    assert result.laps_completed == len(running) == (8 if replacement else 2)
    assert result.status == (DriverStatus.FINISHED if replacement else DriverStatus.DNF)
    assert len(services) == result.pit_stops == int(replacement)
    if replacement:
        assert result.pit_laps == [3]
        assert result.tire_set_history[-1]["set_id"] == "I2"
        assert running == [3, 4, 1, 2, 3, 4, 5, 6]
    conserved(result, records)


def test_automatic_selection_uses_pool_without_mutating_inputs_or_rng():
    sim = RaceSimulator(np.random.default_rng(3))
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    weather = Weather(change_probability=0)
    records = [{"id": "S", "compound": "soft", "age": 5},
               {"id": "H", "compound": "hard"}]
    before = deepcopy((driver, car, track, weather, records, sim.rng.bit_generator.state))
    from f1sim.simulation.race import TeamStrategyArchetype

    pool, selected = sim._inventory_opening_set(
        driver, car, track, weather, TeamStrategyArchetype.BALANCED, records,
    )
    assert selected.id in {"S", "H"}
    assert pool.current_set_id is None
    assert (driver, car, track, weather, records, sim.rng.bit_generator.state) == before


def state_with_pool(records, selected, age):
    sim = RaceSimulator(np.random.default_rng(0))
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1)
    pool = TireInventory.from_sets(records)
    sim._initialize_inventory(state, pool, pool.sets[selected])
    state.tire_laps = state.driver.current_tire_laps = age
    track = Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90)
    return sim, state, track


def test_free_refit_can_retain_only_usable_set_without_resetting_wear():
    records = [{"id": "I", "compound": "intermediate", "age": 4}]
    sim, state, track = state_with_pool(records, "I", 7)
    state.pit_stops = 2
    before = deepcopy(sim.rng.bit_generator.state)
    assert sim._refit_inventory_free(
        state, track, Weather(track_wetness=.5, rain_intensity=.3), 3,
    )
    assert state.tire_laps == 7 and state.prior_tire_laps == 4
    assert state.pit_stops == 2 and len(state.tire_set_history) == 1
    assert sim.rng.bit_generator.state == before


def test_unrun_paid_set_returns_unchanged_when_red_flag_fits_another_set():
    records = [{"id": "S", "compound": "soft", "age": 2},
               {"id": "H", "compound": "hard", "age": 5},
               {"id": "I", "compound": "intermediate", "age": 3}]
    sim, state, track = state_with_pool(records, "S", 4)
    state.inventory_pit_proposal = (3, "H")
    sim._execute_pit_stop(state, track, Weather(), 3, sample_service=False)
    paid = deepcopy(state.pit_stop_details)
    state.pit_stops = 1
    assert sim._refit_inventory_free(
        state, track, Weather(track_wetness=.5, rain_intensity=.3), 2,
    )
    assert state.tire_inventory.current_set_id == "I"
    assert state.tire_inventory.sets["H"].age == 5
    assert state.pit_stop_details == paid and state.pit_stops == 1
    assert [(item["set_id"], item["laps_used"]) for item in state.tire_set_history] == [
        ("S", 2), ("H", 0), ("I", 0),
    ]
    assert state.tire_compound_history == ["soft", "intermediate"]
    result = sim._inventory_result_fields(state)
    result["tire_set_history"][0]["age_at_fit"] = 100
    result["tire_inventory"][0]["age"] = 100
    assert state.tire_set_history[0]["age_at_fit"] == 2
    assert state.tire_inventory.sets["S"].age == 4


@pytest.mark.parametrize("replacement", [False, True])
def test_red_flag_repairs_puncture_from_pool_or_withdraws_without_service(
    race, monkeypatch, replacement,
):
    _, sim, _, run, services, running = race
    control = sim.event_manager
    monkeypatch.setattr(control, "process_lap", EventManager.process_lap.__get__(control))
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a: None)
    monkeypatch.setattr(control, "_check_random_incident",
                        lambda drivers, track, weather, lap, **kw:
                        RaceEvent(EventType.PUNCTURE, lap, ["A"], forces_pit_stop=True)
                        if lap == 2 else None)
    control.set_forced_red_flag(2)
    records = [{"id": "I1", "compound": "intermediate", "age": 3}]
    if replacement:
        records.append({"id": "I2", "compound": "intermediate", "age": 1})
    result = run(records, "intermediate", 3,
                 Weather(track_wetness=.5, rain_intensity=.3, change_probability=0))
    assert result.pit_stops == 0 and services == []
    assert result.status == (DriverStatus.FINISHED if replacement else DriverStatus.DNF)
    assert result.laps_completed == len(running) == (8 if replacement else 2)
    if replacement:
        assert result.tire_set_history[-1]["kind"] == "red_flag"
        assert result.tire_set_history[-1]["age_at_fit"] == 1
    assert result.tire_inventory[0]["unavailable"]
    conserved(result, records)


def test_chronological_closed_exit_returns_unrun_paid_set_and_fits_restart_pool(monkeypatch):
    sim = RaceSimulator(np.random.default_rng(7))
    engine = ChronologicalRace(sim)
    control = sim.event_manager
    control.set_forced_red_flag(2)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a: None)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **kw: None)
    # The field reaches damp restart conditions while B is still in paid service.
    def evolve(weather, rng):
        return weather.model_copy(update={"track_wetness": .5, "rain_intensity": .3}
                                  if control.red_flag_active else {})

    monkeypatch.setattr(Weather, "evolve", evolve)
    services, samples = [], []
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time",
                        lambda car: services.append(car.team_id) or 100)
    monkeypatch.setattr(sim.overtaking_model, "attempt_overtake", lambda *a, **kw: (True, False))

    def policy(state, states, track, lap, *a, **kw):
        state.inventory_pit_proposal = (lap, "H")
        return state.driver.id == "B" and lap == 2

    def pace(driver, car, track, tire, weather, lap, *a, **kw):
        samples.append((driver.id, lap, driver.current_tire_laps, tire.compound.value))
        return 90 if driver.id == "A" else 100

    monkeypatch.setattr(sim, "_should_pit", policy)
    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", pace)
    records = [{"id": "S", "compound": "soft", "age": 2},
               {"id": "H", "compound": "hard", "age": 5},
               {"id": "I", "compound": "intermediate", "age": 3}]
    results = engine.run(
        [Driver(id=key, name=key, team_id=key) for key in "AB"],
        {key: Car(team_id=key, team_name=key) for key in "AB"},
        Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), list("AB"),
        starting_tires={key: "soft" for key in "AB"},
        starting_tire_ages={key: 2 for key in "AB"},
        tire_inventory={key: records for key in "AB"},
    )
    assert engine.suspensions == [(180, 820, ("A", "B"))]
    assert engine.pit_exits == [("B", 2, 820)]
    assert services == ["B"] and engine.expected_box_releases == {}
    assert [sample for sample in samples if sample[:2] == ("B", 2)] == [
        ("B", 2, 3, "intermediate"),
    ]
    b = next(item for item in results if item.driver_id == "B")
    assert b.pit_stops == 1 and b.laps_completed == 3
    assert [(item["set_id"], item["laps_used"]) for item in b.tire_set_history] == [
        ("S", 1), ("H", 0), ("I", 2),
    ]
    assert b.strategy == ["soft", "intermediate"]
    assert all(item.status == DriverStatus.FINISHED for item in results)
    for result in results:
        conserved(result, records)


def test_automatic_pool_policy_handles_shortened_finish(race, monkeypatch):
    from f1sim.simulation import race_timing

    monkeypatch.setattr(race_timing, "RACING_TIME_LIMIT_SECONDS", 100)
    _, _, _, run, _, _ = race
    records = [{"id": "S", "compound": "soft", "age": 5},
               {"id": "H", "compound": "hard"}]
    result = run(records, compound=None)
    assert result.status == DriverStatus.FINISHED
    assert result.race_time_limited and result.laps_completed == 3
    assert len(set(result.strategy)) == 2
    assert sum(item["laps_used"] for item in result.tire_set_history) == 3
    conserved(result, records)


@pytest.mark.parametrize("paid_fit", [False, True])
def test_mechanical_retirement_does_not_credit_failed_lap_to_set(race, monkeypatch, paid_fit):
    _, sim, _, run, _, running = race
    control = sim.event_manager
    monkeypatch.setattr(control, "process_lap", EventManager.process_lap.__get__(control))
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a: None)

    def fail(driver, car, track, lap, weather):
        if lap == 2:
            driver.dnf, driver.dnf_reason = True, "Controlled failure"
            return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])
        return None

    monkeypatch.setattr(control, "_check_mechanical_failure", fail)
    records = [{"id": "I", "compound": "intermediate", "age": 3}]
    if paid_fit:
        records.append({"id": "I2", "compound": "intermediate", "age": 6})

        def policy(state, states, track, lap, *a, **kw):
            state.inventory_pit_proposal = (lap, "I2")
            return lap == 2

        monkeypatch.setattr(sim, "_should_pit", policy)
    result = run(records, "intermediate", 3,
                 Weather(track_wetness=.5, rain_intensity=.3, change_probability=0))
    assert result.status == DriverStatus.DNF and result.laps_completed == 1
    assert running == [3, 6 if paid_fit else 4]
    assert result.tire_set_history[0]["laps_used"] == 1
    if paid_fit:
        assert result.pit_stops == 1 and result.tire_set_history[-1]["laps_used"] == 0
    conserved(result, records)
