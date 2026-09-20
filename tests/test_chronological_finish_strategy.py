"""Chronological integration of the bounded elective-stop finish guard."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import race as race_module
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype
from f1sim.simulation.tire_inventory import TireInventory, TireSet


def setup(monkeypatch, *, pit_lane_delta=80.0, stop_mode="elective"):
    simulator = RaceSimulator(np.random.default_rng(21))
    engine = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="T", name="T", country="T", total_laps=4, base_lap_time=100,
                  pit_lane_delta=pit_lane_delta)
    decisions = []

    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *a, **k: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **k: (True, False))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, *a, **k: 90.0 if driver.id == "A" else 100.0)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 0.0)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    # The fixture intentionally supplies a legal already-satisfied obligation;
    # the guard is testing distance, not the mandatory tyre-rule branch.
    monkeypatch.setattr(simulator, "_stay_satisfies_tire_rule", lambda state: True)

    def decide(state, states, planning, lap, *args, **kwargs):
        if state.driver.id == "B" and lap == 1:
            state.dry_pit_proposal = (lap, TireCompound.SOFT)
            decisions.append((state.driver.id, lap, True))
            if stop_mode == "forced":
                state.force_pit_next_lap = True
            return True
        return False

    monkeypatch.setattr(simulator, "_should_pit", decide)

    def run():
        return engine.run(
            drivers,
            cars,
            track,
            Weather(),
            list("AB"),
            starting_tires={key: TireCompound.HARD for key in "AB"},
        )

    return engine, run, decisions


def test_native_elective_stop_is_vetoed_when_it_loses_a_lap(monkeypatch):
    engine, run, decisions = setup(monkeypatch)

    result = next(row for row in run() if row.driver_id == "B")

    assert decisions == [("B", 1, True)]
    assert result.pit_laps == []
    assert result.laps_completed == 4
    assert engine.pit_exits == []


def test_equal_finish_distance_leaves_native_stop_untouched(monkeypatch):
    engine, run, _ = setup(monkeypatch, pit_lane_delta=0.1)

    result = next(row for row in run() if row.driver_id == "B")

    assert result.pit_laps == [1]
    assert result.laps_completed == 4
    assert len(engine.pit_exits) == 1


def test_forced_stop_survives_finish_guard(monkeypatch):
    engine, run, _ = setup(monkeypatch, stop_mode="forced")

    result = next(row for row in run() if row.driver_id == "B")

    assert result.pit_laps == [1]
    assert result.laps_completed == 3
    assert len(engine.pit_exits) == 1


def _faithful_real_physics_run(monkeypatch, *, guarded):
    """Reproduce the late two-lap native proposal with mean B physics."""
    with monkeypatch.context() as patch:
        drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
        cars = {
            key: Car(team_id=key, team_name=key, pit_stop_avg=1.8, pit_stop_std=0.1)
            for key in "AB"
        }
        track = Track(id="timed-pit-distance", name="Timed pit distance", country="Test",
                      total_laps=90, base_lap_time=113.4, pit_lane_delta=20)
        simulator = RaceSimulator(np.random.default_rng(123))
        simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
        engine = ChronologicalRace(simulator)
        original_should_pit = simulator._should_pit
        original_lap_time = simulator.lap_simulator.calculate_lap_time
        original_plan = race_module.plan_dry_stop
        service = expected_stationary_time(cars["B"])
        target = {}

        def capture_plan(*args, **kwargs):
            decision = original_plan(*args, **kwargs)
            if args[0].id == "B" and args[2].total_laps == 62:
                target.update(
                    pit_now=float(decision.pit_now_cost),
                    wait_cost=float(decision.wait_cost),
                    chosen_compound=decision.compound,
                )
            return decision

        patch.setattr(race_module, "plan_dry_stop", capture_plan)

        def decide(state, states, planning, lap, *args, **kwargs):
            if state.driver.id == "B" and lap == 17:
                state.dry_pit_proposal = (lap, TireCompound.SOFT)
                return True
            if state.driver.id == "B" and lap < 61:
                return False
            if state.driver.id == "B" and lap == 61:
                decision = bool(original_should_pit(state, states, planning, lap, *args, **kwargs))
                target.update(
                    native_stop=decision,
                    proposal=state.dry_pit_proposal,
                    planning_horizon=planning.total_laps,
                    projected_flag=float(engine._projected_flag_time(state.total_time)),
                )
                return decision
            return False

        patch.setattr(simulator, "_should_pit", decide)
        if not guarded:
            patch.setattr(engine, "_protect_elective_finish_distance",
                          lambda *args, **kwargs: False)
        patch.setattr(simulator.event_manager, "process_lap", lambda *a, **k: [])
        patch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
        patch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **k: None)
        patch.setattr(simulator.overtaking_model, "attempt_overtake",
                      lambda *a, **k: (True, False))

        def running(driver, car, lap_track, tire, weather, lap, total, *args, **kwargs):
            if driver.id == "A":
                return 88.0
            kwargs.pop("sample_variation", None)
            return original_lap_time(
                driver, car, lap_track, tire, weather, lap, total,
                *args, sample_variation=False, **kwargs,
            )

        patch.setattr(simulator.lap_simulator, "calculate_lap_time", running)
        patch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: service)
        patch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
        results = engine.run(
            drivers,
            cars,
            track,
            Weather(),
            ["A", "B"],
            starting_tires={"A": TireCompound.HARD, "B": TireCompound.HARD},
        )
        return next(row for row in results if row.driver_id == "B"), target


def test_faithful_late_native_stop_is_vetoed_by_real_mean_physics(monkeypatch):
    guarded, target = _faithful_real_physics_run(monkeypatch, guarded=True)
    baseline, baseline_target = _faithful_real_physics_run(monkeypatch, guarded=False)

    assert target["native_stop"] is True
    assert target["proposal"] == (61, TireCompound.SOFT)
    assert target["planning_horizon"] == 62
    assert target["projected_flag"] == 7304
    assert target["pit_now"] < target["wait_cost"]
    assert target["chosen_compound"] == TireCompound.SOFT
    assert baseline_target == target
    assert guarded.laps_completed == 62
    assert guarded.pit_laps == [17]
    assert baseline.laps_completed == 61
    assert baseline.pit_laps == [17, 61]


def decision_snapshot(monkeypatch):
    engine = ChronologicalRace(RaceSimulator(np.random.default_rng(42)))
    engine.control_intervals = 0
    engine.track = Track(id="t", name="T", country="T", total_laps=10,
                         base_lap_time=100, pit_lane_delta=80)
    engine.weather = Weather()
    state = DriverRaceState(
        Driver(id="B", name="B", team_id="B"), Car(team_id="B", team_name="B"), 2,
        tire_compound_history=["hard", "medium"],
        dry_pit_proposal=(1, TireCompound.SOFT),
    )
    leader = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                             Car(team_id="A", team_name="A"), 1)
    monkeypatch.setattr(engine, "_forecast_leader", lambda: leader)
    monkeypatch.setattr(engine, "_projected_flag_time", lambda *a, **k: 250)
    monkeypatch.setattr(engine, "_weather_projection_clock", lambda *a, **k: (100, 100, 2))
    return engine, state, leader


@pytest.mark.parametrize("exclusion", [
    "repair", "critical", "compound_rule", "unavailable", "leader",
    "safety_car_active", "vsc_active", "red_flag_active", "missing_clock",
])
def test_compulsory_and_uncertain_decisions_are_not_vetoed(monkeypatch, exclusion):
    engine, state, _ = decision_snapshot(monkeypatch)
    assert engine._protect_elective_finish_distance(state, engine.track, 0, 0, None)
    if exclusion == "repair":
        state.force_pit_next_lap = True
    elif exclusion == "critical":
        engine.weather = Weather(track_wetness=.9, rain_intensity=.9)
    elif exclusion == "compound_rule":
        state.tire_compound_history = ["medium"]
    elif exclusion == "unavailable":
        state.tire_inventory = TireInventory([TireSet("current", TireCompound.MEDIUM, 0)])
        state.tire_inventory.fit("current")
        state.tire_inventory.unavailable_ids.add("current")
    elif exclusion == "leader":
        monkeypatch.setattr(engine, "_forecast_leader", lambda: state)
    elif exclusion == "missing_clock":
        monkeypatch.setattr(engine, "_weather_projection_clock", lambda *a, **k: None)
    else:
        setattr(engine.simulator.event_manager, exclusion, True)
    assert not engine._protect_elective_finish_distance(state, engine.track, 0, 0, None)


def test_selected_physical_set_and_forecast_leave_ledger_and_rng_unchanged(monkeypatch):
    engine, state, _ = decision_snapshot(monkeypatch)
    state.tire_inventory = TireInventory([
        TireSet("current", TireCompound.MEDIUM, 0),
        TireSet("selected", TireCompound.HARD, 7),
        TireSet("faster", TireCompound.SOFT, 0),
    ])
    state.tire_inventory.fit("current")
    state.inventory_pit_proposal = (1, "selected")
    before = deepcopy((state, engine.weather, engine.simulator.rng.bit_generator.state))
    options = engine._finish_replacement_options(state, lap=1)
    assert [(option.identifier, option.compound, option.age) for option in options] == [
        ("selected", TireCompound.HARD, 7),
    ]
    assert engine._protect_elective_finish_distance(state, engine.track, 0, 0, None)
    assert state.tire_inventory.__dict__ == before[0].tire_inventory.__dict__
    assert state.tire_set_history == before[0].tire_set_history
    assert state.inventory_pit_proposal == before[0].inventory_pit_proposal
    assert engine.weather == before[1]
    assert engine.simulator.rng.bit_generator.state == before[2]


def test_surface_at_expected_exit_counts_updates_but_excludes_flag(monkeypatch):
    engine, _, _ = decision_snapshot(monkeypatch)
    engine.weather = Weather(track_wetness=.1, rain_intensity=.8)
    initial = engine.weather.model_copy(deep=True)
    assert engine._projected_surface_at(99, now=0) == initial
    assert engine._projected_surface_at(100, now=0) == initial.project_surface()
    expected = initial.project_surface().project_surface()
    assert engine._projected_surface_at(200, now=0) == expected
    assert engine._projected_surface_at(300, now=0) == expected
    assert engine.weather == initial


def test_weather_choice_precedes_stale_dry_proposal(monkeypatch):
    engine, state, _ = decision_snapshot(monkeypatch)
    engine.weather = Weather(track_wetness=.8, rain_intensity=.8)
    state.current_tire = TIRE_COMPOUNDS[TireCompound.WET]
    assert engine._finish_replacement_options(state, lap=1)[0].compound == TireCompound.WET
