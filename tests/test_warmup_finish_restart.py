"""Finish forecasts and free red-flag refits account for fitting elapsed time."""

from types import SimpleNamespace

import numpy as np

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.finish_strategy import ReplacementOption, evaluate_finish_protection
from f1sim.simulation.race import RaceSimulator


class FixedPace:
    def __init__(self, seconds):
        self.seconds = seconds

    def calculate_lap_time(self, *args, **kwargs):
        return self.seconds


def test_pending_fit_fee_changes_finish_distance_and_only_delays_later_weather_entries():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="T", total_laps=5,
                  base_lap_time=90, pit_lane_delta=80)
    weather = Weather()
    current = TIRE_COMPOUNDS[TireCompound.MEDIUM]

    def run(profile=None, current_fit_pending=False):
        entries = []

        def surface_at(absolute_time):
            entries.append(absolute_time)
            return Weather()

        result = evaluate_finish_protection(
            driver,
            car,
            track,
            current,
            0,
            1,
            weather,
            0,
            195,
            5,
            lap_simulator=FixedPace(90),
            expected_lane_loss=80,
            replacements=(ReplacementOption(TireCompound.MEDIUM),),
            projected_surface_at=surface_at,
            tire_warmup=profile,
            current_fit_pending=current_fit_pending,
        )
        return result, entries

    baseline, baseline_entries = run()
    fitted, fitted_entries = run({"medium": 30}, current_fit_pending=True)

    assert baseline.retained_laps == 3
    assert fitted.retained_laps == 2
    assert baseline.stop_laps == 2
    assert fitted.stop_laps == 1
    # The fresh set's outlap sees the surface at the actual 80-second pit exit.
    # The 30-second fit cost moves its crossing later, while the retained path's
    # next entry moves from 90 to 120 seconds.
    assert 80 in fitted_entries and 110 not in fitted_entries
    assert 90 in baseline_entries and 120 in fitted_entries


def test_chronological_red_flag_different_inventory_set_pays_one_restart_fee(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(7), tire_warmup={"hard": 30})
    engine = ChronologicalRace(simulator, red_flag_pause_seconds=100)
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90)
    simulator.event_manager.set_forced_red_flag(1)
    monkeypatch.setattr(simulator.event_manager, "_deploy_safety_measure", lambda *a: None)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))
    monkeypatch.setattr(simulator, "_should_pit", lambda *a, **k: False)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **k: None)
    actual_plan = simulator._plan_inventory

    def choose_restart_set(state, *args, free_fit=False, **kwargs):
        if free_fit:
            return SimpleNamespace(set_id="restart")
        return actual_plan(state, *args, free_fit=free_fit, **kwargs)

    monkeypatch.setattr(simulator, "_plan_inventory", choose_restart_set)
    running = []

    def record_running(driver, car, track, tire, weather, lap, total_laps, **kwargs):
        running.append((lap, tire.compound, driver.current_tire_laps))
        return 10.0

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", record_running)
    records = [
        {"id": "old", "compound": "soft", "age": 2},
        {"id": "restart", "compound": "hard", "age": 4},
    ]
    result, = engine.run(
        [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
        starting_tires={"A": TireCompound.SOFT},
        starting_tire_ages={"A": 2},
        tire_inventory={"A": records},
    )

    assert engine.suspensions == [(10, 110, ("A",))]
    assert engine.crossings == [("A", 1, 10), ("A", 2, 150), ("A", 3, 160), ("A", 4, 170)]
    assert running == [
        (1, TireCompound.SOFT, 2),
        (2, TireCompound.HARD, 4),
        (3, TireCompound.HARD, 5),
        (4, TireCompound.HARD, 6),
    ]
    assert [(stint["set_id"], stint["age_at_fit"], stint["age_at_end"], stint["laps_used"])
            for stint in result.tire_set_history] == [
        ("old", 2, 3, 1), ("restart", 4, 7, 3),
    ]
    assert {item["id"]: item["age"] for item in result.tire_inventory} == {
        "old": 3, "restart": 7,
    }
