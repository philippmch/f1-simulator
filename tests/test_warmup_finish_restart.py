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


def test_automatic_restart_reuses_worn_set_after_unrun_paid_fit_and_shortened_finish(monkeypatch):
    monkeypatch.setattr("f1sim.simulation.race_timing.RACING_TIME_LIMIT_SECONDS", 650)
    simulator = RaceSimulator(
        np.random.default_rng(19), tire_warmup={"soft": 2, "medium": 14, "hard": 35},
    )
    engine = ChronologicalRace(simulator, red_flag_pause_seconds=100)
    control = simulator.event_manager
    control.set_forced_red_flag(3)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a, **k: None)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *a, **k: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **k: None)
    plans, restart_events, warmups, running = {}, [], [], []
    forecast_inputs, plan_inputs = {}, {}
    actual_planning = engine._planning_track

    def record_forecast(state, now, *, restart=False):
        planning = actual_planning(state, now, restart=restart)
        if restart:
            restart_events.append(("horizon", state.driver.id))
            forecast_inputs[state.driver.id] = [planning]
        return planning

    monkeypatch.setattr(engine, "_planning_track", record_forecast)
    actual_cadence = engine._weather_intervals

    def record_cadence(state, now, planning, *, restart=False):
        cadence = actual_cadence(state, now, planning, restart=restart)
        if restart:
            restart_events.append(("cadence", state.driver.id))
            forecast_inputs[state.driver.id].append(cadence)
        return cadence

    monkeypatch.setattr(engine, "_weather_intervals", record_cadence)
    actual_clock = engine._strategy_weather_clock

    def record_clock(state, now, planning, queue_delay, *, restart=False):
        clock = actual_clock(state, now, planning, queue_delay, restart=restart)
        # Capture the initial free-refit input, including a leader's None clock.
        # Starting its new running lap may subsequently request a paid-stop clock.
        if restart and len(forecast_inputs[state.driver.id]) == 2:
            restart_events.append(("clock", state.driver.id))
            forecast_inputs[state.driver.id].append(clock)
        return clock

    monkeypatch.setattr(engine, "_strategy_weather_clock", record_clock)
    actual_plan = simulator._plan_inventory

    def record_plan(state, track, weather, lap, **kwargs):
        decision = actual_plan(state, track, weather, lap, **kwargs)
        if kwargs.get("free_fit"):
            plan_inputs[state.driver.id] = [
                track, kwargs["weather_intervals"], kwargs["weather_clock"],
            ]
            plans[state.driver.id] = {
                "planning_laps": track.total_laps,
                "remaining_laps": track.total_laps - lap + 1,
                "physical_laps": kwargs["physical_total_laps"],
                "current_set": state.tire_inventory.current_set_id,
                "age": state.tire_laps,
                "fit_pending": state.fit_lap_pending,
                "paid_stops": state.pit_stops,
                "pool_ages": {
                    item["id"]: item["age"]
                    for item in state.tire_inventory.snapshot(state.tire_laps)
                },
                "selected_set": decision.set_id,
            }
        return decision

    monkeypatch.setattr(simulator, "_plan_inventory", record_plan)
    actual_fit = simulator._fit_inventory_tire

    def record_fit(state, set_id, lap, kind):
        actual_fit(state, set_id, lap, kind)
        if kind == "red_flag":
            restart_events.append(("fit", state.driver.id))

    monkeypatch.setattr(simulator, "_fit_inventory_tire", record_fit)
    actual_consume = simulator._consume_tire_warmup

    def record_warmup(state):
        fee = actual_consume(state)
        if fee:
            warmups.append((state.driver.id, state.current_tire.compound, fee))
        return fee

    monkeypatch.setattr(simulator, "_consume_tire_warmup", record_warmup)
    actual_service = simulator.lap_simulator.calculate_pit_stop_time
    service_count = 0

    def closed_exit_service(car):
        nonlocal service_count
        if car.team_id == "B":
            service_count += 1
            if service_count == 2:
                # Hold the paid H fit behind the closed exit before it runs.
                return 100
        return actual_service(car)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", closed_exit_service)
    actual_lap = simulator.lap_simulator.calculate_lap_time

    def record_running(driver, car, track, tire, weather, lap, total_laps, *args, **kwargs):
        pace = actual_lap(driver, car, track, tire, weather, lap, total_laps, *args, **kwargs)
        running.append((driver.id, lap, tire.compound, driver.current_tire_laps, total_laps))
        return pace

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", record_running)
    drivers = [Driver(id=key, name=key, team_id=key, skill_rating=skill)
               for key, skill in [("A", 0), ("B", 1)]]
    cars = {key: Car(team_id=key, team_name=key, base_pace=pace)
            for key, pace in [("A", 0), ("B", 1)]}
    track = Track(id="T", name="T", country="T", total_laps=16, base_lap_time=90,
                  pit_lane_delta=20, safety_car_probability=0, overtake_difficulty=1)
    records = {key: [{"id": "S", "compound": "soft", "age": 0},
                     {"id": "M", "compound": "medium", "age": 0},
                     {"id": "H", "compound": "hard", "age": 0}]
               for key in "AB"}
    results = engine.run(
        drivers, cars, track, Weather(change_probability=0), list("AB"),
        starting_tires={key: TireCompound.SOFT for key in "AB"}, tire_inventory=records,
        pit_plans={"B": [{"lap": 2, "compound": "medium"}, {"lap": 3, "compound": "hard"}]},
    )

    assert plans["B"] == {
        "planning_laps": 8, "remaining_laps": 6, "physical_laps": 16,
        "current_set": "H", "age": 0, "fit_pending": True, "paid_stops": 2,
        "pool_ages": {"S": 1, "M": 1, "H": 0}, "selected_set": "S",
    }
    assert (plans["A"]["planning_laps"], plans["A"]["remaining_laps"],
            plans["A"]["physical_laps"]) == (8, 5, 16)
    # Horizons, weather cadence and clocks freeze before any free inventory fitting.
    assert set(restart_events[:6]) == {
        (kind, driver) for kind in ["horizon", "cadence", "clock"] for driver in "AB"
    }
    assert {driver for kind, driver in restart_events[6:] if kind == "fit"} == {"A", "B"}
    assert len(restart_events) == 8
    for driver in "AB":
        assert plan_inputs[driver] == forecast_inputs[driver]
        assert plan_inputs[driver][0] is forecast_inputs[driver][0]
    red, restart, survivors = engine.suspensions[0]
    assert set(survivors) == {"A", "B"}
    assert red < restart
    assert ("B", 3, restart) in engine.pit_exits
    assert running and {row[4] for row in running} == {16}
    assert [(compound, fee) for driver, compound, fee in warmups if driver == "B"] == [
        (TireCompound.MEDIUM, 14), (TireCompound.SOFT, 2),
    ]
    result = next(result for result in results if result.driver_id == "B")
    assert result.race_time_limited
    assert result.pit_stops == 2 and result.pit_laps == [2, 3]
    assert [(stint["set_id"], stint["age_at_fit"], stint["age_at_end"], stint["laps_used"])
            for stint in result.tire_set_history] == [
        ("S", 0, 1, 1), ("M", 0, 1, 1), ("H", 0, 0, 0), ("S", 1, 6, 5),
    ]
    assert {item["id"]: item["age"] for item in result.tire_inventory} == {
        "S": 6, "M": 1, "H": 0,
    }
