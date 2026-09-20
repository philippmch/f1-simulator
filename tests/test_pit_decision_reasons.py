"""Paid-stop decision metadata is truthful and consumed at execution."""

from math import isfinite

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.tire_inventory import TireInventory


def _race(engine, *, weather, records=None, laps=20, seed=4):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=20)
    simulator = RaceSimulator(np.random.default_rng(seed))
    simulator.event_manager.process_lap = lambda *args, **kwargs: []
    kwargs = {"starting_tires": {"A": TireCompound.MEDIUM}}
    if records is not None:
        kwargs["tire_inventory"] = {"A": records}
    execute = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator).run)
    return execute([driver], {"A": car}, track, weather, ["A"], **kwargs)[0]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_dry_planner_records_forecast_saving_for_both_engines(engine):
    result = _race(engine, weather=Weather(change_probability=0))

    stops = [stop for stop in result.pit_stop_details
             if stop["decision_reason"] == "dry_forecast"]
    assert stops
    assert all(isinstance(stop["forecast_saving_seconds"], float)
               and isfinite(stop["forecast_saving_seconds"])
               for stop in stops)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_final_lap_compound_requirement_has_null_forecast_saving(engine):
    result = _race(engine, weather=Weather(change_probability=0), laps=2, seed=5)

    assert result.pit_stop_details[0]["lap"] == 2
    assert result.pit_stop_details[0]["decision_reason"] == "compound_requirement"
    assert result.pit_stop_details[0]["forecast_saving_seconds"] is None


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_finite_inventory_forecast_records_a_paid_saving(engine):
    result = _race(
        engine,
        weather=Weather(change_probability=0),
        records=[{"id": "M", "compound": "medium"},
                 {"id": "S", "compound": "soft"},
                 {"id": "H", "compound": "hard"}],
        seed=4,
    )

    stop = next(stop for stop in result.pit_stop_details
                if stop["decision_reason"] == "inventory_forecast")
    assert stop["from_set_id"] == "M"
    assert stop["to_set_id"] == "S"
    assert stop["forecast_saving_seconds"] == pytest.approx(.0578571428571)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
def test_critical_weather_stop_reason_is_recorded_with_finite_pool_support(engine, finite):
    records = ([{"id": "S", "compound": "soft"},
                {"id": "I", "compound": "intermediate"},
                {"id": "M", "compound": "medium"}] if finite else None)
    result = _race(
        engine,
        weather=Weather(condition=WeatherCondition.HEAVY_RAIN,
                        rain_intensity=.8, track_wetness=.9, change_probability=0),
        records=records,
        laps=8,
        seed=9,
    )

    assert result.pit_stop_details
    stop = result.pit_stop_details[0]
    assert stop["decision_reason"] == "critical_weather"
    assert stop["forecast_saving_seconds"] is None
    if finite:
        assert stop["from_set_id"] in {"S", "M"}
        assert stop["to_set_id"] == "I"


def test_forced_repair_overrides_any_elective_context_and_direct_execution_is_consumed():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=4,
                  base_lap_time=90, pit_lane_delta=20)
    weather = Weather(change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(2))
    state = DriverRaceState(driver, car, 1, tire_laps=4)
    state.pit_decision_context = {
        "lap": 2,
        "decision_reason": "dry_forecast",
        "forecast_saving_seconds": 4.0,
    }
    state.force_pit_next_lap = True

    simulator._execute_pit_stop(state, track, weather, current_lap=2,
                                 sample_service=False)

    stop = state.pit_stop_details[0]
    assert stop["decision_reason"] == "forced_repair"
    assert stop["forecast_saving_seconds"] is None
    assert state.pit_decision_context is None


def test_unavailable_finite_set_uses_forced_repair_reason_without_forecast_saving():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=20,
                  base_lap_time=90, pit_lane_delta=20)
    simulator = RaceSimulator(np.random.default_rng(1))
    inventory = TireInventory.from_sets([
        {"id": "M", "compound": "medium"},
        {"id": "S", "compound": "soft"},
    ])
    state = DriverRaceState(driver, car, 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy())
    simulator._initialize_inventory(state, inventory, inventory.sets["M"])
    state.tire_laps = 3
    inventory.mark_current_unavailable(state.tire_laps)

    assert simulator._should_pit(
        state, [state], track, 5, False, Weather(change_probability=0),
    )
    assert state.pit_decision_context == {
        "lap": 5,
        "decision_reason": "forced_repair",
        "forecast_saving_seconds": None,
    }
    simulator._execute_pit_stop(
        state, track, Weather(change_probability=0), 5, sample_service=False,
    )
    assert state.pit_stop_details[0]["decision_reason"] == "forced_repair"
    assert state.pit_stop_details[0]["forecast_saving_seconds"] is None


def test_inventory_forecast_saving_uses_wait_minus_pit_cost(monkeypatch):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=20,
                  base_lap_time=90, pit_lane_delta=20)
    weather = Weather(change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(1))
    inventory = TireInventory.from_sets([
        {"id": "M", "compound": "medium"},
        {"id": "S", "compound": "soft"},
    ])
    state = DriverRaceState(driver, car, 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy())
    simulator._initialize_inventory(state, inventory, inventory.sets["M"])

    class Decision:
        pit_now_cost = 7.25
        wait_cost = 12.5
        set_id = "S"

        def should_pit(self, bias=0):
            return True

    monkeypatch.setattr(simulator, "_plan_inventory", lambda *args, **kwargs: Decision())
    assert simulator._should_pit(state, [state], track, 5, False, weather)
    assert state.pit_decision_context["forecast_saving_seconds"] == pytest.approx(5.25)

    simulator._execute_pit_stop(state, track, weather, 5, sample_service=False)
    assert state.pit_stop_details[0]["forecast_saving_seconds"] == pytest.approx(5.25)


def test_stale_context_and_free_refit_do_not_label_a_later_paid_stop():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="Test", country="Test", total_laps=4,
                  base_lap_time=90, pit_lane_delta=20)
    weather = Weather(change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(2))
    state = DriverRaceState(driver, car, 1, tire_laps=4)
    state.pit_decision_context = {
        "lap": 1,
        "decision_reason": "dry_forecast",
        "forecast_saving_seconds": 4.0,
    }

    simulator._execute_pit_stop(state, track, weather, current_lap=2,
                                 sample_service=False)
    stop = state.pit_stop_details[0]
    assert stop["decision_reason"] is None
    assert stop["forecast_saving_seconds"] is None

    state.pit_decision_context = {
        "lap": 3,
        "decision_reason": "dry_forecast",
        "forecast_saving_seconds": 4.0,
    }
    simulator._fit_red_flag_tires([state], weather, track, current_lap=2)
    assert state.pit_decision_context is None


def test_chronological_finish_veto_clears_decision_context():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    state = DriverRaceState(driver, car, 1)
    state.pit_decision_context = {
        "lap": 2,
        "decision_reason": "dry_forecast",
        "forecast_saving_seconds": 4.0,
    }

    ChronologicalRace._clear_one_lap_pit_proposals(state)

    assert state.pit_decision_context is None
