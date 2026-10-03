"""Recorded savings compare equal distances and the replacement actually fitted."""

from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.inventory_strategy import InventoryDecision
from f1sim.simulation.pit_strategy import DryPitDecision
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.rain_strategy import RainStopDecision, RainTransitionDecision
from f1sim.simulation.tire_inventory import TireInventory


def decision(kind, pit_cost, wait_cost, pit_laps, wait_laps):
    if kind == "inventory":
        return InventoryDecision(pit_cost, wait_cost, "proposed", TireCompound.SOFT,
                                 pit_laps, wait_laps)
    if kind == "dry":
        return DryPitDecision(pit_cost, wait_cost, TireCompound.SOFT, pit_laps, wait_laps)
    return RainTransitionDecision(pit_cost, wait_cost, TireCompound.SOFT,
                                  pit_now_laps=pit_laps, wait_laps=wait_laps)


@pytest.mark.parametrize("kind", ["dry", "rain", "inventory"])
@pytest.mark.parametrize("pit_laps,wait_laps,expected", [
    (4, 3, None), (2, 3, None), (4, 4, -150.), (None, None, -150.),
])
def test_time_saving_is_unavailable_when_known_completed_distances_differ(
    kind, pit_laps, wait_laps, expected,
):
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1)
    forecast = decision(kind, 450., 300., pit_laps, wait_laps)
    simulator = RaceSimulator(np.random.default_rng(1))
    simulator._capture_pit_decision_context(state, 2, f"{kind}_forecast", forecast)
    assert state.pit_decision_context["forecast_saving_seconds"] == expected
    if pit_laps == 4 and wait_laps == 3:
        # Distance still wins even though running the extra lap takes longer.
        assert forecast.should_pit()


@pytest.mark.parametrize("kind", ["dry", "rain", "inventory"])
def test_partial_continuations_never_claim_a_finishing_time_saving(kind):
    forecast = decision(kind, inf, inf, 4, 3)
    assert RaceSimulator._decision_forecast_saving(forecast) is None


@pytest.mark.parametrize("replacement", [None, "soft", "hard"])
def test_reprepared_physical_replacement_cannot_reuse_the_other_sets_saving(
    monkeypatch, replacement,
):
    simulator = RaceSimulator(np.random.default_rng(1))
    driver, car = Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A")
    state = DriverRaceState(driver, car, 1, laps_completed=1)
    track = Track(id="T", name="T", country="Test", total_laps=20,
                  base_lap_time=90., pit_lane_delta=20.)
    weather = Weather(change_probability=0.)
    stock = TireInventory.from_sets([
        dict(id="M", compound="medium"), dict(id="proposed", compound="soft"),
        dict(id="alternate", compound=replacement or "soft", age=5)])
    simulator._initialize_inventory(state, stock, stock.sets["M"])
    state.tire_laps = state.driver.current_tire_laps = 1
    offered = decision("inventory", 7.25, 12.5, 19, 19)
    replanned = InventoryDecision(20., inf, "alternate", stock.sets["alternate"].compound,
                                  19, -1)
    monkeypatch.setattr(simulator, "_plan_inventory", lambda *args, **kwargs:
                        replanned if kwargs.get("force_stop") else offered)
    assert simulator._should_pit(state, [state], track, 2, False, weather)
    assert state.pit_decision_context["forecast_saving_seconds"] == 5.25
    if replacement is not None:
        stock.unavailable_ids.add("proposed")
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False)
    stop, = state.pit_stop_details
    assert stop["to_set_id"] == ("proposed" if replacement is None else "alternate")
    assert stop["forecast_saving_seconds"] == (5.25 if replacement is None else None)
    assert stop["decision_reason"] == "inventory_forecast"
    assert state.pit_decision_context is None


@pytest.mark.parametrize("kind", ["dry", "same_rain"])
def test_paid_weather_correction_cannot_reuse_the_original_replacements_saving(kind):
    simulator = RaceSimulator(np.random.default_rng(1))
    compound = TireCompound.MEDIUM if kind == "dry" else TireCompound.INTERMEDIATE
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1, laps_completed=1,
                            current_tire=TIRE_COMPOUNDS[compound], tire_laps=1)
    track = Track(id="T", name="T", country="Test", total_laps=8,
                  base_lap_time=90., pit_lane_delta=20.)
    offered = (decision("dry", 7.25, 12.5, 7, 7) if kind == "dry" else
               RainStopDecision(7.25, 12.5, pit_now_laps=7, wait_laps=7))
    simulator._capture_pit_decision_context(
        state, 2, "dry_forecast" if kind == "dry" else "rain_forecast", offered)
    assert state.pit_decision_context["forecast_saving_seconds"] == 5.25
    if kind == "dry":
        state.dry_pit_proposal = (2, TireCompound.SOFT)
    weather = Weather(track_wetness=.5 if kind == "dry" else .85,
                      rain_intensity=.5 if kind == "dry" else .85, change_probability=0.)
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False)
    stop, = state.pit_stop_details
    assert stop["to_compound"] in ({"intermediate", "wet"} if kind == "dry" else {"wet"})
    assert stop["forecast_saving_seconds"] is None
    assert state.pit_decision_context is None
