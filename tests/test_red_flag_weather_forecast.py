"""Free unlimited sets use a pure forecast on the actual restart weather clock."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock


def inputs():
    sim = RaceSimulator(np.random.default_rng(5))
    track = Track(id="T", name="T", country="T", total_laps=20,
                  base_lap_time=90, pit_lane_delta=20)
    state = DriverRaceState(
        Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"), 1,
        current_tire=TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], tire_laps=3,
        pit_stops=4, laps_completed=17,
    )
    clock = StrategyWeatherClock((0, 100, 200), 10, 25, 12, 23, 23)
    return sim, state, track, clock


def test_external_drying_clock_changes_free_choice_without_mutation():
    sim, state, track, clock = inputs()
    weather = Weather(track_wetness=.3, rain_intensity=0)
    before = deepcopy((state, weather, track, sim.rng.bit_generator.state))

    ordinary = sim._choose_red_flag_tire(state, weather, track, 17)
    accelerated = sim._choose_red_flag_tire(
        state, weather, track, 17, weather_intervals=(0, 4, 8), weather_clock=clock,
    )

    # More drainage events between this car's laps make an immediate free slick
    # preferable. All four prior stops remain spent; the free fit buys no budget.
    assert ordinary == TireCompound.INTERMEDIATE
    assert accelerated == TireCompound.SOFT
    assert (state, weather, track, sim.rng.bit_generator.state) == before


def test_chronological_free_fit_forwards_cadence_and_paid_stop_clock(monkeypatch):
    sim, state, track, clock = inputs()
    engine = ChronologicalRace(sim)
    engine.track = track
    engine.weather = Weather(track_wetness=.3, rain_intensity=0)
    engine.free_refits = {state.driver.id}
    original = sim._choose_red_flag_tire
    calls = []

    def choose(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(sim, "_choose_red_flag_tire", choose)
    engine._fit_red_flag_set(state, track, (0, 4, 8), clock)

    assert calls == [dict(physical_total_laps=20, weather_intervals=(0, 4, 8),
                          weather_clock=clock)]
    assert state.current_tire.compound == TireCompound.SOFT
    assert state.pit_stops == 4
    assert state.tire_laps == 0
    assert not engine.free_refits


def test_critically_dry_intermediates_are_not_available_for_free():
    sim, state, track, _ = inputs()
    assert sim._choose_red_flag_tire(state, Weather(), track, 17) in {
        TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD,
    }


@pytest.mark.parametrize("reverse,old_exit", [(False, 0), (True, 1000000)])
def test_native_unlimited_restart_freezes_clocks_for_pending_and_running_cars(
    monkeypatch, reverse, old_exit,
):
    from test_chronological_race import fixture

    from f1sim.simulation.events import EventManager

    engine, args, *_ = fixture(monkeypatch, {"A": 90, "B": 150}, laps=8)
    control = engine.simulator.event_manager
    monkeypatch.setattr(control, "process_lap", EventManager.process_lap.__get__(control))
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a: None)
    control.set_forced_red_flag(2)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(
        update={"track_wetness": .28} if control.red_flag_active else {}))
    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_pit_stop_time", lambda car: 200)

    def policy(state, states, track, lap, *args, **kwargs):
        state.dry_pit_proposal = (lap, TireCompound.HARD)
        return state.driver.id == "B" and lap == 2

    monkeypatch.setattr(engine.simulator, "_should_pit", policy)
    resume = engine._resume_if_collected
    choose = engine.simulator._choose_red_flag_tire
    observed = {}

    def collected(now):
        if set(engine.states) <= engine.red_waiting:
            assert "B" in engine.pending and "A" not in engine.pending
            engine.pending["B"].expected_exit = old_exit
            if reverse:
                engine.resumption_order.reverse()
        return resume(now)

    def inspect(state, weather, planning, lap, **kwargs):
        observed[state.driver.id] = kwargs
        return choose(state, weather, planning, lap, **kwargs)

    monkeypatch.setattr(engine, "_resume_if_collected", collected)
    monkeypatch.setattr(engine.simulator, "_choose_red_flag_tire", inspect)
    results = engine.run(*args, starting_tires={key: "soft" for key in "AB"},
                         starting_tire_ages={key: 2 for key in "AB"})
    assert observed["A"]["weather_intervals"] == (0, 1, 2, 3, 4, 5)
    assert observed["B"]["weather_intervals"] == (0, 1, 3, 5)
    assert observed["A"].get("weather_clock") is None  # No external leader for A.
    assert observed["B"]["weather_clock"] is not None
    for options in observed.values():
        assert options["physical_total_laps"] == 8
    result = next(item for item in results if item.driver_id == "B")
    assert result.pit_stops == 1
    assert engine.pit_exits == [("B", 2, 951)]
