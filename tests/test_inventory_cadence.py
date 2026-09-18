"""Forced fits and shared restarts retain the observed leader weather clock."""

from copy import deepcopy

import pytest
from test_chronological_race import fixture

from f1sim.models import Weather


def test_forced_puncture_keeps_observed_cadence_and_rejoin_snapshot(monkeypatch):
    engine, args, *_ = fixture(monkeypatch, {"A": 90, "B": 180}, laps=8)
    args = (*args[:3], Weather(track_wetness=.30), args[4])
    start = engine._start_lap
    prepare = engine.simulator._prepare_inventory_pit
    plans = []

    def puncture(state, now, **kwargs):
        if state.driver.id == "B" and state.laps_completed == 1:
            engine.simulator._damage_inventory_tire(state)
            state.force_pit_next_lap = True
        return start(state, now, **kwargs)

    def inspect(state, track, weather, lap, **kwargs):
        if (state.driver.id == "B" and lap == 2 and kwargs.get("weather_intervals")
                and state.inventory_pit_proposal is None):
            cadence = kwargs["weather_intervals"]
            assert cadence == (0, 2, 4)
            expected = engine.simulator._plan_inventory(
                state, track, weather, lap, force_stop=True, **kwargs)
            ordinary_options = dict(kwargs, weather_intervals=None, weather_clock=None)
            ordinary = engine.simulator._plan_inventory(
                state, track, weather, lap, force_stop=True, **ordinary_options)
            assert expected.set_id == "set-2"
            assert ordinary.set_id == "set-3"
            assert kwargs["current_traffic_gaps"] is not None
            plans.append(expected.set_id)
        return prepare(state, track, weather, lap, **kwargs)

    monkeypatch.setattr(engine, "_start_lap", puncture)
    monkeypatch.setattr(engine.simulator, "_prepare_inventory_pit", inspect)
    results = engine.run(*args, starting_tires={"A": "intermediate", "B": "soft"},
                         tire_inventory={"B": [{"compound": c} for c in
                                               ["soft", "hard", "intermediate"]]})
    result = next(item for item in results if item.driver_id == "B")
    assert plans == ["set-2"]
    assert result.pit_stop_details[0]["to_set_id"] == "set-2"


@pytest.mark.parametrize("reverse,old_exit", [(False, 0), (True, 1000000)])
def test_closed_exit_free_fits_use_shared_restart_forecast(monkeypatch, reverse, old_exit):
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
        state.inventory_pit_proposal = (lap, "H")
        return state.driver.id == "B" and lap == 2

    monkeypatch.setattr(engine.simulator, "_should_pit", policy)
    resume = engine._resume_if_collected
    refit = engine.simulator._refit_inventory_free
    observed = {}

    def collected(now):
        if set(engine.states) <= engine.red_waiting:
            assert "B" in engine.pending
            engine.pending["B"].expected_exit = old_exit
            if reverse:
                engine.resumption_order.reverse()
        return resume(now)

    def inspect(state, track, weather, lap, **kwargs):
        cadence = kwargs["weather_intervals"]
        before = deepcopy(state.tire_inventory.__dict__)
        expected = engine.simulator._plan_inventory(state, track, weather, lap + 1,
                                                    free_fit=True, **kwargs)
        assert state.tire_inventory.__dict__ == before
        result = refit(state, track, weather, lap, **kwargs)
        observed[state.driver.id] = (cadence, state.tire_inventory.current_set_id)
        assert state.tire_inventory.current_set_id == expected.set_id
        return result

    monkeypatch.setattr(engine, "_resume_if_collected", collected)
    monkeypatch.setattr(engine.simulator, "_refit_inventory_free", inspect)
    records = [{"id": "S", "compound": "soft", "age": 2},
               {"id": "H", "compound": "hard", "age": 5},
               {"id": "I", "compound": "intermediate", "age": 3}]
    results = engine.run(*args, starting_tires={key: "soft" for key in "AB"},
                         starting_tire_ages={key: 2 for key in "AB"},
                         tire_inventory={key: records for key in "AB"})
    assert observed["A"][0] == (0, 1, 2, 3, 4, 5)
    assert observed["B"][0] == (0, 1, 3, 5)
    assert observed["B"][1] == "I"
    result = next(item for item in results if item.driver_id == "B")
    assert result.pit_stops == 1
    assert engine.pit_exits == [("B", 2, 951)]
