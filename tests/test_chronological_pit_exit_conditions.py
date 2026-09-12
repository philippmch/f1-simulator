"""Paid service does not commit the weather or control of unrun track work."""

import pytest
from test_chronological_race import fixture, run

from f1sim.models import Weather


@pytest.mark.parametrize("flag", ["safety_car_active", "vsc_active"])
@pytest.mark.parametrize("deploy", [True, False])
def test_running_conditions_refresh_after_service(monkeypatch, flag, deploy):
    engine, args, calls, _, _, _ = fixture(monkeypatch, {"A": 90, "B": 100}, laps=4)
    args = (*args[:3], Weather(track_wetness=.19), args[4])
    control = engine.simulator.event_manager

    def events(lap, *args, **kwargs):
        if lap == 1:
            setattr(control, flag, not deploy)
        elif lap == 2:
            setattr(control, flag, deploy)
        return []

    monkeypatch.setattr(control, "process_lap", events)
    monkeypatch.setattr(engine.simulator, "_should_pit", lambda state, states, track, lap,
                        *args, **kwargs: state.driver.id == "B" and lap == 2)
    services = []
    monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda car: services.append(1) or 200)
    observed = []
    begin = engine._begin_running

    def capture(state, pending, now):
        if state.driver.id != "B" or pending.lap != 2:
            return begin(state, pending, now)
        fitted = pending.tire.model_dump()
        old_weather = pending.weather.track_wetness
        modifier = control.get_lap_time_modifier()
        count = len(calls)
        begin(state, pending, now)
        assert len(calls) == count + 1
        observed.append(now)
        assert pending.weather.track_wetness == engine.weather.track_wetness
        assert pending.weather.track_wetness != old_weather
        assert pending.neutralized is deploy
        assert pending.active_aero_enabled is (not deploy)
        assert pending.safety_car is (deploy and flag == "safety_car_active")
        assert pending.lap_time_modifier == modifier
        assert pending.restart_boost == control.is_restart_lap(engine.control_intervals + 1)
        assert pending.mode_allowed == control.is_overtake_mode_allowed(
            engine.control_intervals + 1, engine.weather)
        assert pending.mode_active is False
        assert pending.tire.model_dump() == fitted
        if not pending.safety_car:
            assert pending.ready == now + 100 * modifier
            assert pending.sc_queue_pace is None
        else:
            assert pending.sc_queue_pace is not None

    monkeypatch.setattr(engine, "_begin_running", capture)
    results = run(engine, args)
    assert len(observed) == len(services) == 1
    assert next(result for result in results if result.driver_id == "B").pit_stops == 1


@pytest.mark.parametrize("flag", ["safety_car_active", "vsc_active"])
@pytest.mark.parametrize("deploy", [True, False])
@pytest.mark.parametrize("on_track", [True, False])
def test_forecasts_refresh_only_unrun_service_laps(flag, deploy, on_track):
    from copy import deepcopy

    from test_chronological_projected_finish import fixture as forecast_fixture

    from f1sim.models import Track

    engine = forecast_fixture(scheduled=4)
    engine.track = Track(id="t", name="T", country="T", total_laps=4, base_lap_time=90)
    engine.running_paces = {"A": 90, "B": 90}
    for state in engine.states.values():
        state.laps_completed = 1
    pending = engine.pending["A"]
    pending.lap = 2
    pending.on_track = on_track
    pending.running_start = 10
    pending.ready = 200 if on_track else 999  # Sampled service is private to execution.
    pending.expected_exit = 20 if deploy else 0
    control = engine.simulator.event_manager
    setattr(control, flag, not deploy)
    pending.lap_time_modifier = control.get_lap_time_modifier()
    setattr(control, flag, deploy)
    modifier = control.get_lap_time_modifier()
    before = deepcopy(pending), deepcopy(engine.simulator.rng.bit_generator.state)
    first = 200 if on_track else (20 if deploy else 10) + 90 * modifier
    assert engine._projected_flag_time(10) == first + 180
    intervals = engine._weather_intervals(engine.states["B"], 10, engine.track)
    expected_intervals = (0, 0, 1) if deploy else ((0, 0, 0) if on_track else (0, 1, 2))
    assert intervals == expected_intervals
    start = 10 if on_track else max(10, pending.expected_exit)
    when = start + 20
    expected = 20 / (190 if on_track else 90 * modifier)
    assert engine._projected_progress("A", when, 2, now=10) == pytest.approx(expected)
    assert (pending, engine.simulator.rng.bit_generator.state) == before
