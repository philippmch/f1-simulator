"""Known paid rival fits use native mean physics at conditional track entry."""

import cProfile
import pickle
from copy import deepcopy
from dataclasses import replace
from itertools import count

import numpy as np
import pytest
from test_chronological_field_finish import (
    FieldPhysics,
    field,
    ledger_signature,
    native_path,
    snapshot_engine,
)

from f1sim.models import Car, Driver, TireCompound, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import (
    CommittedRivalFit,
    ObservedChronologicalField,
    evaluate_chronological_finish_protection,
)
from f1sim.simulation.finish_strategy import ReplacementOption
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.weather_schedule import WeatherForecastContext


def fitted_field(*, control="green", age=0, remaining=20., surface="dry"):
    intervals = 0 if control == "green" else 2
    inputs = field(control=control, off_track=True, remaining=remaining, fee=7.,
                   intervals=intervals)
    driver, car, track, context, now = inputs
    wetness = {"dry": 0., "drying": .24, "rain": .5, "schedule": .5}[surface]
    weather = Weather(track_wetness=wetness,
                      rain_intensity=wetness if surface in ("rain", "schedule") else 0.,
                      change_probability=0.)
    forecast = (WeatherForecastContext.from_schedule(
        [dict(lap=72, rain_intensity=0.)], leading_lap=71) if surface == "schedule" else None)
    compound = TireCompound.HARD if wetness <= .3 else TireCompound.INTERMEDIATE
    fit = CommittedRivalFit(
        Driver(id="B", name="B", team_id="B", tire_management=.87),
        Car(team_id="B", team_name="B", tire_degradation_factor=1.1),
        track.model_copy(deep=True), TIRE_COMPOUNDS[compound].model_copy(deep=True), age,
        weather.model_copy(deep=True), forecast,
    )
    rival = replace(context.rivals[0], committed_fit=fit)
    context = replace(context, rivals=(rival,), forecast_context=forecast)
    return (driver, car, track, context, now), weather, forecast


def run_forecast(context, now, *, stopped, delay, native):
    projection = ObservedChronologicalField(context, now)
    projection._native_crossings = native
    origin = context.timeline.states[context.identifier].completed_laps
    first = True
    while not projection.finished:
        projection.enter(delay if first and stopped else None)
        projection.cross(95. if stopped else 99., 5. if first and stopped else 0.)
        first = False
    return projection.timeline.states[context.identifier].completed_laps - origin, projection.now


@pytest.mark.parametrize("control", ["green", "vsc", "sc"])
@pytest.mark.parametrize("surface", ["dry", "drying", "rain", "schedule"])
@pytest.mark.parametrize("age", [0, 8])
@pytest.mark.parametrize("remaining", [20., 140.])
@pytest.mark.parametrize("stopped", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_fitted_field_crossings_match_native_execution(
    monkeypatch, control, surface, age, remaining, stopped, native,
):
    inputs, weather, forecast = fitted_field(control=control, surface=surface,
                                            age=age, remaining=remaining)
    context, now = inputs[3:]
    before = deepcopy(context.rivals), ledger_signature(context.timeline)
    direct = LapSimulator(np.random.default_rng(27))
    original = FieldPhysics.calculate_lap_time
    entries = []

    def first_fitted_lap(self, driver, car, track, tire, entry, lap, total, **options):
        if driver.id != "B" or lap != context.rivals[0].completed_laps + 1:
            return original(self, driver, car, track, tire, entry, lap, total, **options)
        value = direct.calculate_lap_time(driver, car, track, tire, entry, lap, total,
                                         sample_variation=False, **options)
        entries.append((entry.track_wetness, driver.current_tire_laps, value))
        return value

    monkeypatch.setattr(FieldPhysics, "calculate_lap_time", first_fitted_lap)
    delay = 160.
    expected = native_path(inputs, stopped=stopped, delay=delay, weather=weather,
                           warmup={"soft": 5.}, forecast_context=forecast)[:2]
    actual = run_forecast(context, now, stopped=stopped, delay=delay, native=native)
    assert actual[0] == expected[0]
    assert actual[1] == pytest.approx(expected[1], abs=1.e-8)
    assert len(entries) == 1
    assert entries[0][1] == age
    if remaining == 140. and not stopped and surface in ("drying", "schedule"):
        assert entries[0][0] < weather.track_wetness
    assert before == (context.rivals, ledger_signature(context.timeline))


@pytest.mark.parametrize("control", ["green", "vsc", "sc"])
@pytest.mark.parametrize("age", [0, 8])
@pytest.mark.parametrize("sampled_end", [7063., 7300.])
def test_engine_captures_known_fit_without_future_draws(control, age, sampled_end):
    engine, state, now = snapshot_engine(control=control, sampled_end=sampled_end)
    rival = engine.states["B"]
    pending = engine.pending["B"]
    rival.tire_laps = rival.driver.current_tire_laps = pending.tire_age = age
    pending.paid_stop = True
    before = pickle.dumps((engine.states, engine.pending, engine.queue, engine.weather,
                           engine.simulator.rng.bit_generator.state))
    context = engine._chronological_finish_context(state, now)
    fit = context.rivals[0].committed_fit
    assert fit is not None and fit.age == age
    assert fit.driver is not rival.driver and fit.tire is not pending.tire
    assert fit.track.total_laps == engine.track.total_laps
    observed = engine._observed_control_projection(now)
    assert observed is not None
    assert engine._projected_flag_time(now) == observed.flag_time
    assert engine._field_finish_required(context.control_intervals)
    assert before == pickle.dumps((engine.states, engine.pending, engine.queue, engine.weather,
                                  engine.simulator.rng.bit_generator.state))


def test_alternate_future_service_draws_share_the_same_fitted_clock():
    clocks = []
    for sampled_end in (7063., 7300.):
        engine, state, now = snapshot_engine(control="green", sampled_end=sampled_end)
        engine.pending["B"].paid_stop = True
        context = engine._chronological_finish_context(state, now)
        clocks.append((context.rivals[0], engine._observed_control_projection(now)))
    assert clocks[0] == clocks[1]


@pytest.mark.parametrize("replacement", ["instance_sampler", "class_sampler", "control",
                                         "driver", "tire", "age", "unpaid", "running"])
def test_unknown_fit_physics_keeps_the_existing_observation(monkeypatch, replacement):
    engine, state, now = snapshot_engine(control="green")
    pending = engine.pending["B"]
    pending.paid_stop = True
    if replacement == "instance_sampler":
        monkeypatch.setattr(engine.simulator.lap_simulator, "calculate_lap_time",
                            lambda *a, **k: pytest.fail("forecast called the live sampler"))
    elif replacement == "class_sampler":
        monkeypatch.setattr(LapSimulator, "calculate_lap_time",
                            lambda *a, **k: pytest.fail("forecast called a custom sampler"))
    elif replacement == "control":
        monkeypatch.setattr(engine.simulator.event_manager, "is_active_aero_allowed", lambda: True)
    elif replacement == "driver":
        class ExtendedDriver(Driver):
            pass
        engine.states["B"].driver = ExtendedDriver(**engine.states["B"].driver.model_dump())
    elif replacement == "tire":
        pending.tire = TIRE_COMPOUNDS[TireCompound.SOFT]
    elif replacement == "age":
        pending.tire_age += 1
    elif replacement == "unpaid":
        pending.paid_stop = False
    else:
        pending.on_track, pending.running_start, pending.running = True, now - 5., 100.
        engine.order.insert(0, "B")
        time, distance, serial, _, key, generation = engine.queue[0]
        engine.queue[0] = time, distance, serial, "cross", key, generation
    context = engine._chronological_finish_context(state, now)
    assert context is not None and context.rivals[0].committed_fit is None
    assert context.rivals[0].free_running == 100.
    assert not engine._has_committed_rival_fit()


def test_branches_own_fit_consumption_and_ignore_later_live_edits():
    inputs, _, _ = fitted_field(remaining=140.)
    context, now = inputs[3:]
    parent = ObservedChronologicalField(context, now)
    branch, sibling = parent.fork(), parent.fork()
    fit = context.rivals[0].committed_fit
    branch.enter(160.)
    pace = branch.free_paces["B"]
    assert not branch.rival_fits
    assert parent.rival_fits and sibling.rival_fits
    fit.driver.tire_management = .1
    fit.tire.initial_grip *= .9
    fit.weather.track_wetness = .3
    sibling.enter(160.)
    assert sibling.free_paces["B"] == pace
    assert context.timeline.chequered_time is None


@pytest.mark.parametrize("age", [0, 8])
def test_rejoin_traffic_uses_the_known_fitted_outlap(age):
    engine, state, now = snapshot_engine(control="green")
    rival, pending = engine.states["B"], engine.pending["B"]
    pending.paid_stop = True
    rival.tire_laps = rival.driver.current_tire_laps = pending.tire_age = age
    context = engine._chronological_finish_context(state, now)
    rival_exit = context.rivals[0].ready
    candidate_exit = now + engine.track.pit_lane_delta + expected_stationary_time(state.car)
    assert candidate_exit > rival_exit
    before = deepcopy((engine.states, engine.pending, engine.simulator.rng.bit_generator.state))
    result = engine._strategy_traffic(state, now, 0.)
    projection = LapSimulator(np.random.default_rng(28))
    mean = projection.calculate_lap_time(
        rival.driver, rival.car, engine.track, pending.tire, engine.weather,
        pending.lap, engine.track.total_laps, sample_variation=False)
    progress = (candidate_exit - rival_exit) / (mean + 7.)
    assert result.current_traffic_gaps[1] == pytest.approx(progress * 99.)
    assert (engine.states, engine.pending, engine.simulator.rng.bit_generator.state) == before


@pytest.mark.parametrize("control", ["green", "vsc", "sc"])
@pytest.mark.parametrize("scheduled", [False, True])
def test_fitted_leader_supplies_explicit_weather_events(control, scheduled):
    engine, state, now = snapshot_engine(control=control, relative_laps=1)
    engine.pending["B"].paid_stop = True
    engine.weather = Weather(track_wetness=.24, rain_intensity=.24 if scheduled else 0.,
                             change_probability=0.)
    if scheduled:
        engine.simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
            [dict(lap=73, rain_intensity=0.)], leading_lap=72)
    observed = engine._observed_control_projection(now)
    assert observed is not None and observed.leading_updates
    planning = engine._planning_track(state, now)
    assert planning.total_laps == state.laps_completed + len(observed.own_crossings)
    clock = engine._strategy_weather_clock(state, now, planning, 0.)
    assert clock is not None
    assert clock.update_offsets == tuple(time - now for time in observed.leading_updates)
    starts = (now, *observed.own_crossings[:-1])
    expected = tuple(sum(time <= start + 1.e-9 for time in observed.leading_updates)
                     for start in starts)
    assert engine._weather_intervals(state, now, planning) == expected
    for time in observed.leading_updates:
        assert engine._weather_updates_at(time, engine._weather_projection_clock(now), now=now) \
            == sum(update <= time + 1.e-9 for update in observed.leading_updates)


@pytest.mark.parametrize("control", ["green", "vsc", "sc"])
@pytest.mark.parametrize("age", [0, 8])
@pytest.mark.parametrize("remaining", [20., 140.])
def test_finish_guard_matches_native_conditional_distances(monkeypatch, control, age, remaining):
    inputs, weather, _ = fitted_field(control=control, age=age, remaining=remaining)
    driver, car, track, context, now = inputs
    direct = LapSimulator(np.random.default_rng(31))
    original = FieldPhysics.calculate_lap_time

    def conditional_lap(self, driver, car, track, tire, entry, lap, total, **options):
        if driver.id == "B" and lap == context.rivals[0].completed_laps + 1:
            return direct.calculate_lap_time(driver, car, track, tire, entry, lap, total,
                                             sample_variation=False, **options)
        return original(self, driver, car, track, tire, entry, lap, total, **options)

    monkeypatch.setattr(FieldPhysics, "calculate_lap_time", conditional_lap)
    retained = native_path(inputs, stopped=False, weather=weather)[:2]
    stopped = native_path(inputs, stopped=True, delay=160., weather=weather,
                          warmup={"soft": 5.})[:2]
    result = evaluate_chronological_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, weather, now, context,
        expected_lane_loss=160., replacements=(ReplacementOption(TireCompound.SOFT),),
        lap_simulator=FieldPhysics({"B": 100.}), tire_warmup={"soft": 5.})
    assert (result.retained_laps, result.stop_laps) == (retained[0], stopped[0])
    assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    assert result.veto == (retained[0] > stopped[0])


def ready_engine():
    engine, state, now = snapshot_engine(control="green", relative_laps=1)
    engine.pending["B"].paid_stop = True
    engine.free_refits = set()
    engine.box_releases, engine.expected_box_releases = {}, {}
    engine.serial = count(100)
    engine.weather = Weather(track_wetness=.24, change_probability=0.)
    return engine, state, now


def test_native_decision_views_share_one_field_snapshot(monkeypatch):
    engine, state, now = ready_engine()
    monkeypatch.setattr(engine.simulator, "_should_pit", lambda *a, **k: False)
    original = type(engine)._observed_control_projection.__code__
    profile = cProfile.Profile()
    profile.runcall(engine._start_lap, state, now)
    calls = sum(row.callcount for row in profile.getstats() if row.code is original)
    assert calls == 1
    assert engine.pending[state.driver.id].on_track
    assert state.pit_stops == 0


def test_custom_view_retains_its_original_call_signature(monkeypatch):
    engine, state, now = ready_engine()
    original = engine._weather_intervals
    calls = []

    def custom(state, now, planning, *, restart=False):
        calls.append((state.driver.id, now))
        return original(state, now, planning, restart=restart)

    monkeypatch.setattr(engine, "_weather_intervals", custom)
    monkeypatch.setattr(engine.simulator, "_should_pit", lambda *a, **k: False)
    assert engine._strategy_projection_options(state, now) == {}
    engine._start_lap(state, now)
    assert calls == [(state.driver.id, now)]


def test_shared_view_does_not_survive_the_policy_callback(monkeypatch):
    engine, state, now = ready_engine()
    earlier = engine._observed_control_projection(now)
    checked = []

    def policy(*args, **kwargs):
        engine.states["B"].car.base_pace = .6
        return True

    def guard(state, planning, now, delay, traffic, **kwargs):
        updated = engine._observed_control_projection(now)
        assert updated is not None and updated != earlier
        context = engine._chronological_finish_context(state, now)
        assert context.rivals[0].committed_fit.car.base_pace == .6
        checked.append(updated)
        return False

    monkeypatch.setattr(engine.simulator, "_should_pit", policy)
    monkeypatch.setattr(engine, "_protect_elective_finish_distance", guard)
    engine._start_lap(state, now)
    assert len(checked) == 1 and state.pit_stops == 1
