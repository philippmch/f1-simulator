"""Known SC/VSC duration reaches native distance and leading weather events."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from test_chronological_field_finish import field, ledger_signature, native_path
from test_custom_pit_replacements import snapshot

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import project_observed_chronological_clock
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.race_timing import RaceFinishClock, forecast_final_lap
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_neutralization import observed_control_intervals
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import projected_surfaces
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("modifier", [1.2, 1.4])
@pytest.mark.parametrize("duration", [0, 1, 2, 4, 8, np.int64(4)])
@pytest.mark.parametrize("now", [6500., 6850., 7199.999])
@pytest.mark.parametrize("scheduled", [68, 90])
def test_duration_horizon_matches_independent_finish_controller(modifier, duration, now, scheduled):
    clock = RaceFinishClock(scheduled)
    for lap in range(1, 66):
        clock.observe_leader_crossing(lap, lap * now / 65)
    time, lap, remaining = now, 65, int(duration)
    while clock.winner_time is None:
        time += 100. * (modifier if remaining else 1.)
        remaining = max(0, remaining - 1)
        lap += 1
        clock.observe_leader_crossing(lap, time)
    assert forecast_final_lap(scheduled, 65, now, 100., 7200., modifier,
                              controlled_laps=duration) == lap


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
@pytest.mark.parametrize("control_name", ["vsc", "safety_car"])
@pytest.mark.parametrize("duration", [1, 2, 4, 8])
@pytest.mark.parametrize("finite", [False, True])
def test_native_duration_clock_preserves_completed_horizon_and_flag(
    monkeypatch, engine_name, control_name, duration, finite,
):
    simulator = RaceSimulator(np.random.default_rng(34))
    engine = ChronologicalRace(simulator) if engine_name == "chronological" else None
    track = Track(id="T", name="Timed", country="Test", total_laps=90, base_lap_time=100.)
    driver, car = Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A")
    captured, fuel = {}, []
    control = simulator.event_manager

    def update(lap, *args, **kwargs):
        remaining = max(0, 65 + duration - lap)
        active = 65 <= lap < 65 + duration
        control.safety_car_active = control_name == "safety_car" and active
        control.vsc_active = control_name == "vsc" and active
        control.safety_car_laps_remaining = remaining if control.safety_car_active else 0
        control.vsc_laps_remaining = remaining if control.vsc_active else 0
        return []

    def physics(driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        fuel.append(total_laps)
        # One early elapsed loss leaves a late decision at 6850, followed by
        # an observed recurring free pace of 100 seconds in both engines.
        return 450. if lap_number == 1 else 100.

    def decide(state, states, planning, lap, *args, **kwargs):
        if lap == 66:
            captured["horizon"] = planning.total_laps
            if engine is not None:
                before = (snapshot(state, simulator), deepcopy(engine.pending),
                          engine.weather.model_copy(deep=True), ledger_signature(engine.timeline))
                captured["flag"] = engine._projected_flag_time(state.total_time)
                assert before == (snapshot(state, simulator), engine.pending,
                                  engine.weather, ledger_signature(engine.timeline))
        return False

    monkeypatch.setattr(control, "process_lap", update)
    monkeypatch.setattr(control, "_check_mechanical_failure", lambda *a, **k: None)
    monkeypatch.setattr(control, "_check_random_incident", lambda *a, **k: None)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", physics)
    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
    execute = engine.run if engine is not None else simulator.simulate_race
    inventory = ({"A": [dict(id="I", compound="intermediate", age=0),
                         dict(id="W", compound="wet", age=0)]} if finite else None)
    result, = execute(
        [driver], {"A": car}, track,
        Weather(track_wetness=.4, rain_intensity=.4, change_probability=0.), ["A"],
        starting_tires={"A": TireCompound.INTERMEDIATE}, tire_inventory=inventory)
    assert captured["horizon"] == result.laps_completed
    if engine is not None:
        assert captured["flag"] == pytest.approx(result.total_time, abs=1.e-8)
    assert result.race_time_limited and result.pit_laps == []
    assert fuel and set(fuel) == {90}


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("duration", [2, 4])
@pytest.mark.parametrize("relative_laps", [-1, 1])
@pytest.mark.parametrize("remaining", [5., 250.])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("own_fee", [0., 3.])
def test_observed_field_clock_matches_native_crossings_and_weather_events(
    monkeypatch, control, duration, relative_laps, remaining, off_track, own_fee,
):
    inputs = field(control=control, intervals=duration, relative_laps=relative_laps,
                   remaining=remaining, neutralized=True, off_track=off_track,
                   fee=7. if off_track else 0.)
    _, _, _, context, now = inputs
    own, updates, flags = [], [], []
    after, resolve = ChronologicalRace._after_leader_crossing, ChronologicalRace._resolve_crossing

    def leader(engine, time, red):
        if engine.timeline.chequered_time is None:
            updates.append(time)
        else:
            flags.append(engine.timeline.chequered_time)
        return after(engine, time, red)

    def crossing(engine, key, time):
        completed = resolve(engine, key, time)
        if completed and key == "A":
            own.append(time)
        return completed

    monkeypatch.setattr(ChronologicalRace, "_after_leader_crossing", leader)
    monkeypatch.setattr(ChronologicalRace, "_resolve_crossing", crossing)
    native_path(inputs, stopped=False, pending_fit=own_fee > 0., warmup={"medium": own_fee})
    before = ledger_signature(context.timeline), context.rivals
    projected = project_observed_chronological_clock(context, now, own_fitting_cost=own_fee)
    assert projected is not None and projected.identifier == "A"
    assert projected.flag_time == pytest.approx(flags[0], abs=1.e-8)
    assert projected.own_crossings == pytest.approx(own, abs=1.e-8)
    assert projected.leading_updates == pytest.approx(updates, abs=1.e-8)
    assert projected.flag_time not in projected.leading_updates
    assert before == (ledger_signature(context.timeline), context.rivals)
    # Weather instructions do not change the held free-pace clock or impose
    # safety rules on its internal pace adapter's placeholder tyre.
    rainy = replace(context, forecast_context=WeatherForecastContext(((72, 1., "heavy_rain"),)))
    assert project_observed_chronological_clock(rainy, now, own_fitting_cost=own_fee) == projected


@pytest.mark.parametrize("kind", ["same", "transition", "inventory", "bound"])
def test_all_weather_cost_paths_use_nonuniform_updates_without_cache_poisoning(kind):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="Test", total_laps=4, base_lap_time=90.)
    weather = Weather(track_wetness=.4, rain_intensity=.6)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    regular = StrategyWeatherClock((0., 170., 340., 510.), 10., 90., 6, 20., 20.)
    controlled = replace(regular, update_offsets=(10., 136., 262., 352., 442., 532.))

    def evaluate(clock):
        kwargs = dict(weather_clock=clock, physical_total_laps=90,
                      current_lap_time_modifier=1.4, active_aero_enabled=False)
        if kind == "same":
            return plan_rain_stop(driver, car, track, weather, tire, 9, 1, 0, **kwargs)
        if kind == "transition":
            return plan_rain_transition(driver, car, track, weather, tire, 9, 1, 0, **kwargs)
        if kind == "inventory":
            pool = TireInventory.from_sets([dict(id="I", compound="intermediate", age=9)])
            pool.fit("I")
            return plan_inventory_strategy(driver, car, track, weather, pool, 1,
                                           tire_age=9, remaining_stops=0, **kwargs)
        return weather_stop_costs(driver, car, track, weather, tire, 9, 1,
                                  traffic_possible=False, **kwargs)

    def reference(clock):
        events = clock.update_offsets or tuple(10. + 90. * index for index in range(6))
        counts = [0, *(sum(event <= time for event in events) for time in (170., 340., 510.))]
        surfaces = projected_surfaces(weather, 4, tuple(counts))
        projection = driver.model_copy(deep=True)
        result = 0.
        for offset, surface in enumerate(surfaces):
            projection.current_tire_laps = 9 + offset
            running = LapSimulator().calculate_lap_time(
                projection, car, track, tire, surface, 1 + offset, 90,
                active_aero_enabled=offset > 0, sample_variation=False)
            result += running * (1.4 if offset == 0 else 1.)
        return result

    first, second = evaluate(regular), evaluate(controlled)
    value = "stay_cost" if kind == "bound" else "wait_cost"
    assert getattr(first, value) == pytest.approx(reference(regular), abs=1.e-8)
    assert getattr(second, value) == pytest.approx(reference(controlled), abs=1.e-8)
    assert getattr(second, value) != pytest.approx(getattr(first, value), abs=1.e-5)
    assert evaluate(regular) == first and evaluate(controlled) == second


@pytest.mark.parametrize("kind", ["same", "transition", "inventory"])
@pytest.mark.parametrize("fitting_fee", [0., 40.])
@pytest.mark.parametrize("observed_running", [None, (130., 80.)])
def test_paid_replacements_price_nonuniform_entry_surfaces_and_unscaled_fitting(
    kind, fitting_fee, observed_running,
):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="Test", total_laps=4, base_lap_time=90.)
    weather = Weather(track_wetness=.4, rain_intensity=.6)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    delay = track.pit_lane_delta * .5 + expected_stationary_time(car) + 2.
    clock = StrategyWeatherClock(
        (0., 170., 340., 510.), 10., 90., 6, delay,
        track.pit_lane_delta + expected_stationary_time(car), observed_running,
        update_offsets=(10., 136., 262., 352., 442., 532.),
    )
    warmup = {"intermediate": fitting_fee, "wet": 3.}
    kwargs = dict(weather_clock=clock, physical_total_laps=90, tire_warmup=warmup,
                  current_lap_time_modifier=1.4, active_aero_enabled=False,
                  pit_lane_factor=.5, additional_current_stop_cost=2.)
    if kind == "same":
        result = plan_rain_stop(driver, car, track, weather, tire, 9, 1, 1, **kwargs)
    elif kind == "transition":
        result = plan_rain_transition(driver, car, track, weather, tire, 9, 1, 1, **kwargs)
    else:
        pool = TireInventory.from_sets([dict(id="old", compound="intermediate", age=9),
                                       dict(id="fresh", compound="intermediate", age=0)])
        pool.fit("old")
        result = plan_inventory_strategy(driver, car, track, weather, pool, 1,
                                         tire_age=9, remaining_stops=1, force_stop=True,
                                         require_compound_rule=False, **kwargs)

    def reference(compound):
        fee = warmup[compound.value]
        total = delay + fee
        projection = driver.model_copy(deep=True)
        for offset, start in enumerate(clock.lap_start_offsets):
            if offset and observed_running is not None:
                start = observed_running[1] + start - clock.lap_start_offsets[1]
            entry = start + delay + (fee if offset else 0.)
            updates = sum(event <= entry + 90.e-12 for event in clock.update_offsets)
            surface = weather.model_copy(deep=True)
            for _ in range(updates):
                surface = surface.project_surface()
            assert surface.tire_mismatch(compound) != "critical"
            projection.current_tire_laps = offset
            running = LapSimulator().calculate_lap_time(
                projection, car, track, TIRE_COMPOUNDS[compound], surface, 1 + offset, 90,
                active_aero_enabled=offset > 0, sample_variation=False)
            total += running * (1.4 if offset == 0 else 1.)
        return total

    candidates = ([TireCompound.INTERMEDIATE, TireCompound.WET] if kind == "transition"
                  else [TireCompound.INTERMEDIATE])
    assert result.pit_now_cost == pytest.approx(min(map(reference, candidates)), abs=1.e-8)


@pytest.mark.parametrize("duration", [0, 1, 4, 10 ** 400])
def test_extreme_duration_and_tiny_observed_pace_retain_bounded_horizon(duration):
    assert forecast_final_lap(90, 65, 6850., 1.e-320, 7200., 1.4,
                              controlled_laps=duration) == 90


@pytest.mark.parametrize("duration", [True, 2.5, -1, None])
def test_unknown_duration_cannot_invent_a_shorter_horizon(duration):
    assert forecast_final_lap(90, 65, 6850., 100., 7200., 1.4,
                              controlled_laps=duration) == 90


@pytest.mark.parametrize("times", [(10.,), (11., 20.), (10., 10.), (10., float("inf")),
                                    (10., True), [10., 20.]])
def test_nonuniform_clock_rejects_incomplete_or_invalid_update_observations(times):
    with pytest.raises(ValueError):
        StrategyWeatherClock((0., 30.), 10., 10., 2, 0., 0., update_offsets=times)


@pytest.mark.parametrize("control", ["vsc", "safety_car"])
@pytest.mark.parametrize("remaining", [np.int64(0), np.int64(4), True, 2.5, None])
def test_duration_observation_is_pure_and_does_not_coerce_unknown_counts(control, remaining):
    manager = RaceSimulator(np.random.default_rng(9)).event_manager
    manager.vsc_active = control == "vsc"
    manager.safety_car_active = control == "safety_car"
    setattr(manager, control + "_laps_remaining", remaining)
    before = deepcopy({key: value for key, value in manager.__dict__.items() if key != "rng"})
    random_state = deepcopy(manager.rng.bit_generator.state)
    expected = (max(1, int(remaining)) if isinstance(remaining, np.integer) else None)
    assert observed_control_intervals(manager) == expected
    assert {key: value for key, value in manager.__dict__.items() if key != "rng"} == before
    assert manager.rng.bit_generator.state == random_state
