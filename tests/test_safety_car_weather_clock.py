"""Observed SC running must reach the same weather events as execution."""

from copy import deepcopy
from dataclasses import replace

import pytest
from test_chronological_weather_intervals import setup

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.custom_pit_strategy import choose_custom_pit_replacement
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_neutralization import SafetyCarBranch, StrategySafetyCarSnapshot
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.surface_projection import projected_surfaces
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("pace", [100., 130., 140.])
@pytest.mark.parametrize("stopped", [False, True])
@pytest.mark.parametrize("warmup", [0., 3.])
@pytest.mark.parametrize("wetness,rain", [(.4, .6), (.25, 0.)])
def test_clock_reaches_executed_next_lap_weather_after_safety_car_catchup(
    monkeypatch, pace, stopped, warmup, wetness, rain,
):
    engine, run, actual, _ = setup(monkeypatch, pace=pace, wetness=wetness, rain=rain, laps=8)
    simulator = engine.simulator
    simulator.tire_warmup = {compound.value: warmup for compound in TireCompound} if warmup else {}
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        expected_stationary_time)
    observations = {}

    def control(lap, *args, **kwargs):
        simulator.event_manager.safety_car_active = lap == 1
        return []

    def decide(state, states, track, lap, *args, **kwargs):
        if state.driver.id == "B" and lap == 2:
            clock = kwargs["weather_clock"]
            snapshot = state.strategy_safety_car_snapshot
            assert clock is not None and snapshot is not None
            observations.update(clock=clock, snapshot=snapshot)
        return state.driver.id == "B" and lap == 2 and stopped

    monkeypatch.setattr(simulator.event_manager, "process_lap", control)
    monkeypatch.setattr(simulator, "_should_pit", decide)
    begin = engine._begin_running

    def running(state, pending, now):
        begin(state, pending, now)
        if state.driver.id == "B" and pending.lap == 2:
            # The next entry precedes the leader's following green crossing,
            # so a later pass or blue-flag hold cannot change this oracle.
            assert pending.ready < engine.pending["A"].ready + 90.

    monkeypatch.setattr(engine, "_begin_running", running)
    run()
    clock = observations["clock"]
    snapshot = observations["snapshot"]
    updates = clock.updates(1, int(stopped), stopped, fit_delay=warmup if stopped else 0.)
    expected = projected_surfaces(actual["B", 2], 2, (0, updates))[-1]
    assert expected == actual["B", 3]
    assert updates == 1
    assert clock.current_running_times == pytest.approx(tuple(
        branch.running_time(pace, 1.4) for branch in (snapshot.retained, snapshot.stopped)))
    if pace >= 130:
        uniform = replace(clock, current_running_times=None)
        assert uniform.updates(1, int(stopped), stopped, fit_delay=warmup if stopped else 0.) == 2


@pytest.mark.parametrize("reason", ["observed", "missing", "green", "vsc", "red", "restart"])
def test_weather_clock_only_uses_a_current_observed_queue_without_mutation(monkeypatch, reason):
    engine, run, _, _ = setup(monkeypatch, pace=130., wetness=.4, rain=.6, laps=8)
    control = engine.simulator.event_manager
    checked = []

    def events(lap, *args, **kwargs):
        control.safety_car_active = lap == 1
        return []

    def decide(state, states, track, lap, *args, **kwargs):
        if state.driver.id != "B" or lap != 2:
            return False
        snapshot = state.strategy_safety_car_snapshot
        assert snapshot is not None
        before = deepcopy((engine.states, engine.pending, engine.weather,
                           engine.simulator.rng.bit_generator.state))
        with monkeypatch.context() as patch:
            patch.setattr(control, "safety_car_active", reason not in ("green", "vsc"))
            patch.setattr(control, "vsc_active", reason == "vsc")
            patch.setattr(control, "red_flag_active", reason == "red")
            patch.setattr(state, "strategy_safety_car_snapshot",
                          None if reason == "missing" else snapshot)
            clock = engine._strategy_weather_clock(
                state, state.total_time, track, 0., restart=reason == "restart")
        assert before == (engine.states, engine.pending, engine.weather,
                          engine.simulator.rng.bit_generator.state)
        if reason == "red":
            assert clock is None
        else:
            assert clock is not None
            assert (clock.current_running_times is not None) == (reason == "observed")
        checked.append(clock)
        return False

    monkeypatch.setattr(control, "process_lap", events)
    monkeypatch.setattr(engine.simulator, "_should_pit", decide)
    run()
    assert len(checked) == 1


@pytest.mark.parametrize("kind", ["same", "transition", "inventory", "bound", "custom"])
@pytest.mark.parametrize("wetness,rain", [(.44, .6), (.19, 0.)])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "intermediate": 3.}])
def test_all_weather_planners_preserve_separate_retained_and_paid_running_clocks(
    kind, wetness, rain, warmup,
):
    driver = Driver(id="D", name="D", team_id="T")
    car = Car(team_id="T", team_name="T")
    track = Track(id="T", name="T", country="T", total_laps=4,
                  base_lap_time=90., pit_lane_delta=3.)
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(deep=True)
    queue = StrategySafetyCarSnapshot(SafetyCarBranch(126., ahead_progress=.08),
                                     SafetyCarBranch(126., ahead_progress=.3))
    delay = 3. * .55 + expected_stationary_time(car) + 2.
    clock = StrategyWeatherClock((0., 126., 216., 306.), 30., 90., 6,
                                 delay, 3. + expected_stationary_time(car),
                                 current_running_times=(117., 100.))

    def evaluate(selected_clock):
        options = dict(weather_clock=selected_clock, tire_warmup=warmup,
                       current_lap_time_modifier=1.4, pit_lane_factor=.55,
                       additional_current_stop_cost=2., active_aero_enabled=False,
                       safety_car=queue)
        if kind == "same":
            return plan_rain_stop(driver, car, track, weather, tire, 19, 1, 2, **options)
        options["used_compounds"] = {TireCompound.HARD, TireCompound.MEDIUM}
        if kind == "transition":
            return plan_rain_transition(driver, car, track, weather, tire, 19, 1, 2, **options)
        if kind == "inventory":
            inventory = TireInventory.from_sets([
                dict(id=compound.value, compound=compound.value, age=0)
                for compound in TireCompound])
            inventory.fit("intermediate")
            return plan_inventory_strategy(driver, car, track, weather, inventory, 1,
                                           tire_age=19, remaining_stops=2, **options)
        if kind == "custom":
            return choose_custom_pit_replacement(
                driver, car, track, weather, tire, 19, 1,
                [{"lap": 3, "compound": "medium"}], **options)
        options.pop("used_compounds")
        return weather_stop_costs(driver, car, track, weather, tire, 19, 1,
                                  traffic_possible=False, **options)

    # Each branch is independently rebased onto its observed first running
    # interval. These clocks have no branch override; they exercise the old
    # absolute-start contract, including future paid fits and warmup delays.
    retained = replace(clock, lap_start_offsets=(0., 117., 207., 297.),
                       current_running_times=None)
    stopped = replace(clock, lap_start_offsets=(0., 100., 190., 280.),
                      current_running_times=None)
    before = deepcopy((driver, car, track, weather, tire, clock, queue))
    actual = evaluate(clock)
    stopped_reference = evaluate(stopped)
    if kind == "custom":
        assert actual == stopped_reference
    else:
        assert actual.pit_now_cost == pytest.approx(stopped_reference.pit_now_cost, abs=1.e-9)
        retained_reference = evaluate(retained)
        field = "stay_cost" if kind == "bound" else "wait_cost"
        assert getattr(actual, field) == pytest.approx(
            getattr(retained_reference, field), abs=1.e-9)
        if kind in {"transition", "inventory"}:
            assert actual.compound == stopped_reference.compound
    # Interleaving different raw clocks must not poison cached future states.
    assert evaluate(clock) == actual
    assert before == (driver, car, track, weather, tire, clock, queue)
