"""Queue-aware first-lap costs agree with the engines' actual mean running."""

from copy import deepcopy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap
from f1sim.simulation.custom_pit_strategy import choose_custom_pit_replacement
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time, plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_strategy import weather_stop_costs


def setup(engine_name, wetness, warmup):
    simulator = RaceSimulator(np.random.default_rng(7), tire_warmup=warmup)
    simulator.event_manager.safety_car_active = True
    physics = simulator.lap_simulator.calculate_lap_time
    simulator.lap_simulator.calculate_lap_time = (
        lambda *args, **kwargs: physics(*args, **dict(kwargs, sample_variation=False)))
    track = Track(id="sc", name="SC", country="test", total_laps=5,
                  base_lap_time=90., pit_lane_delta=3., tire_stress=.5)
    weather = Weather(track_wetness=wetness, rain_intensity=wetness)
    compound = TireCompound.MEDIUM if wetness == 0 else TireCompound.INTERMEDIATE
    states = [DriverRaceState(Driver(id=key, name=key, team_id=key),
                              Car(team_id=key, team_name=key), position,
                              current_tire=TIRE_COMPOUNDS[compound].model_copy(deep=True),
                              tire_laps=12 if key == "A" else 10, total_time=0. if key == "A"
                              else 6., laps_completed=4)
              for position, key in enumerate("AB", 1)]
    own = states[1]
    own.fit_lap_pending = bool(warmup)
    states[0].fit_lap_pending = bool(warmup)
    delay = track.pit_lane_delta * .55 + expected_stationary_time(own.car) + 2.
    engine = ChronologicalRace(simulator)
    engine.track = track
    engine.weather = weather
    engine.states = {state.driver.id: state for state in states}
    engine.order = list("AB")
    engine.running_paces = {key: 93. for key in "AB"}
    engine.control_intervals = 4
    if engine_name == "chronological":
        # This is an already-running predecessor; its queue and fitting delay
        # are visible and held fixed while we price the candidate's entry.
        ahead = _PendingLap(5, 0., 0., weather, True, states[0].current_tire, 12, 0.)
        engine.pending = {"A": ahead}
        engine._begin_running(states[0], ahead, 0.)
        context = engine._safety_car_strategy_snapshot(own, own.total_time, 2.)
    else:
        context = simulator._standard_safety_car_snapshot(
            own, states, states, track, weather, 5, 2., {})
    assert context is not None

    def executed(compound, stopped):
        candidate = deepcopy(own)
        if stopped:
            candidate.current_tire = TIRE_COMPOUNDS[compound].model_copy(deep=True)
            candidate.tire_laps = 0
            candidate.total_time += delay
            candidate.fit_lap_pending = True
        candidate.driver.current_tire_laps = candidate.tire_laps
        fee = warmup.get(compound.value, 0.) if candidate.fit_lap_pending else 0.
        if engine_name == "chronological":
            pending = _PendingLap(5, own.total_time, candidate.total_time,
                                  weather, True, candidate.current_tire, candidate.tire_laps, 0.,
                                  paid_stop=stopped)
            engine.states["B"] = candidate
            engine.pending["B"] = pending
            engine._begin_running(candidate, pending, candidate.total_time)
            engine.states["B"] = own
            engine.pending.pop("B")
            running = max(pending.ready, engine.pending["A"].ready + 1.e-9) - candidate.total_time
        else:
            rows = [deepcopy(states[0]), candidate]
            simulator._handle_pit_batch_position_changes([candidate] if stopped else [], rows)
            free = {}
            for row in rows:
                row.driver.current_tire_laps = row.tire_laps
                free[row.driver.id] = simulator.lap_simulator.calculate_lap_time(
                    row.driver, row.car, track, row.current_tire, weather, 5, 5,
                    gap_to_car_ahead=simulator._get_gap_to_car_ahead(row, rows),
                    active_aero_enabled=False,
                )
            controlled = simulator._safety_car_lap_times(free, rows, 1.4)
            entry = candidate.total_time
            for row in rows:
                fee = warmup.get(row.current_tire.compound.value, 0.) if row.fit_lap_pending else 0.
                row.total_time += controlled[row.driver.id] + fee
            simulator._reconcile_racing_times(rows)
            running = candidate.total_time - entry
        return running + (delay if stopped else 0.)

    return own, track, weather, context, delay, executed


CASES = [("dry", 0.), ("rain", .5), ("transition", 0.), ("transition", .5),
         ("inventory", 0.), ("inventory", .5), ("custom", 0.), ("custom", .5),
         ("bound", 0.), ("bound", .5)]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("kind,wetness", CASES)
@pytest.mark.parametrize("clock", [False, True])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "intermediate": 1.5}])
def test_paid_and_retained_costs_match_actual_queue_running(engine, kind, wetness, clock, warmup):
    own, track, weather, context, delay, executed = setup(engine, wetness, warmup)
    before = deepcopy((own, track, weather, context))
    options = dict(pit_lane_factor=.55, additional_current_stop_cost=2.,
                   current_lap_time_modifier=1.4, active_aero_enabled=False,
                   safety_car=context, tire_warmup=warmup,
                   current_fit_pending=own.fit_lap_pending)
    if clock and kind != "dry":
        options["weather_clock"] = StrategyWeatherClock((0.,), 1000., 90., 0, delay, delay)
    compounds = [compound for compound in TireCompound
                 if weather.tire_mismatch(compound) != "critical"]
    if kind == "dry":
        compounds = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
        decision = plan_dry_stop(own.driver, own.car, track, own.current_tire, 10, 1, 1,
                                 {TireCompound.HARD, TireCompound.MEDIUM}, **options)
    elif kind == "rain":
        compounds = [own.current_tire.compound]
        decision = plan_rain_stop(own.driver, own.car, track, weather, own.current_tire,
                                  10, 5, 1, **options)
    elif kind == "transition":
        decision = plan_rain_transition(own.driver, own.car, track, weather, own.current_tire,
                                        10, 5, 1, used_compounds={TireCompound.HARD,
                                                                TireCompound.MEDIUM}, **options)
    elif kind == "inventory":
        inventory = TireInventory.from_sets([
            dict(id=compound.value, compound=compound.value, age=0) for compound in TireCompound])
        inventory.fit(own.current_tire.compound.value)
        original = deepcopy(inventory.__dict__)
        compounds = [compound for compound in compounds if compound != own.current_tire.compound]
        decision = plan_inventory_strategy(
            own.driver, own.car, track, weather, inventory, 5, tire_age=10, remaining_stops=1,
            used_compounds={TireCompound.HARD, TireCompound.MEDIUM}, **options)
        assert inventory.__dict__ == original
    elif kind == "custom":
        options.pop("current_fit_pending")
        decision = choose_custom_pit_replacement(
            own.driver, own.car, track, weather, own.current_tire, 10, 5, [],
            used_compounds={TireCompound.HARD, TireCompound.MEDIUM}, **options)
    else:
        decision = weather_stop_costs(own.driver, own.car, track, weather, own.current_tire,
                                      10, 5, traffic_possible=False, **options)
    costs = {compound: executed(compound, True) for compound in compounds}
    best = min(costs.values(), default=inf)
    pit_cost = decision.cost if kind == "custom" else decision.pit_now_cost
    assert pit_cost == pytest.approx(best, abs=1.e-9)
    if kind != "custom":
        wait = decision.stay_cost if kind == "bound" else decision.wait_cost
        assert wait == pytest.approx(executed(own.current_tire.compound, False), abs=1.e-9)
    if kind in {"dry", "transition", "custom", "inventory"}:
        assert costs[decision.compound] == pytest.approx(best, abs=1.e-9)
    assert before == (own, track, weather, context)


@pytest.mark.parametrize("clock", [False, True])
def test_queue_forecast_only_replaces_current_running_and_keeps_future_green(clock):
    own, track, weather, context, delay, executed = setup("standard", .5, {})
    extended = track.model_copy(update={"total_laps": 8})
    options = dict(pit_lane_factor=.55, additional_current_stop_cost=2.,
                   current_lap_time_modifier=1.4, active_aero_enabled=False)
    if clock:
        options["weather_clock"] = StrategyWeatherClock((0., 90., 180., 270.),
                                                       1000., 90., 0, delay, delay)
    baseline = plan_rain_stop(own.driver, own.car, extended, weather, own.current_tire,
                              10, 5, 1, current_traffic_gaps=context.traffic_gaps, **options)
    projected = plan_rain_stop(own.driver, own.car, extended, weather, own.current_tire,
                               10, 5, 1, safety_car=context, **options)
    driver = own.driver.model_copy(deep=True)
    physics = LapSimulator()
    for stopped in (False, True):
        driver.current_tire_laps = 0 if stopped else 10
        free = physics.calculate_lap_time(driver, own.car, extended, own.current_tire, weather,
                                          5, 8, active_aero_enabled=False,
                                          gap_to_car_ahead=context.traffic_gaps[int(stopped)],
                                          sample_variation=False)
        # Fuel changes the free pace but this compact SC queue holds the
        # observed predecessor's clock. No future green cost is transformed.
        correction = executed(own.current_tire.compound, stopped) - free * 1.4
        if stopped:
            correction -= delay
        old = baseline.pit_now_cost if stopped else baseline.wait_cost
        new = projected.pit_now_cost if stopped else projected.wait_cost
        assert new - old == pytest.approx(correction, abs=1.e-9)
