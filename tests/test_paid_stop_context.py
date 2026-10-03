"""Paid replacement forecasts use observed outlap conditions in every path."""

from copy import deepcopy
from itertools import combinations, product
from math import inf

import numpy as np
import pytest
from test_custom_pit_replacements import inputs, snapshot
from test_paid_weather_compounds import exhaustive_safe_actions, models

from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.inventory_strategy import InventoryDecision
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.weather_schedule import WeatherForecastContext

SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)


def dry_execution_costs(state, track, *, gap, maximum_future_stops=2):
    """Execute every short dry schedule after a committed fresh root set.

    Two prior compounds already satisfy the rule. Every future paid service
    fits a fresh physical alternative, and its outlap consumes one lap before
    another service. The root lane, service and queue costs are common.
    """
    horizon = track.total_laps - 2
    physics = LapSimulator(np.random.default_rng(31))
    driver = state.driver.model_copy(deep=True)
    costs = dict.fromkeys(SLICKS, inf)
    for root in SLICKS:
        for count in range(maximum_future_stops + 1):
            for offsets in combinations(range(1, horizon), count):
                for compounds in product(SLICKS, repeat=count):
                    stops = dict(zip(offsets, compounds, strict=True))
                    compound, age, total = root, 0, 0.
                    for offset in range(horizon):
                        if offset in stops:
                            compound, age = stops[offset], 0
                            total += track.pit_lane_delta + expected_stationary_time(state.car)
                        driver.current_tire_laps = age
                        total += physics.calculate_lap_time(
                            driver, state.car, track, TIRE_COMPOUNDS[compound], Weather(),
                            offset + 3, track.total_laps, sample_variation=False,
                            gap_to_car_ahead=gap if offset == 0 else None,
                        )
                        age += 1
                    costs[root] = min(costs[root], total)
    return costs


@pytest.mark.parametrize("stress,zones,gain", [(.7, 8, 1.), (.7, 11, .75), (1., 11, .75)])
def test_forced_dry_choice_matches_every_executed_schedule_at_the_lap_floor(stress, zones, gain):
    simulator, state, track = inputs(laps=16)
    state.pit_plan, state.force_pit_next_lap = None, True
    simulator.tire_warmup = {}
    state.driver.skill_rating = state.car.base_pace = state.car.straight_line_speed = 1.
    track.tire_stress = stress
    track.active_aero_zones = [ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=gain)
                               for i in range(zones)]
    clean = simulator._choose_committed_dry_compound(state, track, 3, Weather())
    executed = dry_execution_costs(state, track, gap=.3)
    simulator._execute_pit_stop(state, track, Weather(), 3, sample_service=False,
                                current_traffic_gaps=(1., .3))
    assert clean == TireCompound.MEDIUM
    assert state.current_tire.compound == TireCompound.SOFT
    assert executed[state.current_tire.compound] == pytest.approx(min(executed.values()),
                                                               rel=0, abs=1.e-8)
    assert executed[clean] - min(executed.values()) > .1


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("floor", [False, True])
def test_finite_immediate_fallback_prices_actual_wear_first_gap_and_fit_fee(
    monkeypatch, free, control, floor,
):
    simulator, state, track = inputs(True, [], laps=5)
    state.pit_plan = None
    simulator.event_manager.safety_car_active = control
    if floor:
        state.driver.skill_rating = state.car.base_pace = state.car.straight_line_speed = 1.
        track.active_aero_zones = [ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1.)
                                   for i in range(8)]
    inventory = state.tire_inventory
    candidates = list(inventory.replacements())
    if free:
        candidates.insert(0, inventory.sets[inventory.current_set_id])
    candidates = [item for item in candidates
                  if Weather().tire_mismatch(item.compound) != "critical"]
    gap = 1. if free else .3
    physics = LapSimulator()
    expected = {}
    for item in candidates:
        driver = state.driver.model_copy(deep=True)
        driver.current_tire_laps = (state.tire_laps if item.id == inventory.current_set_id
                                   else item.age)
        value = physics.calculate_lap_time(
            driver, state.car, track, TIRE_COMPOUNDS[item.compound], Weather(), 3, 40,
            sample_variation=False, gap_to_car_ahead=gap,
            active_aero_enabled=not control,
        ) * simulator.event_manager.get_lap_time_modifier()
        if item.id != inventory.current_set_id or state.fit_lap_pending:
            value += simulator.tire_warmup[item.compound.value]
        expected[item.id] = value
    before = snapshot(state, simulator)
    calculate, observed = LapSimulator.calculate_lap_time, []

    def inspect(*args, **kwargs):
        observed.append(kwargs.get("gap_to_car_ahead"))
        return calculate(*args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", inspect)
    selected = simulator._inventory_immediate_set(
        state, track, Weather(), 3, free_fit=free, physical_total_laps=40,
        current_traffic_gaps=(1., .3),
    )
    assert expected[selected] == pytest.approx(min(expected.values()), rel=0, abs=1.e-8)
    assert observed == [gap] * len(candidates)
    assert snapshot(state, simulator) == before
    if not free:
        monkeypatch.setattr(simulator, "_plan_inventory",
                            lambda *args, **kwargs: InventoryDecision(inf, inf, None, None))
        observed.clear()
        assert simulator._prepare_inventory_pit(state, track, Weather(), 3,
                                                physical_total_laps=40,
                                                current_traffic_gaps=(1., .3))
        assert state.inventory_pit_proposal == (3, selected)
        assert observed == [gap] * len(candidates)


@pytest.mark.parametrize("fallback", [False, True])
def test_running_cost_hooks_cannot_mutate_live_forecast_inputs(monkeypatch, fallback):
    simulator, state, track = inputs(True, [], laps=5)
    weather = Weather()
    for compound, tire in TIRE_COMPOUNDS.items():
        monkeypatch.setitem(TIRE_COMPOUNDS, compound, tire.model_copy(deep=True))
    calculate = LapSimulator.calculate_lap_time

    def hook(self, driver, car, track, tire, weather, *args, **kwargs):
        driver.total_race_time = 12345.
        car.pit_stop_avg = 99.
        track.pit_lane_delta = 999.
        tire.initial_grip = .8
        weather.track_wetness = .1
        return calculate(self, driver, car, track, tire, weather, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", hook)
    before = (snapshot(state, simulator), deepcopy(track), deepcopy(weather),
              deepcopy(TIRE_COMPOUNDS))
    if fallback:
        simulator._inventory_immediate_set(state, track, weather, 3)
    else:
        simulator.lap_simulator.projected_stint_lap_cost(
            state.driver, state.car, track, TIRE_COMPOUNDS[TireCompound.SOFT],
            3, 3, weather,
        )
    assert (snapshot(state, simulator), track, weather, TIRE_COMPOUNDS) == before


def test_delayed_fallback_surface_hooks_cannot_change_observed_weather(monkeypatch):
    simulator, state, track = inputs(laps=5)
    state.current_tire = TIRE_COMPOUNDS[TireCompound.HARD]
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    project = Weather.project_surface

    def hook(self):
        self.track_temperature = 45.
        return project(self)

    monkeypatch.setattr(Weather, "project_surface", hook)
    clock = StrategyWeatherClock((0., 90., 180.), 1., 90., 3, 11., 11.)
    before = snapshot(state, simulator), deepcopy(weather), deepcopy(track)
    simulator._rank_stint_compounds(state, track, 3,
                                    [TireCompound.INTERMEDIATE, TireCompound.WET],
                                    weather=weather, weather_clock=clock)
    assert (snapshot(state, simulator), weather, track) == before


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("warmup", [0., 12.])
@pytest.mark.parametrize("control", [None, "safety_car_active", "vsc_active"])
@pytest.mark.parametrize("spent", [0, 4])
def test_forced_weather_forecast_matches_executed_queue_and_rejoin_context(
    monkeypatch, external, warmup, control, spent,
):
    driver, car, track = models(laps=5)
    simulator = RaceSimulator(np.random.default_rng(17),
                              tire_warmup={c.value: warmup for c in TireCompound} if warmup else {})
    state = DriverRaceState(driver, car, 1, current_tire=TIRE_COMPOUNDS[TireCompound.SOFT],
                            tire_laps=5, laps_completed=1, pit_stops=spent,
                            force_pit_next_lap=True)
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
        [dict(lap=3, rain_intensity=1.)], leading_lap=2,
    )
    if control:
        setattr(simulator.event_manager, control, True)
    clock = (StrategyWeatherClock((0., 90., 180., 270.), 10., 90., 6, 18., 11.)
             if external else None)
    expected = exhaustive_safe_actions(
        driver, car, track, weather, state.current_tire, 5, 2, 4 if spent == 0 else 1,
        simulator.weather_forecast_context, clock=clock, used=(TireCompound.SOFT,),
        dry=3 if spent == 0 else 1, damp=1, warmup=warmup, physical=8,
        lane=simulator._pit_lane_factor(), queue=7.,
        modifier=simulator.event_manager.get_lap_time_modifier(),
        aero=simulator.event_manager.is_active_aero_allowed(), gaps=(1., .3),
    )
    from f1sim.simulation import race

    planner = race.plan_rain_transition
    observed = []

    def inspect(*args, **kwargs):
        result = planner(*args, **kwargs)
        observed.append(result)
        return result

    monkeypatch.setattr(race, "plan_rain_transition", inspect)
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False,
                                physical_total_laps=8, weather_clock=clock,
                                current_traffic_gaps=(1., .3), additional_current_stop_cost=7.)
    assert len(observed) == 1
    assert observed[0].pit_now_cost == pytest.approx(expected[0], rel=0, abs=1.e-8)
    assert expected[3][state.current_tire.compound] == pytest.approx(expected[0], rel=0, abs=1.e-8)


@pytest.mark.parametrize("compound", [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD])
@pytest.mark.parametrize("control", [(1., True), (1.4, False)])
@pytest.mark.parametrize("zones", [0, 7, 14])
def test_fresh_stint_projection_uses_only_the_observed_first_gap(compound, control, zones):
    driver, car, track = models(laps=30)
    driver.skill_rating = car.base_pace = car.straight_line_speed = 1.
    track.active_aero_zones = [ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1.)
                               for i in range(zones)]
    weather = Weather(track_wetness=.08, rain_intensity=0.)
    tire = TIRE_COMPOUNDS[compound]
    modifier, aero = control
    simulator = LapSimulator(np.random.default_rng(19))
    before = deepcopy((driver, car, track, weather, tire, simulator.rng.bit_generator.state))
    expected = 0.
    running_driver, surface = driver.model_copy(deep=True), weather.model_copy(deep=True)
    for offset in range(5):
        running_driver.current_tire_laps = offset
        expected += simulator.calculate_lap_time(
            running_driver, car, track, tire, surface, 24 + offset, 40,
            gap_to_car_ahead=.3 if offset == 0 else None, sample_variation=False,
            active_aero_enabled=aero if offset == 0 else True,
        ) * (modifier if offset == 0 else 1.)
        surface = surface.project_surface()
    projected = simulator.projected_stint_lap_cost(
        driver, car, track, tire, 5, 24, weather, physical_total_laps=40,
        current_lap_time_modifier=modifier, active_aero_enabled=aero, gap_to_car_ahead=.3,
    )
    assert projected == pytest.approx(expected, rel=0, abs=1.e-8)
    assert (driver, car, track, weather, tire, simulator.rng.bit_generator.state) == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("sampled_service", [4., 100.])
def test_automatic_stops_keep_committed_context_independent_of_sampled_service(
    monkeypatch, engine, finite, sampled_service,
):
    simulator = RaceSimulator(np.random.default_rng(23))
    chronology = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id="team") for key in "AB"]
    car = Car(team_id="team", team_name="team", pit_stop_avg=2.75, pit_stop_std=.1)
    _, _, track = models(laps=8, lane=20.)
    records = {key: [dict(id=c.value, compound=c.value, age=0) for c in TireCompound]
               for key in "AB"}
    observed, committed = {}, {}
    if engine == "standard":
        traffic = simulator._standard_pit_traffic_snapshot

        def inspect_traffic(state, *args, **kwargs):
            result = traffic(state, *args, **kwargs)
            observed[state.driver.id, state.laps_completed + 1] = result, args[3]
            return result

        monkeypatch.setattr(simulator, "_standard_pit_traffic_snapshot", inspect_traffic)
    else:
        traffic = chronology._strategy_traffic

        def inspect_traffic(state, now, delay):
            result = traffic(state, now, delay)
            observed[state.driver.id, state.laps_completed + 1] = result, delay
            return result

        monkeypatch.setattr(chronology, "_strategy_traffic", inspect_traffic)
    execute = simulator._execute_pit_stop

    def inspect_stop(state, track, weather, current_lap, **kwargs):
        snapshot, delay = observed[state.driver.id, current_lap]
        assert kwargs["current_traffic_gaps"] == snapshot.current_traffic_gaps
        assert kwargs["additional_current_stop_cost"] == delay
        committed[state.driver.id, current_lap] = delay
        return execute(state, track, weather, current_lap, **kwargs)

    monkeypatch.setattr(simulator, "_execute_pit_stop", inspect_stop)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                        lambda car: sampled_service)
    monkeypatch.setattr(simulator.event_manager, "process_lap", lambda *args, **kwargs: [])
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *args: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident",
                        lambda *args, **kwargs: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *args, **kwargs: (False, False))
    run = simulator.simulate_race if engine == "standard" else chronology.run
    rows = run(drivers, {"team": car}, track, Weather(change_probability=0.), list("AB"),
               starting_tires={key: TireCompound.WET for key in "AB"},
               tire_inventory=records if finite else None)
    assert {("A", 1), ("B", 1)} <= committed.keys()
    assert committed["B", 1] == expected_stationary_time(car)
    follower = next(row for row in rows if row.driver_id == "B")
    assert follower.pit_stop_details[0]["queue_time"] == sampled_service
    assert all(row.pit_plan_history is None for row in rows)
