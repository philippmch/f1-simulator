"""Custom replacements agree with independent execution of the remaining policy."""

import importlib.util
from copy import deepcopy
from math import inf
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.models import ActiveAeroZone, Car, Driver, Sector, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_plans import initialize_pit_plan_state
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext

SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)
PLANS = ([], [{"lap": 4, "compound": "hard"}, {"lap": 7, "compound": "soft"}],
         [{"lap": 4, "compound": "wet"}, {"lap": 6, "compound": "medium"}])


def snapshot(state, simulator):
    # TireInventory is a mutable ledger with identity equality. Compare its
    # records and cursor explicitly instead of comparing it with a deep copy.
    values = dict(vars(state))
    if state.tire_inventory is not None:
        values["tire_inventory"] = vars(state.tire_inventory)
    return deepcopy((values, simulator.rng.bit_generator.state))


def inputs(finite=False, plan=(), *, laps=8):
    simulator = RaceSimulator(np.random.default_rng(33),
                              tire_warmup={"soft": 9., "medium": 4., "hard": 3.,
                                           "intermediate": 5., "wet": 6.})
    track = Track(id="custom", name="Synthetic", country="Synthetic", total_laps=laps,
                  base_lap_time=90., pit_lane_delta=8., tire_stress=1.,
                  sectors=[Sector(number=i, base_time=30., is_high_speed=i != 2)
                           for i in (1, 2, 3)],
                  active_aero_zones=[ActiveAeroZone(zone_id=1, sector=1, time_gain=.7),
                                     ActiveAeroZone(zone_id=2, sector=3, time_gain=.9)])
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A", tire_degradation_factor=1.5), 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM], tire_laps=8,
                            prior_tire_laps=2, laps_completed=2,
                            tire_compound_history=["soft", "medium"])
    if finite:
        records = [dict(id="M", compound="medium", age=2),
                   dict(id="S", compound="soft", age=0),
                   dict(id="H-used", compound="hard", age=9),
                   dict(id="H-fresh", compound="hard", age=0),
                   dict(id="I", compound="intermediate", age=1),
                   dict(id="W", compound="wet", age=0)]
        inventory = TireInventory.from_sets(records)
        simulator._initialize_inventory(state, inventory, inventory.sets["M"])
        state.tire_laps, state.tire_compound_history = 8, ["soft", "medium"]
    initialize_pit_plan_state(state, list(plan))
    return simulator, state, track


def execution_costs(simulator, state, track, weather, lap, *, free_fit=False,
                    physical_total_laps=None, weather_intervals=None, weather_clock=None):
    """Enumerate compulsory alternatives using execution helpers, never a planner.

    Every branch fits a real set, runs public lap physics, consumes the real
    pending-fit flag, and lets the engine prepare/skip the requested instruction.
    The external clock is hand-stepped from absolute event times. Short test
    horizons keep this exhaustive reference inexpensive without memoization.
    """
    original_execute = RaceSimulator._execute_pit_stop
    first_control = (simulator.event_manager.safety_car_active, simulator.event_manager.vsc_active)
    physical = track.total_laps if physical_total_laps is None else physical_total_laps
    context = simulator.weather_forecast_context

    def surface(offset, paid, first_stop, warmup_delay):
        if weather_clock is not None:
            elapsed = weather_clock.lap_start_offsets[offset]
            elapsed += paid * weather_clock.future_stop_delay
            if first_stop:
                elapsed += weather_clock.current_stop_delay - weather_clock.future_stop_delay
            if offset:
                elapsed += warmup_delay
            updates = 0
            if offset or paid:
                while updates < weather_clock.max_updates and (
                    weather_clock.first_update_after + updates * weather_clock.update_interval
                    <= elapsed + 1.e-10
                ):
                    updates += 1
        else:
            updates = offset if weather_intervals is None else weather_intervals[offset]
        result = weather.model_copy(deep=True)
        for update in range(updates):
            result = (result.project_surface() if context is None else
                      context.advanced(update).project_next(result))
        return result

    def options(state, weather, *, free=False):
        if state.tire_inventory is None:
            return [(compound, None) for compound in TireCompound
                    if weather.tire_mismatch(compound) != "critical"]
        inventory = state.tire_inventory
        available = list(inventory.replacements())
        if free and inventory.current_set_id not in inventory.unavailable_ids:
            available.insert(0, inventory.sets[inventory.current_set_id])
        return [(item.compound, item.id) for item in available
                if weather.tire_mismatch(item.compound) != "critical"]

    def running(state, offset, paid, first_stop, delay):
        after = surface(offset, paid, first_stop, delay)
        if after.tire_mismatch(state.current_tire.compound) == "critical":
            return inf, inf
        state.driver.current_tire_laps = state.tire_laps
        seconds = LapSimulator().calculate_lap_time(
            state.driver, state.car, track, state.current_tire, after, lap + offset,
            physical, sample_variation=False,
            active_aero_enabled=simulator.event_manager.is_active_aero_allowed(),
        ) * simulator.event_manager.get_lap_time_modifier()
        fee = simulator._consume_tire_warmup(state)
        state.tire_laps += 1
        state.laps_completed = lap + offset
        requested, later = tail(state, offset + 1, paid, first_stop, delay + fee)
        return requested, seconds + fee + later

    def service(state, offset, paid, first_stop, delay, option):
        state = deepcopy(state)
        state.pit_plan_target = option[0]
        if state.tire_inventory is not None:
            state.inventory_pit_proposal = (lap + offset, option[1])
        seconds = original_execute(simulator, state, track,
                                   surface(offset, paid, first_stop, delay), lap + offset,
                                   sample_service=False, physical_total_laps=physical)
        simulator._commit_pit_plan_if_due(state,
                                         overridden=state.pit_plan_override_reason is not None)
        state.pit_plan_target = None
        state.force_pit_next_lap = False
        requested, later = running(state, offset, paid + 1, first_stop or offset == 0, delay)
        return requested, seconds + later

    def tail(state, offset, paid, first_stop, delay):
        if lap + offset > track.total_laps:
            if simulator._stay_satisfies_tire_rule(state) or physical <= 1:
                return -sum(item["status"] == "executed" for item in state.pit_plan_history), 0.
            return inf, inf
        simulator.event_manager.safety_car_active = first_control[0] if offset == 0 else False
        simulator.event_manager.vsc_active = first_control[1] if offset == 0 else False
        entry = surface(offset, paid, first_stop, delay)
        request = simulator._custom_pit_plan_decision(state, track, entry, lap + offset)
        if request is True:
            selected = (state.pit_plan_target, state.pit_plan_target_set_id)
            return service(state, offset, paid, first_stop, delay, selected)
        if simulator._pit_plan_compulsory_reason(state, track, entry, lap + offset) is not None:
            candidates = options(state, entry)
            if lap + offset >= max(2, track.total_laps):
                candidates = [option for option in candidates
                              if simulator._pit_plan_satisfies_rule(state, option[0])]
            return min((service(state, offset, paid, first_stop, delay, option)
                        for option in candidates), default=(inf, inf))
        return running(state, offset, paid, first_stop, delay)

    def forbid_planning(*args, **kwargs):
        raise AssertionError("Exhaustive execution reference called a replacement planner")

    costs = {}
    with patch.object(simulator, "_custom_plan_replacement_choice", forbid_planning):
        for compound, identifier in options(state, weather, free=free_fit):
            simulator.event_manager.safety_car_active, simulator.event_manager.vsc_active = (
                first_control)
            candidate = deepcopy(state)
            if free_fit:
                if candidate.tire_inventory is None:
                    simulator._fit_tire(candidate, compound)
                else:
                    simulator._fit_inventory_tire(candidate, identifier, lap, "red_flag")
                candidate.force_pit_next_lap = False
                cost = tail(candidate, 0, 0, False, 0.)
            else:
                cost = service(candidate, 0, 0, False, 0., (compound, identifier))
            costs[identifier or compound] = cost
    simulator.event_manager.safety_car_active, simulator.event_manager.vsc_active = first_control
    return costs


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("plan", PLANS)
@pytest.mark.parametrize("surface_case", ["dry", "drying", "scheduled"])
@pytest.mark.parametrize("control", ["green", "safety_car"])
def test_custom_choice_matches_exhaustive_execution(finite, free, plan, surface_case, control):
    simulator, state, track = inputs(finite, plan)
    weather = Weather(track_wetness=.3 if surface_case == "drying" else 0.)
    if surface_case == "scheduled":
        simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
            [{"lap": 4, "rain_intensity": .7}, {"lap": 7, "rain_intensity": 0.}],
            leading_lap=3,
        )
    simulator.event_manager.safety_car_active = control == "safety_car"
    options = dict(free_fit=free, physical_total_laps=40)
    if surface_case != "dry":
        options.update(weather_intervals=(0, 0, 1, 3, 4, 7),
                       weather_clock=StrategyWeatherClock((0., 15., 45., 60., 120., 180.),
                                                         20., 25., 10, 45., 11.))
    before = (snapshot(state, simulator), deepcopy(track), deepcopy(weather))
    expected = execution_costs(simulator, deepcopy(state), track, weather, 3, **options)
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 3, **options)
    assert choice.cost == pytest.approx(min(expected.values())[1], rel=0, abs=1.e-8)
    if choice.compound is not None:
        assert expected[choice.set_id or choice.compound][1] == pytest.approx(choice.cost,
                                                                             rel=0, abs=1.e-8)
    assert (snapshot(state, simulator), track, weather) == before


@pytest.fixture(scope="module")
def harness():
    path = Path(__file__).resolve().parents[1] / "examples/check_weather_openings.py"
    spec = importlib.util.spec_from_file_location("replacement_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case_name", ["empty", "early", "late", "warmup", "scheduled", "timed"])
def test_real_compulsory_replacement_beats_every_explicit_alternative(harness, engine, finite,
                                                                   case_name):
    case = dict(water=0., rain=0., laps=60, base=90., lane=1., stress=1., degradation=1.5,
                pit_plan=[])
    if case_name in {"early", "warmup"}:
        case["pit_plan"] = [{"lap": 10, "compound": "hard"}]
    elif case_name == "late":
        case["pit_plan"] = [{"lap": 55, "compound": "soft"}]
    elif case_name == "scheduled":
        case.update(laps=30, lane=20.,
                    pit_plan=[{"lap": 5, "compound": "hard"}, {"lap": 22, "compound": "soft"}],
                    schedule=[{"lap": 8, "rain_intensity": .7}, {"lap": 20, "rain_intensity": 0.}])
    elif case_name == "timed":
        case.update(laps=10, base=1800., lane=20., degradation=1.,
                    pit_plan=[{"lap": 8, "compound": "soft"}])
    if case_name == "warmup":
        case["warmup"] = {"soft": 10., "medium": 4., "hard": 3.}
    if finite:
        case["inventory"] = [dict(id=c.value, compound=c.value, age=0) for c in TireCompound]
    original_execute = RaceSimulator._execute_pit_stop

    def run(compound=None):
        def execute(simulator, state, track, weather, current_lap, *args, **kwargs):
            saved_target = state.pit_plan_target
            if current_lap == 1 and compound is not None:
                state.pit_plan_target = compound
                if finite:
                    state.inventory_pit_proposal = (current_lap, compound.value)
            result = original_execute(simulator, state, track, weather, current_lap,
                                      *args, **kwargs)
            state.pit_plan_target = saved_target
            return result

        with patch.object(RaceSimulator, "_execute_pit_stop", execute):
            return harness.run_race(case, engine, TireCompound.WET)

    automatic = run()
    alternatives = [run(compound) for compound in SLICKS]
    def rank(row):
        return (-row["laps_completed"],
                -sum(item["status"] == "executed" for item in row["pit_plan_history"]),
                row["total_seconds"])

    best = min(alternatives, key=rank)
    assert automatic["laps_completed"] == best["laps_completed"]
    assert rank(automatic)[:2] == rank(best)[:2]
    assert automatic["total_seconds"] == pytest.approx(best["total_seconds"], rel=0, abs=1.e-8)
    if case_name == "timed":
        assert automatic["race_time_limited"]
        assert automatic["pit_plan_history"][0]["status"] == "not_reached"
    else:
        assert automatic["laps_completed"] == case["laps"]
    if case_name == "empty":
        assert automatic["compounds"][0] == "hard"
        assert "wet" not in automatic["compounds"]
        assert automatic["pit_laps"] == [1, 60]
    if case_name == "early":
        assert automatic["compounds"][0] == "soft"


@pytest.mark.parametrize("finite", [False, True])
def test_free_choice_follows_the_plan_and_preserves_the_live_state(finite):
    simulator, state, track = inputs(finite, [], laps=60)
    track.pit_lane_delta = 1.
    state.pit_stops = 1
    before = snapshot(state, simulator)
    choice = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, free_fit=True)
    assert choice.compound == TireCompound.HARD
    assert snapshot(state, simulator) == before
    if finite:
        simulator._refit_inventory_free(state, track, Weather(), 2)
        assert state.tire_inventory.current_set_id == choice.set_id == "H-fresh"
    else:
        assert simulator._choose_red_flag_tire(state, Weather(), track, 2) == choice.compound
    assert state.pit_stops == 1
    assert state.pit_plan_index == 0


@pytest.mark.parametrize("finite", [False, True])
def test_cancelled_replacement_projection_preserves_state(finite):
    simulator, state, track = inputs(finite, PLANS[1])
    before = snapshot(state, simulator)
    with cancellation_scope(lambda: True), pytest.raises(SimulationCancelled):
        simulator._custom_plan_replacement_choice(state, track, Weather(), 3)
    assert snapshot(state, simulator) == before


@pytest.mark.parametrize("case_name", ["pending_retention", "damaged", "worn", "repeat", "unrun"])
def test_physical_reuse_and_unrun_fits_match_execution(case_name):
    simulator, state, track = inputs(True, PLANS[1])
    if case_name == "pending_retention":
        simulator.tire_warmup = {c.value: 60. for c in TireCompound}
        state.fit_lap_pending = True
        track.total_laps = 3
    elif case_name == "damaged":
        state.tire_inventory.mark_current_unavailable(state.tire_laps)
        state.tire_inventory.unavailable_ids.add("H-fresh")
    elif case_name == "worn":
        state.tire_laps = 1201  # Live wear has no input-age validation ceiling.
    elif case_name == "repeat":
        records = [dict(id=item.id, compound=item.compound.value, age=item.age)
                   for item in state.tire_inventory.sets.values()]
        records.append(dict(id="H-equal", compound="hard", age=0))
        inventory = TireInventory.from_sets(records)
        simulator._initialize_inventory(state, inventory, inventory.sets["M"])
        state.tire_laps = 8
        state.tire_compound_history = ["soft", "medium"]
        initialize_pit_plan_state(state, [{"lap": lap, "compound": "hard"}
                                         for lap in range(3, 9)])
    elif case_name == "unrun":
        simulator._fit_inventory_tire(state, "W", 3, "red_flag")
        state.tire_compound_history = ["soft", "wet"]
        track.total_laps = 3
        initialize_pit_plan_state(state, [{"lap": 3, "compound": "soft"}])
        assert simulator._actually_used_compounds(state) == {TireCompound.SOFT}
    before = snapshot(state, simulator)
    expected = execution_costs(simulator, deepcopy(state), track, Weather(), 3,
                               free_fit=True, physical_total_laps=40)
    choice = simulator._custom_plan_replacement_choice(state, track, Weather(), 3,
                                                       free_fit=True, physical_total_laps=40)
    assert choice.cost == pytest.approx(min(expected.values())[1], rel=0, abs=1.e-8)
    assert expected[choice.set_id][1] == pytest.approx(choice.cost, rel=0, abs=1.e-8)
    assert snapshot(state, simulator) == before
    if case_name == "unrun":
        assert choice.compound != TireCompound.SOFT


def test_exhausted_pool_keeps_existing_compulsory_retirement():
    simulator, state, track = inputs(True, [])
    state.tire_inventory.unavailable_ids.update(state.tire_inventory.sets)
    before = snapshot(state, simulator)
    choice = simulator._custom_plan_replacement_choice(state, track, Weather(), 3)
    assert choice.set_id is choice.compound is None
    assert choice.cost == inf
    assert snapshot(state, simulator) == before
    assert not simulator._prepare_inventory_pit(state, track, Weather(), 3)
    assert state.status == DriverStatus.DNF
    assert state.dnf_reason == "No suitable replacement tyre set available"


@pytest.mark.parametrize("alternative_available", [False, True])
def test_repair_reserves_requested_set_when_an_alternative_can_honor_it(alternative_available):
    simulator, state, track = inputs(True, [{"lap": 4, "compound": "hard"}], laps=6)
    state.laps_completed, state.tire_laps = 1, 3
    state.tire_compound_history = ["medium"]
    state.force_pit_next_lap = True
    state.tire_inventory.mark_current_unavailable(state.tire_laps)
    state.tire_inventory.unavailable_ids.add("H-used")
    if not alternative_available:
        state.tire_inventory.unavailable_ids.add("S")
    choice = simulator._custom_plan_replacement_choice(state, track, Weather(), 2)
    assert choice.set_id == ("S" if alternative_available else "H-fresh")
    assert choice.instructions == int(alternative_available)
    simulator._execute_pit_stop(state, track, Weather(), 2, sample_service=False)
    assert state.tire_inventory.current_set_id == choice.set_id
    assert state.pit_plan_index == 0
    assert state.pit_plan_history[0]["status"] is None
    state.force_pit_next_lap = False
    state.tire_laps += 2
    assert simulator._custom_pit_plan_decision(state, track, Weather(), 4) is alternative_available
    if alternative_available:
        assert state.pit_plan_target_set_id == "H-fresh"
    else:
        assert state.pit_plan_history[0]["reason"] == "requested_compound_unavailable"


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("free", [False, True])
def test_prepared_custom_costs_preserve_public_physics_dispatch(monkeypatch, finite, free):
    simulator, state, track = inputs(finite, PLANS[1])
    native = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, free_fit=free)
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    public = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, free_fit=free)
    assert public.compound == native.compound
    assert public.set_id == native.set_id
    assert public.laps == native.laps
    assert public.cost == pytest.approx(native.cost, rel=0, abs=1.e-8)

    calculate = LapSimulator.calculate_lap_time

    def extension(self, driver, car, track, tire, *args, **kwargs):
        driver.total_race_time = 12345.  # Forecast-local mutations stay isolated.
        return calculate(self, driver, car, track, tire, *args, **kwargs) + (
            100. if tire.compound == TireCompound.SOFT else 0.)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", extension)
    before = snapshot(state, simulator)
    expected = execution_costs(simulator, deepcopy(state), track, Weather(), 3, free_fit=free)
    extended = simulator._custom_plan_replacement_choice(state, track, Weather(), 3, free_fit=free)
    assert extended.cost == pytest.approx(min(expected.values())[1], rel=0, abs=1.e-8)
    assert snapshot(state, simulator) == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case_name", ["empty", "early", "timed"])
def test_actual_restart_choice_preserves_paid_plan_and_matches_alternatives(engine, finite,
                                                                          case_name):
    timed = case_name == "timed"
    laps, red_lap = (10, 2) if timed else (60, 5)
    plan = ([{"lap": 8, "compound": "soft"}] if timed else
            [{"lap": 10, "compound": "hard"}] if case_name == "early" else [])
    track = Track(id="restart", name="Synthetic", country="Synthetic", total_laps=laps,
                  base_lap_time=1800. if timed else 90., pit_lane_delta=1., tire_stress=1.)
    car = Car(team_id="A", team_name="A", tire_degradation_factor=1.5)
    records = [dict(id=c.value, compound=c.value, age=0) for c in TireCompound]

    def run(override=None):
        simulator = RaceSimulator(np.random.default_rng(12), red_flag_pause_seconds=0.)
        manager = simulator.event_manager
        manager.set_forced_red_flag(red_lap)
        manager._deploy_safety_measure = lambda *args, **kwargs: None
        manager._check_mechanical_failure = lambda *args, **kwargs: None
        manager._check_random_incident = lambda *args, **kwargs: None
        calculate = simulator.lap_simulator.calculate_lap_time

        def mean(*args, **kwargs):
            kwargs["sample_variation"] = False
            return calculate(*args, **kwargs)

        simulator.lap_simulator.calculate_lap_time = mean
        simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
        refits = []
        if finite:
            original = simulator._refit_inventory_free

            def fit(state, track, weather, current_lap, **kwargs):
                before = (state.pit_stops, state.pit_plan_index)
                if override is None:
                    success = original(state, track, weather, current_lap, **kwargs)
                else:
                    simulator._fit_inventory_tire(state, override.value, current_lap + 1,
                                                   "red_flag")
                    state.force_pit_next_lap = False
                    success = True
                assert (state.pit_stops, state.pit_plan_index) == before
                refits.append(state.current_tire.compound)
                return success

            simulator._refit_inventory_free = fit
        else:
            original = simulator._choose_red_flag_tire

            def choose(state, weather, track, current_lap, **kwargs):
                choice = override if override is not None else original(
                    state, weather, track, current_lap, **kwargs)
                refits.append(choice)
                return choice

            simulator._choose_red_flag_tire = choose
        options = dict(starting_tires={"A": TireCompound.MEDIUM}, pit_plans={"A": plan})
        if finite:
            options["tire_inventory"] = {"A": records}
        runner = (simulator.simulate_race if engine == "standard" else
                  ChronologicalRace(simulator, red_flag_pause_seconds=0.).run)
        result, = runner([Driver(id="A", name="A", team_id="A")], {"A": car}, track,
                          Weather(change_probability=0.), ["A"], **options)
        assert result.status == DriverStatus.FINISHED
        assert len(refits) == 1
        assert any(event.event_type == EventType.RED_FLAG for event in manager.events)
        return result, refits[0]

    actual, chosen = run()
    alternatives = [run(compound)[0] for compound in SLICKS]
    best = min(alternatives, key=lambda result: (
        -result.laps_completed,
        -sum(item["status"] == "executed" for item in result.pit_plan_history),
        result.total_time,
    ))
    assert actual.laps_completed == best.laps_completed
    assert actual.total_time == pytest.approx(best.total_time, rel=0, abs=1.e-8)
    if case_name == "empty":
        assert chosen == TireCompound.HARD
        assert actual.pit_stops == 0
    elif case_name == "early":
        assert chosen == TireCompound.SOFT
        assert actual.pit_plan_history[0]["status"] == "executed"
        assert actual.pit_laps == [10]
    else:
        assert actual.race_time_limited
        assert actual.pit_plan_history[0]["status"] == "not_reached"
