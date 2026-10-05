"""Compulsory wets govern physical running, stock and requests until SC return."""

from copy import deepcopy
from itertools import product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.control_schedule import (
    WET_RESUMPTION_CONTROL_SCHEDULE_POLICY,
    control_schedule_policy,
    parse_control_schedule_spec,
    validate_control_schedule_snapshot,
)
from f1sim.simulation.events import EventManager
from f1sim.simulation.finish_strategy import SafetyCarFinishBranch, SafetyCarFinishField
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.strategy_control_clock import StandardControlContext, StrategyControlContext
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory


def run_case(monkeypatch, engine, *, inventory=None, pit_plan=None, after=2, action="resume_wet"):
    sim = RaceSimulator(np.random.default_rng(91), red_flag_pause_seconds=100,
                        control_schedule=[{"lap": after, "control": "red_flag", "action": action}])
    for name in ("_check_random_incident", "_check_mechanical_failure",
                 "_check_red_flag_conditions"):
        monkeypatch.setattr(sim.event_manager, name, lambda *a, **kw: None)
    monkeypatch.setattr(sim, "_process_overtakes", lambda *a, **kw: 0)
    monkeypatch.setattr(sim.overtaking_model, "attempt_overtake", lambda *a, **kw: (False, False))
    samples, services = [], []
    physics = sim.lap_simulator.calculate_lap_time

    def observe(driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        samples.append({"lap": lap_number, "compound": tire.compound.value,
                        "age": driver.current_tire_laps,
                        "required": sim.event_manager.is_wet_tire_required()})
        kwargs["sample_variation"] = False
        return physics(driver, car, track, tire, weather, lap_number, total_laps, **kwargs)

    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", observe)
    monkeypatch.setattr(sim.lap_simulator, "calculate_pit_stop_time",
                        lambda car: services.append(car.team_id) or expected_stationary_time(car))
    driver = Driver(id="A", name="A", team_id="A", consistency=1.)
    car = Car(team_id="A", team_name="A", reliability=1.)
    track = Track(id="T", name="T", country="Synthetic", total_laps=8, base_lap_time=90.,
                  safety_car_probability=0.)
    run = sim.simulate_race if engine == "standard" else ChronologicalRace(
        sim, red_flag_pause_seconds=100).run
    result = run([driver], {"A": car}, track, Weather(change_probability=0.), ["A"],
                 starting_tires={"A": "medium"},
                 tire_inventory=None if inventory is None else {"A": inventory},
                 pit_plans=None if pit_plan is None else {"A": pit_plan})[0]
    return result, sim, samples, services


def test_wet_resumption_is_explicit_and_old_schemas_cannot_claim_it():
    schedule = parse_control_schedule_spec("2:red:resume_wet,4:vsc:1")
    assert schedule == [{"lap": 2, "control": "red_flag", "action": "resume_wet"},
                        {"lap": 4, "control": "vsc", "duration_laps": 1}]
    assert control_schedule_policy(schedule) == WET_RESUMPTION_CONTROL_SCHEDULE_POLICY
    snapshot = {"schema_version": 13, "control_schedule": schedule,
                "control_schedule_policy": WET_RESUMPTION_CONTROL_SCHEDULE_POLICY}
    validate_control_schedule_snapshot(snapshot, schedule)
    for version, policy in [(11, "observed_control_schedule_v1"),
                            (12, "observed_control_schedule_v2")]:
        with pytest.raises(ValueError, match="schema 13|Schema 11"):
            validate_control_schedule_snapshot(snapshot | {"schema_version": version,
                                                "control_schedule_policy": policy}, schedule)
    with pytest.raises(ValueError):
        parse_control_schedule_spec("2:red:resume_wet,3:vsc:1")


def test_director_instruction_expires_at_sc_return_and_reset():
    manager = EventManager(control_schedule=[{"lap": 2, "control": "red_flag",
                                             "action": "resume_wet"}])
    track = Track(id="T", name="T", country="Synthetic", total_laps=8, base_lap_time=90.)
    manager.process_lap(2, [], {}, track, Weather())
    assert manager.red_flag_active and not manager.is_wet_tire_required()
    manager.end_red_flag()
    assert manager.is_wet_tire_required()
    manager.end_red_flag()
    assert manager.safety_car_deployments == 1
    manager.process_lap(3, [], {}, track, Weather())
    assert not manager.is_wet_tire_required()
    manager.mandatory_wet_tires = True
    manager.reset()
    assert not manager.mandatory_wet_tires


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_full_wets_run_once_then_green_policy_can_change_compound(monkeypatch, engine):
    result, sim, samples, _ = run_case(monkeypatch, engine)
    assert result.status == DriverStatus.FINISHED
    required = [row for row in samples if row["required"]]
    assert required == [{"lap": 3, "compound": "wet", "age": 0, "required": True}]
    assert all(row["compound"] != "wet" for row in samples if row["lap"] >= 4)
    assert not sim.event_manager.is_wet_tire_required()
    assert "full-wet tyres compulsory" in sim.event_manager.events[-1].description


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_free_fit_and_fixed_request_cannot_erase_compulsory_wet_running(monkeypatch, engine):
    result, _, samples, _ = run_case(monkeypatch, engine, pit_plan=[
        {"lap": 3, "compound": "soft"}, {"lap": 4, "compound": "hard"},
    ])
    assert next(row for row in samples if row["lap"] == 3)["compound"] == "wet"
    assert result.pit_plan_history[0]["status"] == "overridden"
    assert result.pit_plan_history[0]["reason"] == "mandatory_wet_tires"
    assert result.pit_plan_history[0]["actual_compound"] is None
    assert result.pit_plan_history[1]["actual_compound"] == "hard"
    assert result.pit_laps == [4]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_disallowed_early_window_preserves_its_green_deadline(monkeypatch, engine):
    result, _, samples, _ = run_case(monkeypatch, engine, pit_plan=[
        {"earliest_lap": 3, "lap": 4, "trigger": "neutralized", "compound": "soft"},
    ])
    assert next(row for row in samples if row["lap"] == 3)["compound"] == "wet"
    assert result.pit_plan_history[0]["actual_lap"] == 4
    assert result.pit_plan_history[0]["actual_compound"] == "soft"
    assert result.pit_laps == [4]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("wet_stock", [[], [{"id": "W", "compound": "wet", "remaining_laps": 0}]])
def test_missing_or_expired_wets_prevent_resumption_without_paid_service(
    monkeypatch, engine, wet_stock,
):
    inventory = [{"id": "M", "compound": "medium"}, {"id": "H", "compound": "hard"}]
    result, _, samples, services = run_case(monkeypatch, engine, inventory=inventory + wet_stock,
                                           pit_plan=[])
    assert result.status == DriverStatus.DNF and result.laps_completed == 2
    assert result.dnf_reason == "No usable full-wet tyre set for compulsory resumption"
    assert [row["lap"] for row in samples] == [1, 2]
    assert result.pit_stops == 0 and not services
    assert sum(stint["laps_used"] for stint in result.tire_set_history) == 2


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_limited_wet_set_keeps_its_prior_wear_and_consumes_only_a_real_lap(monkeypatch, engine):
    inventory = [{"id": "M", "compound": "medium"}, {"id": "H", "compound": "hard"},
                 {"id": "W", "compound": "wet", "age": 9, "remaining_laps": 1}]
    before = deepcopy(inventory)
    result, _, samples, _ = run_case(monkeypatch, engine, inventory=inventory, pit_plan=[])
    assert inventory == before and result.status == DriverStatus.FINISHED
    assert next(row for row in samples if row["lap"] == 3)["age"] == 9
    wet = next(row for row in result.tire_set_history if row["set_id"] == "W")
    assert wet["kind"] == "red_flag" and wet["lap"] == 3
    assert (wet["age_at_fit"], wet["age_at_end"], wet["laps_used"]) == (9, 10, 1)
    assert wet["remaining_laps_at_end"] == 0
    assert result.pit_laps == [4]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_terminal_wet_request_does_not_fit_or_leave_a_mandate(monkeypatch, engine):
    result, sim, samples, _ = run_case(monkeypatch, engine, after=8)
    assert result.status == DriverStatus.FINISHED and result.laps_completed == 8
    assert not any(row["required"] for row in samples)
    assert not sim.event_manager.mandatory_wet_tires
    assert sim.event_manager.get_control_schedule_history()[0]["reason"] == "race_finished"


@pytest.mark.parametrize("wetness", [0., .3, .8])
@pytest.mark.parametrize("limited", [False, True])
@pytest.mark.parametrize("external_clock", [False, True])
def test_free_wet_selection_matches_independent_physical_schedule_enumeration(
    wetness, limited, external_clock,
):
    driver = Driver(id="A", name="A", team_id="A", consistency=1.)
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="Synthetic", total_laps=4,
                  base_lap_time=90., pit_lane_delta=3.)
    weather = Weather(track_wetness=wetness, rain_intensity=wetness, change_probability=0.)
    records = [{"id": "M", "compound": "medium"},
               {"id": "W1", "compound": "wet", "age": 4},
               {"id": "W2", "compound": "wet", "age": 1},
               {"id": "I", "compound": "intermediate"}]
    if limited:
        records[2]["remaining_laps"] = 1
    inventory = TireInventory.from_sets(records)
    inventory.fit("M")
    delay = track.pit_lane_delta + expected_stationary_time(car)
    clock = (StrategyWeatherClock((0., 90., 180., 270.), 25., 90., 4, delay, delay)
             if external_clock else None)
    decision = plan_inventory_strategy(driver, car, track, weather, inventory, 1,
        free_fit=True, required_wet_tires=True, remaining_stops=3,
        current_lap_time_modifier=1.4, active_aero_enabled=False, weather_clock=clock)
    physics = LapSimulator(np.random.default_rng(0))
    expected, selected = inf, None
    ids = list(inventory.sets)
    for sequence in product(ids, repeat=4):
        if not sequence[0].startswith("W"):
            continue
        ages = {row["id"]: row.get("age", 0) for row in records}
        uses, total, previous = dict.fromkeys(ids, 0), 0., None
        feasible = True
        for offset, identifier in enumerate(sequence):
            record = next(row for row in records if row["id"] == identifier)
            compound = TireCompound(record["compound"])
            if (uses[identifier] >= record.get("remaining_laps", inf)
                    or offset > 0 and weather.tire_mismatch(compound) == "critical"):
                feasible = False
                break
            if offset > 0 and identifier != previous:
                total += track.pit_lane_delta + expected_stationary_time(car)
            driver.current_tire_laps = ages[identifier]
            total += physics.calculate_lap_time(driver, car, track, TIRE_COMPOUNDS[compound],
                weather, offset + 1, 4, sample_variation=False,
                active_aero_enabled=offset > 0) * (1.4 if offset == 0 else 1.)
            ages[identifier] += 1
            uses[identifier] += 1
            previous = identifier
        if feasible and total < expected:
            expected, selected = total, sequence[0]
    assert decision.pit_now_cost == pytest.approx(expected, abs=1.e-9)
    assert decision.set_id == selected
    assert decision.wait_cost == inf


@pytest.mark.parametrize("intervals", [2, 4])
@pytest.mark.parametrize("limited", [False, True])
def test_known_control_prefix_matches_all_compulsory_wet_and_green_schedules(intervals, limited):
    driver = Driver(id="A", name="A", team_id="A", consistency=1.)
    car = Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="Synthetic", total_laps=4,
                  base_lap_time=90., pit_lane_delta=3.)
    weather = Weather(change_probability=0.)
    records = [{"id": "M", "compound": "medium"},
               {"id": "W1", "compound": "wet", "age": 4},
               {"id": "W2", "compound": "wet", "age": 1}]
    if limited:
        records[2]["remaining_laps"] = 1
    inventory = TireInventory.from_sets(records)
    inventory.fit("M")
    empty = SafetyCarFinishBranch((None,))
    delay = track.pit_lane_delta * .55 + expected_stationary_time(car)
    field = StandardControlContext(1, 50., SafetyCarFinishField(empty, empty),
                                   1.4, True, intervals, delay)
    context = StrategyControlContext(field, 50., delay)
    decision = plan_inventory_strategy(driver, car, track, weather, inventory, 1,
        required_wet_tires=True, remaining_stops=4, control_context=context,
        current_lap_time_modifier=1.4, active_aero_enabled=False,
        tire_warmup={"wet": 3.})
    physics = LapSimulator(np.random.default_rng(0))
    expected, selected = inf, None
    for sequence in product([row["id"] for row in records], repeat=4):
        ages = {row["id"]: row.get("age", 0) for row in records}
        uses, total, previous, feasible = dict.fromkeys(ages, 0), 0., "M", True
        for offset, identifier in enumerate(sequence):
            record = next(row for row in records if row["id"] == identifier)
            compound = TireCompound(record["compound"])
            controlled = offset < intervals
            if (uses[identifier] >= record.get("remaining_laps", inf)
                    or (compound != TireCompound.WET if controlled else
                        weather.tire_mismatch(compound) == "critical")):
                feasible = False
                break
            if identifier != previous:
                total += track.pit_lane_delta * (.55 if controlled else 1.) \
                    + expected_stationary_time(car) + (3. if compound == TireCompound.WET else 0.)
            driver.current_tire_laps = ages[identifier]
            total += physics.calculate_lap_time(driver, car, track, TIRE_COMPOUNDS[compound],
                weather, offset + 1, 4, sample_variation=False,
                active_aero_enabled=not controlled) * (1.4 if controlled else 1.)
            ages[identifier] += 1
            uses[identifier] += 1
            previous = identifier
        if feasible and total < expected:
            expected, selected = total, sequence[0]
    assert decision.pit_now_cost == pytest.approx(expected, abs=1.e-9)
    assert decision.set_id == selected and decision.wait_cost == inf


@pytest.mark.parametrize("replacement_available", [False, True])
def test_puncture_replacement_remains_wet_and_unavailable_wets_sample_no_service(
    monkeypatch, replacement_available,
):
    sim = RaceSimulator(np.random.default_rng(91))
    sim.event_manager.safety_car_active = sim.event_manager.mandatory_wet_tires = True
    records = [{"id": "damaged", "compound": "wet", "age": 4},
               {"id": "H", "compound": "hard"}]
    if replacement_available:
        records.append({"id": "replacement", "compound": "wet", "age": 2})
    inventory = TireInventory.from_sets(records)
    inventory.fit("damaged")
    inventory.mark_current_unavailable(4)
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.WET].model_copy(deep=True),
                            tire_laps=4, tire_inventory=inventory, force_pit_next_lap=True)
    services = []
    monkeypatch.setattr(sim, "_sample_pit_service", lambda state: services.append(True) or 3.)
    sim._execute_pit_stop(state,
        Track(id="T", name="T", country="Synthetic", total_laps=8, base_lap_time=90.),
        Weather(change_probability=0.), 3)
    if replacement_available:
        assert state.status == DriverStatus.RACING and services == [True]
        assert state.current_tire.compound == TireCompound.WET and state.tire_laps == 2
        assert state.pit_stop_details[-1]["decision_reason"] == "forced_repair"
    else:
        assert state.status == DriverStatus.DNF and not services
        assert not state.pit_stop_details
