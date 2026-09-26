"""Behavioral checks for the optional first-running-lap fitting cost."""

from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.lap import minimum_lap_time
from f1sim.simulation.pit_strategy import plan_dry_stop
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.rain_strategy import plan_rain_stop
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.warmup import parse_tire_warmup_spec, validate_tire_warmup
from f1sim.simulation.weather_strategy import weather_stop_costs


@pytest.mark.parametrize("value", [None, {}, {"soft": 0}, {"soft": 0.0}])
def test_absent_or_zero_profile_normalizes_to_exact_off_state(value):
    assert validate_tire_warmup(value) == {}


def test_profile_normalizes_to_native_seconds_and_discards_zero_entries():
    assert validate_tire_warmup({"soft": 0, "medium": 1}) == {"medium": 1.0}
    assert parse_tire_warmup_spec("soft=0, medium=1.25") == {"medium": 1.25}
    assert parse_tire_warmup_spec("  ") == {}


@pytest.mark.parametrize("value", [
    [], "soft=1", {"unknown": 1}, {1: 1}, {"soft": True},
    {"soft": float("nan")}, {"soft": float("inf")}, {"soft": -0.01},
    {"soft": 60.01},
])
def test_profile_rejects_noncanonical_or_out_of_range_values(value):
    with pytest.raises(ValueError):
        validate_tire_warmup(value)


@pytest.mark.parametrize("spec", [
    "soft=nan", "soft=inf", "soft=60.01", "soft=-1", "soft=1,soft=2",
    "slick=1", "soft", "=1", "soft=",
])
def test_cli_profile_parser_rejects_invalid_entries(spec):
    with pytest.raises(ValueError):
        parse_tire_warmup_spec(spec)


def test_reused_physical_set_keeps_wear_and_pays_for_each_refit_once():
    simulator = RaceSimulator(tire_warmup={"soft": 4.5, "hard": 2.0})
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    state = DriverRaceState(driver, car, 1)
    inventory = TireInventory.from_sets([
        {"id": "S", "compound": "soft", "age": 5},
        {"id": "H", "compound": "hard", "age": 2},
    ])
    simulator._initialize_inventory(state, inventory, inventory.sets["S"])
    state.tire_laps = state.driver.current_tire_laps = 7

    simulator._fit_inventory_tire(state, "H", 3, "pit")
    assert inventory.sets["S"].age == 7
    assert state.fit_lap_pending
    assert simulator._consume_tire_warmup(state) == 2.0
    assert simulator._consume_tire_warmup(state) == 0.0

    # Returning to S restores its conserved age, then creates a new fitting
    # event and therefore a new one-time fee for the first lap after service.
    state.tire_laps = state.driver.current_tire_laps = 4
    simulator._fit_inventory_tire(state, "S", 6, "pit")
    assert state.current_tire.compound == TireCompound.SOFT
    assert state.tire_laps == state.driver.current_tire_laps == 7
    assert inventory.sets["H"].age == 4
    assert state.fit_lap_pending
    assert simulator._consume_tire_warmup(state) == 4.5
    assert simulator._consume_tire_warmup(state) == 0.0


def test_red_flag_retaining_same_physical_set_does_not_create_a_new_fit(monkeypatch):
    simulator = RaceSimulator(tire_warmup={"soft": 4.5})
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1)
    inventory = TireInventory.from_sets([
        {"id": "S", "compound": "soft", "age": 3},
        {"id": "H", "compound": "hard"},
    ])
    simulator._initialize_inventory(state, inventory, inventory.sets["S"])
    state.tire_laps = state.driver.current_tire_laps = 6
    monkeypatch.setattr(simulator, "_plan_inventory",
                        lambda *args, **kwargs: SimpleNamespace(set_id="S"))
    track = Track(id="t", name="T", country="T", total_laps=10, base_lap_time=90)

    assert simulator._refit_inventory_free(state, track, Weather(), 6)
    assert inventory.current_set_id == "S"
    assert inventory.sets["S"].age == state.tire_laps == 6
    assert not state.fit_lap_pending
    assert simulator._consume_tire_warmup(state) == 0.0


def _run_controlled_fitting_race(profile, control, *, floor_case=False, engine="standard"):
    simulator = RaceSimulator(np.random.default_rng(17), tire_warmup=profile)
    driver = Driver(id="A", name="A", team_id="A", skill_rating=1.0)
    car = Car(team_id="A", team_name="A", base_pace=1.0)
    active_aero_zones = []
    if floor_case:
        from f1sim.models.track import ActiveAeroZone

        active_aero_zones = [
            ActiveAeroZone(zone_id=index, sector=1, time_gain=1.0)
            for index in range(1, 11)
        ]
    track = Track(id="t", name="T", country="T", total_laps=2,
                  base_lap_time=90, pit_lane_delta=3,
                  active_aero_zones=active_aero_zones)
    manager = simulator.event_manager
    manager.process_lap = lambda *args, **kwargs: []
    reset = manager.reset

    def reset_with_control():
        reset()
        manager.vsc_active = control == "vsc"
        manager.safety_car_active = control == "sc"

    manager.reset = reset_with_control
    def force_first_stop(state, _states, _track, lap, *_args, **_kwargs):
        if lap != 1:
            return False
        state.dry_pit_proposal = (lap, TireCompound.MEDIUM)
        return True

    simulator._should_pit = force_first_stop

    raw_times = []
    calculate = simulator.lap_simulator.calculate_lap_time

    def record_raw(*args, **kwargs):
        value = calculate(*args, **kwargs)
        tire = kwargs.get("tire", args[3] if len(args) > 3 else None)
        lap_number = kwargs.get("lap_number", args[5] if len(args) > 5 else None)
        if lap_number == 1 and tire.compound == TireCompound.MEDIUM:
            raw_times.append(value)
        return value

    simulator.lap_simulator.calculate_lap_time = record_raw
    if engine == "standard":
        result, = simulator.simulate_race(
            [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
            starting_tires={"A": TireCompound.HARD},
        )
    else:
        result, = ChronologicalRace(simulator).run(
            [driver], {"A": car}, track, Weather(change_probability=0), ["A"],
            starting_tires={"A": TireCompound.HARD},
        )
    return result, raw_times, track


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("control", ["green", "vsc", "sc"])
def test_fitting_cost_is_added_after_control_scaling(control, engine):
    # A fee is absolute elapsed time. VSC's 1.2 and SC's 1.4 modifiers apply
    # to the underlying lap pace, while the fee remains exactly five seconds.
    baseline, _, _ = _run_controlled_fitting_race({}, control, engine=engine)
    fitted, _, _ = _run_controlled_fitting_race(
        {"medium": 5.0}, control, engine=engine,
    )
    assert fitted.pit_laps == baseline.pit_laps == [1]
    assert fitted.total_time - baseline.total_time == pytest.approx(5.0)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_fitting_cost_is_added_after_the_existing_lap_time_floor(engine):
    baseline, baseline_raw, track = _run_controlled_fitting_race(
        {}, "green", floor_case=True, engine=engine,
    )
    fitted, fitted_raw, _ = _run_controlled_fitting_race(
        {"medium": 5.0}, "green", floor_case=True, engine=engine,
    )
    floor = minimum_lap_time(track)
    assert baseline_raw == pytest.approx([floor])
    assert fitted_raw == pytest.approx([floor])
    assert fitted.total_time - baseline.total_time == pytest.approx(5.0)


def test_differential_compound_cost_changes_the_direct_strategy_ranking():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A", tire_degradation_factor=1.5)
    track = Track(id="t", name="T", country="T", total_laps=3,
                  base_lap_time=90, pit_lane_delta=3, tire_stress=1)
    options = dict(tire_age=0, remaining_laps=3, remaining_stops=3,
                   used_compounds={TireCompound.SOFT}, physical_total_laps=3)
    baseline = plan_dry_stop(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT], **options,
    )
    penalized_medium = plan_dry_stop(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.SOFT],
        tire_warmup={"medium": 15.0}, **options,
    )
    assert baseline.compound == TireCompound.MEDIUM
    assert penalized_medium.compound == TireCompound.HARD


def test_differential_compound_cost_changes_red_flag_restart_ranking():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A", tire_degradation_factor=1.5)
    track = Track(id="t", name="T", country="T", total_laps=3,
                  base_lap_time=90, pit_lane_delta=3, tire_stress=1)
    state = DriverRaceState(driver, car, 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.SOFT], tire_laps=0)
    weather = Weather(change_probability=0)
    baseline = RaceSimulator()._choose_red_flag_tire(state, weather, track, 0)
    penalized_soft = RaceSimulator(tire_warmup={"soft": 15.0})._choose_red_flag_tire(
        state, weather, track, 0,
    )
    assert baseline == TireCompound.SOFT
    assert penalized_soft == TireCompound.MEDIUM


def test_weather_stop_projection_charges_current_and_new_fits_once():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=1,
                  base_lap_time=90, pit_lane_delta=1)
    weather = Weather(track_wetness=0.73, change_probability=0)
    current = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    baseline = weather_stop_costs(
        driver, car, track, weather, current, 2, 1,
        traffic_possible=False, current_lap_time_modifier=1.2,
    )
    profiled = weather_stop_costs(
        driver, car, track, weather, current, 2, 1,
        traffic_possible=False, current_lap_time_modifier=1.2,
        tire_warmup={"intermediate": 7.0, "wet": 5.0},
        current_fit_pending=True,
    )
    assert profiled.stay_cost - baseline.stay_cost == pytest.approx(7.0)
    assert profiled.pit_now_cost - baseline.pit_now_cost == pytest.approx(5.0)


def test_rain_stop_projection_charges_fresh_fit_once():
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=1,
                  base_lap_time=90, pit_lane_delta=1)
    current = TIRE_COMPOUNDS[TireCompound.WET]
    weather = Weather(track_wetness=0.5, change_probability=0)
    baseline = plan_rain_stop(driver, car, track, weather, current, 0, 1, 1)
    profiled = plan_rain_stop(driver, car, track, weather, current, 0, 1, 1,
                              tire_warmup={"wet": 6.0})
    assert profiled.pit_now_cost - baseline.pit_now_cost == pytest.approx(6.0)
    assert profiled.wait_cost == pytest.approx(baseline.wait_cost)


def test_rain_stop_projection_charges_recursive_and_pending_fits_once():
    driver = Driver(id="A", name="A", team_id="A", tire_management=0)
    car = Car(team_id="A", team_name="A", tire_degradation_factor=1.5,
              pit_stop_avg=1.5, pit_stop_std=0.1)
    track = Track(id="t", name="T", country="T", total_laps=8,
                  base_lap_time=900, pit_lane_delta=0.1, tire_stress=1)
    weather = Weather(track_wetness=0.5, rain_intensity=0, change_probability=0)
    current = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE]
    common = dict(active_aero_enabled=False)
    baseline = plan_rain_stop(driver, car, track, weather, current, 0, 1, 3, **common)
    profile = {"intermediate": 4.0}
    profiled = plan_rain_stop(
        driver, car, track, weather, current, 0, 1, 3,
        tire_warmup=profile, **common,
    )
    already_fitted = plan_rain_stop(
        driver, car, track, weather, current, 0, 1, 3,
        tire_warmup=profile, current_fit_pending=True, **common,
    )

    # At this horizon the optimal pit-now forecast fits the same compound
    # three times (now and twice more). The waiting forecast also uses three
    # fittings; a currently pending set adds one more absolute fee to every
    # viable wait branch, including a branch that stops before the horizon.
    assert profiled.pit_now_cost - baseline.pit_now_cost == pytest.approx(12.0)
    assert profiled.wait_cost - baseline.wait_cost > 8.0
    assert already_fitted.wait_cost - profiled.wait_cost == pytest.approx(4.0)
