"""Fallback choices preserve real accepted laps across compulsory continuations."""

import runpy
from copy import deepcopy
from math import inf
from pathlib import Path

import pytest
from test_custom_pit_replacements import execution_costs, inputs, snapshot

from f1sim.models import Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.custom_pit_strategy import CustomPitFinishContext
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.tire_inventory import TireInventory
from f1sim.simulation.weather_schedule import WeatherForecastContext

DIAGNOSTIC = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                / "examples" / "check_inventory_continuations.py"))


def drying_inputs(plan=None, *, water=.24, laps=8):
    simulator, state, track = inputs(True, plan or [], laps=laps)
    records = [dict(id="H", compound="hard", remaining_laps=1),
               dict(id="I", compound="intermediate", remaining_laps=4),
               dict(id="W", compound="wet", remaining_laps=1)]
    stock = TireInventory.from_sets(records)
    simulator._initialize_inventory(state, stock, stock.sets["H"])
    state.tire_laps = state.driver.current_tire_laps = state.laps_completed = 1
    state.pit_plan = plan
    simulator.tire_warmup = {}
    return simulator, state, track, Weather(track_wetness=water, change_probability=0)


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("plan", [None, [], [{"lap": 4, "compound": "intermediate"}],
                                 [{"lap": 7, "compound": "wet"}]])
@pytest.mark.parametrize("context", ["own", "external", "scheduled", "sc", "vsc"])
def test_compulsory_fallback_matches_independent_retirement_execution(free, plan, context):
    simulator, state, track, weather = drying_inputs(
        plan, water=.5 if context == "scheduled" else .24)
    simulator.tire_warmup = {"intermediate": 3., "wet": 7.}
    options = dict(free_fit=free, physical_total_laps=40,
                   current_traffic_gaps=(.3, .8), additional_current_stop_cost=11.)
    if context == "external":
        stop = track.pit_lane_delta + expected_stationary_time(state.car)
        options["weather_clock"] = StrategyWeatherClock(
            tuple(80. * i for i in range(7)), 35., 95., 10, stop + 11., stop)
    if context == "scheduled":
        simulator.weather_forecast_context = WeatherForecastContext.from_schedule(
            [{"lap": 4, "rain_intensity": .85}, {"lap": 6, "rain_intensity": 0.}],
            leading_lap=2)
        options["weather_intervals"] = (0, 1, 2, 3, 4, 5, 6)
    simulator.event_manager.safety_car_active = context == "sc"
    simulator.event_manager.vsc_active = context == "vsc"
    before = snapshot(state, simulator)
    oracle_state = deepcopy(state)
    if plan is None:
        oracle_state.pit_plan = []  # Automatic fallback forecasts compulsory stops only.
    expected = execution_costs(simulator, oracle_state, track, weather, 2,
                               allow_incomplete=True, **options)
    best = min(expected.values())
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 2, **options)
    assert best[0] == 1  # Every available pool has fewer laps than this horizon.
    assert choice.cost == inf
    assert choice.laps == -best[1] > 0
    assert choice.instructions == 0
    assert choice.partial_time == pytest.approx(best[3], abs=1.e-8)
    assert expected[choice.set_id] == pytest.approx(best, abs=1.e-8)
    assert simulator._inventory_immediate_set(state, track, weather, 2, **options) == choice.set_id
    assert snapshot(state, simulator) == before


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("plan", [None, []])
def test_drying_fallback_keeps_the_extra_lap_without_sampling_service(free, plan, monkeypatch):
    simulator, state, track, weather = drying_inputs(plan)
    before = snapshot(state, simulator)

    def forbid_sampling(*args, **kwargs):
        raise AssertionError("Strategy sampled pit service")

    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", forbid_sampling)
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 2, free_fit=free)
    assert choice.set_id == "W" and choice.laps == 5 and choice.cost == inf
    assert snapshot(state, simulator) == before
    if free:
        assert simulator._refit_inventory_free(state, track, weather, 1)
        assert state.tire_inventory.current_set_id == "W"
        assert state.pit_stops == 0
        assert state.tire_set_history[-1]["kind"] == "red_flag"
    else:
        assert simulator._prepare_inventory_pit(state, track, weather, 2)
        assert state.inventory_pit_proposal == (2, "W")


@pytest.mark.parametrize("free", [False, True])
def test_illegal_final_dry_crossing_is_not_counted_as_survival(free):
    simulator, state, track, _ = drying_inputs([])
    stock = TireInventory.from_sets([dict(id="old", compound="soft", age=40),
                                     dict(id="fresh", compound="soft")])
    simulator._initialize_inventory(state, stock, stock.sets["old"])
    state.tire_laps = state.driver.current_tire_laps = 41
    state.tire_compound_history = ["soft"]
    weather = Weather(change_probability=0)
    expected = execution_costs(simulator, deepcopy(state), track, weather, 2,
                               free_fit=free, allow_incomplete=True)
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 2, free_fit=free)
    best = min(expected.values())
    assert choice.set_id == "fresh"
    assert choice.cost == inf and choice.laps == 6 and choice.instructions == 0
    assert best[1] == -6
    assert choice.partial_time == pytest.approx(best[3], abs=1.e-8)


@pytest.mark.parametrize("free", [False, True])
def test_shorter_legal_timed_finish_beats_a_longer_retirement(free):
    simulator, state, track, weather = drying_inputs([], water=.23)
    state.laps_completed = 2
    stock = TireInventory.from_sets([
        dict(id="old", compound="hard"), dict(id="I", compound="intermediate", remaining_laps=3),
        dict(id="W", compound="wet", age=1000, remaining_laps=2)])
    simulator._initialize_inventory(state, stock, stock.sets["old"])
    stock.mark_current_unavailable(state.tire_laps)
    simulator.tire_warmup = {"wet": 60.}
    finish = CustomPitFinishContext(now=6890., time_limit_seconds=7200.)
    options = dict(free_fit=free, finish_context=finish)
    expected = execution_costs(simulator, deepcopy(state), track, weather, 3,
                               allow_incomplete=True, **options)
    # The integration receives the observed clock through race state.
    state.strategy_finish_context = finish
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 3, free_fit=free)
    assert expected["W"][0] == 0 and expected["W"][1] == -2
    assert expected["I"][0] == 1 and expected["I"][1] == -3
    assert choice.set_id == "W" and choice.laps == 2
    assert choice.cost == pytest.approx(expected["W"][3], abs=1.e-8)
    assert choice.partial_time == inf


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("plan", list(DIAGNOSTIC["PLANS"].values()))
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("warmup", [None, {"intermediate": 3., "wet": 7.}])
def test_actual_race_keeps_the_sixth_lap_and_accounts_for_every_physical_use(
    engine, free, plan, reverse, warmup,
):
    run = DIAGNOSTIC["run_continuation"]
    options = dict(free=free, plan=plan, reverse=reverse, warmup=warmup)
    result = run(engine, **options)
    wet = run(engine, forced_set="W", **options)
    intermediate = run(engine, forced_set="I", **options)
    assert result["status"] == wet["status"] == intermediate["status"] == "dnf"
    assert result["laps_completed"] == wet["laps_completed"] == 6
    assert intermediate["laps_completed"] == 5
    assert result["total_seconds"] == pytest.approx(wet["total_seconds"], abs=1.e-8)
    assert [s["set_id"] for s in result["tire_set_history"]] == ["H", "W", "I"]
    assert sum(s["laps_used"] for s in result["tire_set_history"]) == 6
    assert all(s["remaining_laps"] == 0 for s in result["tire_inventory"])
    assert result["pit_laps"] == ([3] if free else [2, 3])
    assert result["paid_stops"] == (1 if free else 2)
    assert result["tire_set_history"][1]["kind"] == ("red_flag" if free else "pit")


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_executable_request_keeps_its_compound_even_when_it_sacrifices_distance(engine):
    result = DIAGNOSTIC["run_continuation"](engine, plan=[dict(lap=2, compound="intermediate")])
    assert result["laps_completed"] == 5
    assert result["pit_laps"] == [2]
    assert result["tire_set_history"][1]["set_id"] == "I"
    assert result["pit_plan_history"][0]["status"] == "executed"


def test_diagnostic_reports_actual_dnf_paths_and_zero_distance_and_time_gaps():
    rows = DIAGNOSTIC["compare_continuations"]()
    assert len(rows) == 16
    assert {row["engine"] for row in rows} == {"standard", "chronological"}
    assert {row["free_fit"] for row in rows} == {True, False}
    for row in rows:
        assert row["best_alternative"] == "W"
        assert row["distance_gap"] == 0
        assert row["time_gap_seconds"] == pytest.approx(0, abs=1.e-8)
        assert row["selected"]["laps_completed"] == 6
        assert row["selected"]["status"] == "dnf"


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("identifiers", [("Z", "A"), ("A", "Z")])
def test_equal_partial_choices_keep_physical_input_order(free, identifiers):
    simulator, state, track, weather = drying_inputs([])
    records = [dict(id="H", compound="hard", remaining_laps=1),
               dict(id="I", compound="intermediate", remaining_laps=4),
               *(dict(id=key, compound="wet", remaining_laps=1) for key in identifiers)]
    stock = TireInventory.from_sets(records)
    simulator._initialize_inventory(state, stock, stock.sets["H"])
    state.tire_laps = state.driver.current_tire_laps = 1
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 2, free_fit=free)
    assert choice.set_id == identifiers[0]
    expected = execution_costs(simulator, deepcopy(state), track, weather, 2,
                               free_fit=free, allow_incomplete=True)
    assert choice.cost == inf and choice.laps == 6 and choice.instructions == 0
    assert choice.partial_time == pytest.approx(min(expected.values())[3], abs=1.e-8)


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("plan", [None, []])
def test_zero_legal_laps_still_retire_before_service(free, plan, monkeypatch):
    simulator, state, track, _ = drying_inputs(plan)
    stock = TireInventory.from_sets([dict(id="old", compound="soft"),
                                     dict(id="spare", compound="soft")])
    simulator._initialize_inventory(state, stock, stock.sets["old"])
    state.laps_completed = 7
    state.tire_laps = state.driver.current_tire_laps = 7
    before_history = deepcopy(state.tire_set_history)

    def forbid_sampling(*args, **kwargs):
        raise AssertionError("Unavailable final compound sampled service")

    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", forbid_sampling)
    choice = simulator._custom_plan_replacement_choice(
        state, track, Weather(change_probability=0), 8, free_fit=free)
    assert choice.set_id is None and choice.cost == choice.partial_time == inf
    assert choice.laps == choice.instructions == 0
    if free:
        assert not simulator._refit_inventory_free(state, track, Weather(), 7)
    else:
        assert not simulator._prepare_inventory_pit(state, track, Weather(), 8)
    assert state.status.value == "dnf" and state.pit_stops == 0
    assert state.tire_set_history == before_history


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("plan", [None, []])
def test_compulsory_retirement_paths_survive_workers_and_saved_replay(tmp_path, engine, plan):
    from test_inventory_analysis import save_result

    from f1sim.analysis.montecarlo import MonteCarloRunner
    from f1sim.analysis.replay import replay_saved_simulation
    from f1sim.models import TireCompound

    simulator, state, track, weather = drying_inputs(plan)
    runner = MonteCarloRunner([state.driver], {"A": state.car}, track, weather,
                              race_engine=engine, seed=7, starting_tires={"A": TireCompound.HARD},
                              tire_inventory={"A": list(DIAGNOSTIC["POOL"])},
                              **({"pit_plans": {"A": plan}} if plan is not None else {}))
    serial = runner.run(2, parallel=False)
    parallel = runner.run(2, parallel=True, max_workers=2)
    assert serial.race_results == parallel.race_results
    assert serial.input_snapshot == parallel.input_snapshot
    assert serial.input_snapshot["schema_version"] == 9
    assert any([stint["set_id"] for stint in row.tire_set_history] == ["H", "W", "I"]
               for race in serial.race_results for row in race)
    assert all(row.status.value == "dnf" and row.laps_completed <= 6
               for race in serial.race_results for row in race)
    replay = replay_saved_simulation(save_result(tmp_path, serial), simulation=2)
    assert replay.race_results[0] == serial.race_results[1]
    assert replay.input_snapshot == serial.input_snapshot


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_failed_free_paths_get_no_credit_for_an_extra_requested_service(engine):
    run = DIAGNOSTIC["run_continuation"]
    options = dict(free=True, plan=[dict(lap=2, compound="intermediate")])
    result = run(engine, **options)
    extra_service = run(engine, forced_set="W", **options)
    assert result["laps_completed"] == extra_service["laps_completed"] == 5
    assert result["status"] == extra_service["status"] == "dnf"
    assert result["total_seconds"] < extra_service["total_seconds"]
    assert result["pit_plan_history"][0]["status"] == "skipped"
    assert extra_service["pit_plan_history"][0]["status"] == "executed"
    wet_fit, = [row for row in extra_service["tire_set_history"] if row["set_id"] == "W"]
    assert wet_fit["laps_used"] == 0 and wet_fit["remaining_laps_at_end"] == 1


@pytest.mark.parametrize("free", [False, True])
@pytest.mark.parametrize("plan", [None, []])
def test_partial_continuation_cost_hooks_cannot_mutate_live_or_shared_models(
    monkeypatch, free, plan,
):
    simulator, state, track, weather = drying_inputs(plan)
    for compound, tire in TIRE_COMPOUNDS.items():
        monkeypatch.setitem(TIRE_COMPOUNDS, compound, tire.model_copy(deep=True))
    calculate = LapSimulator.calculate_lap_time

    def mutate(self, driver, car, track, tire, weather, *args, **kwargs):
        driver.total_race_time = 12345.
        car.pit_stop_avg = 99.
        track.pit_lane_delta = 999.
        tire.initial_grip = .8
        weather.track_wetness = .1
        return calculate(self, driver, car, track, tire, weather, *args, **kwargs)

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", mutate)
    before = (snapshot(state, simulator), deepcopy(track), deepcopy(weather),
              deepcopy(TIRE_COMPOUNDS))
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 2, free_fit=free)
    assert choice.cost == inf and choice.partial_time < inf and choice.laps > 0
    assert (snapshot(state, simulator), track, weather, TIRE_COMPOUNDS) == before


@pytest.mark.parametrize("free", [False, True])
def test_many_identical_sets_keep_every_permitted_lap_and_the_first_identity(free):
    from f1sim.models import TireCompound

    simulator, state, track, _ = drying_inputs([], laps=20)
    stock = TireInventory.from_sets([
        dict(id="old", compound="hard", remaining_laps=1),
        *(dict(id=f"S{i}", compound="soft", remaining_laps=1) for i in range(12))])
    simulator._initialize_inventory(state, stock, stock.sets["old"])
    state.tire_laps = state.driver.current_tire_laps = 1
    weather = Weather(change_probability=0)
    before = snapshot(state, simulator)
    driver = state.driver.model_copy(deep=True)
    driver.current_tire_laps = 0
    physics = LapSimulator()
    running = sum(physics.calculate_lap_time(
        driver, state.car, track, TIRE_COMPOUNDS[TireCompound.SOFT], weather, lap, 20,
        sample_variation=False) for lap in range(2, 14))
    # Each of twelve distinct one-lap sets is fitted once; there is no usable
    # removed set left to run a thirteenth lap. Only the first fit can be free.
    service = (12 - int(free)) * (track.pit_lane_delta + expected_stationary_time(state.car))
    choice = simulator._custom_plan_replacement_choice(state, track, weather, 2, free_fit=free)
    assert choice.set_id == "S0" and choice.laps == 12 and choice.cost == inf
    assert choice.partial_time == pytest.approx(running + service, abs=1.e-8)
    assert snapshot(state, simulator) == before
