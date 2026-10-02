"""Starting tyres are priced against the custom policy that will be executed."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation import opening_strategy as opening
from f1sim.simulation import race as race_module
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype
from f1sim.simulation.weather_schedule import WeatherForecastContext

STYLE = TeamStrategyArchetype.BALANCED
SELECTORS = ("dry", "weather", "inventory")


@pytest.fixture(scope="module")
def harness():
    path = Path(__file__).resolve().parents[1] / "examples/check_weather_openings.py"
    spec = importlib.util.spec_from_file_location("custom_opening_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(laps=60, base=90.):
    simulator = RaceSimulator(np.random.default_rng(4))
    args = (Driver(id="D", name="Synthetic", team_id="T"),
            Car(team_id="T", team_name="Synthetic", tire_degradation_factor=1.5),
            Track(id="custom", name="Synthetic", country="Synthetic", total_laps=laps,
                  base_lap_time=base, pit_lane_delta=1., tire_stress=1.),
            Weather(change_probability=0), STYLE,
            simulator.strategy_tuning, simulator.strategy_profiles)
    records = [{"id": "S", "compound": "soft", "age": 5},
               {"id": "M", "compound": "medium", "age": 0},
               {"id": "H", "compound": "hard", "age": 0}]
    return args, records, simulator


def selector(name, args, records, **options):
    if name == "inventory":
        return opening.inventory_opening_policy_costs(*args, records, **options)
    function = (opening.dry_opening_policy_costs if name == "dry"
                else opening.opening_policy_costs)
    return function(*args, **options)


def cache_for(name):
    return {"dry": opening._cached_dry_policy_costs,
            "weather": opening._cached_policy_costs,
            "inventory": opening._cached_inventory_policy_costs}[name]


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("case_index", range(20))
def test_custom_opening_scores_match_every_executed_alternative(harness, engine, case_index):
    case = deepcopy(harness.CUSTOM_PLAN_CASES[case_index])
    before = deepcopy(case)
    row, = harness.compare_openings(cases=(case,), engines=(engine,))
    driver = Driver(id="D", name="Other identity", team_id="T", current_tire_laps=99, dnf=True)
    car = Car(team_id="T", team_name="Other identity",
              **({"tire_degradation_factor": case["degradation"]} if "degradation" in case else {}))
    track = Track(id="opening", name="Synthetic", country="Synthetic", total_laps=case["laps"],
                  base_lap_time=case["base"], pit_lane_delta=case["lane"],
                  **({"tire_stress": case["stress"]} if "stress" in case else {}))
    simulator = RaceSimulator(np.random.default_rng(17))
    args = (driver, car, track, harness.weather_for(case), STYLE,
            simulator.strategy_tuning, simulator.strategy_profiles)
    options = dict(pit_plan=case["pit_plan"], tire_warmup=case.get("warmup"))
    if "schedule" in case:
        options["forecast_context"] = WeatherForecastContext.from_schedule(case["schedule"])
    records = case.get("inventory", [])
    mode = ("inventory" if records else "weather" if "schedule" in case
            or case["water"] >= .08 or case["rain"] >= .15 else "dry")
    input_before = deepcopy((args, simulator.rng.bit_generator.state))
    scores = selector(mode, args, records, **options)
    for candidate, score in scores:
        if records:
            item = next(item for item in records if item["id"] == candidate)
            label = f"{item['compound']}@{item.get('age', 0)}"
        else:
            label = candidate.value
        actual = row["alternatives"][label]
        assert score.negative_mean_laps == -actual["mean_laps"]
        assert score.negative_mean_instructions == -actual["mean_executed_instructions"]
        assert score.mean_time == pytest.approx(actual["mean_seconds"], rel=0, abs=1.e-8)
    assert row["mean_distance_gap"] == 0
    assert row["mean_instruction_gap"] == 0
    assert row["mean_time_gap_seconds"] == pytest.approx(0., abs=1.e-8)
    assert (args, simulator.rng.bit_generator.state) == input_before
    assert case == before
    if case["name"] in {"dry_no_elective", "finite_no_elective"}:
        assert row["selected"]["compounds"][0] == "hard"
        assert row["alternatives"]["medium" if not records else "medium@4"]["mean_seconds"] - (
            row["selected"]["total_seconds"]
        ) > 100.
    if case["name"] == "dry_early_hard":
        assert row["selected"]["compounds"][0] == "soft"
        assert row["selected"]["pit_laps"] == [2]
    if case["name"] == "unsafe_requested_slick":
        assert row["selected"]["pit_plan_history"][0]["status"] == "skipped"
    if case["name"] == "timed_unreached_plan":
        assert row["selected"]["laps_completed"] < case["laps"]
        assert row["selected"]["pit_plan_history"][0]["status"] == "not_reached"
    if case["name"].startswith("reserve_requested_wet"):
        assert row["selected"]["opening"] == "intermediate@0"
        assert row["selected"]["pit_plan_history"][0]["status"] == "executed"
        assert row["selected"]["total_seconds"] > row["alternatives"]["wet@0"]["mean_seconds"]
    if case["name"] == "reuse_requested_wet":
        assert [item["status"] for item in row["selected"]["pit_plan_history"]] == [
            "executed", "executed", "executed"]
    if case["name"] == "two_wet_sets":
        assert row["alternatives"]["wet@0"]["mean_executed_instructions"] == 1
        fitted = harness.run_race(case, engine, TireCompound.WET)
        assert fitted["pit_plan_history"][0]["status"] == "executed"
        assert [item["set_id"] for item in fitted["tire_set_history"]] == [
            "wet-first", "wet-second"]
    if case["name"] == "reserve_wet_before_timed_finish":
        assert row["selected"]["opening"] == "intermediate@0"
        assert row["selected"]["laps_completed"] < case["laps"]
        assert [item["status"] for item in row["selected"]["pit_plan_history"]] == [
            "executed", "not_reached"]
    if case.get("warmup"):
        assert all(item["lap"] != 1 for item in row["selected"]["warmup_laps"])


@pytest.mark.parametrize("mode", SELECTORS)
def test_plan_cache_keeps_automatic_empty_and_changed_schedules_distinct(mode):
    args, records, simulator = inputs()
    cache = cache_for(mode)
    cache.cache_clear()
    original = selector(mode, args, records)
    assert selector(mode, args, records, pit_plan=None) is original
    empty = selector(mode, args, records, pit_plan=[])
    assert empty != original
    plan = [{"lap": 4, "compound": "hard"}]
    planned = selector(mode, args, records, pit_plan=plan)
    before = deepcopy((args, records, plan, simulator.rng.bit_generator.state))
    assert selector(mode, args, records, pit_plan=deepcopy(plan)) is planned
    assert (args, records, plan, simulator.rng.bit_generator.state) == before
    assert cache.cache_info().misses == 3
    plan[0]["lap"] = 5
    assert selector(mode, args, records, pit_plan=plan) is not planned
    plan[0]["compound"] = "soft"
    selector(mode, args, records, pit_plan=plan)
    assert cache.cache_info().misses == 5
    assert (args, records, simulator.rng.bit_generator.state) == (before[0], before[1], before[3])
    assert cache.cache_info().maxsize == 128


@pytest.mark.parametrize("mode", SELECTORS)
@pytest.mark.parametrize("helper", ["initializer", "decision"])
def test_custom_policy_extensions_bypass_warmed_opening_cache(monkeypatch, mode, helper):
    args, records, _ = inputs(laps=12)
    cache = cache_for(mode)
    cache.cache_clear()
    automatic = selector(mode, args, records)
    planned = selector(mode, args, records, pit_plan=[])
    assert planned != automatic
    hits = cache.cache_info().hits
    if helper == "initializer":
        original = opening.initialize_pit_plan_state
        monkeypatch.setattr(opening, "initialize_pit_plan_state", lambda state, plan:
                            original(state, None))
    else:
        monkeypatch.setattr(RaceSimulator, "_custom_pit_plan_decision", lambda *args: None)
    observed = selector(mode, args, records, pit_plan=[])
    assert [candidate for candidate, _ in observed] == [candidate for candidate, _ in automatic]
    for (_, score), (_, expected) in zip(observed, automatic, strict=True):
        assert score.negative_mean_laps == expected.negative_mean_laps
        assert score.mean_time == pytest.approx(expected.mean_time, rel=0, abs=1.e-8)
    assert cache.cache_info().hits == hits


@pytest.mark.parametrize("mode", SELECTORS)
@pytest.mark.parametrize("plan", [{}, [{}], [{"lap": True, "compound": "soft"}],
                                 [{"lap": 7, "compound": "soft"}],
                                 [{"lap": 2, "compound": "hard", "set_id": "H"}],
                                 [{"lap": 4, "compound": "hard"},
                                  {"lap": 3, "compound": "soft"}]])
def test_direct_opening_scores_reject_invalid_plans_before_projection(monkeypatch, mode, plan):
    args, records, simulator = inputs(laps=6)
    before = deepcopy((args, records, plan, simulator.rng.bit_generator.state))
    monkeypatch.setattr(opening, "_policy_path_outcome", lambda *args, **kwargs:
                        pytest.fail("Invalid plan reached race projection"))
    with pytest.raises(ValueError):
        selector(mode, args, records, pit_plan=plan)
    assert (args, records, plan, simulator.rng.bit_generator.state) == before


def test_finite_opening_scores_reject_requests_absent_from_the_pool():
    args, records, _ = inputs(laps=6)
    with pytest.raises(ValueError, match="absent"):
        opening.inventory_opening_policy_costs(*args, records,
                                               pit_plan=[{"lap": 3, "compound": "wet"}])


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
def test_custom_plan_applies_only_to_the_listed_drivers_opening(engine, finite):
    args, records, simulator = inputs()
    driver, car, track, weather = args[:4]
    other = driver.model_copy(update={"id": "E", "name": "Other", "team_id": "U"}, deep=True)
    other_car = car.model_copy(update={"team_id": "U", "team_name": "Other"}, deep=True)
    simulator._infer_team_strategy = lambda *args: STYLE
    simulator.event_manager.process_lap = lambda *args, **kwargs: []
    simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
    simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
    execute = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator).run)
    options = dict(pit_plans={"D": []})
    if finite:
        options["tire_inventory"] = {"D": records}
    rows = execute([driver, other], {"T": car, "U": other_car}, track, weather, ["D", "E"],
                   **options)
    rows = {row.driver_id: row for row in rows}
    assert rows["D"].strategy[0] == "hard"
    assert rows["D"].pit_plan_history == []
    assert rows["E"].strategy[0] in {"soft", "medium"}
    assert rows["E"].pit_plan_history is None


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
def test_explicit_used_opening_remains_authoritative_with_custom_plan(
    monkeypatch, harness, engine, finite,
):
    for module, name in ((race_module, "dry_opening_policy_costs"),
                         (race_module, "opening_policy_costs"),
                         (opening, "inventory_opening_policy_costs")):
        monkeypatch.setattr(module, name, lambda *args, **kwargs:
                            pytest.fail("Explicit opening was rescored"))
    case = dict(name="explicit", water=0., rain=0., laps=12, base=90., lane=22., pit_plan=[])
    if finite:
        case["inventory"] = list(harness.CUSTOM_POOL)
    row = harness.run_race(case, engine, TireCompound.SOFT, seed=4, age=5)
    assert row["compounds"][0] == "soft"
    assert row["pit_plan_history"] == []
    if finite:
        assert row["tire_set_history"][0]["set_id"] == "soft-used"
        assert row["tire_set_history"][0]["age_at_fit"] == 5


def test_custom_plan_opening_diagnostic_is_reproducible():
    script = Path(__file__).resolve().parents[1] / "examples/check_weather_openings.py"
    command = [sys.executable, str(script), "--custom-plans"]
    first = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    second = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    assert first.stdout == second.stdout
    rows = json.loads(first.stdout)
    assert len(rows) == 40
    assert {row["engine"] for row in rows} == {"standard", "chronological"}
    assert all(row["reaction_seeds"] == list(range(8)) for row in rows)
    assert all(row["mean_distance_gap"] == row["mean_instruction_gap"] == 0
               and abs(row["mean_time_gap_seconds"]) < 1.e-7
               for row in rows)
