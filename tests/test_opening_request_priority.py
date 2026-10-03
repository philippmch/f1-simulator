"""Opening choices preserve executable custom requests before comparing time."""

import importlib.util
from copy import deepcopy
from dataclasses import replace
from math import inf
from pathlib import Path

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation import opening_strategy as opening
from f1sim.simulation import race as race_module
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator, TeamStrategyArchetype
from f1sim.simulation.weather_schedule import WeatherForecastContext

STYLE = TeamStrategyArchetype.BALANCED
Score = opening.OpeningPolicyScore


@pytest.fixture(scope="module")
def harness():
    path = Path(__file__).resolve().parents[1] / "examples/check_weather_openings.py"
    spec = importlib.util.spec_from_file_location("request_priority_harness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(case, harness):
    simulator = RaceSimulator(np.random.default_rng(17))
    args = (Driver(id="A", name="Synthetic", team_id="A"),
            Car(team_id="A", team_name="Synthetic"),
            Track(id="opening", name="Synthetic", country="Synthetic", total_laps=case["laps"],
                  base_lap_time=case["base"], pit_lane_delta=case["lane"]),
            harness.weather_for(case), STYLE,
            simulator.strategy_tuning, simulator.strategy_profiles)
    return args, simulator


@pytest.mark.parametrize("preferred,other", [
    (Score(-10, 2000, 0), Score(-9, 900, -2)),
    (Score(-10, 1200, -2), Score(-10, 900, -1)),
    (Score(-10, 1200, -1.5), Score(-10, 900, -1)),
    (Score(-10, 900, -1), Score(-10, 1200, -1)),
    (Score(-10, 900), Score(-10, 1200)),
    (Score(-10, 1200), Score(inf, inf)),
])
def test_score_order_prioritizes_distance_then_requests_then_time(preferred, other):
    assert preferred < other
    assert preferred <= other
    assert other > preferred
    assert other >= preferred
    assert min((other, preferred)) == preferred
    assert preferred != other
    assert preferred == replace(preferred)
    assert preferred <= replace(preferred)
    assert hash(preferred) == hash(replace(preferred))


def test_score_retains_legacy_constructor_and_rejects_unrelated_ordering():
    assert Score(-10, 900) == Score(-10, 900, 0)
    with pytest.raises(TypeError):
        Score(-10, 900) < (10, 900)


def test_dry_weighted_ties_require_the_same_request_fulfillment(monkeypatch):
    costs = ((TireCompound.SOFT, Score(-60, 10, 0)),
             (TireCompound.MEDIUM, Score(-60, 20, -1)),
             (TireCompound.HARD, Score(-60, 20, -1)))
    monkeypatch.setattr(race_module, "dry_opening_policy_costs", lambda *args, **kwargs: costs)
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="t", name="T", country="T", total_laps=60, base_lap_time=90.)
    simulator = RaceSimulator(np.random.default_rng(19))
    chosen = {simulator._choose_starting_compound(
        STYLE, track, Weather(), driver, car, pit_plan=[dict(lap=4, compound="hard")],
    ) for _ in range(32)}
    assert chosen == {TireCompound.MEDIUM, TireCompound.HARD}


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("pit_plan", [None, [], [dict(lap=10, compound="intermediate")]])
def test_dry_scores_with_known_rain_match_actual_eight_seed_paths(harness, engine, pit_plan):
    case = dict(name="dry_known_rain", water=0., rain=0., laps=20, base=90., lane=22.,
                schedule=[dict(lap=4, rain_intensity=.35), dict(lap=12, rain_intensity=.85)])
    if pit_plan is not None:
        case["pit_plan"] = pit_plan
    args, simulator = inputs(case, harness)
    context = WeatherForecastContext.from_schedule(case["schedule"])
    before = deepcopy((args, context, case, simulator.rng.bit_generator.state))
    scores = opening.dry_opening_policy_costs(*args, forecast_context=context, pit_plan=pit_plan)
    general = dict(opening.opening_policy_costs(*args, forecast_context=context, pit_plan=pit_plan))
    for compound, score in scores:
        rows = [harness.run_race(case, engine, compound, seed) for seed in opening.REACTION_SEEDS]
        assert score.negative_mean_laps == -sum(row["laps_completed"] for row in rows) / len(rows)
        assert score.negative_mean_instructions == -sum(
            item["status"] == "executed"
            for row in rows for item in row.get("pit_plan_history", ())
        ) / len(rows)
        assert score.mean_time == pytest.approx(
            sum(row["total_seconds"] for row in rows) / len(rows), rel=0, abs=1.e-8)
        assert score == general[compound]
    assert (args, context, case, simulator.rng.bit_generator.state) == before


@pytest.mark.parametrize("schedule,seeds", [
    (None, (0,)), ([], (0,)), ([dict(lap=4, rain_intensity=.35)], opening.REACTION_SEEDS),
])
def test_dry_sample_count_includes_prescribed_changes(monkeypatch, harness, schedule, seeds):
    args, _ = inputs(dict(water=0., rain=0., laps=20, base=90., lane=22.), harness)
    calls = []

    def outcome(*values, **options):
        compound, seed = values[-2:]
        calls.append((compound, seed))
        return 20, 1800 + seed

    monkeypatch.setattr(opening, "_policy_path_outcome", outcome)
    context = None if schedule is None else WeatherForecastContext.from_schedule(schedule)
    scores = opening.dry_opening_policy_costs(*args, forecast_context=context)
    assert calls == [(compound, seed) for compound in opening.SLICKS for seed in seeds]
    assert all(score == Score(-20, 1800 + sum(seeds) / len(seeds)) for _, score in scores)


@pytest.mark.parametrize("mode", ["dry", "weather", "inventory"])
def test_score_aggregation_extensions_bypass_warm_native_cache(monkeypatch, harness, mode):
    args, _ = inputs(dict(water=0., rain=0., laps=6, base=90., lane=22.), harness)
    records = [dict(id="S", compound="soft", age=0), dict(id="H", compound="hard", age=0)]
    options = dict(pit_plan=[dict(lap=4, compound="hard")])
    selector = {"dry": opening.dry_opening_policy_costs, "weather": opening.opening_policy_costs,
                "inventory": opening.inventory_opening_policy_costs}[mode]
    values = (*args, records) if mode == "inventory" else args
    cache = {"dry": opening._cached_dry_policy_costs, "weather": opening._cached_policy_costs,
             "inventory": opening._cached_inventory_policy_costs}[mode]
    cache.cache_clear()
    original_scores = selector(*values, **options)
    assert selector(*values, **options) is original_scores
    hits = cache.cache_info().hits
    original = opening._mean_policy_score
    phase = 0

    def score(outcomes):
        return replace(original(outcomes), negative_mean_instructions=-phase)

    monkeypatch.setattr(opening, "_mean_policy_score", score)
    assert all(value.negative_mean_instructions == 0 for _, value in selector(*values, **options))
    phase = 2
    assert all(value.negative_mean_instructions == -2 for _, value in selector(*values, **options))
    assert cache.cache_info().hits == hits


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_executed_requests_do_not_make_an_infeasible_opening_win(harness, engine):
    case = dict(water=0., rain=0., laps=6, base=90., lane=22.)
    args, simulator = inputs(case, harness)
    records = [dict(id="S", compound="soft", age=0), dict(id="H", compound="hard", age=0)]
    plan = [dict(lap=2, compound="hard")]
    schedule = [dict(lap=2, rain_intensity=.9)]
    context = WeatherForecastContext.from_schedule(schedule)
    scores = opening.inventory_opening_policy_costs(*args, records, pit_plan=plan,
                                                   forecast_context=context)
    assert all(score.mean_time == score.negative_mean_laps == inf for _, score in scores)
    assert all(score.negative_mean_instructions == 0 for _, score in scores)
    laps, elapsed, instructions = opening._policy_path_outcome(
        *args, TireCompound.SOFT, 0, tire_inventory=records, opening_set_id="S", pit_plan=plan,
        forecast_context=context, include_instructions=True,
    )
    simulator._infer_team_strategy = lambda *args: STYLE
    simulator.event_manager.process_lap = lambda *args, **kwargs: []
    simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
    simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
    calculate = simulator.lap_simulator.calculate_lap_time

    def mean_lap(*args, **kwargs):
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean_lap
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    execute = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result, = execute([args[0]], {"A": args[1]}, args[2], args[3], ["A"],
                      starting_tires={"A": TireCompound.SOFT}, tire_inventory={"A": records},
                      pit_plans={"A": plan}, weather_schedule=schedule)
    assert result.status == DriverStatus.DNF
    executed = sum(item["status"] == "executed" for item in result.pit_plan_history)
    assert instructions == executed == 1
    assert elapsed == inf
    assert laps == result.laps_completed < case["laps"]
    score = dict(scores)["S"]
    assert score.incomplete_fraction == 1
    assert score.negative_partial_mean_laps == -laps
    assert score.partial_mean_time == pytest.approx(result.total_time, abs=1.e-8)
