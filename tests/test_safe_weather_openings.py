"""Weather openings compare every safe set against executed later policies."""

import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather, WeatherCondition
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import opening_strategy
from f1sim.simulation import race as race_module
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype
from f1sim.simulation.weather_schedule import WeatherForecastContext


def models():
    return (Driver(id="D", name="Driver", team_id="T"), Car(team_id="T", team_name="Team"),
            Track(id="opening", name="Opening", country="Synthetic", total_laps=12,
                  base_lap_time=90., pit_lane_delta=22.))


@pytest.fixture(scope="module")
def opening_harness():
    # Examples are executable scripts, not part of the installed package.
    # Loading by file works under both pytest and python -m pytest.
    path = Path(__file__).resolve().parents[1] / "examples" / "check_weather_openings.py"
    specification = importlib.util.spec_from_file_location("weather_opening_harness", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("water,rain,eligible", [
    (0., 0., {"soft", "medium", "hard"}),
    (.079, .149, {"soft", "medium", "hard"}),
    (.08, 0., {"soft", "medium", "hard", "intermediate"}),
    (.2, .29, {"soft", "medium", "hard", "intermediate", "wet"}),
    (.45, .35, {"soft", "medium", "hard", "intermediate", "wet"}),
    (.46, .35, {"intermediate", "wet"}),
    (.8, .85, {"intermediate", "wet"}),
])
@pytest.mark.parametrize("condition", [WeatherCondition.CLOUDY, WeatherCondition.HEAVY_RAIN])
@pytest.mark.parametrize("context", [None, WeatherForecastContext.from_schedule([]),
                                     WeatherForecastContext.from_schedule(
                                         [{"lap": 4, "rain_intensity": .8}],
                                     )])
def test_opening_scores_price_all_and_only_noncritical_sets_for_every_reaction_seed(
    monkeypatch, water, rain, eligible, condition, context,
):
    driver, car, track = models()
    weather = Weather(condition=condition, track_wetness=water, rain_intensity=rain)
    simulator = RaceSimulator(np.random.default_rng(17))
    calls = []

    def outcome(*args, **kwargs):
        compound, seed = args[-2:]
        calls.append((compound.value, seed))
        assert kwargs.get("forecast_context") is context
        return 12, 1000. + seed

    monkeypatch.setattr(opening_strategy, "_policy_path_outcome", outcome)
    before = deepcopy((driver, car, track, weather, simulator.rng.bit_generator.state))
    scores = opening_strategy.opening_policy_costs(
        driver, car, track, weather, TeamStrategyArchetype.BALANCED,
        simulator.strategy_tuning, simulator.strategy_profiles, forecast_context=context,
    )
    assert {compound.value for compound, _ in scores} == eligible
    assert len(calls) == len(eligible) * 8
    assert set(calls) == {(compound, seed) for compound in eligible for seed in range(8)}
    assert all(score == opening_strategy.OpeningPolicyScore(-12, 1003.5)
               for _, score in scores)
    assert (driver, car, track, weather, simulator.rng.bit_generator.state) == before


@pytest.mark.parametrize("style", list(TeamStrategyArchetype))
def test_drying_wet_start_prefers_faster_safe_intermediates_without_race_rng(style):
    driver, car, track = models()
    weather = Weather(track_wetness=.75, rain_intensity=0., change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(42))
    before = deepcopy(simulator.rng.bit_generator.state)
    scores = dict(opening_strategy.opening_policy_costs(
        driver, car, track, weather, style, simulator.strategy_tuning, simulator.strategy_profiles,
    ))
    assert scores[TireCompound.INTERMEDIATE] < scores[TireCompound.WET]
    assert scores[TireCompound.WET].mean_time - scores[TireCompound.INTERMEDIATE].mean_time > 3.
    assert simulator._choose_starting_compound(style, track, weather, driver, car) == (
        TireCompound.INTERMEDIATE
    )
    assert simulator.rng.bit_generator.state == before
    # Without physics inputs, the existing numeric weather fallback remains available.
    assert simulator._choose_starting_compound(style, track, weather) == TireCompound.WET


def test_safe_full_wet_can_win_when_live_tyre_physics_changes_after_cache_warming(
    monkeypatch, opening_harness,
):
    driver, car, track = models()
    weather = Weather(track_wetness=.45, rain_intensity=.35, change_probability=0)
    simulator = RaceSimulator(np.random.default_rng(42))
    style = TeamStrategyArchetype.BALANCED
    assert simulator._choose_starting_compound(style, track, weather, driver, car) == (
        TireCompound.INTERMEDIATE
    )
    before = deepcopy(simulator.rng.bit_generator.state)
    monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.INTERMEDIATE,
                        TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(
                            update={"degradation_rate": .1, "cliff_threshold": 1,
                                    "cliff_multiplier": 10.},
                        ))
    scores = dict(opening_strategy.opening_policy_costs(
        driver, car, track, weather, style, simulator.strategy_tuning, simulator.strategy_profiles,
    ))
    assert min(scores, key=scores.get) == TireCompound.WET
    assert simulator._choose_starting_compound(style, track, weather, driver, car) == (
        TireCompound.WET
    )
    assert simulator.rng.bit_generator.state == before
    case = dict(name="changed_physics", water=.45, rain=.35, laps=12, base=90., lane=22.)
    for row in opening_harness.compare_openings(cases=(case,)):
        assert row["selected"]["compounds"][0] == row["best_opening"] == "wet"
        assert row["mean_distance_gap"] == 0
        assert row["mean_time_gap_seconds"] == pytest.approx(0., abs=1.e-8)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("warmup", [None, {"intermediate": 8., "wet": 35., "soft": 2.}])
def test_automatic_wet_opening_matches_every_safe_executed_policy(engine, warmup, opening_harness):
    case = dict(name="drying", water=.75, rain=0., laps=12, base=90., lane=22.)
    if warmup:
        case["warmup"] = warmup
    row, = opening_harness.compare_openings(cases=(case,), engines=(engine,))
    assert row["reaction_seeds"] == list(range(8))
    assert set(row["alternatives"]) == {"intermediate", "wet"}
    assert row["selected"]["compounds"][0] == row["best_opening"] == "intermediate"
    assert row["mean_distance_gap"] == 0
    assert row["mean_time_gap_seconds"] == pytest.approx(0., abs=1.e-8)
    assert 1 not in row["selected"]["pit_laps"]
    assert all(cost["lap"] != 1 for cost in row["selected"]["warmup_laps"])


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_timed_weather_opening_scores_match_completed_distance_in_execution(
    engine, opening_harness,
):
    case = dict(name="timed", water=.45, rain=.35, laps=10, base=1800., lane=22.)
    row, = opening_harness.compare_openings(cases=(case,), engines=(engine,))
    assert row["selected"]["race_time_limited"]
    assert row["selected"]["laps_completed"] < case["laps"]
    assert row["mean_distance_gap"] == 0
    assert row["mean_time_gap_seconds"] == pytest.approx(0., abs=1.e-8)


@pytest.mark.parametrize("finite", [False, True])
def test_explicit_opening_compound_and_age_bypass_automatic_weather_comparison(monkeypatch, finite):
    driver, car, track = models()
    track.total_laps = 3
    simulator = RaceSimulator(np.random.default_rng(17))
    ages = []
    running = simulator.lap_simulator.calculate_lap_time

    def observed_lap(*args, **kwargs):
        age = (kwargs["driver"] if "driver" in kwargs else args[0]).current_tire_laps
        ages.append(age)
        return running(*args, **kwargs)

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", observed_lap)
    monkeypatch.setattr(race_module, "opening_policy_costs",
                        lambda *args, **kwargs: pytest.fail("Explicit opening was rescored"))
    options = dict(starting_tires={"D": TireCompound.WET}, starting_tire_ages={"D": 3},
                   pit_plans={"D": []})
    if finite:
        options["tire_inventory"] = {"D": [{"id": "wet-used", "compound": "wet", "age": 3}]}
    result, = simulator.simulate_race(
        [driver], {"T": car}, track,
        Weather(track_wetness=.75, rain_intensity=.7, change_probability=0), ["D"], **options,
    )
    assert result.strategy[0] == "wet"
    assert ages[0] == 3


def test_weather_opening_diagnostic_is_reproducible_and_checks_all_executed_safe_sets():
    script = Path(__file__).resolve().parents[1] / "examples" / "check_weather_openings.py"
    command = [sys.executable, str(script), "--engine", "both"]
    first = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    second = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    assert first.stdout == second.stdout
    rows = json.loads(first.stdout)
    assert len(rows) == 14
    assert {row["engine"] for row in rows} == {"standard", "chronological"}
    for row in rows:
        assert row["reaction_seeds"] == list(range(8))
        assert row["mean_distance_gap"] == 0
        assert abs(row["mean_time_gap_seconds"]) < 1.e-7
        assert row["selected"]["compounds"][0] in row["alternatives"]
    light = [row for row in rows if row["case"]["name"] == "light_rain"]
    assert all(set(row["alternatives"]) == {compound.value for compound in TireCompound}
               for row in light)
