"""Incomplete finite-pool openings preserve accepted distance and elapsed time."""

import json
import subprocess
import sys
from copy import deepcopy
from math import inf
from pathlib import Path

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation import opening_strategy as opening
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverStatus, RaceSimulator, TeamStrategyArchetype


def models(records, weather, plan):
    simulator = RaceSimulator(np.random.default_rng(7))
    inputs = (Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
              Track(id="T", name="T", country="T", total_laps=8, base_lap_time=90),
              weather, TeamStrategyArchetype.BALANCED,
              simulator.strategy_tuning, simulator.strategy_profiles)
    scores = dict(opening.inventory_opening_policy_costs(*inputs, records, pit_plan=plan))
    return simulator, inputs, scores


def execute(engine, records, weather, plan, compound=None, age=0):
    simulator, inputs, _ = models(records, weather, plan)
    simulator._infer_team_strategy = lambda *a: TeamStrategyArchetype.BALANCED
    simulator.event_manager.process_lap = lambda *a, **kw: []
    simulator.event_manager._check_mechanical_failure = lambda *a, **kw: None
    simulator.event_manager._check_random_incident = lambda *a, **kw: None
    calculate = simulator.lap_simulator.calculate_lap_time

    def mean(*args, **kwargs):
        surface = kwargs.get("weather", args[4] if args else None)
        tire = kwargs.get("tire", args[3] if args else None)
        assert surface.tire_mismatch(tire.compound) != "critical"
        kwargs["sample_variation"] = False
        return calculate(*args, **kwargs)

    simulator.lap_simulator.calculate_lap_time = mean
    simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    result, = run([inputs[0]], {"A": inputs[1]}, inputs[2], inputs[3], ["A"],
                  tire_inventory={"A": records},
                  **({"pit_plans": {"A": plan}} if plan is not None else {}),
                  **({"starting_tires": {"A": compound}, "starting_tire_ages": {"A": age}}
                     if compound is not None else {}))
    return result


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("plan", [None, [], [{"lap": 2, "compound": "intermediate"}],
                                 [{"lap": 4, "compound": "wet"}]])
@pytest.mark.parametrize("reverse", [False, True])
def test_incomplete_opening_keeps_the_extra_accepted_lap(engine, plan, reverse):
    records = [dict(id="I", compound="intermediate", remaining_laps=4),
               dict(id="W", compound="wet", remaining_laps=1)]
    if reverse:
        records.reverse()
    weather = Weather(track_wetness=.2, rain_intensity=0., change_probability=0)
    before = deepcopy((records, weather, plan))
    simulator, inputs, scores = models(records, weather, plan)
    rng_before = deepcopy(simulator.rng.bit_generator.state)
    _, selected = simulator._inventory_opening_set(*inputs[:4], inputs[4], records, pit_plan=plan)
    assert selected.id == "W"
    assert simulator.rng.bit_generator.state == rng_before
    assert all(score.mean_time == score.negative_mean_laps == inf for score in scores.values())
    assert all(score.negative_mean_instructions == 0 for score in scores.values())
    assert scores["W"] < scores["I"]
    alternatives = {compound: execute(engine, records, weather, plan, compound)
                    for compound in ("intermediate", "wet")}
    result = execute(engine, records, weather, plan)
    assert result.status == DriverStatus.DNF
    assert result.dnf_reason == "No suitable replacement tyre set available"
    assert alternatives["intermediate"].laps_completed == 4
    assert result.laps_completed == alternatives["wet"].laps_completed == 5
    assert result.total_time == pytest.approx(alternatives["wet"].total_time, abs=1.e-8)
    assert result.tire_set_history[0]["set_id"] == "W"
    assert result.pit_laps == [2]
    for compound, row in alternatives.items():
        score = scores["I" if compound == "intermediate" else "W"]
        assert score.negative_partial_mean_laps == -row.laps_completed
        assert score.partial_mean_time == pytest.approx(row.total_time, abs=1.e-8)
    assert sum(row["laps_used"] for row in result.tire_set_history) == 5
    assert all(row["remaining_laps"] == 0 for row in result.tire_inventory)
    assert (records, weather, plan) == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("plan", [None, []])
@pytest.mark.parametrize("reverse", [False, True])
def test_equal_incomplete_distance_prefers_the_faster_policy(engine, plan, reverse):
    records = [dict(id="worn", compound="soft", age=40),
               dict(id="fresh", compound="soft", age=0)]
    if reverse:
        records.reverse()
    weather = Weather(change_probability=0)
    _, _, scores = models(records, weather, plan)
    assert scores["fresh"].negative_partial_mean_laps == scores["worn"].negative_partial_mean_laps
    assert scores["fresh"] < scores["worn"]
    result = execute(engine, records, weather, plan)
    fresh = execute(engine, records, weather, plan, "soft", 0)
    worn = execute(engine, records, weather, plan, "soft", 40)
    assert result.status == fresh.status == worn.status == DriverStatus.DNF
    assert result.laps_completed == fresh.laps_completed == worn.laps_completed == 7
    assert result.tire_set_history[0]["set_id"] == "fresh"
    assert result.total_time == pytest.approx(fresh.total_time, abs=1.e-8)
    assert result.total_time < worn.total_time
    assert scores["fresh"].partial_mean_time == pytest.approx(result.total_time, abs=1.e-8)


@pytest.mark.parametrize("preferred,other", [
    ([(3, 1000., 0, 1000.)], [(7, inf, 5, 700.)]),
    ([(3, 1000., 0, 1000.), (2, inf, 5, 200.)], [(7, inf, 10, 700.)] * 2),
    ([(5, inf, 0, 1000.)], [(4, inf, 10, 200.)]),
    ([(5, inf, 0, 700.)], [(5, inf, 10, 1000.)]),
])
def test_finishability_then_partial_distance_and_time_precede_failed_request_credit(
    preferred, other,
):
    better = opening._mean_policy_score(preferred)
    worse = opening._mean_policy_score(other)
    assert better < worse
    assert worse > better
    assert min((worse, better)) == better
    if worse.mean_time == inf:
        assert worse.negative_mean_instructions == 0


def test_legacy_failed_projection_does_not_invent_elapsed_time_or_plan_credit():
    score = opening._mean_policy_score([(5, inf, 9)])
    assert score == opening.OpeningPolicyScore(inf, inf)
    observed = opening._mean_policy_score([(5, inf, 0, 700.)])
    assert observed < score


def test_equal_incomplete_identities_keep_input_order_and_share_one_projection(monkeypatch):
    records = [dict(id=identifier, compound="soft", age=0) for identifier in ("Z", "A", "B")]
    original = opening._policy_path_outcome
    calls = []

    def observe(*args, **kwargs):
        calls.append(kwargs["opening_set_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(opening, "_policy_path_outcome", observe)
    opening._cached_inventory_policy_costs.cache_clear()
    simulator, inputs, scores = models(records, Weather(change_probability=0), [])
    assert calls == ["Z"]
    assert len(set(scores.values())) == 1
    assert all(score.negative_partial_mean_laps == -7 for score in scores.values())
    pool, selected = simulator._inventory_opening_set(*inputs[:4], inputs[4], records, pit_plan=[])
    assert selected.id == "Z"
    assert list(pool.sets) == ["Z", "A", "B"]


def test_projection_retains_infinite_completion_cost_and_separate_crossing_time():
    records = [dict(id="I", compound="intermediate", remaining_laps=4),
               dict(id="W", compound="wet", remaining_laps=1)]
    _, inputs, _ = models(records, Weather(track_wetness=.2, change_probability=0), [])
    legacy = opening._policy_path_outcome(*inputs, TireCompound.WET, 0,
                                         tire_inventory=records, opening_set_id="W",
                                         pit_plan=[], include_instructions=True)
    detailed = opening._policy_path_outcome(*inputs, TireCompound.WET, 0,
                                           tire_inventory=records, opening_set_id="W",
                                           pit_plan=[], include_instructions=True,
                                           include_incomplete_time=True)
    assert detailed[:3] == legacy == (5, inf, 0)
    assert detailed[3] == pytest.approx(512.4749416852804, abs=1.e-8)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_incomplete_opening_survives_workers_and_saved_replay(tmp_path, engine):
    from test_inventory_analysis import save_result

    from f1sim.analysis.montecarlo import MonteCarloRunner
    from f1sim.analysis.replay import replay_saved_simulation

    records = [dict(id="I", compound="intermediate", remaining_laps=4),
               dict(id="W", compound="wet", remaining_laps=1)]
    _, inputs, _ = models(records, Weather(track_wetness=.2, change_probability=0), [])
    runner = MonteCarloRunner([inputs[0]], {"A": inputs[1]}, inputs[2], inputs[3],
                              race_engine=engine, seed=7, tire_inventory={"A": records},
                              pit_plans={"A": []})
    serial = runner.run(2, parallel=False)
    parallel = runner.run(2, parallel=True, max_workers=2)
    assert serial.race_results == parallel.race_results
    assert serial.input_snapshot["schema_version"] == 9
    assert serial.input_snapshot["tire_inventory"] == parallel.input_snapshot["tire_inventory"]
    assert all(row.tire_set_history[0]["set_id"] == "W"
               for race in serial.race_results for row in race)
    assert all(row.status == DriverStatus.DNF and row.laps_completed <= 5
               for race in serial.race_results for row in race)
    replay = replay_saved_simulation(save_result(tmp_path, serial), simulation=2)
    assert replay.race_results[0] == serial.race_results[1]
    assert replay.input_snapshot == serial.input_snapshot


def test_offline_incomplete_diagnostic_compares_finite_retirement_times():
    path = Path(__file__).resolve().parents[1] / "examples/check_weather_openings.py"
    result = subprocess.run([sys.executable, str(path), "--incomplete"],
                            capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = json.loads(result.stdout)
    assert len(rows) == 8
    assert {row["engine"] for row in rows} == {"standard", "chronological"}
    assert all(row["selected"]["status"] == "dnf" for row in rows)
    assert all(row["mean_finish_fraction_gap"] == row["mean_distance_gap"] == 0 for row in rows)
    assert all(row["mean_time_gap_seconds"] == pytest.approx(0., abs=1.e-8) for row in rows)
    for row in rows:
        if row["case"]["name"].startswith("drying"):
            assert row["selected"]["laps_completed"] == 5
            assert row["best_opening"] == "wet@0"
            assert "soft@0" not in row["alternatives"]
        else:
            assert row["selected"]["laps_completed"] == 7
            assert row["best_opening"] == "soft@0"
