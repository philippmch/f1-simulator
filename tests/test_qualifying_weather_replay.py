"""Session weather survives worker, snapshot, replay and comparison boundaries."""

import json
from copy import deepcopy
from itertools import product

import pytest
from test_qualifying_session_weather import runner

from f1sim.analysis.montecarlo import _run_single_simulation
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.strategy_comparison import (
    _runner_variant,
    compare_saved_pit_plans,
    compare_saved_race_engines,
    compare_saved_starting_tires,
)
from f1sim.output import Exporter

OVERRIDES = {"Q1": {"track_wetness": .85, "rain_intensity": .9},
             "Q3": {"track_wetness": .35, "rain_intensity": .4}}


def options(ages=False, finite=False, plans=False, warmup=False):
    values = {"starting_tires": {"A": "soft"}}
    if ages:
        values["starting_tire_ages"] = {"A": 2}
    if finite:
        values["tire_inventory"] = {"A": [
            {"id": "s", "compound": "soft", "age": 2 if ages else 0},
            {"id": "m", "compound": "medium"},
        ]}
    if plans:
        values["pit_plans"] = {"A": [{"lap": 3, "compound": "medium"}]}
    if warmup:
        values["tire_warmup"] = {"soft": .5, "medium": 1.}
    return values


@pytest.mark.parametrize("ages,finite,plans,warmup", list(product((False, True), repeat=4)))
def test_schema_seven_round_trip_all_independent_overlays(tmp_path, ages, finite, plans, warmup):
    original = runner(OVERRIDES, **options(ages, finite, plans, warmup)).run(1, parallel=False)
    assert original.input_snapshot["schema_version"] == 7
    assert original.input_snapshot["qualifying_weather"]
    assert ("tire_warmup" in original.input_snapshot) is warmup
    path = Exporter(tmp_path).export_statistics_json(original)
    restored, count = _load_saved_runner(path)
    assert count == 1
    assert restored.qualifying_weather == original.input_snapshot["qualifying_weather"]
    replay = replay_saved_simulation(path)
    assert replay.race_results == original.race_results
    assert replay.qualifying_results == original.qualifying_results
    assert replay.input_snapshot == original.input_snapshot
    assert paired_comparison_statistics({"old": original, "replay": replay}, "old")[
        "variants"]["replay"]["status"] == "paired"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_actual_process_trials_and_second_trial_replay_match(tmp_path, engine):
    configured = runner(OVERRIDES, engine=engine, **options(True, True, True, True))
    sequential = configured.run(2, parallel=False)
    parallel = configured.run(2, parallel=True, max_workers=1)
    assert sequential.race_results == parallel.race_results
    assert sequential.qualifying_results == parallel.qualifying_results
    assert sequential.event_stats == parallel.event_stats
    assert sequential.input_snapshot == parallel.input_snapshot
    path = Exporter(tmp_path).export_statistics_json(sequential)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.seed == sequential.seed + 1
    assert replay.race_results == [sequential.race_results[1]]
    assert replay.qualifying_results == [sequential.qualifying_results[1]]


@pytest.mark.parametrize("version", range(1, 7))
def test_old_saved_schemas_reject_session_weather_presence(tmp_path, version):
    original = runner().run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"].update(schema_version=version, qualifying_weather={})
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="qualifying_weather"):
        replay_saved_simulation(path)
    variant = deepcopy(original)
    variant.input_snapshot.update(schema_version=version, qualifying_weather={})
    assert paired_comparison_statistics({"a": original, "b": variant}, "a")[
        "variants"]["b"]["status"] == "unavailable"


@pytest.mark.parametrize("changes", [
    {"qualifying_weather": None}, {"qualifying_weather": {}},
    {"qualifying_weather": {"Q4": {}}},
    {"qualifying_weather": {"Q1": {"track_wetness": "0.5"}}},
    {"tire_warmup": {"soft": .5}},
    {"tire_warmup_policy": "post_fit_first_lap_v1"},
    {"tire_warmup": {}, "tire_warmup_policy": "post_fit_first_lap_v1"},
    {"tire_warmup": {"soft": .5}, "tire_warmup_policy": "unknown"},
])
def test_schema_seven_malformed_weather_or_warmup_rejected_for_replay_and_pair(tmp_path, changes):
    original = runner(OVERRIDES).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"].update(changes)
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        replay_saved_simulation(path)
    variant = deepcopy(original)
    variant.input_snapshot.update(changes)
    assert paired_comparison_statistics({"a": original, "b": variant}, "a")[
        "variants"]["b"]["status"] == "unavailable"


def test_pairing_compares_effective_weather_including_equal_legacy_fallback():
    legacy_runner = runner()
    original = legacy_runner.run(1, parallel=False)
    explicit = runner({"Q1": legacy_runner.weather}).run(1, parallel=False)
    assert explicit.qualifying_results == original.qualifying_results
    assert explicit.race_results == original.race_results
    assert paired_comparison_statistics({"a": original, "b": explicit}, "a")[
        "variants"]["b"]["status"] == "paired"
    # Humidity changes no current qualifying pace; equal sampled records must
    # still not make different configured session weather compatible.
    changed = deepcopy(explicit)
    changed.input_snapshot["qualifying_weather"]["Q1"]["humidity"] = .9
    assert paired_comparison_statistics({"a": original, "b": changed}, "a")[
        "variants"]["b"]["status"] == "unavailable"


def test_saved_engine_starting_tyre_and_pit_plan_variants_inherit_sessions(tmp_path):
    original = runner(OVERRIDES, **options(plans=True, warmup=True)).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    results = list(compare_saved_race_engines(path, num_simulations=1).values())
    results += list(compare_saved_starting_tires(
        path, "A", ("automatic", "soft", "hard"), num_simulations=1,
    ).values())
    results += list(compare_saved_pit_plans(
        path, "A", {"saved": [{"lap": 3, "compound": "medium"}], "automatic": None},
        num_simulations=1,
    ).values())
    for result in results:
        assert result.input_snapshot["qualifying_weather"] == original.input_snapshot[
            "qualifying_weather"]
        assert result.qualifying_results == original.qualifying_results
    restored, _ = _load_saved_runner(path)
    rival_variant = _runner_variant(restored, pit_plans={"B": []})
    assert rival_variant.qualifying_weather == restored.qualifying_weather
    rival_variant.qualifying_weather["Q1"]["track_wetness"] = .7
    assert restored.qualifying_weather["Q1"]["track_wetness"] == .85


def test_explicit_thirteen_item_worker_and_legacy_calls_retain_compatibility():
    original = runner().run(1, parallel=False)
    snapshot = original.input_snapshot
    args = tuple(snapshot[key] for key in ("drivers", "cars", "track", "weather")) + (
        19, "standard", {}, snapshot["rng_policy"], {}, {}, {},
    )
    assert _run_single_simulation(args + ({}, {})) == _run_single_simulation(args)
    actual = _run_single_simulation(args + ({}, OVERRIDES))
    expected = runner(OVERRIDES).run(1, parallel=False)
    assert actual[0] == expected.race_results[0]
    assert actual[1] == expected.qualifying_results[0]
    with pytest.raises(ValueError, match="5 through 13"):
        _run_single_simulation(args + ({}, OVERRIDES, None))
