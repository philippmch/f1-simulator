"""Prescribed rainfall survives workers, persistence and saved variants."""
import json
from copy import deepcopy
from itertools import product

import pytest
from test_qualifying_session_weather import runner
from test_qualifying_weather_replay import OVERRIDES, options

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

SCHEDULE = [{"lap": 2, "rain_intensity": .8, "condition": "heavy_rain"},
            {"lap": 4, "rain_intensity": 0, "condition": "dry"}]


@pytest.mark.parametrize("finite,plans,warmup,qualifying", list(product((False, True), repeat=4)))
def test_schema_eight_independent_overlays_round_trip(tmp_path, finite, plans, warmup, qualifying):
    original = runner(OVERRIDES if qualifying else None, weather_schedule=SCHEDULE,
                      **options(False, finite, plans, warmup)).run(1, parallel=False)
    assert original.input_snapshot["schema_version"] == 8
    assert original.input_snapshot["weather_schedule"] == SCHEDULE
    assert ("tire_warmup" in original.input_snapshot) is warmup
    path = Exporter(tmp_path).export_statistics_json(original)
    restored, count = _load_saved_runner(path)
    assert count == 1 and restored.weather_schedule == SCHEDULE
    replay = replay_saved_simulation(path)
    assert replay.race_results == original.race_results
    assert replay.qualifying_results == original.qualifying_results
    assert replay.input_snapshot == original.input_snapshot
    assert paired_comparison_statistics({"a": original, "b": replay}, "a")[
        "variants"]["b"]["status"] == "paired"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_process_equivalence_and_second_trial_replay(tmp_path, engine):
    configured = runner(OVERRIDES, engine=engine, weather_schedule=SCHEDULE)
    sequential = configured.run(2, parallel=False)
    parallel = configured.run(2, parallel=True, max_workers=1)
    assert sequential.race_results == parallel.race_results
    assert sequential.qualifying_results == parallel.qualifying_results
    assert sequential.event_stats == parallel.event_stats
    path = Exporter(tmp_path).export_statistics_json(sequential)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [sequential.race_results[1]]


@pytest.mark.parametrize("version", range(1, 8))
def test_old_schemas_reject_schedule_key_even_empty(tmp_path, version):
    original = runner().run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"].update(schema_version=version, weather_schedule=[])
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="weather_schedule"):
        replay_saved_simulation(path)
    variant = deepcopy(original)
    variant.input_snapshot.update(schema_version=version, weather_schedule=[])
    assert paired_comparison_statistics({"a": original, "b": variant}, "a")[
        "variants"]["b"]["status"] == "unavailable"


@pytest.mark.parametrize("value", [None, [], [{"lap": 6, "rain_intensity": 0}],
                                  [{"lap": 2, "rain_intensity": True}]])
def test_schema_eight_requires_valid_nonempty_schedule(tmp_path, value):
    original = runner(weather_schedule=SCHEDULE).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"]["weather_schedule"] = value
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        replay_saved_simulation(path)
    original.input_snapshot["weather_schedule"] = value
    assert paired_comparison_statistics({"a": original, "b": original}, "a")[
        "variants"]["b"]["status"] == "unavailable"


def test_empty_parity_isolation_worker_and_pair_signature():
    plain = runner().run(1, parallel=False)
    empty = runner(weather_schedule=[]).run(1, parallel=False)
    assert plain.input_snapshot == empty.input_snapshot
    assert plain.race_results == empty.race_results
    configured = runner(weather_schedule=SCHEDULE)
    configured.weather_schedule[0]["rain_intensity"] = .7
    assert SCHEDULE[0]["rain_intensity"] == .8
    scheduled = runner(weather_schedule=SCHEDULE).run(1, parallel=False)
    changed = deepcopy(scheduled)
    changed.input_snapshot["weather_schedule"][0]["rain_intensity"] = .7
    assert paired_comparison_statistics({"a": scheduled, "b": changed}, "a")[
        "variants"]["b"]["status"] == "unavailable"
    snapshot = plain.input_snapshot
    args = tuple(snapshot[key] for key in ("drivers", "cars", "track", "weather")) + (
        19, "standard", {}, snapshot["rng_policy"], {}, {}, {}, {}, {}, SCHEDULE,
    )
    assert _run_single_simulation(args)[0] == scheduled.race_results[0]
    assert _run_single_simulation(args)[1] == plain.qualifying_results[0]


def test_saved_engine_tyre_plan_and_rival_variants_inherit_schedule(tmp_path):
    original = runner(weather_schedule=SCHEDULE).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    results = list(compare_saved_race_engines(path, num_simulations=1).values())
    results += list(compare_saved_starting_tires(path, "A", ("automatic", "soft"),
                                               num_simulations=1).values())
    results += list(compare_saved_pit_plans(path, "A", {"auto": None, "hold": []},
                                          num_simulations=1).values())
    assert all(result.input_snapshot["weather_schedule"] == SCHEDULE for result in results)
    restored, _ = _load_saved_runner(path)
    variant = _runner_variant(restored, pit_plans={"B": []})
    variant.weather_schedule[0]["rain_intensity"] = .2
    assert restored.weather_schedule == SCHEDULE
