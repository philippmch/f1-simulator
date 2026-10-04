"""Observed control assumptions survive processes, saved schemas and strategy variants."""

import json
from copy import deepcopy
from inspect import signature
from itertools import product

import pytest
from test_qualifying_session_weather import runner
from test_qualifying_weather_replay import OVERRIDES, options

from f1sim.analysis.montecarlo import _run_single_simulation
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.analysis.rival_strategy_selection import evaluate_saved_rival_pit_plan_selection
from f1sim.analysis.strategy_comparison import (
    _runner_variant,
    compare_saved_pit_plans,
    compare_saved_race_engines,
    compare_saved_starting_tires,
)
from f1sim.output import Exporter
from f1sim.simulation.control_schedule import CONTROL_SCHEDULE_POLICY
from f1sim.simulation.lap import LapSimulator

SCHEDULE = [{"lap": 2, "control": "safety_car", "duration_laps": 1},
            {"lap": 4, "control": "vsc", "duration_laps": 2}]


def test_mutated_runner_schedule_preflight_is_atomic(monkeypatch):
    import f1sim.analysis.montecarlo as montecarlo

    configured = runner(control_schedule=SCHEDULE)
    configured.track.total_laps = 3
    before = deepcopy(configured.control_schedule)
    monkeypatch.setattr(montecarlo, "_run_single_simulation",
                        lambda *a: pytest.fail("trial attempted"))
    with pytest.raises(ValueError, match="total_laps"):
        configured.run(1, parallel=False)
    assert configured.control_schedule == before


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_unobserved_future_control_cannot_change_openings_or_earlier_laps(monkeypatch, engine):
    original = LapSimulator.calculate_lap_time
    contract = signature(original)
    observed = []

    def capture(self, *args, **kwargs):
        values = contract.bind(self, *args, **kwargs)
        values.apply_defaults()
        outcome = original(self, *args, **kwargs)
        fields = values.arguments
        if fields["sample_variation"] and fields["lap_number"] <= 4:
            observed.append((fields["driver"].id, fields["lap_number"],
                             fields["tire"].compound, outcome))
        return outcome

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", capture)
    plain = runner(engine=engine, control_schedule=[]).run(1, parallel=False)
    prefix = observed.copy()
    observed.clear()
    future = runner(engine=engine, control_schedule=[SCHEDULE[1]]).run(1, parallel=False)
    assert prefix and prefix == observed
    assert future.qualifying_results == plain.qualifying_results
    for reference, variant in zip(plain.race_results[0], future.race_results[0], strict=True):
        assert reference.driver_id == variant.driver_id
        assert reference.strategy[0] == variant.strategy[0]
    default = runner(engine=engine).run(1, parallel=False)
    explicit_auto = runner(engine=engine, control_schedule=None).run(1, parallel=False)
    assert default.race_results == explicit_auto.race_results
    assert default.input_snapshot == explicit_auto.input_snapshot


@pytest.mark.parametrize("schedule", [[], SCHEDULE])
@pytest.mark.parametrize("finite,windows,warmup,weather", list(product((False, True), repeat=4)))
def test_schema_eleven_overlay_round_trip(tmp_path, schedule, finite, windows, warmup, weather):
    settings = options(True, finite, False, warmup)
    if finite:
        settings["tire_inventory"]["A"][0]["remaining_laps"] = 3
    if windows:
        settings["pit_plans"] = {"A": [{"lap": 4, "earliest_lap": 3,
                                      "trigger": "neutralized", "compound": "medium"}]}
    original = runner(OVERRIDES, control_schedule=schedule,
                      weather_schedule=[{"lap": 3, "rain_intensity": .5}] if weather else None,
                      **settings).run(1, parallel=False)
    assert original.input_snapshot["schema_version"] == 11
    assert original.input_snapshot["control_schedule"] == schedule
    assert original.input_snapshot["control_schedule_policy"] == CONTROL_SCHEDULE_POLICY
    path = Exporter(tmp_path).export_statistics_json(original)
    restored, _ = _load_saved_runner(path)
    assert restored.control_schedule == schedule
    replay = replay_saved_simulation(path)
    assert replay.race_results == original.race_results
    assert replay.qualifying_results == original.qualifying_results
    assert replay.input_snapshot == original.input_snapshot
    assert replay.control_schedule_histories == original.control_schedule_histories
    assert replay.get_control_schedule_statistics()["valid_history_races"] == 1
    assert paired_comparison_statistics({"a": original, "b": replay}, "a")[
        "variants"]["b"]["status"] == "paired"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("schedule", [[], SCHEDULE])
def test_worker_process_equivalence_and_second_trial_replay(tmp_path, engine, schedule):
    configured = runner(engine=engine, control_schedule=schedule)
    serial = configured.run(2, parallel=False)
    parallel = configured.run(2, parallel=True, max_workers=1)
    assert serial.race_results == parallel.race_results
    assert serial.qualifying_results == parallel.qualifying_results
    assert serial.event_stats == parallel.event_stats
    assert serial.control_schedule_histories == parallel.control_schedule_histories
    path = Exporter(tmp_path).export_statistics_json(serial)
    replay = replay_saved_simulation(path, simulation=2)
    assert replay.race_results == [serial.race_results[1]]
    assert replay.control_schedule_histories == [serial.control_schedule_histories[1]]
    snapshot = serial.input_snapshot
    args = tuple(snapshot[key] for key in ("drivers", "cars", "track", "weather")) + (
        serial.seed, engine, {}, snapshot["rng_policy"], {}, {}, {}, {}, {}, None, schedule,
    )
    race, qualifying, events = _run_single_simulation(args)
    assert race == serial.race_results[0] and qualifying == serial.qualifying_results[0]
    assert events["control_schedule_history"] == serial.control_schedule_histories[0]
    with pytest.raises(ValueError, match="require control_schedule"):
        _run_single_simulation(args[:-1] + (None,))


@pytest.mark.parametrize("version", range(1, 11))
@pytest.mark.parametrize("schedule", [[], SCHEDULE])
def test_downgraded_schemas_cannot_drop_control_source(tmp_path, version, schedule):
    original = runner(control_schedule=schedule).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"]["schema_version"] = version
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="control_schedule"):
        replay_saved_simulation(path)
    original.input_snapshot["schema_version"] = version
    assert paired_comparison_statistics({"a": original, "b": original}, "a")[
        "variants"]["b"]["status"] == "unavailable"


@pytest.mark.parametrize("change", [{"control_schedule": None}, {"control_schedule": "automatic"},
                                    {"control_schedule_policy": None},
                                    {"control_schedule_policy": "future_known"},
                                    {"control_schedule": [{"lap": 6, "control": "vsc",
                                                           "duration_laps": 1}]}])
def test_invalid_source_policy_or_distance_is_not_replayable(tmp_path, change):
    result = runner(control_schedule=[]).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(result)
    saved = json.loads(path.read_text())
    saved["simulation_inputs"].update(change)
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        replay_saved_simulation(path)
    result.input_snapshot.update(change)
    assert paired_comparison_statistics({"a": result, "b": result}, "a")[
        "variants"]["b"]["status"] == "unavailable"


@pytest.mark.parametrize("schedule", [[], SCHEDULE])
def test_all_saved_variants_inherit_an_isolated_source(tmp_path, schedule):
    original = runner(control_schedule=schedule).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    results = list(compare_saved_race_engines(path, num_simulations=1).values())
    results += list(compare_saved_starting_tires(path, "A", ("automatic", "soft"),
                                               num_simulations=1).values())
    results += list(compare_saved_pit_plans(path, "A", {"auto": None, "hold": []},
                                          num_simulations=1).values())
    assert all(result.input_snapshot["control_schedule"] == schedule for result in results)
    restored, _ = _load_saved_runner(path)
    variant = _runner_variant(restored, pit_plans={"B": []})
    variant.control_schedule.append({"lap": 5, "control": "vsc", "duration_laps": 1})
    assert restored.control_schedule == schedule
    changed = deepcopy(original)
    changed.input_snapshot["control_schedule"] = [] if schedule else SCHEDULE
    assert paired_comparison_statistics({"a": original, "b": changed}, "a")[
        "variants"]["b"]["status"] == "unavailable"
    automatic = runner().run(1, parallel=False)
    assert paired_comparison_statistics({"a": original, "b": automatic}, "a")[
        "variants"]["b"]["status"] == "unavailable"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("method", ["weighted_mean", "minimax_regret"])
def test_frozen_selection_varies_only_declared_control_assumptions(tmp_path, engine, method):
    source = Exporter(tmp_path).export_statistics_json(
        runner(engine=engine, control_schedule=SCHEDULE).run(1, parallel=False),
    )
    scenarios = {"inherit": {"weight": 1, "pit_plans": {}},
                 "none": {"weight": 2, "pit_plans": {}, "control_schedule": []},
                 "vsc": {"weight": 1, "pit_plans": {}, "control_schedule": [SCHEDULE[1]]}}
    before = deepcopy(scenarios)
    outcome = evaluate_saved_rival_pit_plan_selection(
        source, {"reference": None, "window": [{"lap": 5, "earliest_lap": 3,
                                                "trigger": "neutralized", "compound": "hard"}]},
        "reference", scenarios, driver_id="A", training_simulations=2, validation_simulations=2,
        selection_method=method,
    )
    assert scenarios == before
    effective = {row["name"]: row["control_schedule"]
                 for row in outcome["selection"]["rival_scenarios"]}
    assert effective == {"inherit": SCHEDULE, "none": [], "vsc": [SCHEDULE[1]]}
    for phase in ("training_results", "validation_results"):
        qualifying = next(iter(outcome[phase].values()))["reference"].qualifying_results
        for name, variants in outcome[phase].items():
            for result in variants.values():
                assert result.input_snapshot["control_schedule"] == effective[name]
                assert result.qualifying_results == qualifying
                assert result.get_control_schedule_statistics()["valid_history_races"] == 2
