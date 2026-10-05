"""Native countback results survive processes, constrained replay and app exports."""

import csv
import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.output import Exporter
from f1sim.output.comparison import render_comparison_report
from f1sim.output.console import ConsoleOutput
from f1sim.simulation.abandonment import race_abandonment_context, serialize_abandonment
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.control_schedule import (
    RED_FLAG_CONTROL_SCHEDULE_POLICY,
    validate_control_schedule,
)
from f1sim.simulation.race import RaceSimulator
from f1sim.web.server import _summarize_scenario_results


def inputs():
    drivers = [Driver(id=key, name=key, team_id=key, consistency=1.) for key in ("A", "B")]
    cars = {key: Car(team_id=key, team_name=key, reliability=1., base_pace=.8)
            for key in ("A", "B")}
    track = Track(id="t", name="Native countback", country="Synthetic", total_laps=8,
                  base_lap_time=90., safety_car_probability=0.)
    return drivers, cars, track, Weather(change_probability=0.)


def runner(engine, *, after=4, action="abandon", **kwargs):
    return MonteCarloRunner(*inputs(), seed=91, race_engine=engine,
                           rng_policy="isolated_race_v1",
                           starting_tires={key: "medium" for key in ("A", "B")},
                           control_schedule=[{"lap": after, "control": "red_flag",
                                              "action": action}], **kwargs)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_worker_results_and_abandonment_evidence_match_across_processes(engine):
    serial = runner(engine).run(2, parallel=False)
    parallel = runner(engine).run(2, parallel=True, max_workers=2)
    assert native_physics()
    assert parallel.race_results == serial.race_results
    assert parallel.control_schedule_histories == serial.control_schedule_histories
    assert serial.input_snapshot["schema_version"] == 12
    assert serial.input_snapshot["control_schedule_policy"] == RED_FLAG_CONTROL_SCHEDULE_POLICY
    assert all(context is not None for context in serial.get_race_abandonment_contexts())
    assert serial.get_abandonment_statistics()["recorded_abandoned_races"] == 2
    assert serial.get_control_schedule_statistics()["valid_history_races"] == 2
    paired = paired_comparison_statistics({"serial": serial, "parallel": parallel}, "serial")
    assert paired["variants"]["parallel"]["status"] == "paired"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("overlay", [False, True])
def test_schema_twelve_replays_full_tyres_windows_warmup_and_weather(tmp_path, engine, overlay):
    settings = {}
    if overlay:
        settings = {
            "tire_inventory": {key: [{"id": key + "M", "compound": "medium", "age": 0,
                                      "remaining_laps": 5},
                                     {"id": key + "H", "compound": "hard", "age": 0}]
                               for key in ("A", "B")},
            "pit_plans": {key: [{"lap": 6, "earliest_lap": 2, "trigger": "neutralized",
                                  "compound": "hard"}] for key in ("A", "B")},
            "tire_warmup": {"hard": .7},
            "weather_schedule": [{"lap": 3, "rain_intensity": .1}],
            "qualifying_weather": {"Q1": {"rain_intensity": .1}},
        }
    original = runner(engine, **settings).run(1, parallel=False)
    assert original.get_race_abandonment_context() is not None
    path = Exporter(tmp_path).export_statistics_json(original)
    original_bytes = path.read_bytes()
    restored, _ = _load_saved_runner(path)
    assert restored.control_schedule == original.input_snapshot["control_schedule"]
    replay = replay_saved_simulation(path)
    assert original_bytes == path.read_bytes()
    assert replay.race_results == original.race_results
    assert replay.input_snapshot == original.input_snapshot
    assert replay.get_race_abandonment_contexts() == original.get_race_abandonment_contexts()
    paired = paired_comparison_statistics({"original": original, "replay": replay}, "original")
    assert paired["variants"]["replay"]["status"] == "paired"
    bad = json.loads(original_bytes)
    bad["simulation_inputs"]["schema_version"] = 11
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="policy|red_flag"):
        _load_saved_runner(path)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("after", [1, 4])
def test_exports_app_and_console_explain_countback_without_inventing_a_dnf(
    tmp_path, capsys, engine, after,
):
    result = runner(engine, after=after).run(1, parallel=False)
    context = result.get_race_abandonment_context()
    assert context is not None
    exporter = Exporter(tmp_path)
    stats = json.loads(exporter.export_statistics_json(result).read_text())
    assert stats["race_abandonment_contexts"] == [context]
    assert stats["abandonment_statistics"] == result.get_abandonment_statistics()
    assert stats["abandonment_tire_rules"] == result.get_abandonment_tire_rules()
    csv_rows = list(csv.DictReader(exporter.export_race_results_csv(result).open(encoding="utf-8")))
    assert all(json.loads(row["race_abandonment"]) == context for row in csv_rows)
    assert all("penalty_seconds" in json.loads(row["abandonment_tire_rule"]) for row in csv_rows)
    control = list(csv.DictReader(exporter.export_control_schedule_history_csv(
        result,
    ).open(encoding="utf-8")))
    assert len(control) == 1 and control[0]["action"] == "abandon"
    payload = _summarize_scenario_results({"abandoned": result})["scenarios"]["abandoned"]
    assert payload["sample_race_abandonment_context"] == context
    assert payload["abandonment_tire_rules"] == result.get_abandonment_tire_rules()
    assert all(row["race_abandonment"] == context for row in payload["sample_race"])
    ConsoleOutput.print_race_results(result.race_results[0])
    ConsoleOutput.print_monte_carlo_summary(result)
    console = capsys.readouterr().out
    assert context["description"] in console and "Recorded abandoned races: 1" in console
    for report in (exporter.export_report_html(result).read_text(encoding="utf-8"),
                   render_comparison_report({"abandoned": result})):
        assert "Recorded abandoned races: 1" in report
        assert "Race-control requests" in report
        assert "finish clocks exclude subsequent running and suspension" in " ".join(report.split())
    if after == 1:
        assert "No result" in console and "DNF" not in console.split("RACE RESULTS")[1].split(
            "MONTE CARLO",
        )[0]
        assert all(stats.dnfs == stats.wins == 0 for stats in result.driver_stats.values())


def test_normalized_countback_records_keep_penalty_statistics():
    result = runner("standard", pit_plans={key: [] for key in ("A", "B")}).run(1, parallel=False)
    expected = result.get_abandonment_statistics()
    for row in result.race_results[0]:
        row.race_abandonment = asdict(row.race_abandonment)
        row.abandonment_tire_rule = asdict(row.abandonment_tire_rule)
    assert result.get_race_abandonment_context() is not None
    assert result.get_abandonment_statistics() == expected
    assert expected["recorded_penalized_drivers"] == 2


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_countback_matches_an_independently_observed_native_resumption_prefix(monkeypatch, engine):
    observations = []
    results = None
    for action in ("resume", "abandon"):
        simulator = RaceSimulator(np.random.default_rng(91), control_schedule=[
            {"lap": 4, "control": "red_flag", "action": action},
        ])
        seen, latest = {}, {}
        recharge = simulator._recharge_overtake_mode_energy

        def capture(states, neutralized=False):
            recharge(states, neutralized=neutralized)
            for state in states:
                if state.laps_completed > latest.get(state.driver.id, 0):
                    seen.setdefault(state.driver.id, []).append({
                        "lap": state.laps_completed, "time": state.total_time,
                        "stops": state.pit_stops, "strategy": list(state.tire_compound_history),
                    })
                    latest[state.driver.id] = state.laps_completed

        monkeypatch.setattr(simulator, "_recharge_overtake_mode_energy", capture)
        drivers, cars, track, weather = inputs()
        run = (simulator.simulate_race if engine == "standard"
               else ChronologicalRace(simulator).run)
        results = run(drivers, cars, track, weather, ["A", "B"],
                      starting_tires={key: "medium" for key in ("A", "B")},
                      pit_plans={"A": [{"lap": 3, "compound": "hard"}],
                                 "B": [{"lap": 4, "compound": "hard"}]})
        observations.append(seen)
        assert native_physics()
    evidence = results[0].race_abandonment
    cutoff = min(row["time"] for history in observations[0].values() for row in history
                 if row["lap"] == evidence.countback_lap)
    assert evidence.countback_time == cutoff
    for result in results:
        expected = next(row for row in observations[0][result.driver_id] if row["time"] >= cutoff)
        observed = next(row for row in observations[1][result.driver_id] if row["time"] >= cutoff)
        assert observed == expected  # Future action cannot change the earlier native race.
        assert result.laps_completed == expected["lap"]
        assert result.total_time == expected["time"] + result.abandonment_tire_rule.penalty_seconds
        assert result.pit_stops == expected["stops"] and result.strategy == expected["strategy"]


@pytest.mark.parametrize("bad", [True, "restart", "ABANDON", None, [], 1])
def test_noncanonical_red_decisions_are_rejected_before_work(bad):
    with pytest.raises(ValueError, match="action"):
        validate_control_schedule([{"lap": 4, "control": "red_flag", "action": bad}])


def test_red_flags_can_interrupt_an_earlier_requested_sc_without_overlap_coercion():
    schedule = [{"lap": 1, "control": "safety_car", "duration_laps": 5},
                {"lap": 3, "control": "red_flag", "action": "resume"},
                {"lap": 5, "control": "vsc", "duration_laps": 1}]
    assert validate_control_schedule(schedule) == schedule
    with pytest.raises(ValueError, match="clearance"):
        validate_control_schedule([schedule[0], schedule[1] | {"lap": 1}])


def test_inconsistent_or_partial_countback_evidence_is_unknown():
    result = runner("standard").run(1, parallel=False)
    rows = deepcopy(result.race_results[0])
    assert race_abandonment_context(rows) is not None
    rows[1].race_abandonment = replace(rows[1].race_abandonment, countback_lap=2)
    assert race_abandonment_context(rows) is None
    rows[1].race_abandonment = None
    assert race_abandonment_context(rows) is None
    assert race_abandonment_context(None) is None
    value = asdict(result.race_results[0][0].race_abandonment)
    for bad in (True, "90", float("nan"), float("inf"), 10 ** 1000):
        assert serialize_abandonment(value | {"signal_time": bad}) is None
