"""Compulsory wet scenarios remain explicit across workers, replay and reports."""

import csv
import json

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.analysis.replay import _load_saved_runner, replay_saved_simulation
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.output import Exporter
from f1sim.output.comparison import render_comparison_report
from f1sim.output.console import ConsoleOutput
from f1sim.simulation.control_schedule import WET_RESUMPTION_CONTROL_SCHEDULE_POLICY
from f1sim.web.server import _summarize_scenario_results


def runner(engine, *, overlays=False):
    drivers = [Driver(id=key, name=key, team_id=key, consistency=1.) for key in ("A", "B")]
    cars = {key: Car(team_id=key, team_name=key, reliability=1.) for key in ("A", "B")}
    track = Track(id="T", name="Compulsory wets", country="Synthetic", total_laps=8,
                  base_lap_time=90., safety_car_probability=0.)
    options = {}
    if overlays:
        options = {
            "tire_inventory": {key: [
                {"id": "M", "compound": "medium", "remaining_laps": 5},
                {"id": "W", "compound": "wet", "age": 9, "remaining_laps": 1},
                {"id": "H", "compound": "hard"}, {"id": "I", "compound": "intermediate"},
            ] for key in ("A", "B")},
            "pit_plans": {key: [{"earliest_lap": 3, "lap": 4, "trigger": "neutralized",
                                 "compound": "hard"}] for key in ("A", "B")},
            "tire_warmup": {"wet": 1., "hard": .7},
            "weather_schedule": [{"lap": 5, "rain_intensity": .4}],
            "qualifying_weather": {"Q1": {"rain_intensity": .1}},
        }
    return MonteCarloRunner(drivers, cars, track, Weather(change_probability=0.), seed=91,
        race_engine=engine, rng_policy="isolated_race_v1", starting_tires={"A": "medium",
        "B": "medium"}, control_schedule=[{"lap": 2, "control": "red_flag",
                                              "action": "resume_wet"}], **options)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_workers_preserve_the_director_instruction_and_field_results(engine):
    serial = runner(engine, overlays=True).run(2, parallel=False)
    workers = runner(engine, overlays=True).run(2, parallel=True, max_workers=2)
    assert native_physics()
    assert workers.race_results == serial.race_results
    assert workers.control_schedule_histories == serial.control_schedule_histories
    assert serial.input_snapshot["schema_version"] == 13
    assert (serial.input_snapshot["control_schedule_policy"]
            == WET_RESUMPTION_CONTROL_SCHEDULE_POLICY)
    assert serial.get_control_schedule_statistics()["applied"] == 2
    for race in serial.race_results:
        for result in race:
            wets = [row for row in result.tire_set_history if row["compound"] == "wet"]
            assert len(wets) == 1 and wets[0]["kind"] == "red_flag" and wets[0]["laps_used"] == 1
    paired = paired_comparison_statistics({"serial": serial, "workers": workers}, "serial")
    assert paired["variants"]["workers"]["status"] == "paired"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("overlays", [False, True])
def test_schema_thirteen_replays_every_overlay_and_rejects_older_policy(tmp_path, engine, overlays):
    original = runner(engine, overlays=overlays).run(1, parallel=False)
    path = Exporter(tmp_path).export_statistics_json(original)
    saved = path.read_bytes()
    replay = replay_saved_simulation(path)
    assert path.read_bytes() == saved
    assert replay.race_results == original.race_results
    assert replay.input_snapshot == original.input_snapshot
    paired = paired_comparison_statistics({"original": original, "replay": replay}, "original")
    assert paired["variants"]["replay"]["status"] == "paired"
    bad = json.loads(saved)
    bad["simulation_inputs"].update(schema_version=12,
                                    control_schedule_policy="observed_control_schedule_v2")
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="schema 13"):
        _load_saved_runner(path)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_reports_and_api_explain_the_assumption_and_preserve_canonical_csv(
    tmp_path, capsys, engine,
):
    result = runner(engine, overlays=True).run(1, parallel=False)
    exporter = Exporter(tmp_path)
    html = exporter.export_report_html(result).read_text(encoding="utf-8")
    assert "full-wet tyres compulsory" in html
    csv_rows = list(csv.DictReader(exporter.export_control_schedule_history_csv(
        result).read_text(encoding="utf-8").splitlines()))
    assert len(csv_rows) == 1 and csv_rows[0]["action"] == "resume_wet"
    payload = _summarize_scenario_results({"wet": result})["scenarios"]["wet"]
    assert payload["control_schedule_statistics"]["requested_schedule"][0]["action"] == "resume_wet"
    assert payload["simulation_inputs"]["schema_version"] == 13
    comparison = render_comparison_report({"wet": result})
    assert "full-wet tyres compulsory" in comparison
    ConsoleOutput.print_monte_carlo_summary(result)
    assert "full-wet tyres compulsory" in capsys.readouterr().out
