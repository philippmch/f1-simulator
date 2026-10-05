"""Recorded points eligibility stays consistent across engines, trials and reports."""

import csv
import json
from dataclasses import FrozenInstanceError, asdict

import numpy as np
import pytest

from f1sim.analysis.montecarlo import DriverStatistics, MonteCarloRunner, SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.comparison import render_comparison_report
from f1sim.output.console import ConsoleOutput
from f1sim.output.export import Exporter
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceResult, RaceSimulator
from f1sim.simulation.race_points import (
    RACE_POINTS_POLICY,
    RacePointsContext,
    points_for_result,
    points_reason_for_result,
    race_scoring_context,
    scoring_context_summary,
)
from f1sim.web.server import _serialize_race_result, _summarize_scenario_results


def row(context=None, *, award=25, position=1, classified=True, laps=100, retired=False):
    return RaceResult("A", "Driver A", "Team", position, 9000, 0, 0, 90,
                      DriverStatus.DNF if retired else DriverStatus.FINISHED,
                      classified=classified, laps_completed=laps, points_awarded=award,
                      race_points_context=context)


@pytest.mark.parametrize("laps,green,band,award,reason", [
    (None, False, None, 0, "no_winner"),
    (1, True, "under_25_percent", 0, "fewer_than_two_laps"),
    (100, False, "75_percent_or_more", 0, "no_green_pair"),
    (24, True, "under_25_percent", 6, None),
    (25, True, "25_to_50_percent", 13, None),
    (49, True, "25_to_50_percent", 13, None),
    (50, True, "50_to_75_percent", 19, None),
    (74, True, "50_to_75_percent", 19, None),
    (75, True, "75_percent_or_more", 25, None),
])
def test_summary_records_exact_boundaries_and_reasons(laps, green, band, award, reason):
    context = RacePointsContext(100, laps, green)
    summary = scoring_context_summary(context)
    assert summary["policy"] == RACE_POINTS_POLICY
    assert summary["winner_points"] == award
    assert summary["distance_band"] == band
    assert summary["ineligibility_reason"] == reason
    assert summary["points_eligible"] is (reason is None)
    # Derived descriptions in a JSON round trip are recomputed from raw inputs.
    saved = json.loads(json.dumps(summary))
    saved["description"] = "<script>forged evidence</script>"
    saved["winner_points"] = 99
    assert scoring_context_summary(saved) == summary


@pytest.mark.parametrize("field,value", [
    ("scheduled_laps", True), ("scheduled_laps", 0), ("scheduled_laps", 100.0),
    ("scheduled_laps", float("nan")), ("winner_laps", True), ("winner_laps", 0),
    ("winner_laps", 101), ("winner_laps", 25.0), ("has_two_green_laps", 1),
    ("has_two_green_laps", "true"), ("policy", "unknown"), ("policy", None),
])
def test_invalid_recorded_inputs_do_not_claim_a_scoring_reason(field, value):
    data = asdict(RacePointsContext(100, 100, True))
    data[field] = value
    result = row(data)
    assert scoring_context_summary(data) is None
    assert points_reason_for_result(result) is None
    assert race_scoring_context([result]) is None
    assert points_for_result(result) == 25


@pytest.mark.parametrize("field", ["scheduled_laps", "winner_laps", "has_two_green_laps", "policy"])
def test_missing_context_fields_remain_unknown(field):
    data = asdict(RacePointsContext(100, None, False))
    del data[field]
    assert scoring_context_summary(data) is None


@pytest.mark.parametrize("context,award,position,classified,expected", [
    (RacePointsContext(100, 100, True), 25, 1, True, "full_distance"),
    (RacePointsContext(100, 24, True), 6, 1, True, "reduced_distance"),
    (RacePointsContext(100, 24, True), 0, 6, True, "outside_points_positions"),
    (RacePointsContext(100, 100, True), 0, 11, True, "outside_points_positions"),
    (RacePointsContext(100, 100, True), 15, 3, True, "full_distance"),
    (RacePointsContext(100, 100, True), 0, 3, False, "not_classified"),
    (RacePointsContext(100, 100, False), 0, 3, True, "no_green_pair"),
    (RacePointsContext(100, None, True), 0, 1, False, "no_winner"),
    (None, 0, 1, True, None),
])
def test_driver_reason_retains_classified_retirements_and_explicit_zero(
    context, award, position, classified, expected,
):
    result = row(context, award=award, position=position, classified=classified,
                 retired=position > 1 or (context is not None and context.winner_laps is None))
    assert points_reason_for_result(result) == expected
    assert _serialize_race_result(result)["points_reason"] == expected
    assert points_for_result(result) == award


def test_inconsistent_awards_distances_or_contexts_do_not_establish_race_evidence():
    context = RacePointsContext(100, 100, True)
    assert race_scoring_context([row(context, award=6)]) is None
    assert race_scoring_context([row(context, laps=99)]) is None
    assert race_scoring_context([row(context, retired=True)]) is None
    assert points_reason_for_result(row(context, retired=True)) is None
    assert race_scoring_context([row(RacePointsContext(100, None, False), award=0)]) is None
    assert race_scoring_context([row(RacePointsContext(100, 1, True), award=0, laps=True)]) is None
    assert race_scoring_context([row(context), row(None, position=2, award=18)]) is None
    assert race_scoring_context([row(context)], asdict(RacePointsContext(100, 24, True))) is None
    assert race_scoring_context([], asdict(context)) is None
    assert race_scoring_context([]) is None
    legacy = row(award=None)
    assert points_for_result(legacy) == 25 and points_reason_for_result(legacy) is None


def mixed_results():
    races = [[row(RacePointsContext(100, 100, True))],
             [row(RacePointsContext(100, 24, True), award=6, laps=24)],
             [row(RacePointsContext(100, 100, False), award=0)],
             [row(award=None)]]
    stats = DriverStatistics("A", "Driver A", "Team", positions=[1] * 4,
                             wins=4, total_points=56)
    return SimulationResults(100, "Recorded", {"A": stats}, races, [], seed=7)


def test_coverage_counts_observed_races_and_keeps_missing_evidence_out_of_denominator():
    result = mixed_results()
    assert result.get_race_scoring_statistics() == {
        "recorded_races": 4, "races_with_scoring_evidence": 3,
        "races_without_scoring_evidence": 1, "full_points_races": 1,
        "reduced_points_races": 1, "zero_points_races": 1,
        "zero_points_reasons": {"no_winner": 0, "fewer_than_two_laps": 0, "no_green_pair": 1},
        "zero_points_race_rate": 1 / 3,
    }
    assert result.get_race_scoring_contexts()[-1] is None
    for index in (-1, 4, True, 1.0, None):
        assert result.get_race_scoring_context(index) is None
    empty = SimulationResults(100, "Empty", {}, [[]], [])
    assert empty.get_race_scoring_statistics()["zero_points_race_rate"] is None
    empty.race_points_contexts = [asdict(RacePointsContext(100, None, False))]
    assert empty.get_race_scoring_statistics()["zero_points_reasons"]["no_winner"] == 1
    assert empty.get_race_scoring_context()["ineligibility_reason"] == "no_winner"


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_native_context_is_shared_frozen_and_replaced_on_reuse(monkeypatch, engine):
    simulator = RaceSimulator(np.random.default_rng(7), control_schedule=[])
    assert simulator.race_points_context is None
    run = simulator.simulate_race if engine == "standard" else ChronologicalRace(simulator).run
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    track = Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90)
    weather = Weather(track_wetness=.3, rain_intensity=.3, change_probability=0)
    manager = simulator.event_manager
    monkeypatch.setattr(manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(manager, "_check_red_flag_conditions", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", lambda *a, **kw: 90.)
    options = dict(starting_tires={key: "intermediate" for key in "AB"},
                   pit_plans={key: [] for key in "AB"})
    finished = run(drivers, cars, track, weather, ["A", "B"], **options)
    context = simulator.race_points_context
    assert context == RacePointsContext(4, 4, True)
    assert all(result.race_points_context is context for result in finished)
    with pytest.raises(FrozenInstanceError):
        context.winner_laps = 3
    assert race_scoring_context(finished)["winner_points"] == 25

    def retire(driver, car, track, lap, weather):
        driver.dnf = True
        return RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id])

    monkeypatch.setattr(manager, "_check_mechanical_failure", retire)
    retired = run(drivers, cars, track, weather, ["A", "B"], **options)
    assert simulator.race_points_context == RacePointsContext(4, None, False)
    assert all(points_reason_for_result(result) == "no_winner" for result in retired)
    assert run(drivers, {}, track, weather, ["A", "B"], **options) == []
    assert simulator.race_points_context == RacePointsContext(4, None, False)
    # A subsequent race cannot mutate evidence retained by the first result list.
    assert all(result.race_points_context is context for result in finished)


def test_worker_evidence_survives_process_pool_and_matches_trial_rows():
    drivers = [Driver(id=key, name=key, team_id=key) for key in "AB"]
    cars = {key: Car(team_id=key, team_name=key) for key in "AB"}
    runner = MonteCarloRunner(drivers, cars, Track(id="T", name="T", country="T", total_laps=3,
                                                 base_lap_time=90),
                              Weather(change_probability=0), seed=7, control_schedule=[])
    sequential = runner.run(2, parallel=False)
    parallel = runner.run(2, parallel=True, max_workers=2)
    assert sequential.race_points_contexts == parallel.race_points_contexts
    assert sequential.get_race_scoring_contexts() == parallel.get_race_scoring_contexts()
    assert len(parallel.race_points_contexts) == len(parallel.race_results) == 2
    assert all(context is not None for context in parallel.get_race_scoring_contexts())


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_empty_worker_race_retains_no_winner_evidence(engine):
    result = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="A")], {},
        Track(id="T", name="T", country="T", total_laps=4, base_lap_time=90),
        Weather(change_probability=0), seed=7, race_engine=engine,
    ).run(2, parallel=False)
    assert result.race_results == [[], []]
    assert len(result.race_points_contexts) == 2
    assert result.get_race_scoring_statistics()["zero_points_reasons"]["no_winner"] == 2


def test_dashboard_scoring_context_uses_the_selected_representative_trial():
    full, reduced = RacePointsContext(100, 100, True), RacePointsContext(100, 24, True)
    races = [[row(full), row(full, position=2, award=18)],
             [row(reduced, position=2, award=4, laps=24), row(reduced, award=6, laps=24)]]
    # Use distinct identities, with trial two matching the supplied average places.
    races[0][1].driver_id = races[1][1].driver_id = "B"
    result = SimulationResults(2, "T", {
        "A": DriverStatistics("A", "A", "Team", avg_position=2),
        "B": DriverStatistics("B", "B", "Team", avg_position=1),
    }, races, [])
    payload = _summarize_scenario_results({"choice": result})["scenarios"]["choice"]
    assert payload["sample_index"] == 1
    assert payload["sample_race_scoring_context"]["winner_laps"] == 24
    assert payload["sample_race_scoring_context"]["winner_points"] == 6
    assert payload["sample_race"][1]["driver_id"] == "B"


def test_api_csv_json_html_and_console_share_scoring_evidence(tmp_path, capsys):
    results = mixed_results()
    exporter = Exporter(tmp_path)
    single = json.loads(exporter.export_statistics_json(results).read_text(encoding="utf-8"))
    comparison = json.loads(exporter.export_scenario_comparison_json(
        {"choice": results},
    ).read_text(encoding="utf-8"))["scenarios"]["choice"]
    for key in ("race_scoring_statistics", "race_scoring_contexts"):
        assert single[key] == comparison[key]
    payload = _summarize_scenario_results({"choice": results})["scenarios"]["choice"]
    assert payload["race_scoring_statistics"] == single["race_scoring_statistics"]
    assert payload["sample_race_scoring_context"] == results.get_race_scoring_context(
        payload["sample_index"],
    )
    with exporter.export_race_results_csv(results).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [item["points_awarded"] for item in rows] == ["25", "6", "0", "25"]
    assert [item["points_reason"] for item in rows] == [
        "full_distance", "reduced_distance", "no_green_pair", "",
    ]
    assert rows[1]["winner_laps"] == "24" and rows[1]["scheduled_laps"] == "100"
    assert rows[2]["has_two_green_laps"] == "false"
    assert rows[3]["race_points_policy"] == rows[3]["has_two_green_laps"] == ""
    malicious = '<img src=x onerror="alert(1)">'
    reports = [exporter.export_report_html(results).read_text(encoding="utf-8"),
               render_comparison_report({malicious: results})]
    for report in reports:
        assert "Race points" in report and "3 of 4 recorded races; 1 unknown" in report
        assert "Two consecutive complete green laps were not recorded: 1" in report
        assert malicious not in report
    ConsoleOutput.print_race_results(results.race_results[1])
    assert "Reduced points schedule (6 for first place)" in capsys.readouterr().out
    ConsoleOutput.print_race_results(results.race_results[3])
    assert "Race scoring evidence: Not recorded." in capsys.readouterr().out
    ConsoleOutput.print_monte_carlo_summary(results)
    assert "Full points: 1; reduced points: 1; no points: 1" in capsys.readouterr().out
