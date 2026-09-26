"""Build real HTML exports with markup-like names for an offline browser test."""

import json
from copy import deepcopy
from tempfile import TemporaryDirectory

from f1sim.analysis.montecarlo import DriverStatistics, MonteCarloRunner, SimulationResults
from f1sim.analysis.paired_comparison import paired_comparison_statistics
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.export import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult


def _scalable_plan_result(driver_id, marker, *, hostile_last=False):
    races = []
    for trial in range(1, 1002):
        set_id = f"{marker}-{trial}"
        if hostile_last and trial == 1001:
            set_id = "</script><script>globalThis.planHistoryInjected=true</script>"
        races.append([RaceResult(
            driver_id, driver_id, "Scalability Team", 1, 90, 0, 1, 90,
            DriverStatus.FINISHED,
            pit_plan_history=[{
                "lap": 1, "compound": "hard", "status": "executed",
                "actual_compound": "hard", "actual_set_id": set_id,
            }],
        )])
    if hostile_last:
        races = [[], *races[:500], [], *races[500:]]
    return SimulationResults(
        num_simulations=len(races), track_name="Scalable history", driver_stats={}, seed=71,
        race_results=races, qualifying_results=[],
        input_snapshot={"pit_plans": {driver_id: [{"lap": 1, "compound": "hard"}]}},
    )


def build_fixture():
    script = "</script><script>globalThis.exportInjected=true</script>"
    track = f'Montréal </title>{script}<img src=x onerror="globalThis.exportInjected=true">'
    driver = f"Driver {script} & 'quoted'"
    team = f"Team {script} & <test>"
    filename = "Montréal & report's.html"
    stats_name = "javascript:globalThis.exportInjected=true"
    results = SimulationResults(
        num_simulations=1, track_name=track, seed=42,
        race_results=[[RaceResult(driver, driver, team, 1, 100, 0, 2, 90, DriverStatus.DNF,
                                  strategy=["soft", script, "soft"],
                                  pit_stop_details=[
                                      {"decision_reason": "dry_forecast"},
                                      {"decision_reason": script},
                                  ])]], qualifying_results=[],
        driver_stats={driver: DriverStatistics(driver_id=driver, driver_name=driver, team=team,
                                            wins=1, positions=[1], total_points=25)},
        input_snapshot={"starting_tires": {driver: "soft"},
                        "starting_tire_ages": {driver: 5},
                        "weather": Weather().model_dump(),
                        "rng_policy": "isolated_weather_v1"},
    )
    with TemporaryDirectory() as directory:
        exporter = Exporter(directory)
        empty = SimulationResults(
            num_simulations=5, track_name="No observations", driver_stats={},
            race_results=[], qualifying_results=[], seed=43,
        )
        comparison = exporter.export_scenario_comparison_html(
            {script: results, "No observations": empty}, focus_driver=driver,
        ).read_text(encoding="utf-8")
        paired_variants = {
            compound: MonteCarloRunner(
                [Driver(id="A", name="Driver A", team_id="T")],
                {"T": Car(team_id="T", team_name="Team")},
                Track(id="t", name="Test", country="Test", total_laps=5, base_lap_time=90),
                Weather(change_probability=0), seed=41, starting_tires={"A": compound},
            ).run(3, parallel=False)
            for compound in ("soft", "hard")
        }
        paired = exporter.export_scenario_comparison_html(
            paired_variants, filename="paired.html", focus_driver="A", reference_scenario="hard",
        ).read_text(encoding="utf-8")
        paired_stats = paired_comparison_statistics(paired_variants, "hard")
        report = exporter.export_report_html(results, filename).read_text(encoding="utf-8")
        planned = deepcopy(results)
        planned.input_snapshot["pit_plans"] = {
            driver: [{"lap": lap, "compound": "hard"} for lap in range(2, 6)],
        }
        planned.race_results[0][0].pit_plan_history = [
            {"lap": lap, "compound": "hard", "status": status,
             "reason": reason, "actual_compound": "hard" if lap < 4 else None,
             "actual_set_id": script if lap < 4 else None}
            for lap, status, reason in (
                (2, "executed", "user_plan"),
                (3, "overridden", "forced_repair"),
                (4, "skipped", "requested_compound_unavailable"),
                (5, "not_reached", "retired"),
            )
        ]
        scalable_first = _scalable_plan_result("Scale A", "first", hostile_last=True)
        scalable_second = _scalable_plan_result("Scale B", "second")
        held = deepcopy(planned)
        held.input_snapshot["pit_plans"][driver] = []
        held.race_results[0][0].pit_plan_history = []
        plan_report = exporter.export_scenario_comparison_html(
            {"Custom plan": planned, "No elective stops": held, "Legacy": results},
            filename="plans.html", focus_driver=driver,
        ).read_text(encoding="utf-8")
        scalable_plan_report = exporter.export_scenario_comparison_html(
            {"First large plan": scalable_first, "Second large plan": scalable_second},
            filename="large-plans.html",
        ).read_text(encoding="utf-8")
        scalable_run_report = exporter.export_report_html(
            scalable_first, filename="large-run.html",
        ).read_text(encoding="utf-8")
        exporter._write_history([{
            "timestamp": "<img src=x onerror=globalThis.exportInjected=true>",
            "track": track, "num_simulations": 1, "seed": 42,
            "files": {"report_html": filename, "statistics_json": stats_name},
        }])
        index = exporter.export_run_index_html().read_text(encoding="utf-8")
    return {"report": report, "comparison": comparison, "index": index,
            "paired": paired, "plan_report": plan_report,
            "scalable_plan_report": scalable_plan_report,
            "scalable_run_report": scalable_run_report,
            "paired_stats": paired_stats["variants"]["soft"]["driver_statistics"]["A"],
            "track": track, "driver": driver,
            "team": team, "filename": filename, "stats_name": stats_name}


if __name__ == "__main__":
    print(json.dumps(build_fixture()))
