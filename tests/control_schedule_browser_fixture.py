"""Fresh native SC/VSC evidence for offline dashboard and export checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.output import Exporter
from f1sim.output.comparison import render_comparison_report
from f1sim.web.server import _summarize_scenario_results


def build_fixture():
    drivers = [Driver(id=f"S{i:02}", name=f"Synthetic Driver {i}", team_id=f"team{i // 2}")
               for i in range(22)]
    cars = {driver.team_id: Car(team_id=driver.team_id, team_name=driver.team_id)
            for driver in drivers}
    schedule = [{"lap": 2, "control": "safety_car", "duration_laps": 2},
                {"lap": 6, "control": "vsc", "duration_laps": 1},
                {"lap": 12, "control": "safety_car", "duration_laps": 3}]
    plans = {driver.id: [] for driver in drivers}
    plans["S00"] = [{"lap": 5, "earliest_lap": 3, "trigger": "safety_car", "compound": "hard"}]
    plans["S01"] = [{"lap": 9, "earliest_lap": 7, "trigger": "vsc", "compound": "hard"}]
    result = MonteCarloRunner(
        drivers, cars, Track(id="s", name="Synthetic control case", country="Test",
                            total_laps=12, base_lap_time=90), Weather(change_probability=0),
        seed=91, race_engine="chronological", rng_policy="isolated_race_v1",
        starting_tires={driver.id: "medium" for driver in drivers},
        pit_plans=plans, control_schedule=schedule,
    ).run(2, parallel=False)
    assert native_physics()
    payload = _summarize_scenario_results({"controlled": result})
    with TemporaryDirectory() as folder:
        exporter = Exporter(folder)
        report = exporter.export_report_html(result).read_text(encoding="utf-8")
        csv_text = exporter.export_control_schedule_history_csv(result).read_text(encoding="utf-8")
    return {"payload": payload, "schedule": schedule, "report": report,
            "comparison": render_comparison_report({"controlled": result}),
            "csv": csv_text, "native": native_physics(), "runtime_source": str(Path(__file__))}


if __name__ == "__main__":
    print(json.dumps(build_fixture(), allow_nan=False))
