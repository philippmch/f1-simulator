"""Fresh native compulsory-wet evidence for the dashboard and HTML exports."""

import json
from tempfile import TemporaryDirectory

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.output import Exporter
from f1sim.output.comparison import render_comparison_report
from f1sim.web.server import _summarize_scenario_results


def build_fixture():
    track = Track(id="t", name="Native wet resumption", country="Synthetic", total_laps=8,
                  base_lap_time=90., safety_car_probability=0.)
    results, reports, csvs = {}, {}, {}
    keys = ("A", "B")
    with TemporaryDirectory() as folder:
        exporter = Exporter(folder)
        for engine in ("standard", "chronological"):
            result = MonteCarloRunner(
                [Driver(id=key, name=key, team_id=key, consistency=1.) for key in keys],
                {key: Car(team_id=key, team_name=key, reliability=1.) for key in keys},
                track, Weather(change_probability=0.), seed=91, race_engine=engine,
                rng_policy="isolated_race_v1", starting_tires={key: "medium" for key in keys},
                tire_inventory={key: [
                    {"id": "M", "compound": "medium"}, {"id": "H", "compound": "hard"},
                    {"id": "W", "compound": "wet", "age": 9, "remaining_laps": 1},
                ] for key in keys},
                pit_plans={key: [{"lap": 3, "compound": "hard"},
                                 {"lap": 4, "compound": "hard"}] for key in keys},
                control_schedule=[{"lap": 2, "control": "red_flag", "action": "resume_wet"}],
            ).run(1, parallel=False)
            assert native_physics()
            assert result.input_snapshot["schema_version"] == 13
            results[engine] = result
            reports[engine] = exporter.export_report_html(result).read_text(encoding="utf-8")
            csvs[engine] = exporter.export_control_schedule_history_csv(result).read_text(
                encoding="utf-8")
        payload = _summarize_scenario_results(results)
        payload.update(track=track.name, track_details=track.model_dump(), request={})
    return {"payload": payload, "reports": reports, "csvs": csvs,
            "comparison": render_comparison_report(results), "native": native_physics()}


if __name__ == "__main__":
    print(json.dumps(build_fixture(), allow_nan=False))
