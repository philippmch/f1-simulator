"""Fresh native countback and no-result evidence for the real dashboard."""

import json
from tempfile import TemporaryDirectory

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather
from f1sim.models._native import native_physics
from f1sim.output import Exporter
from f1sim.output.comparison import render_comparison_report
from f1sim.web.server import _summarize_scenario_results


def build_fixture():
    track = Track(id="t", name="Native abandonment", country="Synthetic", total_laps=8,
                  base_lap_time=90., safety_car_probability=0.)
    results, reports, csvs = {}, {}, {}
    with TemporaryDirectory() as folder:
        exporter = Exporter(folder)
        for engine in ("standard", "chronological"):
            for after in (1, 4):
                drivers = [Driver(id=key, name=key, team_id=key, consistency=1.)
                           for key in ("A", "B")]
                cars = {key: Car(team_id=key, team_name=key, reliability=1., base_pace=.8)
                        for key in ("A", "B")}
                result = MonteCarloRunner(
                    drivers, cars, track, Weather(change_probability=0.), seed=91,
                    race_engine=engine, rng_policy="isolated_race_v1",
                    starting_tires={key: "medium" for key in ("A", "B")},
                    pit_plans={key: [] for key in ("A", "B")},
                    control_schedule=[{"lap": after, "control": "red_flag", "action": "abandon"}],
                ).run(1, parallel=False)
                assert native_physics()
                context = result.get_race_abandonment_context()
                assert context is not None and context["countback_lap"] == after - 1
                key = f"{engine}_{after}"
                results[key] = result
                reports[key] = exporter.export_report_html(result).read_text(encoding="utf-8")
                csvs[key] = exporter.export_control_schedule_history_csv(result).read_text(
                    encoding="utf-8",
                )
        payload = _summarize_scenario_results(results)
        payload.update(track=track.name, track_details=track.model_dump(), request={})
    return {"payload": payload, "reports": reports, "native": native_physics(), "csvs": csvs,
            "comparison": render_comparison_report(results)}


if __name__ == "__main__":
    print(json.dumps(build_fixture(), allow_nan=False))
