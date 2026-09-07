"""Build real HTML exports with markup-like names for an offline browser test."""

import json
from tempfile import TemporaryDirectory

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter


def build_fixture():
    script = "</script><script>globalThis.exportInjected=true</script>"
    track = f'Montréal </title>{script}<img src=x onerror="globalThis.exportInjected=true">'
    driver = f"Driver {script} & 'quoted'"
    team = f"Team {script} & <test>"
    filename = "Montréal & report's.html"
    stats_name = "javascript:globalThis.exportInjected=true"
    results = SimulationResults(
        num_simulations=1, track_name=track, seed=42, race_results=[], qualifying_results=[],
        driver_stats={"A": DriverStatistics(driver_id="A", driver_name=driver, team=team,
                                            wins=1, positions=[1], total_points=25)},
    )
    with TemporaryDirectory() as directory:
        exporter = Exporter(directory)
        report = exporter.export_report_html(results, filename).read_text(encoding="utf-8")
        exporter._write_history([{
            "timestamp": "<img src=x onerror=globalThis.exportInjected=true>",
            "track": track, "num_simulations": 1, "seed": 42,
            "files": {"report_html": filename, "statistics_json": stats_name},
        }])
        index = exporter.export_run_index_html().read_text(encoding="utf-8")
    return {"report": report, "index": index, "track": track, "driver": driver,
            "team": team, "filename": filename, "stats_name": stats_name}


if __name__ == "__main__":
    print(json.dumps(build_fixture()))
