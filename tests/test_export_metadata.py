import json

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter


def test_statistics_export_includes_run_metadata(tmp_path) -> None:
    stats = {
        "VER": DriverStatistics(
            driver_id="VER",
            driver_name="Max Verstappen",
            team="Red Bull",
            wins=1,
            positions=[1],
        )
    }
    results = SimulationResults(
        num_simulations=1,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
        seed=123,
        parallel=False,
        max_workers=2,
    )

    exporter = Exporter(output_dir=tmp_path)
    out = exporter.export_statistics_json(results)

    data = json.loads(out.read_text())
    metadata = data["metadata"]
    assert metadata["seed"] == 123
    assert metadata["parallel"] is False
    assert metadata["max_workers"] == 2
    assert "team_championship_projection" in data
    assert "top_3_finish_probabilities" in data
    assert "top_10_finish_probabilities" in data
