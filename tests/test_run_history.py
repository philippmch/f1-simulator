import json

import pytest

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter


def _results(seed: int) -> SimulationResults:
    stats = {
        "VER": DriverStatistics(
            driver_id="VER",
            driver_name="Max Verstappen",
            team="Red Bull",
            wins=1,
            positions=[1],
            total_points=25.0,
        )
    }
    return SimulationResults(
        num_simulations=10,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
        seed=seed,
    )


def test_export_all_writes_run_history_and_index(tmp_path) -> None:
    exporter = Exporter(output_dir=tmp_path)
    files = exporter.export_all(_results(seed=11), prefix="run1")

    history_path = tmp_path / ".run_history.json"
    assert history_path.exists()

    history = json.loads(history_path.read_text())
    assert len(history) == 1
    assert history[0]["prefix"] == "run1"
    assert history[0]["seed"] == 11

    assert "runs_index_html" in files
    assert files["runs_index_html"].exists()


def test_run_history_keeps_latest_first(tmp_path) -> None:
    exporter = Exporter(output_dir=tmp_path)
    exporter.export_all(_results(seed=1), prefix="first")
    exporter.export_all(_results(seed=2), prefix="second")

    history = json.loads((tmp_path / ".run_history.json").read_text())
    assert history[0]["prefix"] == "second"
    assert history[1]["prefix"] == "first"


@pytest.mark.parametrize("prefix", ["", "same_race_dry"])
def test_repeated_prefix_preserves_original_artifacts_and_model_history(tmp_path, prefix):
    exporter = Exporter(output_dir=tmp_path)
    first = exporter.export_all(_results(seed=1), prefix=prefix)
    saved = {key: path.read_bytes() for key, path in first.items() if key != "runs_index_html"}
    next_result = _results(seed=2)
    next_result.race_engine = "chronological"
    second = exporter.export_all(next_result, prefix=prefix)
    for key, content in saved.items():
        assert first[key] != second[key]
        assert first[key].read_bytes() == content
    history = json.loads((tmp_path / ".run_history.json").read_text())
    assert [row["race_engine"] for row in history] == ["chronological", "standard"]
    for row in history:
        stats = json.loads((tmp_path / row["files"]["statistics_json"]).read_text())
        assert stats["metadata"]["seed"] == row["seed"]
        assert stats["metadata"]["race_engine"] == row["race_engine"]
    html = second["runs_index_html"].read_text(encoding="utf-8")
    assert "Race model" in html
    assert "Lap-aware (experimental)" in html
    assert "Standard" in html
