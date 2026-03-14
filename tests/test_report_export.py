from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter


def test_report_html_export_contains_core_sections(tmp_path) -> None:
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
    results = SimulationResults(
        num_simulations=1,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
        seed=42,
    )

    exporter = Exporter(output_dir=tmp_path)
    out = exporter.export_report_html(results)
    html = out.read_text()

    assert "F1 Simulation Report" in html
    assert "Bahrain" in html
    assert "plotly" in html.lower()


def test_export_all_includes_report_html(tmp_path) -> None:
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
    results = SimulationResults(
        num_simulations=1,
        track_name="Bahrain",
        driver_stats=stats,
        race_results=[],
        qualifying_results=[],
        seed=7,
    )

    exporter = Exporter(output_dir=tmp_path)
    files = exporter.export_all(results, prefix="test")

    assert "report_html" in files
    assert files["report_html"].exists()
