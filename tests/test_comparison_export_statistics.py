"""Combined exports retain the observations and uncertainty shown per run."""
import json

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output import Exporter


def test_comparison_statistics_match_single_export_with_observed_denominators(tmp_path):
    stats = DriverStatistics(
        driver_id="A", driver_name="A", team="T", positions=[1, 2, 5, 8],
        wins=1, podiums=2, dnfs=1, total_points=37.5,
    )
    empty = DriverStatistics(driver_id="B", driver_name="B", team="T")
    result = SimulationResults(
        num_simulations=100, track_name="Observed", driver_stats={"A": stats, "B": empty},
        race_results=[], qualifying_results=[], seed=7,
    )
    exporter = Exporter(tmp_path)
    single = json.loads(exporter.export_statistics_json(result).read_text(encoding="utf-8"))
    combined = json.loads(exporter.export_scenario_comparison_json(
        {"first": result},
    ).read_text(encoding="utf-8"))["scenarios"]["first"]
    for key in ("driver_statistics", "probability_intervals", "pit_stop_statistics", "event_rates"):
        assert combined[key] == single[key]
    assert combined["num_simulations"] == 100
    assert combined["driver_statistics"]["A"]["recorded_races"] == 4
    assert combined["driver_statistics"]["A"]["points_per_race"] == 9.375
    assert combined["probability_intervals"]["A"]["trials"] == 4
    assert combined["driver_statistics"]["B"]["points_per_race"] is None
    assert combined["pit_stop_statistics"] == {}
