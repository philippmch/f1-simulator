"""Export simulation results to CSV and JSON."""

import csv
import json
from pathlib import Path
from typing import Any

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.simulation.qualifying import QualifyingResult
from f1sim.simulation.race import RaceResult


class Exporter:
    """Exports simulation results to various formats."""

    def __init__(self, output_dir: str | Path = "output"):
        """Initialize exporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_race_results_csv(
        self,
        results: SimulationResults,
        filename: str = "race_results.csv",
    ) -> Path:
        """Export all race results to CSV.

        Args:
            results: Simulation results
            filename: Output filename

        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "simulation", "position", "driver_id", "driver_name", "team",
                "total_time", "gap_to_leader", "pit_stops", "fastest_lap",
                "status", "dnf_reason", "strategy",
            ])

            for sim_idx, race_results in enumerate(results.race_results, 1):
                for result in race_results:
                    writer.writerow([
                        sim_idx,
                        result.position,
                        result.driver_id,
                        result.driver_name,
                        result.team,
                        f"{result.total_time:.3f}",
                        f"{result.gap_to_leader:.3f}",
                        result.pit_stops,
                        f"{result.fastest_lap:.3f}",
                        result.status.value,
                        result.dnf_reason or "",
                        ",".join(result.strategy),
                    ])

        return filepath

    def export_qualifying_results_csv(
        self,
        results: SimulationResults,
        filename: str = "qualifying_results.csv",
    ) -> Path:
        """Export all qualifying results to CSV.

        Args:
            results: Simulation results
            filename: Output filename

        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "simulation", "position", "driver_id", "driver_name",
                "best_time", "q1_time", "q2_time", "q3_time", "eliminated_in",
            ])

            for sim_idx, quali_results in enumerate(results.qualifying_results, 1):
                for result in quali_results:
                    writer.writerow([
                        sim_idx,
                        result.position,
                        result.driver_id,
                        result.driver_name,
                        f"{result.best_time:.3f}",
                        f"{result.q1_time:.3f}" if result.q1_time else "",
                        f"{result.q2_time:.3f}" if result.q2_time else "",
                        f"{result.q3_time:.3f}" if result.q3_time else "",
                        result.eliminated_in or "",
                    ])

        return filepath

    def export_statistics_json(
        self,
        results: SimulationResults,
        filename: str = "statistics.json",
    ) -> Path:
        """Export aggregated statistics to JSON.

        Args:
            results: Simulation results
            filename: Output filename

        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename

        # Build statistics dictionary
        stats_dict: dict[str, Any] = {
            "metadata": {
                "num_simulations": results.num_simulations,
                "track_name": results.track_name,
            },
            "win_probabilities": results.get_win_probabilities(),
            "championship_projection": results.get_championship_projection(),
            "driver_statistics": {},
        }

        for driver_id, stats in results.driver_stats.items():
            stats_dict["driver_statistics"][driver_id] = {
                "driver_name": stats.driver_name,
                "team": stats.team,
                "wins": stats.wins,
                "win_rate": stats.win_rate,
                "podiums": stats.podiums,
                "podium_rate": stats.podium_rate,
                "points_finishes": stats.points_finishes,
                "dnfs": stats.dnfs,
                "dnf_rate": stats.dnf_rate,
                "total_points": stats.total_points,
                "avg_position": stats.avg_position,
                "avg_qualifying": stats.avg_qualifying,
                "best_position": stats.best_position,
                "worst_position": stats.worst_position,
                "position_distribution": results.get_position_distribution(driver_id),
            }

        with open(filepath, "w") as f:
            json.dump(stats_dict, f, indent=2)

        return filepath

    def export_all(
        self,
        results: SimulationResults,
        prefix: str = "",
    ) -> dict[str, Path]:
        """Export all result formats.

        Args:
            results: Simulation results
            prefix: Optional prefix for filenames

        Returns:
            Dictionary of format -> filepath
        """
        prefix = f"{prefix}_" if prefix else ""

        return {
            "race_csv": self.export_race_results_csv(
                results, f"{prefix}race_results.csv"
            ),
            "qualifying_csv": self.export_qualifying_results_csv(
                results, f"{prefix}qualifying_results.csv"
            ),
            "statistics_json": self.export_statistics_json(
                results, f"{prefix}statistics.json"
            ),
        }
