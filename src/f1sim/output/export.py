"""Export simulation results to CSV and JSON."""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from f1sim.analysis.montecarlo import SimulationResults


class Exporter:
    """Exports simulation results to various formats."""

    _PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"
    _HISTORY_FILE = ".run_history.json"
    _INDEX_FILE = "index.html"

    def __init__(self, output_dir: str | Path = "output"):
        """Initialize exporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _history_path(self) -> Path:
        return self.output_dir / self._HISTORY_FILE

    def _read_history(self) -> list[dict[str, Any]]:
        path = self._history_path()
        if not path.exists():
            return []

        data = json.loads(path.read_text())
        if not isinstance(data, list):
            return []

        return [row for row in data if isinstance(row, dict)]

    def _write_history(self, history: list[dict[str, Any]]) -> None:
        self._history_path().write_text(json.dumps(history, indent=2))

    def _record_export_run(
        self,
        results: SimulationResults,
        files: dict[str, Path],
        prefix: str,
    ) -> None:
        history = self._read_history()
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "track": results.track_name,
            "num_simulations": results.num_simulations,
            "seed": results.seed,
            "prefix": prefix,
            "files": {k: p.name for k, p in files.items()},
        }
        history.insert(0, entry)

        # Keep only latest 100 runs to avoid unbounded growth.
        self._write_history(history[:100])

    def export_run_index_html(self, filename: str = _INDEX_FILE) -> Path:
        """Export a simple HTML index page for browsing run history."""
        filepath = self.output_dir / filename
        history = self._read_history()

        rows = []
        for row in history:
            ts = row.get("timestamp", "-")
            track = row.get("track", "-")
            sims = row.get("num_simulations", "-")
            seed = row.get("seed", "-")
            files = row.get("files", {})
            report = files.get("report_html") if isinstance(files, dict) else None
            stats = files.get("statistics_json") if isinstance(files, dict) else None
            links = []
            if report:
                links.append(f"<a href=\"{report}\">report</a>")
            if stats:
                links.append(f"<a href=\"{stats}\">stats</a>")
            row_links = " | ".join(links) if links else "-"
            rows.append(
                f"<tr><td>{ts}</td><td>{track}</td><td>{sims}</td><td>{seed}</td><td>{row_links}</td></tr>"
            )

        table_rows = "\n".join(rows) if rows else "<tr><td colspan='5'>No runs yet</td></tr>"

        html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>F1 Sim Run History</title>
  <style>
    body {{
      font-family: Inter, system-ui, sans-serif;
      margin: 24px;
      background: #0f1220;
      color: #e8ebff;
    }}
    table {{ width: 100%; border-collapse: collapse; background: #181c30; }}
    th, td {{ border: 1px solid #2a3156; padding: 10px; text-align: left; }}
    a {{ color: #7aa2ff; }}
  </style>
</head>
<body>
  <h1>F1 Simulation Run History</h1>
  <table>
    <thead>
      <tr>
        <th>Timestamp (UTC)</th><th>Track</th><th>Simulations</th><th>Seed</th><th>Artifacts</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
        filepath.write_text(html)
        return filepath

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
                "seed": results.seed,
                "parallel": results.parallel,
                "max_workers": results.max_workers,
            },
            "win_probabilities": results.get_win_probabilities(),
            "top_3_finish_probabilities": results.get_top_n_finish_probabilities(3),
            "top_10_finish_probabilities": results.get_top_n_finish_probabilities(10),
            "championship_projection": results.get_championship_projection(),
            "team_championship_projection": results.get_team_championship_projection(),
            "mechanical_failure_breakdown": results.event_stats.mechanical_failure_breakdown,
            "driver_statistics": {},
        }

        top_5 = results.get_top_n_finish_probabilities(5)
        top_10 = results.get_top_n_finish_probabilities(10)

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
                "position_percentiles": results.get_position_percentiles(driver_id),
                "top_5_finish_probability": top_5.get(driver_id, 0.0),
                "top_10_finish_probability": top_10.get(driver_id, 0.0),
            }

        with open(filepath, "w") as f:
            json.dump(stats_dict, f, indent=2)

        return filepath

    def export_scenario_comparison_json(
        self,
        scenario_results: dict[str, SimulationResults],
        filename: str = "scenario_comparison.json",
    ) -> Path:
        """Export scenario-level win probability comparison to JSON."""
        filepath = self.output_dir / filename

        payload: dict[str, Any] = {"scenarios": {}}
        for name, results in scenario_results.items():
            payload["scenarios"][name] = {
                "num_simulations": results.num_simulations,
                "seed": results.seed,
                "win_probabilities": results.get_win_probabilities(),
                "team_championship_projection": results.get_team_championship_projection(),
            }

        with open(filepath, "w") as f:
            json.dump(payload, f, indent=2)

        return filepath

    def export_report_html(
        self,
        results: SimulationResults,
        filename: str = "report.html",
    ) -> Path:
        """Export interactive HTML report with basic charts."""
        filepath = self.output_dir / filename

        win_probs = results.get_win_probabilities()
        top_items = list(win_probs.items())[:10]
        top_labels = [results.driver_stats[d].driver_name for d, _ in top_items]
        top_values = [v for _, v in top_items]

        team_proj = results.get_team_championship_projection()
        team_labels = list(team_proj.keys())[:10]
        team_values = [team_proj[t] for t in team_labels]

        html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>F1 Simulation Report - {results.track_name}</title>
  <script src=\"{self._PLOTLY_CDN}\"></script>
  <style>
    body {{
      font-family: Inter, system-ui, sans-serif;
      margin: 24px;
      background: #0f1220;
      color: #e8ebff;
    }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
    .card {{ background: #181c30; border: 1px solid #2a3156; border-radius: 12px; padding: 16px; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ color: #b6c0ff; margin-bottom: 16px; }}
  </style>
</head>
<body>
  <h1>F1 Simulation Report</h1>
  <div class=\"meta\">
    Track: {results.track_name} · Simulations: {results.num_simulations} · Seed: {results.seed}
  </div>
  <div class=\"grid\">
    <div class=\"card\"><h2>Top 10 Win Probabilities</h2><div id=\"wins\"></div></div>
    <div class=\"card\"><h2>Team Points Projection (per race)</h2><div id=\"teams\"></div></div>
  </div>
  <script>
    Plotly.newPlot('wins', [{{
      type: 'bar',
      x: {json.dumps(top_labels)},
      y: {json.dumps(top_values)},
      marker: {{ color: '#7aa2ff' }}
    }}], {{
      paper_bgcolor: '#181c30', plot_bgcolor: '#181c30',
      font: {{ color: '#e8ebff' }}, yaxis: {{ title: 'Win %' }}
    }});

    Plotly.newPlot('teams', [{{
      type: 'bar',
      x: {json.dumps(team_labels)},
      y: {json.dumps(team_values)},
      marker: {{ color: '#53d8b8' }}
    }}], {{
      paper_bgcolor: '#181c30', plot_bgcolor: '#181c30',
      font: {{ color: '#e8ebff' }}, yaxis: {{ title: 'Points / race' }}
    }});
  </script>
</body>
</html>
"""
        filepath.write_text(html)
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
        filename_prefix = f"{prefix}_" if prefix else ""

        files = {
            "race_csv": self.export_race_results_csv(
                results, f"{filename_prefix}race_results.csv"
            ),
            "qualifying_csv": self.export_qualifying_results_csv(
                results, f"{filename_prefix}qualifying_results.csv"
            ),
            "statistics_json": self.export_statistics_json(
                results, f"{filename_prefix}statistics.json"
            ),
            "report_html": self.export_report_html(results, f"{filename_prefix}report.html"),
        }

        self._record_export_run(results, files, prefix)
        files["runs_index_html"] = self.export_run_index_html()

        return files
