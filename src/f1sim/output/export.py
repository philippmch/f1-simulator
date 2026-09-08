"""Export simulation results to CSV and JSON."""

import csv
import json
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output.timing import csv_time
from f1sim.simulation.race import result_is_classified
from f1sim.simulation.race_points import points_for_result


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

        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            return []

        return [row for row in data if isinstance(row, dict)]

    def _write_history(self, history: list[dict[str, Any]]) -> None:
        self._history_path().write_text(json.dumps(history, indent=2), encoding="utf-8")

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
            "race_engine": results.race_engine,
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
            ts = escape(str(row.get("timestamp", "-")))
            track = escape(str(row.get("track", "-")))
            sims = escape(str(row.get("num_simulations", "-")))
            seed = escape(str(row.get("seed", "-")))
            engine = row.get("race_engine", "standard")
            engine = escape({
                "standard": "Standard", "chronological": "Lap-aware (experimental)",
            }.get(engine, engine) if isinstance(engine, str) else "Unknown")
            files = row.get("files", {})
            report = files.get("report_html") if isinstance(files, dict) else None
            stats = files.get("statistics_json") if isinstance(files, dict) else None
            links = []
            for label, artifact in (("report", report), ("stats", stats)):
                if not artifact:
                    continue
                if (isinstance(artifact, str) and artifact not in (".", "..")
                        and "/" not in artifact and "\\" not in artifact):
                    href = escape("./" + quote(artifact, safe=""), quote=True)
                    links.append(f'<a href="{href}">{label}</a>')
                else:
                    links.append(escape(str(artifact)))
            row_links = " | ".join(links) if links else "-"
            rows.append(
                f"<tr><td>{ts}</td><td>{track}</td><td>{engine}</td>"
                f"<td>{sims}</td><td>{seed}</td><td>{row_links}</td></tr>"
            )

        table_rows = "\n".join(rows) if rows else "<tr><td colspan='6'>No runs yet</td></tr>"

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
        <th>Timestamp (UTC)</th><th>Track</th><th>Race model</th>
        <th>Simulations</th><th>Seed</th><th>Artifacts</th>
      </tr>
    </thead>
    <tbody>{table_rows}</tbody>
  </table>
</body>
</html>
"""
        filepath.write_text(html, encoding="utf-8")
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

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "simulation", "position", "driver_id", "driver_name", "team",
                "total_time", "gap_to_leader", "pit_stops", "fastest_lap",
                "status", "dnf_reason", "strategy", "laps_completed", "classified",
                "pit_laps", "race_time_limited", "points_awarded",
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
                        getattr(result, "laps_completed", None),
                        str(result_is_classified(result)).lower(),
                        json.dumps(result.pit_laps)
                        if getattr(result, "pit_laps", None) is not None else "",
                        str(getattr(result, "race_time_limited", False)).lower(),
                        points_for_result(result),
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

        with open(filepath, "w", newline="", encoding="utf-8") as f:
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
                        csv_time(result.best_time),
                        csv_time(result.q1_time),
                        csv_time(result.q2_time),
                        csv_time(result.q3_time),
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
                "race_engine": results.race_engine,
                "parallel": results.parallel,
                "max_workers": results.max_workers,
            },
            "win_probabilities": results.get_win_probabilities(),
            "probability_intervals": results.get_probability_intervals(),
            "event_rates": results.get_event_rates(),
            "pit_stop_statistics": results.get_pit_stop_statistics(),
            "race_distance_statistics": results.get_race_distance_statistics(),
            "top_3_finish_probabilities": results.get_top_n_finish_probabilities(3),
            "top_10_finish_probabilities": results.get_top_n_finish_probabilities(10),
            "championship_projection": results.get_championship_projection(),
            "team_championship_projection": results.get_team_championship_projection(),
            "mechanical_failure_breakdown": results.event_stats.mechanical_failure_breakdown,
            "mechanical_failure_component_rates": results.get_mechanical_failure_component_rates(),
            "mechanical_tuning_suggestions": results.get_mechanical_tuning_suggestions(
                {
                    "engine": 0.34,
                    "gearbox": 0.22,
                    "electrical": 0.18,
                    "cooling": 0.14,
                    "brakes": 0.12,
                }
            ),
            "reliability_adjustment_recommendations": (
                results.get_reliability_adjustment_recommendations(
                    {
                        "engine": 0.34,
                        "gearbox": 0.22,
                        "electrical": 0.18,
                        "cooling": 0.14,
                        "brakes": 0.12,
                    }
                )
            ),
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

        with open(filepath, "w", encoding="utf-8") as f:
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
                "race_engine": results.race_engine,
                "win_probabilities": results.get_win_probabilities(),
                "race_distance_statistics": results.get_race_distance_statistics(),
                "team_championship_projection": results.get_team_championship_projection(),
            }

        with open(filepath, "w", encoding="utf-8") as f:
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

        def script_json(value: Any) -> str:
            # JSON quoting alone does not prevent HTML's script-end parser.
            return (json.dumps(value).replace("<", "\\u003c")
                    .replace(">", "\\u003e").replace("&", "\\u0026"))

        track_text = escape(str(results.track_name))
        simulations_text = escape(str(results.num_simulations))
        seed_text = escape(str(results.seed))
        engine_text = escape(results.race_engine)
        distance = results.get_race_distance_statistics()
        recorded = distance["recorded_races"]
        comparable = distance["finishers_with_comparable_distance"]
        recorded_text = f'{recorded} {"race" if recorded == 1 else "races"}'
        known = distance["races_with_known_winner_distance"]
        winning_distance = ("Not recorded" if distance["mean_winner_laps"] is None else
                            f'{distance["mean_winner_laps"]:.1f} laps '
                            f'({known} {"race" if known == 1 else "races"})')
        lapped = ("Not recorded" if not comparable else
                  f'{distance["lapped_finishers"]} of {comparable} finishers with known distance '
                  f'({distance["lapped_finisher_rate"] * 100:.1f}%)')
        timed = ("Not recorded" if not recorded else
                 f'{distance["time_limited_races"]} of {recorded_text} '
                 f'({distance["time_limited_race_rate"] * 100:.1f}%)')
        no_winner = ("Not recorded" if not recorded else
                     f'{distance["races_without_winner"]} of {recorded_text}')

        html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>F1 Simulation Report - {track_text}</title>
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
    Track: {track_text} · Simulations: {simulations_text} · Seed: {seed_text}
    · Race model: {engine_text}
  </div>
  <div class=\"grid\">
    <div class=\"card\" id=\"race-distance\"><h2>Race distance</h2>
      <p>Mean winning distance: {escape(winning_distance)}</p>
      <p>Lapped finishers: {escape(lapped)}</p>
      <p>Time-limited races: {escape(timed)}</p>
      <p>Races without a winner: {escape(no_winner)}</p>
    </div>
    <div class=\"card\"><h2>Top 10 Win Probabilities</h2><div id=\"wins\"></div></div>
    <div class=\"card\"><h2>Team Points Projection (per race)</h2><div id=\"teams\"></div></div>
  </div>
  <script>
    Plotly.newPlot('wins', [{{
      type: 'bar',
      x: {script_json(top_labels)},
      y: {script_json(top_values)},
      marker: {{ color: '#7aa2ff' }}
    }}], {{
      paper_bgcolor: '#181c30', plot_bgcolor: '#181c30',
      font: {{ color: '#e8ebff' }}, yaxis: {{ title: 'Win %' }}
    }});

    Plotly.newPlot('teams', [{{
      type: 'bar',
      x: {script_json(team_labels)},
      y: {script_json(team_values)},
      marker: {{ color: '#53d8b8' }}
    }}], {{
      paper_bgcolor: '#181c30', plot_bgcolor: '#181c30',
      font: {{ color: '#e8ebff' }}, yaxis: {{ title: 'Points / race' }}
    }});
  </script>
</body>
</html>
"""
        filepath.write_text(html, encoding="utf-8")
        return filepath

    def export_all(
        self,
        results: SimulationResults,
        prefix: str = "",
    ) -> dict[str, Path]:
        """Export a uniquely named bundle so later runs preserve history links.

        Args:
            results: Simulation results
            prefix: Optional readable prefix; a unique bundle id is appended

        Returns:
            Dictionary of format -> filepath
        """
        filename_prefix = (f"{prefix}_" if prefix else "") + uuid4().hex + "_"

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
