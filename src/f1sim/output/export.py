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
from f1sim.output.abandonment_context import abandonment_statistics_html
from f1sim.output.control_schedule_context import control_schedule_statistics_html
from f1sim.output.qualifying_context import qualifying_weather_context
from f1sim.output.scoring_context import scoring_statistics_html
from f1sim.output.timing import (
    csv_time,
    format_seconds,
    race_suspension_seconds,
    suspension_statistics,
)
from f1sim.output.warmup_context import warmup_context
from f1sim.output.weather_schedule_context import weather_schedule_context
from f1sim.simulation.abandonment import race_abandonment_context, serialize_abandonment_tire_rule
from f1sim.simulation.control_schedule import validate_control_schedule_history
from f1sim.simulation.race import result_is_classified
from f1sim.simulation.race_points import (
    points_for_result,
    points_reason_for_result,
    race_scoring_context,
)


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
                "standard": "Standard", "chronological": "Lap-aware",
            }.get(engine, engine) if isinstance(engine, str) else "Unknown")
            files = row.get("files", {})
            report = files.get("report_html") if isinstance(files, dict) else None
            stats = files.get("statistics_json") if isinstance(files, dict) else None
            weather = files.get("weather_csv") if isinstance(files, dict) else None
            pits = files.get("pit_stops_csv") if isinstance(files, dict) else None
            links = []
            for label, artifact in (("report", report), ("stats", stats),
                                    ("weather CSV", weather), ("pit stops CSV", pits)):
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
        inventory_fields = (['tire_set_history', 'tire_inventory'] if any(
            getattr(row, 'tire_set_history', None) is not None
            or getattr(row, 'tire_inventory', None) is not None
            for race in results.race_results for row in race
        ) else [])
        snapshot_plans = (results.input_snapshot or {}).get("pit_plans")
        plan_field = ["pit_plan_history"] if (
            isinstance(snapshot_plans, dict) and snapshot_plans
        ) or any(
            getattr(row, "pit_plan_history", None) is not None
            for race in results.race_results for row in race
        ) else []

        getter = getattr(results, "get_race_scoring_context", None)
        scoring_contexts = [
            getter(index) if callable(getter) else race_scoring_context(race)
            for index, race in enumerate(results.race_results)
        ]
        scoring_fields = (["race_points_policy", "scheduled_laps", "winner_laps",
                           "has_two_green_laps", "points_reason"]
                          if any(context is not None for context in scoring_contexts) else [])
        abandonment_contexts = [race_abandonment_context(race) for race in results.race_results]
        abandonment_fields = ["race_abandonment", "abandonment_tire_rule"] if any(
            getattr(row, "race_abandonment", None) is not None
            for race in results.race_results for row in race
        ) else []

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "simulation", "position", "driver_id", "driver_name", "team",
                "total_time", "gap_to_leader", "pit_stops", "fastest_lap",
                "status", "dnf_reason", "strategy", "race_suspension_seconds",
                "laps_completed", "classified",
                "pit_laps", "race_time_limited", "points_awarded",
                "overtake_attempts", "overtake_successes", "overtake_contacts",
                *inventory_fields, *plan_field,
                *scoring_fields,
                *abandonment_fields,
            ])

            for sim_idx, race_results in enumerate(results.race_results, 1):
                suspension = race_suspension_seconds(race_results)
                context = scoring_contexts[sim_idx - 1]
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
                        csv_time(suspension),
                        getattr(result, "laps_completed", None),
                        str(result_is_classified(result)).lower(),
                        json.dumps(result.pit_laps)
                        if getattr(result, "pit_laps", None) is not None else "",
                        str(getattr(result, "race_time_limited", False)).lower(),
                        points_for_result(result),
                        getattr(result, "overtake_attempts", None),
                        getattr(result, "overtake_successes", None),
                        getattr(result, "overtake_contacts", None),
                        *(json.dumps(getattr(result, key))
                          if getattr(result, key, None) is not None else ""
                          for key in inventory_fields),
                        *(json.dumps(getattr(result, "pit_plan_history"))
                          if plan_field and getattr(result, "pit_plan_history", None) is not None
                          else "" for _ in plan_field),
                        *([
                            context["policy"] if context else "",
                            context["scheduled_laps"] if context else "",
                            context["winner_laps"] if context else "",
                            str(context["has_two_green_laps"]).lower() if context else "",
                            points_reason_for_result(result) if context else "",
                        ] if scoring_fields else []),
                        *(json.dumps(value) if value is not None else "" for value in (
                            abandonment_contexts[sim_idx - 1],
                            serialize_abandonment_tire_rule(
                                getattr(result, "abandonment_tire_rule", None),
                            ),
                        ) if abandonment_fields),
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

    def export_weather_history_csv(
        self, results: SimulationResults, filename: str = "weather_history.csv",
    ) -> Path:
        """Export observed shared race-lap weather, with one-based trial indices."""
        filepath = self.output_dir / filename
        with filepath.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["simulation", "race_engine", "weather_interval", "condition",
                             "rain_intensity", "track_wetness"])
            for index, history in enumerate(results.weather_histories, start=1):
                for row in history:
                    writer.writerow([index, results.race_engine, row["lap"], row["condition"],
                                     row["rain_intensity"], row["track_wetness"]])
        return filepath

    @staticmethod
    def _pit_stop_details(results: SimulationResults) -> list[dict]:
        return [
            {"simulation": index, "driver_id": result.driver_id,
             "stops": ([dict(stop) for stop in result.pit_stop_details]
                       if getattr(result, "pit_stop_details", None) is not None else None)}
            for index, race in enumerate(results.race_results, start=1) for result in race
        ]

    def export_control_schedule_history_csv(
        self, results: SimulationResults, filename: str = "control_schedule_history.csv",
    ) -> Path:
        """Export global SC/VSC evidence once per trial, including coverage gaps."""
        filepath = self.output_dir / filename
        stats = results.get_control_schedule_statistics()
        schedule = stats["requested_schedule"]
        histories = results.control_schedule_histories
        action_fields = ["action"] if any(
            row["control"] == "red_flag" for row in schedule or []
        ) else []
        with filepath.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["simulation", "race_engine", "history_status", "lap", "control",
                             "duration_laps", "status", "reason", *action_fields])
            if stats["source"] != "controlled":
                return filepath
            for index in range(len(results.race_results)):
                prefix = [index + 1, results.race_engine]
                if histories is not None and not isinstance(histories, list):
                    writer.writerow(prefix + ["invalid", "", "", "", "", ""]
                                    + [""] * len(action_fields))
                    continue
                history = (histories[index] if isinstance(histories, list)
                           and index < len(histories) else None)
                if history is None:
                    writer.writerow(prefix + ["missing", "", "", "", "", ""]
                                    + [""] * len(action_fields))
                    continue
                try:
                    rows = validate_control_schedule_history(history, schedule)
                except (TypeError, ValueError):
                    writer.writerow(prefix + ["invalid", "", "", "", "", ""]
                                    + [""] * len(action_fields))
                    continue
                if not rows:
                    writer.writerow(prefix + ["complete", "", "", "", "no_requests", ""]
                                    + [""] * len(action_fields))
                for row in rows:
                    writer.writerow(prefix + ["complete"] + [row.get(key, "") for key in (
                        "lap", "control", "duration_laps", "status", "reason",
                        *action_fields,
                    )])
        return filepath

    @staticmethod
    def _tire_set_ledgers(results: SimulationResults) -> list[dict]:
        return [
            {"simulation": index, "driver_id": result.driver_id,
             **{key: ([dict(item) for item in getattr(result, key)]
                      if getattr(result, key, None) is not None else None)
                for key in ("tire_set_history", "tire_inventory")}}
            for index, race in enumerate(results.race_results, 1) for result in race
        ]

    @staticmethod
    def _pit_plan_histories(results: SimulationResults) -> list[dict]:
        """Return per-trial requested-plan histories without inferring missing data."""
        return [
            {
                "simulation": index,
                "driver_id": result.driver_id,
                "pit_plan_history": (
                    [dict(record) for record in result.pit_plan_history]
                    if getattr(result, "pit_plan_history", None) is not None else None
                ),
            }
            for index, race in enumerate(results.race_results, 1)
            for result in race
        ]

    @staticmethod
    def _tire_set_ledger_html(results: SimulationResults) -> str:
        sections = []
        for row in Exporter._tire_set_ledgers(results):
            if row["tire_set_history"] is None:
                continue
            label = escape(f'Trial {row["simulation"]} · {row["driver_id"]}')
            sections.append(f'<details><summary>{label}</summary>')
            for key, caption, fields in (
                ("tire_set_history", "Physical set fittings",
                 ("lap", "kind", "set_id", "compound", "age_at_fit", "age_at_end", "laps_used")),
                ("tire_inventory", "Final race set pool",
                 ("id", "compound", "age", "current", "available", "unavailable")),
            ):
                if key == "tire_inventory" and any("remaining_laps" in item
                                                   for item in row[key] or []):
                    fields += ("remaining_laps",)
                if key == "tire_set_history" and any("remaining_laps_at_fit" in item
                                                     for item in row[key] or []):
                    fields += ("remaining_laps_at_fit", "remaining_laps_at_end")
                sections.append('<div class="table-wrap" tabindex="0"><table><caption>'
                                + caption + '</caption><thead><tr>'
                                + ''.join('<th scope="col">' + field.replace('_', ' ')
                                          + '</th>' for field in fields)
                                + '</tr></thead><tbody>')
                for record in row[key] or []:
                    sections.append('<tr>' + ''.join(
                        '<td>' + escape(str(record.get(field, 'Unlimited' if field.startswith(
                            'remaining_laps') else 'Not recorded'))) + '</td>'
                        for field in fields) + '</tr>')
                sections.append('</tbody></table></div>')
            sections.append('</details>')
        return ''.join(sections) or '<p>No finite race set ledger recorded.</p>'

    def export_pit_stop_details_csv(
        self, results: SimulationResults, filename: str = "pit_stops.csv",
    ) -> Path:
        """Export modeled paid-stop components; free fittings have no rows."""
        fields = ["lap", "from_compound", "to_compound", "tire_age", "condition",
                  "rain_intensity", "track_wetness", "control", "lane_loss",
                  "service_time", "queue_time", "total_loss",
                  "from_set_id", "to_set_id", "incoming_tire_age",
                  "decision_reason", "forecast_saving_seconds"]
        filepath = self.output_dir / filename
        with filepath.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["simulation", "race_engine", "driver_id", *fields])
            for row in self._pit_stop_details(results):
                for stop in row["stops"] or []:
                    writer.writerow([row["simulation"], results.race_engine, row["driver_id"],
                                     *(stop.get(field, "") for field in fields)])
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
            "simulation_inputs": results.input_snapshot,
            "weather_histories": results.weather_histories,
            "control_schedule_histories": results.control_schedule_histories,
            "control_schedule_statistics": results.get_control_schedule_statistics(),
            "race_scoring_contexts": results.get_race_scoring_contexts(),
            "race_abandonment_contexts": results.get_race_abandonment_contexts(),
            "abandonment_statistics": results.get_abandonment_statistics(),
            "abandonment_tire_rules": results.get_abandonment_tire_rules(),
            "race_scoring_statistics": results.get_race_scoring_statistics(),
            "pit_stop_details": self._pit_stop_details(results),
            "tire_set_ledgers": self._tire_set_ledgers(results),
            "pit_plan_histories": self._pit_plan_histories(results),
            "pit_plan_statistics": results.get_pit_plan_statistics(),
            "metadata": {
                "num_simulations": results.num_simulations,
                "track_name": results.track_name,
                "seed": results.seed,
                "race_engine": results.race_engine,
                "parallel": results.parallel,
                "max_workers": results.max_workers,
            },
            "win_probabilities": results.get_win_probabilities(),
            **({"winner_forecast": forecast}
               if (forecast := results.get_winner_forecast()) is not None else {}),
            "probability_intervals": results.get_probability_intervals(),
            "event_rates": results.get_event_rates(),
            "event_rate_trials": results.get_event_rate_trials(),
            "pit_stop_statistics": results.get_pit_stop_statistics(),
            "overtaking_statistics": (
                results.get_overtake_statistics()
                if callable(getattr(results, "get_overtake_statistics", None)) else None
            ),
            "pit_loss_statistics": results.get_pit_loss_statistics(),
            "pit_decision_statistics": results.get_pit_decision_statistics(),
            "strategy_statistics": results.get_strategy_statistics(),
            "race_distance_statistics": results.get_race_distance_statistics(),
            "suspension_statistics": suspension_statistics(results),
            "top_3_finish_probabilities": results.get_top_n_finish_probabilities(3),
            "top_10_finish_probabilities": results.get_top_n_finish_probabilities(10),
            "championship_projection": results.get_championship_projection(),
            "team_championship_projection": results.get_team_championship_projection(),
            "mechanical_failure_breakdown": results.event_stats.mechanical_failure_breakdown,
            "mechanical_failure_component_rates": results.get_mechanical_failure_component_rates(),
            # Retain the fields for consumers of older statistics JSON.  Automatic
            # outputs have no configured reference shares, so there is no basis for
            # calibration or reliability-change recommendations.
            "mechanical_tuning_suggestions": {},
            "reliability_adjustment_recommendations": {},
            "driver_statistics": self._driver_statistics(results),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(stats_dict, f, indent=2)

        return filepath

    @staticmethod
    def _driver_statistics(results: SimulationResults) -> dict[str, dict[str, Any]]:
        """Use identical observed-driver summaries in single and comparison JSON."""
        summaries = {}
        top_5 = results.get_top_n_finish_probabilities(5)
        top_10 = results.get_top_n_finish_probabilities(10)
        win_estimates = results.get_win_probabilities()

        for driver_id, stats in results.driver_stats.items():
            trials = len(stats.positions)
            summaries[driver_id] = {
                "driver_name": stats.driver_name,
                "team": stats.team,
                "recorded_races": trials,
                "points_per_race": stats.total_points / trials if trials else None,
                "wins": stats.wins,
                "win_rate": stats.win_rate,
                "estimated_win_rate": win_estimates.get(driver_id, stats.win_rate),
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

        return summaries

    def export_scenario_comparison_json(
        self,
        scenario_results: dict[str, SimulationResults],
        filename: str = "scenario_comparison.json",
        *, reference_scenario: str | None = None,
    ) -> Path:
        """Export scenario outcomes, observed counts, uncertainty and replay inputs."""
        filepath = self.output_dir / filename

        payload: dict[str, Any] = {"scenarios": {}}
        if reference_scenario is not None:
            from f1sim.analysis.paired_comparison import paired_comparison_statistics

            payload["paired_comparisons"] = paired_comparison_statistics(
                scenario_results, reference_scenario,
            )
        for name, results in scenario_results.items():
            payload["scenarios"][name] = {
                "track_name": results.track_name,
                "num_simulations": results.num_simulations,
                "seed": results.seed,
                "race_engine": results.race_engine,
                "parallel": results.parallel,
                "max_workers": results.max_workers,
                "win_probabilities": results.get_win_probabilities(),
                **({"winner_forecast": forecast}
                   if (forecast := results.get_winner_forecast()) is not None else {}),
                "probability_intervals": results.get_probability_intervals(),
                "driver_statistics": self._driver_statistics(results),
                "pit_stop_statistics": results.get_pit_stop_statistics(),
                "overtaking_statistics": (
                    results.get_overtake_statistics()
                    if callable(getattr(results, "get_overtake_statistics", None)) else None
                ),
                "pit_loss_statistics": results.get_pit_loss_statistics(),
                "pit_decision_statistics": results.get_pit_decision_statistics(),
                "event_rates": results.get_event_rates(),
                "event_rate_trials": results.get_event_rate_trials(),
                "race_distance_statistics": results.get_race_distance_statistics(),
                "suspension_statistics": suspension_statistics(results),
                "weather_histories": results.weather_histories,
                "control_schedule_histories": results.control_schedule_histories,
                "control_schedule_statistics": results.get_control_schedule_statistics(),
                "race_scoring_contexts": results.get_race_scoring_contexts(),
                "race_abandonment_contexts": results.get_race_abandonment_contexts(),
                "abandonment_statistics": results.get_abandonment_statistics(),
                "abandonment_tire_rules": results.get_abandonment_tire_rules(),
                "race_scoring_statistics": results.get_race_scoring_statistics(),
                "pit_stop_details": self._pit_stop_details(results),
                "tire_set_ledgers": self._tire_set_ledgers(results),
                "pit_plan_histories": self._pit_plan_histories(results),
                "pit_plan_statistics": results.get_pit_plan_statistics(),
                "strategy_statistics": results.get_strategy_statistics(),
                "simulation_inputs": results.input_snapshot,
                "team_championship_projection": results.get_team_championship_projection(),
                "mechanical_failure_breakdown": results.event_stats.mechanical_failure_breakdown,
                "mechanical_failure_component_rates": (
                    results.get_mechanical_failure_component_rates()
                ),
                # Keep comparison JSON aligned with single-run JSON while retaining
                # compatibility for consumers that already read these keys.
                "mechanical_tuning_suggestions": {},
                "reliability_adjustment_recommendations": {},
            }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        return filepath

    def export_scenario_comparison_html(
        self, scenario_results: dict[str, SimulationResults],
        filename: str = "scenario_comparison.html", *, focus_driver: str | None = None,
        reference_scenario: str | None = None,
        selection: dict | None = None,
    ) -> Path:
        """Write an offline comparison with observed counts and sampling intervals."""
        from f1sim.output.comparison import render_comparison_report

        filepath = self.output_dir / filename
        filepath.write_text(
            render_comparison_report(scenario_results, focus_driver=focus_driver,
                                     reference_scenario=reference_scenario,
                                     selection=selection), encoding="utf-8",
        )
        return filepath

    def export_rival_strategy_selection_html(
        self,
        manifest: dict,
        filename: str = "rival_selection_summary.html",
        *,
        manifest_filename: str,
    ) -> Path:
        """Write the consolidated offline summary for a weighted rival selection."""
        from f1sim.output.comparison import render_rival_strategy_selection_report

        filepath = self.output_dir / filename
        payload = dict(manifest)
        payload["manifest_filename"] = manifest_filename
        filepath.write_text(
            render_rival_strategy_selection_report(payload), encoding="utf-8",
        )
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
        starting_tires = (results.input_snapshot or {}).get("starting_tires", {})
        starting_ages = (results.input_snapshot or {}).get("starting_tire_ages", {})
        if not isinstance(starting_ages, dict):
            starting_ages = {}
        starting_text = escape(", ".join(
            f"{driver}={compound}"
            + (f"@{starting_ages[driver]}" if starting_ages.get(driver) else "")
            for driver, compound in sorted(starting_tires.items())
        ) or "Automatic")
        from f1sim.output.comparison import (
            _overtaking_statistics_html,
            _pit_plan_history_html,
            _pit_plan_statistics_html,
            _pit_plans,
        )

        warmup_text = escape(warmup_context(results.input_snapshot))
        warmup_html = f"<p>{warmup_text}</p>" if warmup_text else ""
        qualifying_text = escape(qualifying_weather_context(results.input_snapshot))
        qualifying_html = f"<p>{qualifying_text}</p>" if qualifying_text else ""
        schedule_text = escape(weather_schedule_context(results.input_snapshot))
        schedule_html = f"<p>{schedule_text}</p>" if schedule_text else ""
        control_html = control_schedule_statistics_html(results)
        abandonment_html = abandonment_statistics_html(results)
        scoring_html = scoring_statistics_html(results)
        pit_plan_text = escape(_pit_plans(results))
        pit_plan_statistics = _pit_plan_statistics_html(results, "run")
        pit_plan_history = _pit_plan_history_html(results, "run")
        overtaking_html = _overtaking_statistics_html(results, scenario="run")
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
        suspension = suspension_statistics(results)
        suspension_races = suspension["recorded_races"]
        suspension_known = suspension["races_with_recorded_suspension"]
        suspension_coverage = (
            f'{suspension_races} recorded '
            f'{"race" if suspension_races == 1 else "races"}; '
            f'{suspension_known} with suspension'
        )
        suspension_mean = format_seconds(
            suspension["mean_completed_suspension_seconds"]
        )
        suspension_note = (
            "Race-wide suspension includes collection and decision waiting. Countback "
            "finish clocks exclude subsequent running and suspension; earlier resumed "
            "suspensions remain in the historical clocks."
            if results.get_abandonment_statistics()["recorded_abandoned_races"] else
            "Race-wide collection + restart pause are already in finish clocks. This "
            "elapsed-race context is not an individual driver's stopped or driving time."
        )
        strategy_sections = []
        for driver_id, summary in results.get_strategy_statistics().items():
            known = summary["races_with_recorded_strategy"]
            rows = "".join(
                "<tr><td>" + escape(" → ".join(strategy["compounds"])) + "</td>"
                f'<td>{strategy["races"]} ({strategy["share"] * 100:.1f}%)</td>'
                f'<td>{strategy["finished_races"]}</td><td>{strategy["dnf_races"]}</td></tr>'
                for strategy in summary["strategies"]
            )
            strategy_sections.append(
                f'<details><summary>{escape(str(driver_id))}: {known} recorded of '
                f'{summary["races"]} observed '
                f'{"race" if summary["races"] == 1 else "races"}</summary>'
                f'<p>Missing tyre sequences: {summary["missing_strategy_races"]}</p>'
                + ('<div class="table-wrap"><table><thead><tr><th>Tyre sequence</th>'
                   '<th>Races (% of recorded)</th><th>Finished</th><th>Retired</th>'
                   f'</tr></thead><tbody>{rows}</tbody></table></div>' if rows else
                   '<p>No tyre sequences were recorded.</p>') + '</details>'
            )
        strategy_html = "".join(strategy_sections) or "<p>No tyre sequences were recorded.</p>"
        forecast = results.get_winner_forecast()
        forecast_html = ""
        if forecast is not None:
            forecast_rows = "".join(
                f'<tr><th scope="row">{escape(str(driver))}</th>'
                f'<td>{row["probability"] * 100:.1f}%</td>'
                f'<td>{row["native_win_count"]} ({row["native_probability"] * 100:.1f}%)</td></tr>'
                for driver, row in forecast["drivers"].items()
            )
            forecast_html = (
                '<div class="card" id="winner-forecast"><h2>Win estimates and simulated wins</h2>'
                '<p>Estimates use simulated constructor wins and earlier race points. '
                'The simulated counts remain separate.</p><div class="table-wrap"><table>'
                '<thead><tr><th scope="col">Driver</th><th scope="col">Win estimate</th>'
                '<th scope="col">Simulated wins</th></tr></thead>'
                f'<tbody>{forecast_rows}</tbody></table></div></div>'
            )

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
    .grid {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 20px; }}
    .card {{ background: #181c30; border: 1px solid #2a3156; border-radius: 12px;
      padding: 16px; min-width: 0; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .meta {{ color: #b6c0ff; margin-bottom: 16px; overflow-wrap: anywhere; }}
    details {{ margin: 12px 0; }}
    summary {{ cursor: pointer; }}
    .table-wrap {{ overflow-x: auto; max-width: 100%; }}
    table {{ width: 100%; border-collapse: collapse; text-align: left; }}
    th, td {{ padding: 8px; border-bottom: 1px solid #2a3156; }}
  </style>
</head>
<body>
  <h1>F1 Simulation Report</h1>
  <div class=\"meta\">
    Track: {track_text} · Simulations: {simulations_text} · Seed: {seed_text}
    · Race model: {engine_text}
    · Starting tyres: {starting_text}
    · Custom pit plans: {pit_plan_text}
    {warmup_html}
    {qualifying_html}
    {schedule_html}
    {control_html}
    {abandonment_html}
    Input race set pools:
    {escape(json.dumps((results.input_snapshot or {}).get('tire_inventory', {})))}
  </div>
  <div class=\"grid\">
    {forecast_html}
    <div class="card" id="race-scoring"><h2>Race points</h2>
      {scoring_html}
      <p>Points depend on the winner's completed distance and two consecutive complete
      green leader laps. Unknown scoring evidence does not enter these counts.</p>
    </div>
    <div class=\"card\" id=\"race-distance\"><h2>Race distance</h2>
      <p>Mean winning distance: {escape(winning_distance)}</p>
      <p>Lapped finishers: {escape(lapped)}</p>
      <p>Time-limited races: {escape(timed)}</p>
      <p>Races without a winner: {escape(no_winner)}</p>
    </div>
    <div class=\"card\" id=\"suspension-statistics\"><h2>Completed race suspension</h2>
      <p>Mean completed race suspension: {escape(suspension_mean)}</p>
      <p>Known suspension durations: {escape(suspension_coverage)}</p>
      <p>{escape(suspension_note)}</p>
    </div>
    <div class=\"card\" id=\"overtaking-statistics\"><h2>Overtaking attempts and outcomes</h2>
      <p>The counters record attempts that reach the passing model, not rejected gates.
      Contact counts cover passing calls, not every collision; rates describe this model.</p>
      {overtaking_html}
    </div>
    <div class=\"card\" id=\"strategy-statistics\"><h2>Recorded tyre sequences</h2>
      <p>Open a driver to see every sequence. Counts include retirements and free tyre
      changes; sequence length is not the paid-stop count. Shares use races with a
      recorded sequence. Frequency does not establish which strategy is fastest.</p>
      {strategy_html}
    </div>
    <div class="card" id="pit-plan-history"><h2>Custom pit-plan execution</h2>
      <p>Requested laps are each driver's own lap. Statuses describe the recorded
      instruction outcome; missing history is not inferred. Aggregate counts use
      complete valid histories, with coverage shown against recorded trials.</p>
      {pit_plan_statistics}
      {pit_plan_history or '<p>No custom pit-plan history was recorded.</p>'}
    </div>
    <div class="card" id="tire-set-ledgers"><h2>Race tyre sets</h2>
      <p>Fittings include free changes and unrun sets. Ages include prior wear.
      Removed undamaged sets remain reusable; qualifying uses separate sets.</p>
      {self._tire_set_ledger_html(results)}
    </div>
    <div class=\"card\"><h2>Top 10 Win Probabilities</h2><div id=\"wins\"></div></div>
    <div class=\"card\"><h2>Team Points Projection (per race)</h2>
      <p>Sum of listed drivers' points per observed race. Teams with an unobserved
      listed driver are omitted.</p><div id=\"teams\"></div></div>
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
            "weather_csv": self.export_weather_history_csv(
                results, f"{filename_prefix}weather_history.csv"
            ),
            **({"control_schedule_csv": self.export_control_schedule_history_csv(
                results, f"{filename_prefix}control_schedule_history.csv",
            )} if results.control_schedule_histories is not None else {}),
            "pit_stops_csv": self.export_pit_stop_details_csv(
                results, f"{filename_prefix}pit_stops.csv"
            ),
            "statistics_json": self.export_statistics_json(
                results, f"{filename_prefix}statistics.json"
            ),
            "report_html": self.export_report_html(results, f"{filename_prefix}report.html"),
        }

        self._record_export_run(results, files, prefix)
        files["runs_index_html"] = self.export_run_index_html()

        return files
