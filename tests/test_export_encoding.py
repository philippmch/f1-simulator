"""Exported international names must not depend on the host locale."""

import builtins
import csv
import json

import pytest

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.export import Exporter
from f1sim.simulation.qualifying import QualifyingResult
from f1sim.simulation.race import DriverStatus, RaceResult


@pytest.mark.parametrize("locale_encoding", ["cp1252", "ascii"])
def test_all_exports_preserve_unicode_under_restrictive_locale(
    tmp_path, monkeypatch, locale_encoding,
):
    original_open = builtins.open

    def locale_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs:
            kwargs["encoding"] = locale_encoding
        return original_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", locale_open)
    name, team, track = "角田 裕毅", "Équipe 日本", "鈴鹿 · São Paulo"
    result = RaceResult("TSU", name, team, 1, 5400, 0, 1, 90, DriverStatus.FINISHED,
                        strategy=["medium", "hard"], pit_laps=[24])
    quali = QualifyingResult("TSU", name, 1, 88, 90, 89, 88, None)
    stats = DriverStatistics("TSU", name, team, positions=[1], wins=1, total_points=25)
    results = SimulationResults(1, track, {"TSU": stats}, [[result]], [[quali]])
    exporter = Exporter(tmp_path)
    files = exporter.export_all(results)
    for key in ("race_csv", "qualifying_csv"):
        with files[key].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 1
        assert rows[0]["driver_name"] == name
        if key == "race_csv":
            assert rows[0]["team"] == team
            assert json.loads(rows[0]["pit_laps"]) == [24]
    data = json.loads(files["statistics_json"].read_text(encoding="utf-8"))
    assert data["metadata"]["track_name"] == track
    scenario = exporter.export_scenario_comparison_json({"雨": results})
    assert "雨" in json.loads(scenario.read_text(encoding="utf-8"))["scenarios"]
    assert track in files["report_html"].read_text(encoding="utf-8")
    assert track in files["runs_index_html"].read_text(encoding="utf-8")
