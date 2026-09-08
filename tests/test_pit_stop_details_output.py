"""Paid-stop exports retain costs and distinguish absent from empty observations."""

import csv
import json

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.models.tire import TireCompound
from f1sim.output import Exporter
from f1sim.web.server import _summarize_scenario_results


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_paid_stop_exports_and_selected_trial(tmp_path, engine):
    results = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="T")], {"T": Car(team_id="T", team_name="T")},
        Track(id="t", name="Track", country="Test", total_laps=4, base_lap_time=90),
        Weather(condition=WeatherCondition.HEAVY_RAIN, rain_intensity=.8,
                track_wetness=.9, change_probability=0),
        seed=41, race_engine=engine, starting_tires={"A": TireCompound.SOFT},
    ).run(3, parallel=False)
    exporter = Exporter(tmp_path)
    files = exporter.export_all(results)
    with files["pit_stops_csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = [{"simulation": index, "driver_id": race.driver_id,
                 "stops": race.pit_stop_details}
                for index, trial in enumerate(results.race_results, 1) for race in trial]
    stops = [(row, stop) for row in expected for stop in row["stops"]]
    assert len(rows) == len(stops) > 0
    for exported, (row, stop) in zip(rows, stops):
        assert int(exported["simulation"]) == row["simulation"]
        assert exported["driver_id"] == row["driver_id"]
        assert exported["race_engine"] == engine
        for key, value in stop.items():
            assert (float(exported[key]) if isinstance(value, (int, float))
                    else exported[key]) == value
    assert "pit stops CSV" in files["runs_index_html"].read_text(encoding="utf-8")
    assert json.loads(files["statistics_json"].read_text(encoding="utf-8"))[
        "pit_stop_details"] == expected
    comparison = exporter.export_scenario_comparison_json({"rain": results})
    assert json.loads(comparison.read_text(encoding="utf-8"))[
        "scenarios"]["rain"]["pit_stop_details"] == expected
    shown = _summarize_scenario_results({"rain": results})["scenarios"]["rain"]
    assert shown["sample_race"][0]["pit_stop_details"] == results.race_results[
        shown["sample_index"]][0].pit_stop_details

    # Legacy unknown and known zero stops stay distinct in JSON, neither invents CSV rows.
    results.race_results[0][0].pit_stop_details = None
    for trial in results.race_results[1:]:
        trial[0].pit_stop_details = []
    with exporter.export_pit_stop_details_csv(results).open(newline="", encoding="utf-8") as f:
        assert list(csv.DictReader(f)) == []
    saved = json.loads(exporter.export_statistics_json(results).read_text(encoding="utf-8"))
    assert [row["stops"] for row in saved["pit_stop_details"]] == [None, [], []]
