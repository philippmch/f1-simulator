"""CLI and file output preserve classification independently of retirement."""

import csv
import json

import pytest

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output.console import ConsoleOutput
from f1sim.output.export import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult


def race_result(status, laps, classified):
    return RaceResult("A", "Driver A", "Team", 3, 5000, 0, 1, 90, status,
                      dnf_reason="Engine" if status == DriverStatus.DNF else None,
                      laps_completed=laps, classified=classified)


@pytest.mark.parametrize("status,laps,classified,eligible", [
    (DriverStatus.FINISHED, 60, True, True),
    (DriverStatus.DNF, 54, True, True),
    (DriverStatus.DNF, 53, False, False),
    (DriverStatus.DNF, None, None, False),
    (DriverStatus.FINISHED, None, None, True),
])
def test_csv_and_console_agree_on_classification(
    tmp_path, capsys, status, laps, classified, eligible,
):
    result = race_result(status, laps, classified)
    results = SimulationResults(1, "Test", {}, [[result]], [])
    path = Exporter(tmp_path).export_race_results_csv(results)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        # New columns append to the original layout; raw ordinal rank remains available.
        assert reader.fieldnames[-2:] == ["laps_completed", "classified"]
    assert row["position"] == "3"
    assert row["status"] == status.value
    assert row["classified"] == str(eligible).lower()
    assert row["laps_completed"] == (str(laps) if laps is not None else "")
    assert row["dnf_reason"] == (result.dnf_reason or "")

    ConsoleOutput.print_race_results([result])
    lines = capsys.readouterr().out.splitlines()
    line = next(line for line in lines if "Driver A" in line)
    assert line.split()[0] == ("3" if eligible else "NC")
    assert "Laps" in "\n".join(lines)
    if status == DriverStatus.DNF:
        assert "Engine" in line
        assert ("DNF / Classified" if eligible else "DNF / Not classified") in line
    lap_column = next(line for line in lines if "Time/Gap" in line).index("Laps")
    assert line[lap_column:lap_column + 5].strip() == (str(laps) if laps is not None else "-")


def test_statistics_json_includes_sampling_intervals_for_classified_retirement(tmp_path):
    retired = race_result(DriverStatus.DNF, 54, True)
    stats = DriverStatistics("A", "Driver A", "Team", podiums=1, points_finishes=1,
                             dnfs=1, total_points=15, positions=[3])
    results = SimulationResults(1, "Test", {"A": stats}, [[retired]], [])
    path = Exporter(tmp_path).export_statistics_json(results)
    data = json.loads(path.read_text())
    intervals = data["probability_intervals"]["A"]
    assert intervals["scope"] == "monte_carlo_sampling"
    assert intervals["method"] == "wilson"
    assert intervals["confidence"] == 0.95
    assert intervals["trials"] == 1
    assert intervals["dnf"] == intervals["podium"]
    assert intervals["dnf"]["upper"] == 100
    assert 0 < intervals["dnf"]["lower"] < 100
    assert intervals["win"]["lower"] == 0
    assert 0 < intervals["win"]["upper"] < 100
    assert data["driver_statistics"]["A"]["top_5_finish_probability"] == 100
