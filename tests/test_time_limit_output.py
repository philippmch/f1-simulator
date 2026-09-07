"""Race output identifies time-limited distance and actual championship awards."""

import csv
from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.output.console import ConsoleOutput
from f1sim.output.export import Exporter
from f1sim.simulation.race import DriverStatus, RaceResult


def result_with(award, limited):
    return RaceResult(
        "A",
        "Driver A",
        "Team",
        1,
        7300,
        0,
        1,
        90,
        DriverStatus.FINISHED,
        laps_completed=5,
        classified=True,
        race_time_limited=limited,
        points_awarded=award,
    )


@pytest.mark.parametrize(
    "award,limited,expected", [(19, True, 19), (0, True, 0), (None, False, 25)]
)
def test_console_and_csv_preserve_actual_awards(tmp_path, capsys, award, limited, expected):
    result = result_with(award, limited)
    path = Exporter(tmp_path).export_race_results_csv(
        SimulationResults(1, "Test", {}, [[result]], []),
    )
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["race_time_limited"] == str(limited).lower()
    assert int(row["points_awarded"]) == expected
    ConsoleOutput.print_race_results([result])
    output = capsys.readouterr().out
    assert ("two-hour limit" in output) == limited
    assert "Points" in output
    assert next(line for line in output.splitlines() if "Driver A" in line).split()[-1] == str(
        expected
    )


def test_api_preserves_zero_award_and_defaults_legacy_results():
    pytest.importorskip("fastapi")
    from f1sim.web.server import _serialize_race_result

    result = result_with(0, True)
    payload = _serialize_race_result(result)
    assert payload["race_time_limited"] is True
    assert payload["points_awarded"] == 0
    legacy = SimpleNamespace(
        **{
            key: value
            for key, value in vars(result).items()
            if key not in {"points_awarded", "race_time_limited"}
        }
    )
    payload = _serialize_race_result(legacy)
    assert payload["race_time_limited"] is False
    assert payload["points_awarded"] == 25
