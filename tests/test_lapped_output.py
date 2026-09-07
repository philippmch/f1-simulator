"""Lapped finishers show distance gaps without inventing unknown lap counts."""

from types import SimpleNamespace

import pytest

from f1sim.output.console import ConsoleOutput
from f1sim.output.timing import format_lap_deficit
from f1sim.simulation.race import DriverStatus, RaceResult


@pytest.mark.parametrize("completed,leader,expected", [
    (9, 10, "+1 lap"), (8, 10, "+2 laps"), (10, 10, None),
    (None, 10, None), (9, None, None), (True, 10, None),
    (-1, 10, None), (11, 10, None), (9.5, 10, None),
])
def test_distance_gap_requires_known_whole_lap_deficit(completed, leader, expected):
    assert format_lap_deficit(completed, leader) == expected


def result(name, position, time, laps, status=DriverStatus.FINISHED):
    return RaceResult(name, name, "Team", position, time, time - 900, 0, 90, status,
                      laps_completed=laps, classified=True)


def test_console_distinguishes_lapped_finishers_from_same_lap_and_retirement(capsys):
    ConsoleOutput.print_race_results([
        result("Leader", 1, 900, 10), result("SameLap", 2, 905, 10),
        result("OneLap", 3, 990, 9), result("TwoLaps", 4, 1000, 8),
        result("Retired", 5, 880, 9, DriverStatus.DNF),
    ])
    lines = capsys.readouterr().out.splitlines()
    def row(name):
        return next(line for line in lines if name in line)

    assert "+5.000s" in row("SameLap")
    assert "+1 lap" in row("OneLap") and "+1:30" not in row("OneLap")
    assert "+2 laps" in row("TwoLaps")
    assert "DNF" in row("Retired") and "+1 lap" not in row("Retired")


def test_legacy_results_without_lap_metadata_keep_seconds(capsys):
    legacy = [SimpleNamespace(**{key: value for key, value in vars(row).items()
                                if key != "laps_completed"})
              for row in (result("Leader", 1, 900, 10), result("Legacy", 2, 990, 9))]
    ConsoleOutput.print_race_results(legacy)
    line = next(line for line in capsys.readouterr().out.splitlines() if "Legacy" in line)
    assert "+1:30.00" in line
