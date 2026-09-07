"""Incomplete qualifying remains classified while output timing is nullable."""

import csv
import json
import math

import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.console import ConsoleOutput
from f1sim.output.export import Exporter
from f1sim.simulation.qualifying import QualifyingResult, QualifyingSimulator


@pytest.mark.parametrize("all_missing", [False, True])
def test_actual_missing_car_results_export_without_nonfinite_times(tmp_path, capsys, all_missing):
    drivers = [Driver(id=name, name=name, team_id=name) for name in ("A", "B")]
    cars = {} if all_missing else {"A": Car(team_id="A", team_name="A")}
    track = Track(id="t", name="T", country="T", total_laps=10, base_lap_time=90)
    qualifying = QualifyingSimulator().simulate_qualifying(drivers, cars, track, Weather())
    missing = [r for r in qualifying if r.driver_id not in cars]
    assert all(math.isinf(r.best_time) and r.eliminated_in == "Q1" for r in missing)
    results = SimulationResults(1, "T", {}, [], [qualifying])
    path = Exporter(tmp_path).export_qualifying_results_csv(results)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["driver_id"] for row in rows] == [r.driver_id for r in qualifying]
    for row in rows:
        if row["driver_id"] not in cars:
            assert all(row[key] == "" for key in ("best_time", "q1_time", "q2_time", "q3_time"))
            assert row["eliminated_in"] == "Q1"
    ConsoleOutput.print_qualifying_results(list(reversed(qualifying)))
    output = capsys.readouterr().out
    assert output.count("No time") == len(missing)
    assert "inf" not in output.lower() and "nan" not in output.lower()
    pytest.importorskip("fastapi")
    from starlette.responses import JSONResponse

    from f1sim.web.server import _serialize_quali_result

    payload = [_serialize_quali_result(result) for result in qualifying]
    assert json.loads(JSONResponse(payload).body) == payload
    for row in payload:
        if row["driver_id"] not in cars:
            assert row["best_time"] is None
    assert all(math.isinf(r.best_time) for r in missing)  # Internal ordering sentinel retained.


@pytest.mark.parametrize("time", [float("inf"), float("-inf"), float("nan"), None, 0, 90.125])
def test_legacy_timing_sentinels_and_finite_zero(tmp_path, capsys, time):
    result = QualifyingResult("A", "A", 2, time, time, time, time, "Q1")
    valid = time is not None and math.isfinite(time)
    path = Exporter(tmp_path).export_qualifying_results_csv(
        SimulationResults(1, "T", {}, [], [[result]])
    )
    with path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert all(row[key] == (f"{time:.3f}" if valid else "")
               for key in ("best_time", "q1_time", "q2_time", "q3_time"))
    ConsoleOutput.print_qualifying_results([result])
    output = capsys.readouterr().out
    assert (f"{time:.3f}s" if valid else "No time") in output
    assert "nan" not in output.lower() and "inf" not in output.lower()
    pytest.importorskip("fastapi")
    from starlette.responses import JSONResponse

    from f1sim.web.server import _serialize_quali_result

    payload = _serialize_quali_result(result)
    assert json.loads(JSONResponse(payload).body) == payload
    assert all(payload[key] == (time if valid else None)
               for key in ("best_time", "q1_time", "q2_time", "q3_time"))


def test_console_qualifying_uses_position_order(capsys):
    pole = QualifyingResult("A", "A", 1, 90, 90, None, None, None)
    second = QualifyingResult("B", "B", 2, 91, 91, None, None, None)
    ConsoleOutput.print_qualifying_results([second, pole])
    output = capsys.readouterr().out
    assert output.index("90.000s") < output.index("91.000s")


def test_console_shows_sessions_without_cross_session_gap(capsys):
    pole = QualifyingResult("A", "Pole sitter", 1, 89, 89, 90, 92, None)
    second = QualifyingResult("B", "Second on grid", 2, 88, 88, 91, 93, None)
    eliminated = QualifyingResult("C", "Q2 eliminated", 11, 87, 87, 94, None, "Q2")
    ConsoleOutput.print_qualifying_results([eliminated, second, pole])
    output = capsys.readouterr().out
    rows = [line.split() for line in output.splitlines() if line[:1].isdigit()]
    assert rows == [
        ["1", "Pole", "sitter", "89.000s", "90.000s", "92.000s", "89.000s"],
        ["2", "Second", "on", "grid", "88.000s", "91.000s", "93.000s", "88.000s"],
        ["11", "Q2", "eliminated", "87.000s", "94.000s", "--", "87.000s", "out", "in", "Q2"],
    ]
    assert "Gap" not in output and "+-" not in output
    assert "grid order follows session classification" in output
