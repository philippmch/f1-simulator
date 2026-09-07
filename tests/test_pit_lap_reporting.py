"""Paid stop laps remain distinct from free changes and unavailable history."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.analysis.montecarlo import SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.output.export import Exporter
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceResult, RaceSimulator


@pytest.mark.parametrize("retire", [False, True])
def test_results_copy_paid_laps_and_exclude_free_red_flag_change(monkeypatch, retire):
    simulator = RaceSimulator(np.random.default_rng(8))
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A")
    track = Track(id="test", name="Test", country="Test", total_laps=6, base_lap_time=90)
    captured = []

    def pit(state, states, track, lap, *args, **kwargs):
        if not captured:
            captured.append(state)
        return lap in (2, 5)

    def events(lap, drivers, **kwargs):
        if lap == 3:
            return [RaceEvent(EventType.RED_FLAG, lap)]
        if lap == 5 and retire:
            drivers[0].dnf = True
            drivers[0].dnf_reason = "engine"
            return [RaceEvent(EventType.MECHANICAL_FAILURE, lap, ["A"])]
        return []

    monkeypatch.setattr(simulator, "_should_pit", pit)
    monkeypatch.setattr(simulator.event_manager, "process_lap", events)
    result = simulator.simulate_race([driver], {"A": car}, track,
                                     Weather(change_probability=0), ["A"])[0]
    assert result.pit_laps == [2, 5]
    assert result.pit_stops == 2
    assert len(result.strategy) == 4  # Start, two paid changes, one free change.
    assert result.status == (DriverStatus.DNF if retire else DriverStatus.FINISHED)
    if retire:
        assert result.laps_completed == 4
    assert result.pit_laps is not captured[0].pit_laps
    captured[0].pit_laps.append(6)
    assert result.pit_laps == [2, 5]


def result_with(laps):
    return RaceResult("A", "A", "Team", 1, 90, 0, len(laps or []), 90,
                      DriverStatus.FINISHED, pit_laps=laps)


@pytest.mark.parametrize("laps", [[2, 5], [], None])
def test_csv_distinguishes_known_empty_and_unknown_history(tmp_path, laps):
    result = result_with(laps)
    results = SimulationResults(1, "Test", {}, [[result]], [])
    path = Exporter(tmp_path).export_race_results_csv(results)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        assert reader.fieldnames[-1] == "pit_laps"
    assert row["pit_laps"] == (json.dumps(laps) if laps is not None else "")


@pytest.mark.parametrize("laps", [[2, 5], [], None])
def test_api_copies_known_paid_history(laps):
    pytest.importorskip("fastapi")
    from f1sim.web.server import _serialize_race_result

    result = result_with(laps)
    payload = _serialize_race_result(result)
    assert payload["pit_laps"] == laps
    if laps is not None:
        assert payload["pit_laps"] is not result.pit_laps


def test_legacy_result_without_attribute_exports_unknown(tmp_path):
    result = result_with(None)
    assert result.pit_laps is None
    legacy = SimpleNamespace(**{key: value for key, value in vars(result).items()
                                if key != "pit_laps"})
    results = SimulationResults(1, "Test", {}, [[legacy]], [])
    path = Exporter(tmp_path).export_race_results_csv(results)
    with path.open(newline="") as handle:
        assert next(csv.DictReader(handle))["pit_laps"] == ""
    pytest.importorskip("fastapi")
    from f1sim.web.server import _serialize_race_result

    assert _serialize_race_result(legacy)["pit_laps"] is None
