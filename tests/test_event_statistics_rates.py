"""Event occurrence rates count races, not individual deployments."""

import json

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, RaceEventStatistics
from f1sim.models import Track, Weather
from f1sim.output.export import Exporter


def runner():
    return MonteCarloRunner([], {}, Track(id="test", name="Test", country="Test",
                                         total_laps=10, base_lap_time=90), Weather(), seed=1)


def test_empty_event_sample_has_zero_rates():
    stats = runner()._aggregate_event_statistics([])
    assert stats.num_simulations == 0
    assert stats.safety_car_rate == stats.red_flag_rate == 0
    assert RaceEventStatistics().safety_car_rate == 0


def test_multiple_deployments_count_only_once_per_race():
    stats = runner()._aggregate_event_statistics([
        {"safety_car": 3, "red_flag": 2},
        {"safety_car": 1},
        {},
        {},
    ])
    assert stats.num_simulations == 4
    assert stats.safety_car_count == 4
    assert stats.red_flag_count == 2
    assert stats.safety_car_rate == 50
    assert stats.red_flag_rate == 25


def test_runner_properties_and_exported_rates_agree(monkeypatch, tmp_path):
    samples = iter([
        {"safety_car": 2, "red_flag": 3, "vsc": 1, "incidents": 4},
        {},
        {"safety_car": 1},
    ])
    monkeypatch.setattr("f1sim.analysis.montecarlo._run_single_simulation",
                        lambda args: ([], [], next(samples)))
    results = runner().run(3, parallel=False)
    assert results.event_stats.num_simulations == results.num_simulations == 3
    assert results.event_stats.safety_car_rate == pytest.approx(200 / 3)
    assert results.event_stats.red_flag_rate == pytest.approx(100 / 3)
    path = Exporter(tmp_path).export_statistics_json(results)
    rates = json.loads(path.read_text())["event_rates"]
    assert rates["safety_car_race_rate"] * 100 == results.event_stats.safety_car_rate
    assert rates["red_flag_race_rate"] * 100 == results.event_stats.red_flag_rate
    assert rates["avg_safety_cars"] == 1
    assert rates["avg_vsc"] == pytest.approx(1 / 3)
    assert rates["avg_incidents"] == pytest.approx(4 / 3)
