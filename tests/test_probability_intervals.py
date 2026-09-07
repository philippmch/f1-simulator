"""Sampling intervals quantify Monte Carlo noise, not model accuracy."""

from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import (
    DriverStatistics,
    MonteCarloRunner,
    SimulationResults,
    wilson_interval,
)
from f1sim.simulation.race import DriverStatus, RaceResult


@pytest.mark.parametrize(
    ("successes", "trials", "lower", "upper"),
    [
        (0, 0, 0.0, 100.0),
        (0, 1, 0.0, 79.34506856),
        (1, 1, 20.65493144, 100.0),
        (0, 100, 0.0, 3.69934982),
        (100, 100, 96.30065018, 100.0),
        (50, 100, 40.38315303, 59.61684697),
        (10, 100, 5.52291371, 17.43656615),
    ],
)
def test_known_wilson_bounds(successes, trials, lower, upper):
    bounds = wilson_interval(successes, trials)
    assert bounds["lower"] == pytest.approx(lower, abs=1e-6)
    assert bounds["upper"] == pytest.approx(upper, abs=1e-6)


@pytest.mark.parametrize(
    ("successes", "trials"),
    [(-1, 10), (11, 10), (0, -1), (1, 0), (0.5, 1), (1, 1.0), (True, 2), (0, False)],
)
def test_invalid_counts_rejected(successes, trials):
    with pytest.raises(ValueError):
        wilson_interval(successes, trials)


def test_interval_shrinks_with_more_samples():
    widths = []
    for trials in (10, 100, 1000):
        bounds = wilson_interval(trials // 2, trials)
        widths.append(bounds["upper"] - bounds["lower"])
    assert widths[0] > widths[1] > widths[2]
    assert wilson_interval(0, 1000)["upper"] < wilson_interval(0, 10)["upper"]


def _results(stats, races=None):
    return SimulationResults(
        num_simulations=100,  # Requested total is not the per-driver sample size.
        track_name="Test",
        driver_stats=stats,
        race_results=races or [],
        qualifying_results=[],
    )


def test_driver_observations_are_denominator_and_point_estimates_unchanged():
    stats = DriverStatistics("A", "Driver", "Team", wins=1, podiums=2, positions=[1, 2])
    results = _results({"A": stats})
    before = results.get_win_probabilities()
    interval = results.get_probability_intervals()["A"]
    assert interval["trials"] == 2
    assert interval["win"] == wilson_interval(1, 2)
    assert interval["podium"] == wilson_interval(2, 2)
    assert interval["dnf"] == wilson_interval(0, 2)
    assert results.get_win_probabilities() == before == {"A": 50.0}


def test_empty_observations_are_uninformative():
    assert _results({}).get_probability_intervals() == {}
    interval = _results({"A": DriverStatistics("A", "Driver", "Team")})
    bounds = interval.get_probability_intervals()["A"]
    assert bounds["trials"] == 0
    for outcome in ("win", "podium", "dnf"):
        assert bounds[outcome] == {"lower": 0.0, "upper": 100.0}


def test_status_aware_counts_include_retirements_in_trials_only():
    runner = MonteCarloRunner(
        drivers=[SimpleNamespace(id="A", name="Driver", team_id="team")],
        cars={}, track=None, weather=None, seed=0,
    )
    races = [
        [RaceResult("A", "Driver", "Team", 1, 90, 0, 0, 90, status)]
        for status in (DriverStatus.FINISHED, DriverStatus.DNF, "dnf")
    ]
    results = _results(runner._aggregate_statistics(races, []), races)
    interval = results.get_probability_intervals()["A"]
    assert interval["trials"] == 3
    assert interval["win"] == interval["podium"] == wilson_interval(1, 3)
    assert interval["dnf"] == wilson_interval(2, 3)


def test_web_payload_includes_interval_metadata():
    pytest.importorskip("fastapi")
    from f1sim.web.server import _serialize_driver_statistics

    stats = DriverStatistics("A", "Driver", "Team", positions=[4] * 100)
    results = _results({"A": stats})
    payload = _serialize_driver_statistics(results)["A"]
    assert payload["win_rate"] == 0
    assert payload["probability_intervals"] == results.get_probability_intervals()["A"]
    bounds = payload["probability_intervals"]
    assert bounds["confidence"] == 0.95
    assert bounds["method"] == "wilson"
    assert bounds["scope"] == "monte_carlo_sampling"
    assert bounds["trials"] == 100
    assert bounds["win"]["upper"] > 0
