"""Points projections use observed driver races, not nominal requested trials."""

import json

import pytest

from f1sim.analysis.montecarlo import DriverStatistics, SimulationResults
from f1sim.output import Exporter
from f1sim.web.server import _summarize_scenario_results


def results(requested=1000):
    return SimulationResults(requested, "Partial", {
        "A": DriverStatistics("A", "A", "Team", total_points=30, positions=[1, 3, 20, 9]),
        "B": DriverStatistics("B", "B", "Team", total_points=20, positions=[2, 8]),
        "C": DriverStatistics("C", "C", "Other", total_points=0, positions=[20]),
        "D": DriverStatistics("D", "D", "Unknown", total_points=0),
    }, [], [])


@pytest.mark.parametrize("requested", [0, 4, 1000])
def test_points_rank_by_observed_mean_and_keep_observed_zero(requested):
    observed = results(requested)
    assert observed.get_championship_projection() == {"B": 10, "A": 7.5, "C": 0}
    assert list(observed.get_championship_projection()) == ["B", "A", "C"]
    assert observed.get_team_championship_projection() == {"Team": 17.5, "Other": 0}


def test_unobserved_teammate_does_not_become_zero_point_contribution():
    observed = results()
    observed.driver_stats["D"].team = "Team"
    assert observed.get_team_championship_projection() == {"Other": 0}
    # Driver projections still retain the observed members of that team.
    assert observed.get_championship_projection() == {"B": 10, "A": 7.5, "C": 0}


def test_empty_results_have_no_projections_or_division_by_zero():
    observed = SimulationResults(0, "Empty", {}, [], [])
    assert observed.get_championship_projection() == {}
    assert observed.get_team_championship_projection() == {}


def test_api_and_export_points_match_observed_driver_statistics(tmp_path):
    observed = results()
    exported = Exporter(tmp_path).export_statistics_json(observed)
    saved = json.loads(exported.read_text(encoding="utf-8"))
    assert saved["championship_projection"] == {"B": 10, "A": 7.5, "C": 0}
    for driver, points in saved["championship_projection"].items():
        assert points == saved["driver_statistics"][driver]["points_per_race"]
    assert saved["driver_statistics"]["D"]["points_per_race"] is None
    assert saved["team_championship_projection"] == {"Team": 17.5, "Other": 0}
    shown = _summarize_scenario_results({"partial": observed})["scenarios"]["partial"]
    assert dict(shown["championship_projection"]) == saved["championship_projection"]
    assert shown["team_projection"] == saved["team_championship_projection"]
