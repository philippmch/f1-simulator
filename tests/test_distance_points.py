"""Distance awards flow consistently into driver and constructor statistics."""

from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import POINTS_SYSTEM, MonteCarloRunner, SimulationResults
from f1sim.simulation.race import DriverStatus, RaceResult
from f1sim.simulation.race_points import points_for_classification, points_for_result


@pytest.mark.parametrize(('laps', 'expected'), [
    (24, [6, 4, 3, 2, 1, 0, 0, 0, 0, 0]),
    (25, [13, 10, 8, 6, 5, 4, 3, 2, 1, 0]),
    (49, [13, 10, 8, 6, 5, 4, 3, 2, 1, 0]),
    (50, [19, 14, 12, 10, 8, 6, 4, 3, 2, 1]),
    (74, [19, 14, 12, 10, 8, 6, 4, 3, 2, 1]),
    (75, [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]),
    (100, [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]),
])
def test_exact_distance_boundaries(laps, expected):
    assert [points_for_classification(p, True, laps, 100, True)
            for p in range(1, 11)] == expected
    assert points_for_classification(11, True, laps, 100, True) == 0


@pytest.mark.parametrize(('laps', 'scheduled', 'expected'), [
    (2, 9, 6), (2, 8, 13), (2, 5, 13), (2, 4, 19),
    (2, 3, 19), (3, 4, 25), (44, 59, 19), (45, 59, 25),
])
def test_fractional_distance_thresholds_do_not_round(laps, scheduled, expected):
    assert points_for_classification(1, True, laps, scheduled, True) == expected


@pytest.mark.parametrize('laps', [None, 0, 1, 24, 25, 50, 75, 100])
def test_no_two_consecutive_green_laps_means_no_points(laps):
    assert points_for_classification(1, True, laps, 100, False) == 0


@pytest.mark.parametrize('laps', [None, 0, 1])
def test_no_eligible_winner_or_one_lap_race_means_no_points(laps):
    assert points_for_classification(1, True, laps, 100, True) == 0


def test_unclassified_or_invalid_position_never_earns_distance_points():
    assert points_for_classification(1, False, 100, 100, True) == 0
    assert points_for_classification(0, True, 100, 100, True) == 0
    assert points_for_classification(-1, True, 100, 100, True) == 0
    assert points_for_classification(1, True, 100, 0, True) == 0


def make_result(driver, position, award=None, *, retired=False, classified=True):
    result = RaceResult(driver, driver, 'Team', position, 90, 0, 0, 90,
                        DriverStatus.DNF if retired else DriverStatus.FINISHED,
                        classified=classified)
    result.points_awarded = award
    return result


def test_explicit_zero_and_reduced_awards_override_normal_classification_points():
    assert points_for_result(make_result('A', 1, 0)) == 0
    assert points_for_result(make_result('A', 1, 6)) == 6
    assert points_for_result(make_result('A', 3, 3, retired=True)) == 3
    assert points_for_result(make_result('A', 3, 0, retired=True, classified=False)) == 0


def test_legacy_results_keep_classified_normal_points():
    assert POINTS_SYSTEM[1] == 25
    for classified, expected in [(True, 15), (False, 0)]:
        legacy = SimpleNamespace(position=3, status=DriverStatus.DNF, classified=classified)
        assert points_for_result(legacy) == expected
        assert points_for_result(make_result('A', 3, retired=True,
                                             classified=classified)) == expected


def test_aggregation_uses_awards_without_changing_classified_win_and_podium_counts():
    races = [
        [make_result('A', 1, 0), make_result('B', 6, 0)],
        [make_result('A', 1, 6), make_result('B', 3, 3, retired=True)],
        [make_result('A', 1), make_result('B', 3, 0, retired=True, classified=False)],
    ]
    runner = MonteCarloRunner(
        drivers=[SimpleNamespace(id=d, name=d, team_id='team') for d in 'AB'],
        cars={'team': SimpleNamespace(team_name='Team')}, track=None, weather=None,
    )
    stats = runner._aggregate_statistics(races, [])
    assert (stats['A'].total_points, stats['A'].points_finishes,
            stats['A'].wins, stats['A'].podiums) == (31, 2, 3, 3)
    assert (stats['B'].total_points, stats['B'].points_finishes,
            stats['B'].podiums, stats['B'].dnfs) == (3, 1, 1, 2)
    results = SimulationResults(3, 'Test', stats, races, [])
    assert results.get_championship_projection() == pytest.approx({'A': 31 / 3, 'B': 1})
    assert results.get_team_championship_projection() == pytest.approx({'Team': 34 / 3})
    assert results.get_points_finish_probabilities() == pytest.approx({'A': 200 / 3, 'B': 100 / 3})
    assert results.get_top_n_finish_probabilities(10) == pytest.approx({'A': 100, 'B': 200 / 3})
