"""Retirements retain their last completed crossing and may still classify."""

from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.race import DriverStatus, RaceResult, RaceSimulator, result_is_classified


def run_race(monkeypatch, laps, retirements, *, red_flag_lap=None, swap_lap=None):
    simulator = RaceSimulator(rng=np.random.default_rng(4))
    drivers = [Driver(id=name, name=name, team_id=name) for name in 'ABC']
    cars = {d.id: Car(team_id=d.id, team_name=d.id) for d in drivers}
    track = Track(id='test', name='Test', country='Test', total_laps=laps, base_lap_time=90)
    monkeypatch.setattr(simulator, '_should_pit', lambda *a, **kw: False)
    # Failure laps are provisionally faster; they must never become fastest laps.
    monkeypatch.setattr(simulator.lap_simulator, 'calculate_lap_time',
                        lambda **kw: 80.0 if kw['lap_number'] in retirements else 90.0)

    def overtake(states, *args, lap, **kwargs):
        if lap == swap_lap:
            states[1].position, states[2].position = states[2].position, states[1].position
        return 0

    def events(lap, drivers, **kwargs):
        output = []
        for driver in drivers:
            if driver.id in retirements.get(lap, ()):
                driver.dnf = True
                driver.dnf_reason = 'engine'
                output.append(RaceEvent(EventType.MECHANICAL_FAILURE, lap, [driver.id]))
        if lap == red_flag_lap:
            output.append(RaceEvent(EventType.RED_FLAG, lap))
        return output

    monkeypatch.setattr(simulator, '_process_overtakes', overtake)
    monkeypatch.setattr(simulator.event_manager, 'process_lap', events)
    return simulator.simulate_race(drivers, cars, track, Weather(change_probability=0), list('ABC'))


def test_later_retirement_outranks_earlier_retirement(monkeypatch):
    results = run_race(monkeypatch, 10, {2: 'B', 8: 'C'})
    assert [r.driver_id for r in results] == list('ACB')
    assert [r.laps_completed for r in results] == [10, 7, 1]
    assert [r.position for r in results] == [1, 2, 3]
    early = results[-1]
    assert early.total_time == pytest.approx(90)
    assert early.fastest_lap == pytest.approx(90)
    assert early.status == DriverStatus.DNF
    assert early.dnf_reason == 'engine'


def test_same_lap_retirements_use_previous_crossing_not_failure_lap_order(monkeypatch):
    results = run_race(monkeypatch, 10, {8: 'BC'}, swap_lap=8)
    assert [r.driver_id for r in results] == list('ABC')
    assert results[1].laps_completed == results[2].laps_completed == 7


@pytest.mark.parametrize('laps', [60, 61])
@pytest.mark.parametrize(('completed', 'classified'), [(53, False), (54, True)])
def test_ninety_percent_threshold_rounds_down(monkeypatch, laps, completed, classified):
    results = run_race(monkeypatch, laps, {completed + 1: 'C'})
    retired = results[-1]
    assert retired.laps_completed == completed
    assert retired.classified is classified
    assert retired.status == DriverStatus.DNF


def test_first_lap_retirement_has_no_distance_or_fastest_lap(monkeypatch):
    retired = run_race(monkeypatch, 10, {1: 'C'})[-1]
    assert (retired.laps_completed, retired.total_time, retired.fastest_lap) == (0, 0, 0)
    assert retired.classified is False


def test_no_finisher_means_no_classification_even_after_ninety_percent(monkeypatch):
    results = run_race(monkeypatch, 10, {10: 'ABC'})
    assert all(r.laps_completed == 9 and not r.classified for r in results)
    runner = MonteCarloRunner(drivers=[SimpleNamespace(id=n, name=n, team_id=n) for n in 'ABC'],
                             cars={}, track=None, weather=None)
    stats = runner._aggregate_statistics([results], [])
    assert all(s.total_points == s.wins == s.podiums == 0 and s.dnfs == 1
               for s in stats.values())


def test_red_flag_preserves_completed_distance_and_retirement_crossing(monkeypatch):
    results = run_race(monkeypatch, 10, {8: 'BC'}, red_flag_lap=8, swap_lap=8)
    assert [r.driver_id for r in results] == list('ABC')
    assert [r.laps_completed for r in results] == [10, 7, 7]
    assert results[1].total_time == results[2].total_time == pytest.approx(630)
    assert results[1].fastest_lap == 90


@pytest.mark.parametrize('classified', [True, False])
def test_classified_retirement_scores_points_and_podium_but_remains_dnf(classified):
    result = RaceResult('A', 'A', 'A', 3, 5400, 0, 1, 90, DriverStatus.DNF,
                        laps_completed=54, classified=classified)
    runner = MonteCarloRunner(drivers=[SimpleNamespace(id='A', name='A', team_id='A')],
                             cars={}, track=None, weather=None)
    stats = runner._aggregate_statistics([[result]], [])
    driver = stats['A']
    assert driver.total_points == (15 if classified else 0)
    assert driver.podiums == driver.points_finishes == int(classified)
    assert driver.dnfs == 1
    assert driver.positions == [3]
    aggregate = SimulationResults(1, 'Test', stats, [[result]], [])
    assert aggregate.get_top_n_finish_probabilities(3)['A'] == (100 if classified else 0)


@pytest.mark.parametrize('status', [DriverStatus.FINISHED, 'finished', DriverStatus.DNF, 'dnf'])
def test_legacy_classification_and_explicit_override(status):
    result = SimpleNamespace(status=status)
    assert result_is_classified(result) == (status == 'finished')
    result.classified = True
    assert result_is_classified(result)
    result.classified = False
    assert not result_is_classified(result)
