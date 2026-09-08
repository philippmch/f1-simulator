"""Loss summaries require complete paid-stop observations, including retirements."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from f1sim.analysis.montecarlo import MonteCarloRunner, SimulationResults
from f1sim.models import Car, Driver, Track, Weather, WeatherCondition
from f1sim.simulation.race import DriverStatus


def stop(lane=20, service=3, queue=0):
    return dict(lane_loss=lane, service_time=service, queue_time=queue,
                total_loss=lane + service + queue)


def row(stops, details=None, driver="A", retired=False):
    return SimpleNamespace(driver_id=driver, pit_stops=stops, pit_stop_details=details,
                           status=DriverStatus.DNF if retired else DriverStatus.FINISHED)


def results(*rows):
    return SimulationResults(1000, "Test", {}, [[value] for value in rows], [])


def test_complete_observation_denominators_include_zero_and_retired_races():
    sample = results(row(0, []), row(2, [stop(queue=2), stop(queue=4)], retired=True),
                     row(1, (stop(),)), row(1), row(2, [stop()]), row(0, None, "B"))
    before = deepcopy(sample)
    actual = sample.get_pit_loss_statistics()
    assert actual["A"] == {
        "races": 5, "races_with_recorded_details": 3, "missing_details_races": 2,
        "recorded_stops": 3, "queued_stops": 2, "races_with_queue": 1,
        "queue_race_rate": 1 / 3, "mean_total_loss_per_race": 25,
        "mean_lane_loss_per_race": 20, "mean_service_time_per_race": 3,
        "mean_queue_time_per_race": 2,
    }
    assert actual["B"]["races_with_recorded_details"] == 0
    assert actual["B"]["queue_race_rate"] is None
    assert all(value is None for key, value in actual["B"].items() if key.startswith("mean_"))
    assert sample == before
    actual["A"]["recorded_stops"] = 999
    assert sample.get_pit_loss_statistics()["A"]["recorded_stops"] == 3


@pytest.mark.parametrize("count,details", [
    (True, [stop()]), (-1, []), (1.0, [stop()]), (1, []), (0, [stop()]),
    (1, {"stop": stop()}), (1, [None]), (1, [{}]),
    *[(1, [dict(stop(), **{field: value})])
      for field in ("lane_loss", "service_time", "queue_time", "total_loss")
      for value in (True, -1, float("nan"), float("inf"), 10**400, "3", None)],
    (2, [stop(), dict(stop(), total_loss=99)]),
])
def test_malformed_or_partial_race_is_excluded_wholly(count, details):
    summary = results(row(count, details), row(0, [])).get_pit_loss_statistics()["A"]
    assert summary["races"] == 2
    assert summary["races_with_recorded_details"] == summary["missing_details_races"] == 1
    assert summary["recorded_stops"] == summary["queued_stops"] == 0
    assert summary["queue_race_rate"] == summary["mean_total_loss_per_race"] == 0


def test_component_roundoff_tolerance_and_missing_legacy_attribute():
    legacy = SimpleNamespace(driver_id="A", pit_stops=0)
    summary = results(legacy, row(1, [dict(stop(), total_loss=23 + 1e-10)])
                      ).get_pit_loss_statistics()["A"]
    assert summary["races_with_recorded_details"] == 1
    assert summary["missing_details_races"] == 1
    assert summary["mean_total_loss_per_race"] == pytest.approx(23)
    assert results().get_pit_loss_statistics() == {}


def test_unrepresentable_race_totals_are_missing_not_infinite():
    huge = stop(lane=1e308, service=0)
    summary = results(row(2, [huge, huge]), row(0, [])).get_pit_loss_statistics()["A"]
    assert summary["races_with_recorded_details"] == summary["missing_details_races"] == 1
    assert summary["recorded_stops"] == 0
    assert summary["mean_total_loss_per_race"] == 0


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_real_engine_paid_opening_correction_aggregates_loss(engine):
    simulation = MonteCarloRunner(
        [Driver(id="A", name="A", team_id="A")],
        {"A": Car(team_id="A", team_name="A")},
        Track(id="t", name="T", country="T", total_laps=4, base_lap_time=90),
        Weather(condition=WeatherCondition.HEAVY_RAIN, rain_intensity=.8,
                track_wetness=.9, change_probability=0),
        seed=3, race_engine=engine, starting_tires={"A": "soft"},
    ).run(2, parallel=False)
    summary = simulation.get_pit_loss_statistics()["A"]
    assert summary["races_with_recorded_details"] == 2
    assert summary["recorded_stops"] == 2
    assert summary["missing_details_races"] == summary["races_with_queue"] == 0
    totals = [sum(stop["total_loss"] for stop in race[0].pit_stop_details)
              for race in simulation.race_results]
    assert summary["mean_total_loss_per_race"] == pytest.approx(sum(totals) / 2)
