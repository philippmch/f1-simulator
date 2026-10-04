"""Fee-aware bounds cover physical timing and keep exact native decisions."""

import runpy
from copy import deepcopy
from itertools import product
from pathlib import Path

import pytest
from test_controlled_weather_strategy import inputs
from test_inventory_partial_strategies import assert_continuation, independent_schedules

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.models import _native
from f1sim.simulation import inventory_strategy
from f1sim.simulation.inventory_strategy import _inventory_clock_surfaces, plan_inventory_strategy
from f1sim.simulation.strategy_lap import control_lap_scope
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock
from f1sim.simulation.weather_schedule import WeatherForecastContext


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("fees", [(.1, .2), (.5, 60.)])
@pytest.mark.parametrize("current_running", [None, (80., 130.)])
def test_surface_envelope_contains_all_prior_fit_and_paid_stop_histories(
    explicit, fees, current_running,
):
    horizon = 4
    events = tuple(5. + i * 10. for i in range(60))
    clock = StrategyWeatherClock(
        (0., 90., 180., 270.), events[0], 10., len(events), 35., 17.,
        current_running_times=current_running, update_offsets=events if explicit else None)
    rows = _inventory_clock_surfaces(
        horizon, clock, {"soft": fees[0], "wet": fees[1]}, horizon, native=True)
    for offset in range(horizon):
        # A prior lap may retain its set, pay to fit, or fit freely at the root.
        # The current entry has no new fitting fee, even if it pays for service.
        for paid_history in product((False, True), repeat=offset + 1):
            stopped_first = paid_history[0]
            stop_delay = sum(35. if index == 0 else 17.
                             for index, paid in enumerate(paid_history) if paid)
            nominal = clock.lap_start_offsets[offset]
            if offset and current_running is not None:
                nominal = current_running[int(stopped_first)] + (offset - 1) * 90.
            for fit_history in product((0., *fees), repeat=offset):
                delay = 0.
                for fee in fit_history:
                    delay += fee
                elapsed = nominal + stop_delay + delay
                count = (0 if offset == 0 and not stopped_first else
                         sum(event <= elapsed + 10. * 1.e-12 for event in events))
                assert count in rows[offset]
    assert len(rows[0]) < clock.max_updates + 1
    assert len(rows[horizon - 1]) < clock.max_updates + 1


@pytest.mark.parametrize("horizon,native", [(4, False), (101, True)])
def test_custom_and_long_clocks_keep_the_full_fee_envelope(horizon, native):
    clock = StrategyWeatherClock(tuple(90. * i for i in range(horizon)), 5., 10., 12, 35., 17.)
    rows = _inventory_clock_surfaces(horizon, clock, {"soft": .5}, horizon, native=native)
    assert all(set(row) == set(range(13)) for row in rows.values())


def test_surface_envelope_construction_remains_cancellable():
    clock = StrategyWeatherClock((0., 90., 180.), 5., 10., 12, 35., 17.)
    with cancellation_scope(lambda: True), pytest.raises(SimulationCancelled):
        _inventory_clock_surfaces(3, clock, {"soft": .5}, 3, native=True)


def test_shared_own_lap_costs_keep_distinct_fee_profiles_and_ready_sets_separate():
    models = inputs(.2, .2, 4)
    stock = models[-1]
    options = dict(
        tire_age=0, remaining_stops=2, remaining_dry_stops=2, remaining_damp_stops=2,
        used_compounds=("intermediate",), physical_total_laps=12,
        forecast_context=WeatherForecastContext.from_schedule([
            dict(lap=2, rain_intensity=.6), dict(lap=4, rain_intensity=0.),
        ]))
    before = deepcopy(models)

    @control_lap_scope
    def check_profiles():
        for profile in ({"intermediate": .1, "wet": .5}, {"intermediate": 60.},
                        {"wet": 60.}, {"intermediate": .1, "wet": .5}):
            active = dict(options, tire_warmup=profile)
            expected, _ = independent_schedules(models, active)
            actual = plan_inventory_strategy(*models, 1, **active)
            for stopped in (False, True):
                assert_continuation(actual.continuation(stopped), expected[stopped])

    check_profiles()
    assert models[:-1] == before[:-1]
    assert stock.__dict__ == before[-1].__dict__


def test_reachable_fee_surfaces_reduce_search_without_changing_controlled_decisions(monkeypatch):
    benchmark = runpy.run_path(str(
        Path(__file__).parents[1] / "examples/benchmark_controlled_strategy.py"))["benchmark"]
    options = dict(laps=30, drivers=4, intervals=3, profile=True,
                   tire_warmup={compound: .5 for compound in
                                ("soft", "medium", "hard", "intermediate", "wet")})
    bounded = benchmark(**options)
    original = inventory_strategy._inventory_clock_surfaces

    def unrestricted(horizon, clock, warmup, max_paid_stops, *, native=False):
        return original(horizon, clock, warmup, max_paid_stops, native=False)

    monkeypatch.setattr(inventory_strategy, "_inventory_clock_surfaces", unrestricted)
    monkeypatch.setattr(_native, "_HELPERS", [
        (namespace, name, unrestricted if namespace is inventory_strategy.__dict__
         and name == "_inventory_clock_surfaces" else value)
        for namespace, name, value in _native._HELPERS])
    reference = benchmark(**options)
    assert bounded["native"] and reference["native"]
    assert bounded["decision"] == reference["decision"]
    assert bounded["outcome_sha256"] == reference["outcome_sha256"]
    assert 0 < bounded["inventory_state_expansions"] < reference["inventory_state_expansions"]
