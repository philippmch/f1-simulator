"""Transition plans agree with exhaustive legal paid-stop schedules."""

import copy
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.models.track import ActiveAeroZone
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.rain_strategy import plan_rain_transition


def inputs(compound=TireCompound.INTERMEDIATE, wetness=.23, rain=0, laps=7):
    return (Driver(id="A", name="A", team_id="T"), Car(team_id="T", team_name="T"),
            Track(id="t", name="T", country="T", total_laps=laps, base_lap_time=90,
                  pit_lane_delta=5), Weather(track_wetness=wetness, rain_intensity=rain),
            TIRE_COMPOUNDS[compound].model_copy(deep=True))


def exhaustive(args, age, lap, budget, used_compounds, **options):
    """Enumerate complete legal schedules, charging actual mean lap physics."""
    driver, car, track, weather, tire = copy.deepcopy(args)
    sim = LapSimulator(np.random.default_rng(1))
    minima = [inf, inf]
    best_compound = None

    def visit(surface, fitted, tire_age, number, stops, total, first_stop, first_compound, used):
        nonlocal best_compound
        legal = bool(used & {TireCompound.INTERMEDIATE, TireCompound.WET}) or len(used) >= 2
        if number > track.total_laps:
            if not legal:
                return
            index = int(first_stop)
            if total < minima[index]:
                minima[index] = total
                if first_stop:
                    best_compound = first_compound
            return
        critical = surface.tire_mismatch(fitted.compound) == "critical"
        available = [] if critical else [(False, fitted)]
        limit = options.get("remaining_dry_stops" if surface.track_wetness < .08
                            and surface.rain_intensity < .15 else "remaining_damp_stops")
        elective = stops < budget and (
            fitted.compound in (TireCompound.INTERMEDIATE, TireCompound.WET)
            or surface.track_wetness > .3 or limit is None or stops < limit
        )
        if elective or critical or not legal:
            rain = surface.fresh_rain_compound()
            compounds = [rain] if rain else [TireCompound.SOFT, TireCompound.MEDIUM,
                                           TireCompound.HARD]
            available += [(True, TIRE_COMPOUNDS[c]) for c in compounds
                          if surface.tire_mismatch(c) != "critical"
                          and (elective or critical or c not in used)]
        for stop, next_tire in available:
            if number == lap and stop and options.get(
                "required_first_compound", next_tire.compound,
            ) != next_tire.compound:
                continue
            next_age = 0 if stop else tire_age
            driver.current_tire_laps = next_age
            cost = sim.calculate_lap_time(
                driver, car, track, next_tire, surface, number,
                options.get("physical_total_laps", track.total_laps),
                sample_variation=False,
                gap_to_car_ahead=(options.get("current_traffic_gaps", (None, None))[int(stop)]
                                  if number == lap else None),
                active_aero_enabled=options.get("active_aero_enabled", True)
                if number == lap else True,
            )
            if number == lap:
                cost *= options.get("current_lap_time_modifier", 1)
            if stop:
                cost += track.pit_lane_delta * (
                    options.get("pit_lane_factor", 1) if number == lap else 1
                ) + expected_stationary_time(car)
                if number == lap:
                    cost += options.get("additional_current_stop_cost", 0)
            next_surface = surface
            cadence = options.get("weather_intervals")
            updates = (cadence[number - lap + 1] - cadence[number - lap]
                       if cadence is not None and number < track.total_laps else 1)
            for _ in range(updates):
                next_surface = next_surface.project_surface()
            visit(next_surface, next_tire, next_age + 1, number + 1,
                  stops + int(stop), total + cost,
                  stop if number == lap else first_stop,
                  next_tire.compound if number == lap and stop else first_compound,
                  used | {next_tire.compound})

    visit(weather, tire, age, lap, 0, 0, False, None, set(used_compounds))
    return minima[1], minima[0], best_compound


@pytest.mark.parametrize("wetness,rain", [(.1, .1), (.17, 0), (.18, .35), (.35, .5)])
@pytest.mark.parametrize("budget", [0, 1, 3])
@pytest.mark.parametrize("used", [set(), {TireCompound.SOFT},
                                  {TireCompound.SOFT, TireCompound.HARD}])
def test_mixed_slick_schedule_matches_exhaustive(wetness, rain, budget, used):
    args = inputs(TireCompound.SOFT, wetness, rain, laps=5)
    options = dict(remaining_dry_stops=1, remaining_damp_stops=0)
    expected = exhaustive(args, 15, 2, budget, used, **options)
    result = plan_rain_transition(*args, 15, 2, budget, used_compounds=used, **options)
    assert result.pit_now_cost == pytest.approx(expected[0], rel=0., abs=1e-10)
    assert result.wait_cost == pytest.approx(expected[1], rel=0., abs=1e-10)
    if result.compound is None:
        assert expected[2] is None
    else:
        # Reversed stint order can tie to floating-point precision. Verify
        # that the selected first compound can actually achieve the optimum,
        # independently of left-to-right versus suffix summation tie order.
        selected = exhaustive(args, 15, 2, budget, used,
                              required_first_compound=result.compound, **options)
        assert selected[0] == pytest.approx(expected[0], rel=0., abs=1e-10)


@pytest.mark.parametrize("cadence", [(0, 0, 2, 4), (0, 1, 1, 3)])
@pytest.mark.parametrize("queue", [-8, 10])
def test_mixed_controls_cadence_floor_and_fuel(cadence, queue):
    args = inputs(TireCompound.HARD, .17, 0, laps=5)
    args[1].base_pace = 1
    args[0].skill_rating = 1
    args[2].active_aero_zones = [
        ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1) for i in range(16)
    ]
    assert LapSimulator().calculate_lap_time(
        args[0], args[1], args[2], TIRE_COMPOUNDS[TireCompound.SOFT], args[3],
        3, 40, sample_variation=False,
    ) == args[2].base_lap_time * .95
    options = dict(weather_intervals=cadence, remaining_dry_stops=2,
                   remaining_damp_stops=1, current_traffic_gaps=(.2, .7),
                   physical_total_laps=40, current_lap_time_modifier=1.3,
                   active_aero_enabled=False, pit_lane_factor=.4,
                   additional_current_stop_cost=queue)
    used = {TireCompound.HARD}
    expected = exhaustive(args, 18, 2, 2, used, **options)
    result = plan_rain_transition(*args, 18, 2, 2, used_compounds=used, **options)
    assert result.pit_now_cost == pytest.approx(expected[0])
    assert result.wait_cost == pytest.approx(expected[1])


def test_prior_wear_does_not_grant_actual_use_or_wet_exemption():
    args = inputs(TireCompound.INTERMEDIATE, .01, 0, laps=1)
    # The retained rain tyre cannot run; fitting one slick cannot satisfy the rule.
    result = plan_rain_transition(*args, 20, 1, 0, used_compounds=set())
    assert result.pit_now_cost == inf
    assert result.wait_cost == inf
    credited = plan_rain_transition(*args, 20, 1, 0,
                                    used_compounds={TireCompound.INTERMEDIATE})
    assert credited.pit_now_cost < inf
    # Legacy omitted argument intentionally retains the assumed exemption.
    assert plan_rain_transition(*args, 20, 1, 0) == credited


@pytest.mark.parametrize("used", [True, "soft", ["unknown"], [None]])
def test_invalid_actual_use_is_rejected(used):
    with pytest.raises(ValueError, match="used_compounds"):
        plan_rain_transition(*inputs(TireCompound.SOFT), 5, 2, 1,
                             used_compounds=used)


@pytest.mark.parametrize("wetness,rain", [(.1, .1), (.18, .35)])
def test_native_mixed_slick_proposal_is_pure_and_executable(monkeypatch, wetness, rain):
    from types import SimpleNamespace

    from f1sim.simulation.race import DriverRaceState, RaceSimulator
    from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot

    driver, car, track, weather, tire = inputs(TireCompound.SOFT, wetness, rain, laps=6)
    state = DriverRaceState(driver, car, 1, current_tire=tire, tire_laps=12,
                            prior_tire_laps=12, tire_compound_history=["soft"])
    simulator = RaceSimulator(np.random.default_rng(7))
    calls = []

    def plan(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(should_pit=lambda: True, compound=TireCompound.SOFT)

    monkeypatch.setattr("f1sim.simulation.race.plan_rain_transition", plan)
    before = copy.deepcopy(simulator.rng.bit_generator.state)
    assert simulator._should_pit(
        state, [state], track, 2, False, weather, 3, physical_total_laps=40,
        traffic_snapshot=StrategyTrafficSnapshot(None, None, 0, (.2, .5)),
        weather_intervals=(0, 1, 2, 3, 4),
    )
    args, kwargs = calls[0]
    assert args[-1] == (4 if rain > .15 else 1)
    assert kwargs["used_compounds"] == set()
    assert kwargs["physical_total_laps"] == 40
    assert kwargs["current_traffic_gaps"] == (.2, .5)
    assert kwargs["additional_current_stop_cost"] == 3
    assert kwargs["weather_intervals"] == (0, 1, 2, 3, 4)
    assert simulator.rng.bit_generator.state == before
    simulator._execute_pit_stop(state, track, weather, 2, sample_service=False)
    assert state.current_tire.compound == TireCompound.SOFT
    assert state.prior_tire_laps == state.tire_laps == 0
    assert state.weather_pit_proposal is None


def test_future_wet_running_satisfies_rule_without_spurious_slick_correction():
    args = inputs(TireCompound.SOFT, .18, .35, laps=5)
    actual = plan_rain_transition(*args, 5, 2, 0, used_compounds={TireCompound.SOFT})
    expected = exhaustive(args, 5, 2, 0, {TireCompound.SOFT})
    assert actual.wait_cost == pytest.approx(expected[1])
    assert actual.wait_cost < inf


def test_legal_history_is_part_of_shared_cache_key():
    args = inputs(TireCompound.SOFT, .1, .1, laps=4)
    for used in ({TireCompound.SOFT}, {TireCompound.SOFT, TireCompound.HARD}, set(),
                 {TireCompound.INTERMEDIATE}, {TireCompound.SOFT}):
        actual = plan_rain_transition(*args, 10, 2, 1, used_compounds=used)
        expected = exhaustive(args, 10, 2, 1, used)
        assert actual.pit_now_cost == pytest.approx(expected[0])
        assert actual.wait_cost == pytest.approx(expected[1])


def test_slick_callers_must_supply_actual_compound_history():
    with pytest.raises(ValueError, match="used_compounds is required"):
        plan_rain_transition(*inputs(TireCompound.SOFT), 10, 2, 1)
