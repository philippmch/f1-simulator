"""Observed current traffic enters lap physics before clipping and control scaling."""

import copy
from math import inf

import numpy as np
import pytest

from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time, plan_dry_stop
from f1sim.simulation.rain_strategy import plan_rain_stop, plan_rain_transition


def inputs(kind, zones):
    return (Driver(id="D", name="D", team_id="T", skill_rating=1),
            Car(team_id="T", team_name="T", base_pace=1, straight_line_speed=1),
            Track(id="T", name="T", country="T", total_laps=30, base_lap_time=90,
                  pit_lane_delta=1, active_aero_zones=[
                      ActiveAeroZone(zone_id=i + 1, sector=1, time_gain=1) for i in range(zones)
                  ]),
            Weather(track_wetness=.4, rain_intensity=.4) if kind == "same" else Weather(),
            TIRE_COMPOUNDS[TireCompound.MEDIUM if kind == "dry"
                           else TireCompound.INTERMEDIATE].model_copy(deep=True))


def evaluate(kind, models, gaps, modifier=1, aero=True, queue=0):
    driver, car, track, weather, tire = models
    options = dict(current_traffic_gaps=gaps, current_lap_time_modifier=modifier,
                   active_aero_enabled=aero, additional_current_stop_cost=queue,
                   physical_total_laps=40)
    if kind == "dry":
        return plan_dry_stop(driver, car, track, tire, 20, 3, 1,
                             {TireCompound.MEDIUM}, wet_exemption=True, **options)
    planner = plan_rain_stop if kind == "same" else plan_rain_transition
    return planner(driver, car, track, weather, tire, 20, 28, 1, **options)


def oracle(kind, models, gaps, modifier, aero, queue):
    driver, car, track, weather, original = copy.deepcopy(models)
    simulator = LapSimulator(np.random.default_rng(4))
    wait, pit, selected = inf, inf, None
    for stop_at in (None, 0, 1, 2):
        compounds = ([original.compound] if kind == "same" else
                     [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD])
        for compound in compounds:
            tire, age, surface, total, legal = original, 20, weather, 0., True
            for offset in range(3):
                if offset == stop_at:
                    tire, age = TIRE_COMPOUNDS[compound], 0
                    total += track.pit_lane_delta + expected_stationary_time(car)
                    if offset == 0:
                        total += queue
                if kind == "transition" and surface.tire_mismatch(tire.compound) == "critical":
                    legal = False
                    break
                driver.current_tire_laps = age
                total += simulator.calculate_lap_time(
                    driver, car, track, tire, surface, 28 + offset, 40,
                    sample_variation=False, active_aero_enabled=aero if offset == 0 else True,
                    gap_to_car_ahead=(gaps[1] if stop_at == 0 else gaps[0])
                    if offset == 0 else None,
                ) * (modifier if offset == 0 else 1)
                age += 1
                surface = surface.project_surface()
            if legal and stop_at == 0 and total < pit:
                pit, selected = total, compound
            elif legal and stop_at != 0:
                wait = min(wait, total)
    return pit, wait, selected


@pytest.mark.parametrize("kind", ["dry", "same", "transition"])
@pytest.mark.parametrize("zones", [0, 7, 14, 22])
@pytest.mark.parametrize("controls", [(1, True, 0), (1.3, False, 7)])
def test_current_traffic_cost_changes_match_full_executed_laps(kind, zones, controls):
    models = inputs(kind, zones)
    # Keep intermediate retention feasible so both actions can be compared.
    if kind == "transition":
        models[3].track_wetness = .19
    before = copy.deepcopy(models)
    modifier, aero, queue = controls
    clean = evaluate(kind, models, None, *controls)
    traffic = evaluate(kind, models, (1., .3), *controls)
    expected_clean = oracle(kind, models, (None, None), modifier, aero, queue)
    expected_traffic = oracle(kind, models, (1., .3), modifier, aero, queue)
    # Dry fast plans omit common clean running terms; their branch deltas
    # must still equal full execution, including changes of best first set.
    assert traffic.pit_now_cost - clean.pit_now_cost == pytest.approx(
        expected_traffic[0] - expected_clean[0], abs=1e-10,
    )
    assert traffic.wait_cost - clean.wait_cost == pytest.approx(
        expected_traffic[1] - expected_clean[1], abs=1e-10,
    )
    if kind != "same":
        assert traffic.compound == expected_traffic[2]
    assert models == before
    assert evaluate(kind, models, None, *controls) == clean


@pytest.mark.parametrize("kind", ["dry", "same", "transition"])
def test_gap_free_option_preserves_defaults_and_queue_is_independent(kind):
    models = inputs(kind, 22)
    baseline = evaluate(kind, models, None)
    explicit = evaluate(kind, models, (None, None))
    assert explicit == baseline
    queued = evaluate(kind, models, (None, None), queue=8)
    assert queued.pit_now_cost == pytest.approx(baseline.pit_now_cost + 8)
    assert queued.wait_cost == baseline.wait_cost
