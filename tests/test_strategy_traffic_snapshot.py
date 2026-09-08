"""Strategy decisions can consume current traffic without reading old clocks."""

import copy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.pit_strategy import DryPitDecision
from f1sim.simulation.race import DriverRaceState, RaceSimulator
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot


def fixture(wet=False):
    sim = RaceSimulator(np.random.default_rng(18))
    own = DriverRaceState(
        Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
        position=10, total_time=100, tire_laps=15,
        current_tire=TIRE_COMPOUNDS[
            TireCompound.INTERMEDIATE if wet else TireCompound.MEDIUM
        ],
    )
    track = Track(id="test", name="Test", country="Test", total_laps=50,
                  base_lap_time=90, pit_lane_delta=20, overtake_difficulty=0.6)
    weather = Weather(track_wetness=0.3 if wet else 0)
    return sim, own, track, weather


def reject_old_gaps(monkeypatch, sim):
    def unexpected(*args):
        pytest.fail("Snapshot must replace stale crossing-time traffic")

    for method in ("_get_gap_to_car_ahead", "_get_gap_to_car_behind",
                   "_pit_rejoin_traffic_cost"):
        monkeypatch.setattr(sim, method, unexpected)


@pytest.mark.parametrize("cost,expected", [(-0.25, True), (0.25, False)])
def test_none_gaps_are_authoritative_and_rejoin_cost_changes_near_tie(monkeypatch, cost, expected):
    sim, own, track, weather = fixture()
    reject_old_gaps(monkeypatch, sim)
    rng_before = copy.deepcopy(sim.rng.bit_generator.state)

    def plan(*args, **kwargs):
        return DryPitDecision(10 + args[-1], 10, TireCompound.HARD)

    monkeypatch.setattr("f1sim.simulation.race.plan_dry_stop", plan)
    assert sim._should_pit(
        own, [own], track, 17, False, weather,
        traffic_snapshot=StrategyTrafficSnapshot(None, None, cost),
    ) is expected
    assert sim.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("neutral", [None, "safety_car_active", "vsc_active"])
def test_weather_scales_only_green_traffic_and_queue_cost_is_preserved(monkeypatch, neutral):
    sim, own, track, weather = fixture()
    weather.track_wetness = 0.07
    weather.rain_intensity = 0.1
    own.car.wet_performance = 0.5
    if neutral:
        setattr(sim.event_manager, neutral, True)
    reject_old_gaps(monkeypatch, sim)
    observed = []

    def plan(*args, **kwargs):
        observed.append((args[-1], kwargs["current_lap_time_modifier"]))
        return DryPitDecision(10, 11, TireCompound.HARD)

    monkeypatch.setattr("f1sim.simulation.race.plan_dry_stop", plan)
    sim._should_pit(own, [own], track, 17, True, weather,
                    additional_current_stop_cost=2,
                    traffic_snapshot=StrategyTrafficSnapshot(None, None, 0.25))
    factor = sim.lap_simulator.weather_pace_multiplier(own.driver, own.car, weather)
    assert observed == [(pytest.approx(2 if neutral else 2 + 0.25 * factor),
                         sim.event_manager.get_lap_time_modifier())]


@pytest.mark.parametrize("behind,expected", [(None, False), (30, True)])
def test_wet_free_stop_window_uses_snapshot_gap(monkeypatch, behind, expected):
    sim, own, track, weather = fixture(wet=True)
    reject_old_gaps(monkeypatch, sim)
    # Isolate the traffic trigger; a separate cost check can reject its stop.
    monkeypatch.setattr(sim, "_weather_stop_can_pay", lambda *a, **k: True)
    monkeypatch.setattr(sim, "rng", type("NoRandomStop", (), {"random": lambda self: 1})())
    monkeypatch.setattr(sim, "_select_active_pit_plan", lambda *a, **k: [30])
    assert sim._should_pit(
        own, [own], track, 15, True, weather,
        traffic_snapshot=StrategyTrafficSnapshot(None, behind, 0),
    ) is expected


@pytest.mark.parametrize("ahead,expected", [(None, False), (1, True)])
def test_wet_undercut_uses_snapshot_gap(monkeypatch, ahead, expected):
    sim, own, track, weather = fixture(wet=True)
    reject_old_gaps(monkeypatch, sim)
    monkeypatch.setattr(sim, "_weather_stop_can_pay", lambda *a, **k: True)
    monkeypatch.setattr(sim, "rng", type("Threshold", (), {"random": lambda self: 0.3})())
    monkeypatch.setattr(sim, "_select_active_pit_plan", lambda *a, **k: [20])
    assert sim._should_pit(
        own, [own], track, 17, False, weather,
        traffic_snapshot=StrategyTrafficSnapshot(ahead, None, 0),
    ) is expected


def test_drying_track_rejects_costly_rain_refit_despite_large_gap(monkeypatch):
    sim, own, track, weather = fixture(wet=True)
    reject_old_gaps(monkeypatch, sim)
    monkeypatch.setattr(sim, "_select_active_pit_plan", lambda *a, **k: [30])
    assert not sim._should_pit(
        own, [own], track, 15, True, weather,
        traffic_snapshot=StrategyTrafficSnapshot(None, 30, 0),
    )


def test_default_matches_explicit_legacy_observations_and_rng():
    sim, own, track, weather = fixture()
    rival = copy.deepcopy(own)
    rival.driver = Driver(id="B", name="B", team_id="B")
    rival.position, rival.total_time = 9, 99
    field = [own, rival]
    snapshot = StrategyTrafficSnapshot(
        sim._get_gap_to_car_ahead(own, field), sim._get_gap_to_car_behind(own, field),
        sim._pit_rejoin_traffic_cost(own, field, track),
    )
    other_sim, other_field = copy.deepcopy((sim, field))
    assert sim._should_pit(own, field, track, 17, False, weather) == other_sim._should_pit(
        other_field[0], other_field, track, 17, False, weather, traffic_snapshot=snapshot,
    )
    assert own == other_field[0]
    assert sim.rng.bit_generator.state == other_sim.rng.bit_generator.state


def test_snapshot_is_immutable():
    snapshot = StrategyTrafficSnapshot(None, 1, -0.25)
    with pytest.raises(FrozenInstanceError):
        snapshot.gap_ahead = 1
