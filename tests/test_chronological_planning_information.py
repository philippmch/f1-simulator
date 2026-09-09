"""Pit decisions use expected service and current distance/order information."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator


def setup(monkeypatch, service=100, *, laps=4, first_b=91, stop_lap=2, both=True):
    simulator = RaceSimulator(np.random.default_rng(4))
    engine = ChronologicalRace(simulator)
    drivers = [Driver(id=key, name=key, team_id="T") for key in "AB"]
    car = Car(team_id="T", team_name="T")
    track = Track(id="t", name="T", country="T", total_laps=laps,
                  base_lap_time=90, pit_lane_delta=22)
    decisions = {}

    def decide(state, states, planning, lap, *args, **kwargs):
        decisions[state.driver.id, lap] = (planning.total_laps,
                                          kwargs["additional_current_stop_cost"],
                                          kwargs["traffic_snapshot"])
        return both and state.driver.id == "B" and lap == stop_lap

    def control(lap, *args, **kwargs):
        if lap == stop_lap - 1:
            engine.states["A"].force_pit_next_lap = True
        return []

    monkeypatch.setattr(simulator, "_should_pit", decide)
    monkeypatch.setattr(simulator.event_manager, "process_lap", control)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake",
                        lambda *a, **kw: (True, False))
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time",
                        lambda driver, car, track, tire, weather, lap, *a, **kw:
                        first_b if driver.id == "B" and lap == 1 else 90)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_pit_stop_time", lambda car: service)
    monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.model_copy(deep=True))

    def run():
        return engine.run(drivers, {"T": car}, track,
                          Weather(track_wetness=.3, rain_intensity=.3), list("AB"),
                          starting_tires={key: TireCompound.INTERMEDIATE for key in "AB"})

    return engine, run, decisions, car


@pytest.mark.parametrize("service", [3, 8, 100])
def test_forced_teammate_stop_uses_expected_queue_but_charges_actual(monkeypatch, service):
    engine, run, decisions, car = setup(monkeypatch, service)
    results = run()
    assert decisions["B", 2][1] == pytest.approx(expected_stationary_time(car) - 1)
    b = next(result for result in results if result.driver_id == "B")
    assert b.pit_stop_details[0]["queue_time"] == service - 1
    assert b.pit_stop_details[0]["total_loss"] == 22 + service + service - 1
    assert decisions["B", 3][1] == 0
    assert run() == results  # Neither reservation ledger survives run reuse.
    assert engine.expected_box_releases.get("T", 0) <= max(r.total_time for r in results)


def test_completed_service_clears_expected_reservation(monkeypatch):
    _, run, decisions, _ = setup(monkeypatch, .5)
    results = run()
    assert decisions["B", 2][1] == 0
    assert next(r for r in results if r.driver_id == "B").pit_stop_details[0]["queue_time"] == 0


def test_future_sampled_service_cannot_change_rejoin_projection(monkeypatch):
    snapshots = []
    for service in (3, 8, 100):
        _, run, decisions, _ = setup(monkeypatch, service)
        run()
        snapshots.append(decisions["B", 2][2])
    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_collected_red_flag_clears_reservations_and_free_fits_do_not_reserve(monkeypatch):
    engine, run, decisions, _ = setup(monkeypatch, 100, laps=5, both=False)
    control = engine.simulator.event_manager
    control.set_forced_red_flag(2)
    forced_stop = control.process_lap
    process = type(control).process_lap

    def events(*args, **kwargs):
        forced_stop(*args, **kwargs)
        return process(control, *args, **kwargs)

    monkeypatch.setattr(control, "process_lap", events)
    monkeypatch.setattr(control, "_deploy_safety_measure", lambda *a: None)
    results = run()
    assert len(engine.suspensions) == 1
    assert sum(result.pit_stops for result in results) == 1
    assert decisions["B", 3][1] == 0
    assert engine.expected_box_releases == {}


@pytest.mark.parametrize("service", [3, 8, 4000])
def test_same_distance_on_track_leader_sets_timed_horizon(monkeypatch, service):
    _, run, decisions, _ = setup(monkeypatch, service, laps=90, both=False)
    results = run()
    assert decisions["B", 2][0] == 81
    winner = results[0]
    assert winner.driver_id == "B" and winner.laps_completed == 81


@pytest.mark.parametrize("service", [8, 4000])
def test_lap_ahead_pitter_remains_forecast_leader(monkeypatch, service):
    engine, run, decisions, car = setup(monkeypatch, service, laps=90, first_b=181,
                                    stop_lap=3, both=False)
    import f1sim.simulation.chronological_race as module
    forecast = module.forecast_final_lap
    anchors = []

    def observe(*args):
        if engine.states["B"].laps_completed == 1 and engine.states["B"].total_time == 181:
            anchors.append(args[1:3])
        return forecast(*args)

    monkeypatch.setattr(module, "forecast_final_lap", observe)
    run()
    assert anchors[0] == pytest.approx((3, 180 + 22 + expected_stationary_time(car) + 90))
    # A has two laps completed; B has only one. Future sampled service is hidden.
    assert decisions["B", 2][0] < 90


@pytest.mark.parametrize("pace", [67.5, 75, 89.7, 90, 110, 120, 180, 1800])
def test_controlled_forecasts_match_actual_finishes_with_nonrecurring_delays(pace):
    # Compare the planner with the actual event/finish controller, without
    # reproducing its deadline arithmetic. Delays must not become recurring pace.
    for gap in (1, 5, 20, 30, 60):
        for service in (.5, 3, 8, 40, 100):
            with pytest.MonkeyPatch.context() as patch:
                engine, run, decisions, _ = setup(
                    patch, service, laps=int(7200 / pace) + 5, first_b=pace, both=False,
                )
                patch.setattr(engine.simulator.lap_simulator, "calculate_lap_time",
                              lambda *a, **kw: pace)
                begin = engine._begin_running

                def delayed_start(state, pending, now):
                    begin(state, pending, now)
                    if state.driver.id == "B" and pending.lap == 1:
                        pending.ready += gap

                patch.setattr(engine, "_begin_running", delayed_start)
                result = next(row for row in run() if row.driver_id == "B")
                assert decisions["B", 2][0] == result.laps_completed, (pace, gap, service)
