"""Native pit decisions include only an eligible retained first-lap deployment."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import ActiveAeroZone, Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.finish_strategy import ReplacementOption, evaluate_finish_protection
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.race import DriverRaceState, RaceSimulator, TeamStrategyArchetype
from f1sim.simulation.rain_strategy import RainTransitionDecision
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot
from f1sim.simulation.tire_inventory import TireInventory


def near_tie(energy=1, finite=False):
    track = Track(id="T", name="T", country="T", total_laps=20,
                  base_lap_time=90, pit_lane_delta=5, tire_stress=.3,
                  active_aero_zones=[ActiveAeroZone(zone_id=1, sector=2, time_gain=1)])
    state = DriverRaceState(
        Driver(id="D", name="D", team_id="T", tire_management=.8),
        Car(team_id="T", team_name="T", tire_degradation_factor=.8),
        2, tire_laps=8, overtake_mode_energy=energy,
    )
    if finite:
        state.tire_inventory = TireInventory.from_sets([
            dict(id="used", compound="medium", age=8),
            dict(id="soft", compound="soft"), dict(id="hard", compound="hard"),
        ])
        state.tire_inventory.current_set_id = "used"
    return RaceSimulator(np.random.default_rng(1)), state, track


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("energy,stop", [(0, True), (.34, True), (.35, False), (1, False)])
def test_available_deployment_reverses_near_tie_without_spending_energy(finite, energy, stop):
    sim, state, track = near_tie(energy, finite)
    before_rng = deepcopy(sim.rng.bit_generator.state)
    before_driver = state.driver.model_dump()
    snapshot = StrategyTrafficSnapshot(1, None, -.25, (1, None))

    actual = sim._should_pit(state, [state], track, 9, False, Weather(),
                             traffic_snapshot=snapshot)

    assert actual is stop
    assert state.overtake_mode_energy == energy
    assert state.overtake_mode_deployments == 0
    assert not state.overtake_mode_active_lap
    assert sim.rng.bit_generator.state == before_rng
    assert state.driver.model_dump() == before_driver


@pytest.mark.parametrize("gap,allowed", [(None, True), (1.01, True), (1, False)])
def test_detection_or_control_ineligibility_preserves_stop(gap, allowed):
    sim, state, track = near_tie()
    # Hold running traffic fixed to isolate the deployment detection snapshot.
    snapshot = StrategyTrafficSnapshot(gap, None, -.25, (1, None))
    assert sim._should_pit(state, [state], track, 9, False, Weather(),
                           traffic_snapshot=snapshot,
                           current_overtake_mode_allowed=allowed)


@pytest.mark.parametrize("weather", [Weather(track_wetness=.21), Weather(rain_intensity=.26)])
def test_weather_permission_suppresses_projection(weather):
    sim, state, track = near_tie()
    assert not sim._strategy_overtake_mode_active(state, track, 9, weather, 1)


def test_damp_transition_prices_exactly_one_current_deployment(monkeypatch):
    sim, state, track = near_tie()
    weather = Weather(track_wetness=.1, rain_intensity=.1)
    snapshot = StrategyTrafficSnapshot(1, None, -.25, (1, None))
    decisions = []
    original = RainTransitionDecision.should_pit

    def record(decision):
        decisions.append(decision)
        return original(decision)

    monkeypatch.setattr(RainTransitionDecision, "should_pit", record)
    for energy in (0, 1):
        state.overtake_mode_energy = energy
        sim._should_pit(state, [state], track, 9, False, weather, traffic_snapshot=snapshot)
    assert len(decisions) == 2
    assert decisions[0].pit_now_cost == decisions[1].pit_now_cost
    assert decisions[0].compound == decisions[1].compound
    expected = sim._strategy_mode_gain(state, track, weather, 9, True, 1)
    assert expected > 0
    assert decisions[0].wait_cost - decisions[1].wait_cost == pytest.approx(expected)


@pytest.mark.parametrize("energy", [0, .34, .35, 1, 2])
@pytest.mark.parametrize("gap,allowed", [(None, True), (1, False), (1, True), (1.01, True)])
def test_pure_eligibility_matches_execution(energy, gap, allowed):
    sim, state, track = near_tie(energy)
    prediction = sim._strategy_overtake_mode_active(state, track, 9, Weather(), gap, allowed)
    assert state.overtake_mode_energy == energy
    assert prediction == sim._deploy_overtake_mode_if_eligible(state, track, gap, allowed)


@pytest.mark.parametrize("zones", [1, 8, 20])
def test_gain_matches_current_lap_physics_including_floor(zones):
    sim, state, track = near_tie()
    track.active_aero_zones = [
        ActiveAeroZone(zone_id=i + 1, sector=2, time_gain=1) for i in range(zones)
    ]
    state.driver.skill_rating = 1
    state.car.base_pace = 1
    state.car.straight_line_speed = 1
    state.tire_laps = 0
    driver = state.driver.model_copy(deep=True)
    driver.current_tire_laps = 0
    physics = LapSimulator(np.random.default_rng(2))
    args = (driver, state.car, track, state.current_tire, Weather(), 19, 20)
    expected = (physics.calculate_lap_time(*args, gap_to_car_ahead=1,
                                         sample_variation=False)
                - physics.calculate_lap_time(*args, gap_to_car_ahead=1,
                                            sample_variation=False, overtake_mode_active=True))
    actual = sim._strategy_mode_gain(state, track, Weather(), 19, True, 1)
    assert actual == pytest.approx(expected)
    if zones == 20:
        assert actual == 0


def test_finish_distance_uses_mode_only_on_first_retained_lap():
    sim, state, track = near_tie()
    track.base_lap_time = 100

    class Physics:
        def __init__(self):
            self.calls = []

        def calculate_lap_time(self, driver, car, track, tire, weather, lap, total, **kw):
            active = kw["overtake_mode_active"]
            self.calls.append((tire.compound, lap, active))
            return 99 if active else 100

    def project(active):
        physics = Physics()
        result = evaluate_finish_protection(
            state.driver, state.car, track, state.current_tire, 8, 9, Weather(),
            0, 299.5, 20, lap_simulator=physics, expected_lane_loss=100,
            replacements=[ReplacementOption(TireCompound.SOFT)],
            current_overtake_mode_active=active,
        )
        return result, physics.calls

    baseline, _ = project(False)
    deployed, calls = project(True)
    assert baseline.retained_laps == baseline.stop_laps == 3
    assert not baseline.veto
    assert deployed.retained_laps == 4 and deployed.stop_laps == 3
    assert deployed.veto
    assert [(tire, lap) for tire, lap, active in calls if active] == [(TireCompound.MEDIUM, 9)]


def test_chronological_planning_uses_shared_interval_permission(monkeypatch):
    from test_chronological_strategy_traffic import setup

    engine, run, _, _ = setup(monkeypatch)
    checked = []
    monkeypatch.setattr(engine.simulator.event_manager, "is_overtake_mode_allowed",
                        lambda lap, weather: lap % 2 == 0)

    def observe(state, states, track, lap, *args, **kwargs):
        interval = engine.control_intervals + 1
        checked.append((lap, interval))
        assert kwargs["current_overtake_mode_allowed"] == (interval % 2 == 0)
        return False

    monkeypatch.setattr(engine.simulator, "_should_pit", observe)
    run()
    assert any(lap != interval for lap, interval in checked)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("energy,stop", [(0, True), (1, False)])
def test_native_engine_executes_near_tie_choice(monkeypatch, engine, energy, stop):
    sim, state, track = near_tie(energy)
    driver = state.driver
    leader = driver.model_copy(update={"id": "A", "name": "A", "team_id": "A"})
    cars = {"A": state.car.model_copy(update={"team_id": "A"}), "T": state.car}
    chronological = ChronologicalRace(sim)
    snapshot = StrategyTrafficSnapshot(1, None, -.25, (1, None))
    monkeypatch.setattr(sim, "_standard_pit_traffic_snapshot", lambda *a, **kw: snapshot)
    monkeypatch.setattr(chronological, "_strategy_traffic", lambda *a, **kw: snapshot)
    monkeypatch.setattr(sim, "_infer_team_strategy", lambda *a: TeamStrategyArchetype.BALANCED)
    monkeypatch.setattr(sim.event_manager, "process_lap", lambda *a, **kw: [])
    monkeypatch.setattr(sim.event_manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(sim.event_manager, "_check_random_incident", lambda *a, **kw: None)
    native = sim._should_pit
    observations = []

    def choose(candidate, states, planning, lap, *args, **kwargs):
        if candidate.driver.id != "D" or lap != 9:
            return False
        candidate.overtake_mode_energy = energy
        assert candidate.tire_laps == 8
        result = native(candidate, states, planning, lap, *args, **kwargs)
        observations.append(result)
        assert candidate.overtake_mode_energy == energy
        return result

    monkeypatch.setattr(sim, "_should_pit", choose)
    # Isolate strategy from battles and random lap noise; retain native pit execution.
    monkeypatch.setattr(sim, "_process_overtakes", lambda *a, **kw: 0)
    monkeypatch.setattr(sim.overtaking_model, "attempt_overtake", lambda *a, **kw: (True, False))
    calculate = sim.lap_simulator.calculate_lap_time

    def mean(*args, **kwargs):
        return calculate(*args, **dict(kwargs, sample_variation=False))

    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", mean)
    run = sim.simulate_race if engine == "standard" else chronological.run
    results = run([leader, driver], cars, track, Weather(change_probability=0), ["A", "D"],
                  starting_tires={"A": "medium", "D": "medium"})
    result = next(row for row in results if row.driver_id == "D")
    assert observations == [stop]
    assert (9 in result.pit_laps) is stop
