"""Passing retains the earlier detection snapshot for a deployed burst."""

from copy import deepcopy

import numpy as np
import pytest

from f1sim.models import Car, Driver, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.overtaking import OvertakingModel
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator
from f1sim.simulation.strategy_traffic import StrategyTrafficSnapshot


class FixedRoll:
    def __init__(self, value):
        self.value = value
        self.draws = 0

    def random(self):
        self.draws += 1
        return self.value


def setup(current_gap=1.2):
    driver = Driver(id="A", name="A", team_id="T", overtaking_skill=1)
    defender = driver.model_copy(update={"id": "B", "name": "B"})
    car = Car(team_id="T", team_name="T", straight_line_speed=1)
    track = Track(id="t", name="T", country="T", total_laps=5, base_lap_time=90,
                  overtake_difficulty=.95)
    attacker_state = DriverRaceState(driver, car, 2, total_time=100 + current_gap)
    defender_state = DriverRaceState(defender, car, 1, total_time=100)
    simulator = RaceSimulator(np.random.default_rng(42))
    battle = (driver, car, defender, car, track, current_gap)
    baseline = simulator.overtaking_model._calculate_probability(*battle, False, False)
    boosted = simulator.overtaking_model._calculate_probability(*battle, True, False)
    rng = FixedRoll((baseline + boosted) / 2)
    simulator.overtaking_model.rng = rng
    return simulator, attacker_state, defender_state, track, rng


@pytest.mark.parametrize("detected_gap", [.8, 1])
@pytest.mark.parametrize("current_gap", [1.01, 1.2, 1.45])
def test_standard_passing_retains_a_burst_after_the_gap_widens(detected_gap, current_gap):
    simulator, attacker, defender, track, rng = setup(current_gap)
    attacker.overtake_mode_active_lap = simulator._deploy_overtake_mode_if_eligible(
        attacker, track, detected_gap, True,
    )
    assert attacker.overtake_mode_active_lap
    energy = attacker.overtake_mode_energy
    incidents = simulator._process_overtakes(
        [defender, attacker], track, Weather(), lap=5, overtake_mode_allowed=True,
    )
    assert attacker.position == 1
    assert defender.position == 2
    assert incidents == 0
    assert rng.draws == attacker.overtake_attempts == attacker.overtake_successes == 1
    assert attacker.overtake_mode_energy == energy == .65
    assert attacker.overtake_mode_deployments == 1
    assert attacker.overtake_mode_detected_gap == detected_gap


@pytest.mark.parametrize("detected_gap,active", [(0, True), (.8, True), (1, True), (1.01, False)])
@pytest.mark.parametrize("current_gap", [.8, 1.2])
def test_direct_maneuver_uses_supplied_detection_observation(detected_gap, active, current_gap):
    simulator, attacker, defender, track, rng = setup(current_gap)
    result = simulator.overtaking_model.attempt_overtake(
        attacker.driver, attacker.car, defender.driver, defender.car, track, current_gap,
        overtake_mode_active=True, detected_gap=detected_gap,
    )
    assert result[0] is active
    assert rng.draws == 1


def test_standalone_calls_keep_current_gap_gate_without_detection_snapshot():
    simulator, attacker, defender, track, rng = setup()
    result = simulator.overtaking_model.attempt_overtake(
        attacker.driver, attacker.car, defender.driver, defender.car, track, 1.2,
        overtake_mode_active=True,
    )
    assert not result[0]
    assert rng.draws == 1


@pytest.mark.parametrize("energy,allowed,active", [
    (.34, True, False), (.35, True, True), (1, False, False),
])
def test_detection_snapshot_does_not_override_energy_or_control_permission(energy, allowed, active):
    simulator, attacker, defender, track, rng = setup()
    attacker.overtake_mode_energy = energy
    attacker.overtake_mode_active_lap = simulator._deploy_overtake_mode_if_eligible(
        attacker, track, .8, allowed,
    )
    assert attacker.overtake_mode_active_lap is active
    energy_after_detection = attacker.overtake_mode_energy
    simulator._process_overtakes(
        [defender, attacker], track, Weather(), lap=5, overtake_mode_allowed=allowed,
    )
    assert (attacker.position == 1) is active
    assert rng.draws == 1
    assert attacker.overtake_mode_energy == energy_after_detection


def test_earlier_detection_does_not_extend_the_maneuver_opportunity_window():
    simulator, attacker, defender, track, rng = setup(1.51)
    attacker.overtake_mode_active_lap = simulator._deploy_overtake_mode_if_eligible(
        attacker, track, .8, True,
    )
    assert attacker.overtake_mode_active_lap
    simulator._process_overtakes(
        [defender, attacker], track, Weather(), lap=5, overtake_mode_allowed=True,
    )
    assert attacker.position == 2
    assert attacker.overtake_attempts == rng.draws == 0
    assert attacker.overtake_mode_energy == .65


def test_completed_lap_clears_detection_and_next_detection_replaces_it():
    simulator, attacker, _, track, _ = setup()
    attacker.overtake_mode_active_lap = simulator._deploy_overtake_mode_if_eligible(
        attacker, track, .8, True,
    )
    simulator._recharge_overtake_mode_energy([attacker])
    assert not attacker.overtake_mode_active_lap
    assert attacker.overtake_mode_detected_gap is None
    assert attacker.overtake_mode_energy == pytest.approx(.69)
    attacker.overtake_mode_active_lap = simulator._deploy_overtake_mode_if_eligible(
        attacker, track, 1.2, True,
    )
    assert not attacker.overtake_mode_active_lap
    assert attacker.overtake_mode_detected_gap == 1.2
    assert attacker.overtake_mode_energy == pytest.approx(.69)


@pytest.mark.parametrize("finite", [False, True])
def test_strategy_observation_does_not_replace_execution_detection_snapshot(finite):
    from test_strategy_overtake_mode import near_tie

    simulator, state, track = near_tie(finite=finite, aero=False)
    state.overtake_mode_detected_gap = .27
    before = deepcopy(state)
    inventory = state.tire_inventory
    inventory_before = deepcopy(vars(inventory)) if inventory is not None else None
    before.tire_inventory = inventory
    rng_before = deepcopy(simulator.rng.bit_generator.state)
    assert not simulator._should_pit(
        state, [state], track, 9, False, Weather(),
        traffic_snapshot=StrategyTrafficSnapshot(1, None, -.25, (1, None)),
    )
    assert state == before
    if inventory is not None:
        assert vars(inventory) == inventory_before
    assert simulator.rng.bit_generator.state == rng_before


@pytest.mark.parametrize("first_lap", [90.5, 91.2])
def test_chronological_contact_receives_earlier_gap_and_energy_snapshot(monkeypatch, first_lap):
    from test_chronological_traffic import run
    from test_chronological_traffic import setup as chronological_setup

    engine, args, _, _ = chronological_setup(
        monkeypatch, {"A": 90, "B": lambda lap: first_lap if lap == 1 else 60}, laps=2,
    )
    model = OvertakingModel(FixedRoll(.2))
    engine.simulator.overtaking_model = model
    actual = model.attempt_overtake
    observations = []

    def passing(*positional, **options):
        state = engine.states[positional[0].id]
        pending = engine.pending[positional[0].id]
        assert options["detected_gap"] == pending.detected_gap
        assert options["overtake_mode_active"] == pending.mode_active
        before = state.overtake_mode_energy
        result = actual(*positional, **options)
        assert state.overtake_mode_energy == before
        observations.append((positional[-1], options.copy(), state.overtake_mode_detected_gap))
        return result

    monkeypatch.setattr(model, "attempt_overtake", passing)
    run(engine, args)
    assert observations
    current_gap, options, stored_gap = observations[0]
    assert current_gap == 0
    assert options["detected_gap"] == stored_gap
    assert options["overtake_mode_active"] is (first_lap < 91)


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_paid_lap_clears_earlier_detection_without_spending_another_burst(monkeypatch, engine):
    simulator = RaceSimulator(np.random.default_rng(42))
    simulator.overtaking_model.rng = FixedRoll(1)
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.event_manager, "_deploy_safety_measure", lambda *a, **kw: None)
    chronological = ChronologicalRace(simulator)
    actual_deploy = simulator._deploy_overtake_mode_if_eligible
    actual_lap = simulator.lap_simulator.calculate_lap_time
    states, samples = {}, []

    def deploy(state, *args):
        states[state.driver.id] = state
        return actual_deploy(state, *args)

    def running(*args, **options):
        driver = args[0] if args else options["driver"]
        lap = args[5] if args else options["lap_number"]
        if driver.id == "B":
            state = states[driver.id]
            samples.append((lap, options["overtake_mode_active"],
                            state.overtake_mode_detected_gap, state.overtake_mode_deployments))
        return actual_lap(*args, **dict(options, sample_variation=False))

    monkeypatch.setattr(simulator, "_deploy_overtake_mode_if_eligible", deploy)
    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", running)
    drivers = [Driver(id=key, name=key, team_id=key) for key in ("A", "B")]
    cars = {driver.id: Car(team_id=driver.id, team_name=driver.id) for driver in drivers}
    track = Track(id="t", name="T", country="T", total_laps=4, base_lap_time=90,
                  pit_lane_delta=1, safety_car_probability=0)
    run = simulator.simulate_race if engine == "standard" else chronological.run
    run(drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": "medium", "B": "medium"},
        pit_plans={"A": [{"lap": 4, "compound": "hard"}],
                   "B": [{"lap": 3, "compound": "hard"}]})
    eligible = next(row for row in samples if row[0] == 2)
    paid = next(row for row in samples if row[0] == 3)
    assert eligible[1] and eligible[2] <= track.overtake_mode_detection_gap
    assert paid == (3, False, None, eligible[3])


def test_complete_standard_race_keeps_boost_when_native_lap_physics_widens_gap(monkeypatch):
    simulator = RaceSimulator(np.random.default_rng(42))
    monkeypatch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.event_manager, "_check_random_incident", lambda *a, **kw: None)
    monkeypatch.setattr(simulator.event_manager, "_deploy_safety_measure", lambda *a, **kw: None)
    actual_lap = simulator.lap_simulator.calculate_lap_time
    actual_passing = simulator.overtaking_model.attempt_overtake
    observations = []

    class ThirdAttempt:
        draws = 0

        def random(self):
            self.draws += 1
            return .05 if self.draws == 3 else 1

    simulator.overtaking_model.rng = ThirdAttempt()

    def running(*args, **options):
        return actual_lap(*args, **dict(options, sample_variation=False))

    def passing(*args, **options):
        result = actual_passing(*args, **options)
        observations.append((options.copy(), result))
        return result

    monkeypatch.setattr(simulator.lap_simulator, "calculate_lap_time", running)
    monkeypatch.setattr(simulator.overtaking_model, "attempt_overtake", passing)
    drivers = [Driver(id=key, name=key, team_id=key, overtaking_skill=1) for key in ("A", "B")]
    cars = {driver.id: Car(team_id=driver.id, team_name=driver.id, straight_line_speed=1,
                          base_pace=.8 if driver.id == "A" else .75) for driver in drivers}
    track = Track(id="t", name="T", country="T", total_laps=6, base_lap_time=90,
                  overtake_difficulty=.95, safety_car_probability=0)
    results = simulator.simulate_race(
        drivers, cars, track, Weather(change_probability=0), ["A", "B"],
        starting_tires={"A": "medium", "B": "medium"},
        pit_plans={key: [{"lap": 6, "compound": "hard"}] for key in ("A", "B")},
    )
    widened = [(options, result) for options, result in observations
               if options["gap"] > track.overtake_mode_detection_gap
               and options["overtake_mode_active"]]
    assert widened
    assert widened[0][0]["detected_gap"] <= track.overtake_mode_detection_gap
    assert widened[0][1] == (True, False), [
        (options["gap"], options["detected_gap"], options["overtake_mode_active"], result)
        for options, result in observations
    ]
    assert all(result.laps_completed == 6 and result.status == DriverStatus.FINISHED
               for result in results)
