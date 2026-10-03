"""Late native stops are verified against complete mean-physics race execution."""

from copy import deepcopy

import numpy as np
import pytest
from test_custom_pit_replacements import snapshot

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import RaceSimulator, TeamStrategyArchetype


def inputs(lap, age, stress, lane, warmup, *, modifier=1.):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A", tire_degradation_factor=1.5,
              pit_stop_avg=2.75, pit_stop_std=.1)
    track = Track(id="timed", name="timed", country="test", total_laps=90,
                  base_lap_time=100., tire_stress=stress, pit_lane_delta=lane)
    physics = LapSimulator()
    prefix = 0.
    fit_lap = lap - age
    for number in range(1, lap + 1):
        compound = TireCompound.HARD if number < fit_lap else TireCompound.SOFT
        driver.current_tire_laps = number - 1 if number < fit_lap else number - fit_lap
        running = physics.calculate_lap_time(driver, car, track, TIRE_COMPOUNDS[compound],
                                             Weather(), number, 90, sample_variation=False)
        prefix += running * (modifier if number == lap else 1.)
    # Choose a reference pace that places the retained first crossing just
    # before expiry. This exposes the finish phase without changing tyre curves.
    fitting = warmup.get("soft", 0.)
    track.base_lap_time = (
        7200. - .5 - lane - expected_stationary_time(car) - fitting
    ) * 100 / prefix
    driver.reset_race_state()
    return driver, car, track


def run(monkeypatch, engine_name, finite, case, warmup, *, guarded, neutralization=None,
        rival=False, rival_skills=None, passing=True, control_duration=1):
    lap, age, stress, lane = case
    modifier = {None: 1., "vsc": 1.2, "safety_car": 1.4}[neutralization]
    driver, car, track = inputs(lap, age, stress, lane, warmup, modifier=modifier)
    simulator = RaceSimulator(np.random.default_rng(21), tire_warmup=warmup)
    engine = ChronologicalRace(simulator)
    native = simulator._should_pit
    physics = simulator.lap_simulator.calculate_lap_time
    observations = {}
    drivers, cars = [driver], {"A": car}
    if rival:
        for index, skill in enumerate((.5,) if rival_skills is None else rival_skills):
            key = chr(ord("B") + index)
            drivers.append(Driver(id=key, name=key, team_id=key, skill_rating=skill))
            cars[key] = car.model_copy(update={"team_id": key, "team_name": key}, deep=True)
    with monkeypatch.context() as patch:
        patch.setattr(simulator, "_infer_team_strategy",
                      lambda *args: TeamStrategyArchetype.BALANCED)
        def control(lap, *args, **kwargs):
            simulator.event_manager.current_lap = lap
            remaining = max(0, case[0] - 1 + control_duration - lap)
            active = case[0] - 1 <= lap < case[0] - 1 + control_duration
            simulator.event_manager.safety_car_active = (
                neutralization == "safety_car" and active)
            simulator.event_manager.vsc_active = neutralization == "vsc" and active
            simulator.event_manager.safety_car_laps_remaining = (
                remaining if simulator.event_manager.safety_car_active else 0)
            simulator.event_manager.vsc_laps_remaining = (
                remaining if simulator.event_manager.vsc_active else 0)
            return []

        patch.setattr(simulator.event_manager, "process_lap", control)
        patch.setattr(simulator.event_manager, "_check_mechanical_failure", lambda *args: None)
        patch.setattr(simulator.event_manager, "_check_random_incident",
                      lambda *args, **kwargs: None)
        if not passing:
            patch.setattr(simulator.overtaking_model, "should_attempt_overtake",
                          lambda *args, **kwargs: False)
            patch.setattr(simulator.overtaking_model, "attempt_overtake",
                          lambda *args, **kwargs: (False, False))
        patch.setattr(simulator.lap_simulator, "calculate_pit_stop_time",
                      lambda car: expected_stationary_time(car))

        def mean_running(*args, **kwargs):
            kwargs.pop("sample_variation", None)
            return physics(*args, sample_variation=False, **kwargs)

        patch.setattr(simulator.lap_simulator, "calculate_lap_time", mean_running)

        def decide(state, states, planning, number, *args, **kwargs):
            if number < lap:
                if number == lap - age:
                    if finite:
                        state.inventory_pit_proposal = (number, "S")
                    else:
                        state.dry_pit_proposal = (number, TireCompound.SOFT)
                    return True
                return False
            if state.driver.id != "A":
                return False
            stop = native(state, states, planning, number, *args, **kwargs)
            if number == lap:
                observations.update(native_stop=bool(stop), now=state.total_time,
                                    horizon=planning.total_laps,
                                    proposal=state.inventory_pit_proposal if finite
                                    else state.dry_pit_proposal)
                retained = state.driver.model_copy(deep=True)
                retained.current_tire_laps = state.tire_laps
                retained_time = LapSimulator().calculate_lap_time(
                    retained, state.car, track, state.current_tire, Weather(), number, 90,
                    sample_variation=False, active_aero_enabled=neutralization is None,
                )
                observations["retained_first"] = state.total_time + retained_time * modifier
            return stop

        patch.setattr(simulator, "_should_pit", decide)
        if engine_name == "standard":
            protect = simulator._protect_leading_finish_distance
            target, name = simulator, "_protect_leading_finish_distance"
        else:
            protect = engine._protect_elective_finish_distance
            target, name = engine, "_protect_elective_finish_distance"

        def inspect(state, *args, **kwargs):
            before = snapshot(state, simulator), deepcopy(track), deepcopy(engine.weather) if (
                engine_name == "chronological") else None
            result = protect(state, *args, **kwargs) if guarded else False
            assert before == (snapshot(state, simulator), track, engine.weather if (
                engine_name == "chronological") else None)
            if state.laps_completed + 1 == lap:
                observations["veto"] = result
            return result

        patch.setattr(target, name, inspect)
        inventory = [{"id": identifier, "compound": compound, "age": 0}
                     for identifier, compound in (("H", "hard"), ("S", "soft"),
                                                   ("S2", "soft"), ("M", "medium"))]
        execute = simulator.simulate_race if engine_name == "standard" else engine.run
        results = execute(drivers, cars, track, Weather(change_probability=0.), list(cars),
                          starting_tires={key: TireCompound.HARD for key in cars},
                          tire_inventory={key: inventory for key in cars} if finite else None)
        result = next(row for row in results if row.driver_id == "A")
    return result, observations


@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", [(45, 20, .4, 8.), (65, 30, 1., 20.)])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "hard": 1.}])
def test_late_native_choice_keeps_the_additional_executed_lap(
    monkeypatch, engine, finite, case, warmup,
):
    guarded, decision = run(monkeypatch, engine, finite, case, warmup, guarded=True)
    baseline, original = run(monkeypatch, engine, finite, case, warmup, guarded=False)
    lap = case[0]
    assert decision["native_stop"] and original["native_stop"]
    assert decision["retained_first"] == pytest.approx(7199.5, abs=1.e-8)
    assert decision["veto"] and not original["veto"]
    assert lap not in guarded.pit_laps and lap in baseline.pit_laps
    assert guarded.laps_completed == baseline.laps_completed + 1
    assert guarded.laps_completed == lap + 2
    assert guarded.race_time_limited and baseline.race_time_limited


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", [(45, 20, .4, 8.), (65, 30, 1., 20.)])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "hard": 1.}])
def test_standard_vsc_field_keeps_the_additional_executed_lap(
    monkeypatch, finite, case, warmup,
):
    guarded, decision = run(monkeypatch, "standard", finite, case, warmup, guarded=True,
                            neutralization="vsc", rival=True)
    baseline, original = run(monkeypatch, "standard", finite, case, warmup, guarded=False,
                             neutralization="vsc", rival=True)
    lap = case[0]
    assert decision["native_stop"] and original["native_stop"]
    assert decision["retained_first"] == pytest.approx(7199.5, abs=1.e-8)
    assert decision["veto"] and not original["veto"]
    assert guarded.pit_laps == [lap - case[1]]
    assert baseline.pit_laps == [lap - case[1], lap]
    assert guarded.laps_completed == baseline.laps_completed + 1 == lap + 2
    assert guarded.race_time_limited and baseline.race_time_limited


@pytest.mark.parametrize("neutralization", ["vsc", "safety_car"])
@pytest.mark.parametrize("engine", ["standard", "chronological"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", [(45, 20, .4, 8.), (65, 30, 1., 20.)])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "hard": 1.}])
def test_neutralized_native_stop_keeps_the_additional_executed_lap(
    monkeypatch, neutralization, engine, finite, case, warmup,
):
    guarded, decision = run(monkeypatch, engine, finite, case, warmup, guarded=True,
                            neutralization=neutralization)
    baseline, original = run(monkeypatch, engine, finite, case, warmup, guarded=False,
                             neutralization=neutralization)
    lap = case[0]
    assert decision["native_stop"] and original["native_stop"]
    assert decision["proposal"] == original["proposal"] == (
        (lap, "S2") if finite else (lap, TireCompound.SOFT))
    assert decision["retained_first"] == pytest.approx(7199.5, abs=1.e-8)
    assert decision["veto"] and not original["veto"]
    assert lap not in guarded.pit_laps and lap in baseline.pit_laps
    assert guarded.laps_completed == baseline.laps_completed + 1 == lap + 2
    assert guarded.race_time_limited and baseline.race_time_limited


@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", [(45, 20, .4, 8.), (65, 30, 1., 20.)])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "hard": 1.}])
@pytest.mark.parametrize("field_size", [2, 3, 22])
@pytest.mark.parametrize("rival_skill", [.5, .8, .9])
def test_safety_car_field_preserves_distance_and_equal_distance_choices(
    monkeypatch, finite, case, warmup, field_size, rival_skill,
):
    options = dict(neutralization="safety_car", rival=True, passing=False,
                   rival_skills=(rival_skill,) * (field_size - 1))
    guarded, decision = run(monkeypatch, "standard", finite, case, warmup,
                            guarded=True, **options)
    baseline, original = run(monkeypatch, "standard", finite, case, warmup,
                             guarded=False, **options)
    lap = case[0]
    assert decision["native_stop"] and original["native_stop"]
    assert decision["retained_first"] == pytest.approx(7199.5, abs=1.e-8)
    assert guarded.laps_completed == lap + 2
    if rival_skill <= .8:
        assert decision["veto"] is True
        assert guarded.laps_completed == baseline.laps_completed + 1
        assert guarded.pit_laps == [lap - case[1]]
        assert baseline.pit_laps == [lap - case[1], lap]
    else:
        # The faster rival anchors the stopped field before expiry. Both
        # continuations retain the extra lap, so the native choice survives.
        assert decision["veto"] is False
        assert guarded.laps_completed == baseline.laps_completed
        assert guarded.pit_laps == baseline.pit_laps == [lap - case[1], lap]
    assert guarded.race_time_limited and baseline.race_time_limited


@pytest.mark.parametrize("control", ["vsc", "safety_car"])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", [(45, 20, .4, 8.), (65, 30, 1., 20.)])
@pytest.mark.parametrize("warmup", [{}, {"soft": 2., "medium": 1., "hard": 1.}])
@pytest.mark.parametrize("field_size", [2, 3, 22])
@pytest.mark.parametrize("rival_skill", [.5, .8, .9])
def test_chronological_neutralized_field_preserves_finish_distance(
    monkeypatch, control, finite, case, warmup, field_size, rival_skill,
):
    options = dict(neutralization=control, rival=True, passing=False,
                   rival_skills=(rival_skill,) * (field_size - 1))
    guarded, decision = run(monkeypatch, "chronological", finite, case, warmup,
                            guarded=True, **options)
    baseline, original = run(monkeypatch, "chronological", finite, case, warmup,
                             guarded=False, **options)
    lap = case[0]
    assert decision["native_stop"] and original["native_stop"]
    assert guarded.laps_completed == lap + 2
    if rival_skill <= .8:
        assert decision["veto"] is True
        assert guarded.laps_completed == baseline.laps_completed + 1
        assert guarded.pit_laps == [lap - case[1]]
        assert baseline.pit_laps == [lap - case[1], lap]
    else:
        assert decision["veto"] is False
        assert guarded.laps_completed == baseline.laps_completed
        assert guarded.pit_laps == baseline.pit_laps == [lap - case[1], lap]
    assert guarded.race_time_limited and baseline.race_time_limited


@pytest.mark.parametrize("control", ["vsc", "safety_car"])
@pytest.mark.parametrize("duration", [2, 4])
@pytest.mark.parametrize("finite", [False, True])
@pytest.mark.parametrize("case", [(45, 20, .4, 8.), (65, 30, 1., 20.)])
@pytest.mark.parametrize("field_size", [2, 22])
@pytest.mark.parametrize("rival_skill", [.5, .9])
def test_chronological_known_control_duration_preserves_native_distance_choices(
    monkeypatch, control, duration, finite, case, field_size, rival_skill,
):
    options = dict(neutralization=control, rival=True, passing=False,
                   rival_skills=(rival_skill,) * (field_size - 1), control_duration=duration)
    warmup = {"soft": 2., "medium": 1., "hard": 1.}
    guarded, decision = run(monkeypatch, "chronological", finite, case, warmup,
                            guarded=True, **options)
    baseline, original = run(monkeypatch, "chronological", finite, case, warmup,
                             guarded=False, **options)
    lap = case[0]
    assert decision["native_stop"] and original["native_stop"]
    assert guarded.laps_completed == lap + 2
    if rival_skill == .5:
        assert decision["veto"] is True
        assert guarded.laps_completed == baseline.laps_completed + 1
        assert lap not in guarded.pit_laps and lap in baseline.pit_laps
    else:
        assert decision["veto"] is False
        assert guarded.laps_completed == baseline.laps_completed
        assert guarded.pit_laps == baseline.pit_laps
