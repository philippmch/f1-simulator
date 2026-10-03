"""Known-control dry costs and choices match independently completed running."""

from copy import deepcopy
from itertools import product

import numpy as np
import pytest
from test_leading_finish_execution import run as late_native_race

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_race import ChronologicalRace
from f1sim.simulation.finish_strategy import SafetyCarFinishBranch, SafetyCarFinishField
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time, plan_dry_stop
from f1sim.simulation.race import RaceSimulator
from f1sim.simulation.strategy_control_clock import StandardControlContext, StrategyControlContext


@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
@pytest.mark.parametrize("control_name,advantage", [("vsc", .16), ("sc", .52)])
def test_committed_compound_matches_faster_completed_controlled_stint(
    monkeypatch, engine_name, control_name, advantage,
):
    # The fitting fee makes medium faster if only the first lap is controlled.
    # Four native controlled laps make soft faster in both real race engines.
    execute_stop = RaceSimulator._execute_pit_stop

    def mean_stop(simulator, *args, **kwargs):
        kwargs["sample_service"] = False
        return execute_stop(simulator, *args, **kwargs)

    def physics(simulator, driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        assert total_laps == 90
        return (450. if lap_number == 1 else
                99.55 if tire.compound == TireCompound.SOFT else 100.)

    monkeypatch.setattr(RaceSimulator, "_execute_pit_stop", mean_stop)
    monkeypatch.setattr(LapSimulator, "calculate_lap_time", physics)

    def complete(replacement=None):
        simulator = RaceSimulator(np.random.default_rng(34), tire_warmup={"soft": 2.})
        engine = ChronologicalRace(simulator) if engine_name == "chronological" else None
        control = simulator.event_manager

        def update(lap, *args, **kwargs):
            active, remaining = 65 <= lap < 69, max(0, 69 - lap)
            control.safety_car_active = active and control_name == "sc"
            control.vsc_active = active and control_name == "vsc"
            control.safety_car_laps_remaining = remaining if control.safety_car_active else 0
            control.vsc_laps_remaining = remaining if control.vsc_active else 0
            return []

        def stop(state, states, track, lap, *args, **kwargs):
            if lap == 66:
                state.force_pit_next_lap = True
                return True
            return False

        monkeypatch.setattr(control, "process_lap", update)
        monkeypatch.setattr(control, "_check_mechanical_failure", lambda *a, **k: None)
        monkeypatch.setattr(control, "_check_random_incident", lambda *a, **k: None)
        monkeypatch.setattr(simulator, "_should_pit", stop)
        if replacement is not None:
            monkeypatch.setattr(simulator, "_choose_committed_dry_compound",
                                lambda *a, **k: replacement)
        execute = engine.run if engine is not None else simulator.simulate_race
        result, = execute(
            [Driver(id="A", name="A", team_id="A")], {"A": Car(team_id="A", team_name="A")},
            Track(id="T", name="T", country="Test", total_laps=90, base_lap_time=100.),
            Weather(change_probability=0.), ["A"], starting_tires={"A": TireCompound.HARD})
        return result

    chosen, medium, soft = complete(), complete(TireCompound.MEDIUM), complete(TireCompound.SOFT)
    assert chosen.strategy == ["hard", "soft"]
    assert chosen.pit_laps == medium.pit_laps == soft.pit_laps == [66]
    assert chosen.laps_completed == medium.laps_completed == soft.laps_completed == 69
    assert chosen.total_time == pytest.approx(soft.total_time, abs=1.e-8)
    assert medium.total_time - soft.total_time == pytest.approx(advantage, abs=1.e-8)


def single_car_context(track, car, control, intervals):
    modifier, factor = (1.4, .55) if control == "sc" else (1.2, .75)
    empty = SafetyCarFinishBranch((None,))
    delay = track.pit_lane_delta * factor + expected_stationary_time(car)
    field = StandardControlContext(1, 50., SafetyCarFinishField(empty, empty),
                                   modifier, control == "sc", intervals, delay)
    return StrategyControlContext(field, 50., delay)


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("intervals", [2, 4])
@pytest.mark.parametrize("budget", [0, 1, 2])
@pytest.mark.parametrize("pending_fit", [False, True])
def test_costs_match_every_legal_controlled_and_green_paid_schedule(
    monkeypatch, control, intervals, budget, pending_fit,
):
    driver, car = Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="Test", total_laps=4,
                  base_lap_time=100., pit_lane_delta=3.)
    warmup = {"soft": 2., "medium": .4, "hard": 1.1}
    context = single_car_context(track, car, control, intervals)
    current = TIRE_COMPOUNDS[TireCompound.HARD]
    fuel = []

    def physics(simulator, driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        fuel.append(total_laps)
        return ({TireCompound.SOFT: 95., TireCompound.MEDIUM: 97., TireCompound.HARD: 100.
                 }[tire.compound] + 3. * driver.current_tire_laps - .2 * lap_number
                - (.3 if kwargs.get("active_aero_enabled", True) else 0.))

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", physics)
    before = driver.model_copy(deep=True), car.model_copy(deep=True), deepcopy(context)
    result = plan_dry_stop(
        driver, car, track, current, 4, 4, budget, {TireCompound.HARD, TireCompound.MEDIUM},
        tire_warmup=warmup, current_fit_pending=pending_fit,
        physical_total_laps=13, control_context=context)
    expected = {"wait": (float("inf"), None), "pit": (float("inf"), None)}
    for actions in product((None, *SLICKS), repeat=4):
        if sum(action is not None for action in actions) > budget:
            continue
        tire, age, total = current, 4, 0.
        for offset, action in enumerate(actions):
            controlled = offset < intervals
            if action is not None:
                tire, age = TIRE_COMPOUNDS[action], 0
                factor = (.55 if control == "sc" else .75) if controlled else 1.
                total += track.pit_lane_delta * factor + expected_stationary_time(car)
            projection = driver.model_copy(update={"current_tire_laps": age})
            running = physics(None, projection, car, track, tire, Weather(), offset + 1, 13,
                              active_aero_enabled=not controlled)
            total += running * ((1.4 if control == "sc" else 1.2) if controlled else 1.)
            if action is not None or offset == 0 and pending_fit:
                total += warmup[tire.compound.value]
            age += 1
        kind = "wait" if actions[0] is None else "pit"
        if total < expected[kind][0]:
            expected[kind] = total, actions[0]
    assert result.wait_cost == pytest.approx(expected["wait"][0], abs=1.e-8)
    assert result.pit_now_cost == pytest.approx(expected["pit"][0], abs=1.e-8)
    assert result.compound == expected["pit"][1]
    assert fuel and set(fuel) == {13}
    assert before == (driver, car, context)


def test_actual_current_tire_parameters_are_separate_from_same_compound_replacements():
    driver, car = Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A")
    track = Track(id="T", name="T", country="Test", total_laps=3,
                  base_lap_time=100., pit_lane_delta=22.)
    context = single_car_context(track, car, "vsc", 3)
    normal = TIRE_COMPOUNDS[TireCompound.SOFT]
    unusual = normal.model_copy(update={"initial_grip": .7, "degradation_rate": .004})

    def evaluate(tire):
        return plan_dry_stop(driver, car, track, tire, 0, 3, 1,
                             {TireCompound.MEDIUM, TireCompound.HARD}, control_context=context)

    first, second = evaluate(normal), evaluate(unusual)
    assert first.wait_cost != pytest.approx(second.wait_cost, abs=1.e-5)
    assert evaluate(normal) == first and evaluate(unusual) == second


@pytest.mark.parametrize("control", ["vsc", "sc"])
@pytest.mark.parametrize("intervals", [2, 4])
@pytest.mark.parametrize("budget", [1, 2])
def test_native_green_suffix_matches_exhaustive_completed_paid_schedules(
    control, intervals, budget,
):
    driver = Driver(id="A", name="A", team_id="A", skill_rating=.94)
    car = Car(team_id="A", team_name="A", base_pace=.98)
    track = Track(id="T", name="T", country="Test", total_laps=5,
                  base_lap_time=92., pit_lane_delta=2.)
    warmup = {"soft": .5, "medium": 1., "hard": 1.5}
    context = single_car_context(track, car, control, intervals)
    current = TIRE_COMPOUNDS[TireCompound.HARD].model_copy(
        update={"initial_grip": .72, "degradation_rate": .01})
    physics = LapSimulator(np.random.default_rng(19))
    service = expected_stationary_time(car)
    # Cache only the independently executed native lap, then enumerate every
    # paid sequence; no optimizer table or field projection supplies the oracle.
    laps = {}
    for offset in range(5):
        for compound in (None, *SLICKS):
            for age in range(9):
                projection = driver.model_copy(update={"current_tire_laps": age})
                laps[offset, compound, age] = physics.calculate_lap_time(
                    projection, car, track,
                    current if compound is None else TIRE_COMPOUNDS[compound],
                    Weather(), offset + 1, 13, sample_variation=False,
                    active_aero_enabled=offset >= intervals)
    expected = {"wait": (float("inf"), None), "pit": (float("inf"), None)}
    for actions in product((None, *SLICKS), repeat=5):
        if sum(action is not None for action in actions) > budget:
            continue
        compound, age, elapsed = None, 4, 0.
        for offset, action in enumerate(actions):
            controlled = offset < intervals
            if action is not None:
                compound, age = action, 0
                factor = (.55 if control == "sc" else .75) if controlled else 1.
                elapsed += track.pit_lane_delta * factor + service
            elapsed += laps[offset, compound, age] * (
                (1.4 if control == "sc" else 1.2) if controlled else 1.)
            if action is not None:
                elapsed += warmup[compound.value]
            age += 1
        kind = "wait" if actions[0] is None else "pit"
        if elapsed < expected[kind][0]:
            expected[kind] = elapsed, actions[0]
    result = plan_dry_stop(driver, car, track, current, 4, 5, budget,
                           {TireCompound.HARD, TireCompound.MEDIUM}, physical_total_laps=13,
                           tire_warmup=warmup, control_context=context)
    assert result.wait_cost == pytest.approx(expected["wait"][0], abs=1.e-8)
    assert result.pit_now_cost == pytest.approx(expected["pit"][0], abs=1.e-8)
    assert result.compound == expected["pit"][1]
    assert result.wait_laps == result.pit_now_laps == 5


def test_control_prefix_and_green_suffix_preserve_extension_identity_and_race_state(monkeypatch):
    driver = Driver(id="actual", name="Actual driver", team_id="actual", total_race_time=321.)
    car = Car(team_id="actual", team_name="Actual car")
    track = Track(id="actual", name="Actual track", country="Test", total_laps=4,
                  base_lap_time=100., pit_lane_delta=2.)
    context = single_car_context(track, car, "vsc", 2)
    current = TIRE_COMPOUNDS[TireCompound.HARD]
    changed, seen = {"extra": 0.}, []

    def custom_physics(owner, driver, car, track, tire, weather, lap_number, total_laps, **kwargs):
        assert (driver.id, driver.name, driver.team_id, driver.total_race_time) == (
            "actual", "Actual driver", "actual", 321.)
        assert (car.team_id, car.team_name, track.id) == ("actual", "Actual car", "actual")
        assert total_laps == 13
        seen.append(lap_number)
        return 99. + driver.current_tire_laps + changed["extra"]

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", custom_physics)
    before = driver.model_copy(deep=True), car.model_copy(deep=True), deepcopy(context)

    def evaluate():
        return plan_dry_stop(driver, car, track, current, 3, 4, 1,
                             {TireCompound.MEDIUM, TireCompound.HARD},
                             physical_total_laps=13, control_context=context)

    first = evaluate()
    changed["extra"] = 2.
    second = evaluate()
    assert second.wait_cost > first.wait_cost and second.pit_now_cost > first.pit_now_cost
    assert evaluate() == second
    assert set(seen) == {1, 2, 3, 4}
    assert before == (driver, car, context)


@pytest.mark.parametrize("control", ["vsc", "safety_car"])
@pytest.mark.parametrize("duration", [1, 2, 4])
@pytest.mark.parametrize("rival_skill", [.5, .9])
def test_native_control_costs_preserve_distance_before_finish_guard(
    monkeypatch, control, duration, rival_skill,
):
    case = (45, 20, .4, 8.)
    warmup = {"soft": 2., "medium": 1., "hard": 1.}
    options = dict(neutralization=control, rival=True, passing=False,
                   rival_skills=(rival_skill,), control_duration=duration)
    chosen, decision = late_native_race(
        monkeypatch, "chronological", False, case, warmup, guarded=False,
        control_costs=True, **options)
    older, original = late_native_race(
        monkeypatch, "chronological", False, case, warmup, guarded=False,
        control_costs=False, **options)
    assert original["native_stop"]
    assert chosen.laps_completed >= older.laps_completed
    if rival_skill == .5:
        assert not decision["native_stop"]
        assert chosen.laps_completed == older.laps_completed + 1
        assert case[0] not in chosen.pit_laps and case[0] in older.pit_laps
    elif decision["native_stop"]:
        assert chosen.pit_laps == older.pit_laps
    else:
        assert chosen.laps_completed == older.laps_completed + 1
