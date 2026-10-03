"""Known controls conserve physical sets, allowances and completed distance."""

from copy import deepcopy
from dataclasses import replace
from math import inf

import numpy as np
import pytest
from pydantic import PrivateAttr
from test_controlled_dry_strategy import single_car_context
from test_leading_finish_execution import run as native_late_race

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.chronological_finish import ChronologicalFinishContext
from f1sim.simulation.inventory_strategy import InventoryDecision, plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race_timing import RaceFinishTimeline
from f1sim.simulation.strategy_control_clock import StrategyControlContext
from f1sim.simulation.tire_inventory import TireInventory


def pool_fixture(horizon):
    driver = Driver(id="A", name="A", team_id="A")
    car = Car(team_id="A", team_name="A", pit_stop_avg=1.6, pit_stop_std=.1,
              tire_degradation_factor=1.5)
    track = Track(id="T", name="T", country="Test", total_laps=horizon,
                  base_lap_time=100., pit_lane_delta=.3, tire_stress=1.)
    inventory = TireInventory.from_sets([
        {"id": "H", "compound": "hard", "age": 12},
        {"id": "S", "compound": "soft", "age": 3},
        {"id": "S2", "compound": "soft", "age": 3},
        {"id": "M", "compound": "medium", "age": 5},
        {"id": "W", "compound": "wet", "age": 0},
    ])
    inventory.fit("H")
    return driver, car, track, Weather(), inventory


def enumerate_physical_schedules(inputs, control, intervals, budget, dry, used, pending,
                                 force=False):
    """Enumerate actual IDs, including outgoing wear and each reuse."""
    driver, car, track, weather, inventory = deepcopy(inputs)
    physics = LapSimulator(np.random.default_rng(0))
    sets = inventory.sets
    available = tuple(key for key in sets if key not in inventory.unavailable_ids)
    current = inventory.current_set_id
    ages = {key: item.age for key, item in sets.items()}
    ages[current] = 14
    expiries = {key: inf if item.remaining_laps is None else item.age + item.remaining_laps
                for key, item in sets.items()}
    warmup = {"soft": 2., "medium": .4, "hard": 1.1}
    best, selected = {False: inf, True: inf}, None

    def legal(compounds):
        return bool(compounds & {TireCompound.WET, TireCompound.INTERMEDIATE}) or len(compounds) > 1

    def visit(offset, current, wear, compounds, left, allowance, total, first, stopped):
        nonlocal selected
        if offset == track.total_laps:
            if legal(compounds) and total < best[stopped]:
                best[stopped] = total
                if stopped:
                    selected = first
            return
        controlled = offset < intervals
        for target in available:
            if wear[target] >= expiries[target]:
                continue
            changing = target != current
            compound = sets[target].compound
            if weather.tire_mismatch(compound) == "critical":
                continue
            if offset == 0 and force and not changing:
                continue
            if changing and current in available and not (
                left > 0 and allowance > 0 or not legal(compounds) and compound not in compounds
                or wear[current] >= expiries[current]
            ):
                continue
            age = wear[target]
            driver.current_tire_laps = age
            running = physics.calculate_lap_time(
                driver, car, track, TIRE_COMPOUNDS[compound], weather, offset + 1, 12,
                sample_variation=False, active_aero_enabled=not controlled)
            value = running * ((1.4 if control == "sc" else 1.2) if controlled else 1.)
            if changing:
                factor = (.55 if control == "sc" else .75) if controlled else 1.
                value += track.pit_lane_delta * factor + expected_stationary_time(car)
            if changing or offset == 0 and pending:
                value += warmup[compound.value]
            updated = dict(wear)
            updated[target] += 1
            visit(offset + 1, target, updated, compounds | {compound},
                  max(0, left - int(changing)), max(0, allowance - int(changing)), total + value,
                  target if offset == 0 else first, changing if offset == 0 else stopped)

    visit(0, current, ages, set(used), budget, dry, 0., None, False)
    return best, selected


@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("engine_name", ["standard", "chronological"])
@pytest.mark.parametrize("horizon,intervals", [(4, 4), (6, 2)])
@pytest.mark.parametrize("budget,dry,used,force,damaged", [
    (2, 2, {TireCompound.HARD}, False, False),
    (2, 1, {TireCompound.HARD, TireCompound.MEDIUM}, True, False),
    (0, 0, {TireCompound.HARD}, False, False),
    (0, 0, {TireCompound.HARD, TireCompound.MEDIUM}, True, True),
])
@pytest.mark.parametrize("pending", [False, True])
def test_known_control_costs_match_all_physical_schedules(
    control, engine_name, horizon, intervals, budget, dry, used, force, damaged, pending,
):
    inputs = pool_fixture(horizon)
    driver, car, track, weather, inventory = inputs
    if damaged:
        inventory.unavailable_ids.add("H")
    context = single_car_context(track, car, control, intervals)
    if engine_name == "chronological":
        field = ChronologicalFinishContext(
            "A", RaceFinishTimeline(12, ["A"]), ("A",), (), 100.,
            1.4 if control == "sc" else 1.2, control == "sc", control_intervals=intervals)
        context = StrategyControlContext(field, 50., context.current_stop_delay)
    before = deepcopy(inputs), deepcopy(context)
    actual = plan_inventory_strategy(
        driver, car, track, weather, inventory, 1, tire_age=14, remaining_stops=budget,
        remaining_dry_stops=dry, remaining_damp_stops=0, used_compounds=used,
        force_stop=force, physical_total_laps=12,
        tire_warmup={"soft": 2., "medium": .4, "hard": 1.1},
        current_fit_pending=pending, control_context=context)
    expected, selected = enumerate_physical_schedules(
        inputs, control, intervals, budget, dry, used, pending, force)
    assert actual.pit_now_cost == pytest.approx(expected[True], abs=1.e-8)
    assert actual.wait_cost == pytest.approx(expected[False], abs=1.e-8)
    assert actual.set_id == selected
    assert actual.pit_now_laps == (horizon if expected[True] < inf else -1)
    assert actual.wait_laps == (horizon if expected[False] < inf else -1)
    assert before[0][:-1] == inputs[:-1]
    assert before[0][-1].__dict__ == inventory.__dict__
    assert before[1] == context


@pytest.mark.parametrize("control", ["safety_car", "vsc"])
@pytest.mark.parametrize("duration", [2, 4])
@pytest.mark.parametrize("field_size", [2, 22])
def test_native_finite_control_choice_keeps_the_extra_completed_lap(
    monkeypatch, control, duration, field_size,
):
    case = 65, 30, 1., 20.
    options = dict(neutralization=control, rival=True, passing=False,
                   rival_skills=(.5,) * (field_size - 1), control_duration=duration)
    warmup = {"soft": 2., "medium": 1., "hard": 1.}
    chosen, decision = native_late_race(
        monkeypatch, "chronological", True, case, warmup,
        guarded=True, control_costs=True, **options)
    legacy, _ = native_late_race(
        monkeypatch, "chronological", True, case, warmup,
        guarded=False, control_costs=False, **options)
    assert not decision["native_stop"]
    assert chosen.laps_completed == legacy.laps_completed + 1 == 67
    # The native planner can defer service to a crossing that preserves the
    # extra lap. Only the costly decision at 65 must be rejected.
    assert chosen.pit_laps[0] == 35 and 65 not in chosen.pit_laps
    assert legacy.pit_laps == [35, 65]
    assert sum(row["laps_used"] for row in chosen.tire_set_history) == chosen.laps_completed


@pytest.mark.parametrize("pit_laps,wait_laps,expected", [(2, 1, True), (1, 2, False)])
def test_distance_ranking_precedes_cost_and_team_bias(pit_laps, wait_laps, expected):
    decision = InventoryDecision(1000., 1., "S", TireCompound.SOFT, pit_laps, wait_laps)
    assert decision.should_pit(-.1) is expected
    assert decision.should_pit(.1) is expected


@pytest.mark.parametrize("free_fit,paid,wetness,rain", [(True, False, 0., 0.),
                                                     (True, False, .05, 0.),
                                                     (True, False, .2, .8),
                                                     (False, True, .2, .8)])
def test_free_and_already_paid_fits_retain_the_existing_forecast(free_fit, paid, wetness, rain):
    driver, car, track, weather, inventory = pool_fixture(4)
    weather.track_wetness, weather.rain_intensity = wetness, rain
    context = single_car_context(track, car, "sc", 4)
    if paid:
        context = context.for_paid_fit()
    options = dict(free_fit=free_fit, used_compounds={TireCompound.HARD}, tire_age=14)
    assert plan_inventory_strategy(driver, car, track, weather, inventory, 1, **options) == (
        plan_inventory_strategy(driver, car, track, weather, inventory, 1,
                                control_context=context, **options))


def test_custom_car_keeps_actual_identity_and_isolated_state_through_green_suffix():
    calls = []

    class CustomCar(Car):
        _ledger: list = PrivateAttr(default_factory=list)

        def pace_delta_seconds(self, *args, **kwargs):
            self._ledger.append(self.team_id)
            calls.append((self.team_id, len(self._ledger)))
            return super().pace_delta_seconds(*args, **kwargs) + 5.

    driver, car, track, weather, inventory = pool_fixture(6)
    car = CustomCar.model_validate(car.model_dump())
    context = single_car_context(track, car, "sc", 2)
    result = plan_inventory_strategy(
        driver, car, track, weather, inventory, 1, tire_age=14, remaining_stops=0,
        used_compounds={TireCompound.HARD, TireCompound.MEDIUM}, control_context=context)
    clean = driver.model_copy(deep=True)
    expected = 0.
    for offset in range(6):
        clean.current_tire_laps = 14 + offset
        value = LapSimulator().calculate_lap_time(
            clean, Car.model_validate(car.model_dump()), track,
            TIRE_COMPOUNDS[TireCompound.HARD], weather, offset + 1, 6,
            sample_variation=False, active_aero_enabled=offset >= 2)
        expected += (value + 5.) * (1.4 if offset < 2 else 1.)
    assert result.wait_cost == pytest.approx(expected, abs=1.e-8)
    assert calls and set(calls) == {("A", 1)}
    assert car._ledger == []


def test_control_context_rejects_a_different_lap():
    driver, car, track, weather, inventory = pool_fixture(4)
    context = single_car_context(track, car, "sc", 4)
    with pytest.raises(ValueError, match="current lap"):
        plan_inventory_strategy(driver, car, track, weather, inventory, 2,
                                control_context=context)


@pytest.mark.parametrize("control", ["sc", "vsc"])
def test_observed_queue_is_charged_once_without_scaling_fit_cost(control):
    driver, car, track, weather, inventory = pool_fixture(4)
    context = single_car_context(track, car, control, 4)
    options = dict(tire_age=14, remaining_stops=1, used_compounds={TireCompound.HARD},
                   tire_warmup={"soft": 2., "medium": .4, "hard": 1.1})
    plain = plan_inventory_strategy(driver, car, track, weather, inventory, 1,
                                    control_context=context, **options)
    queued = replace(context, field=replace(context.field,
                                           stop_delay=context.current_stop_delay + 9.),
                     current_stop_delay=context.current_stop_delay + 9.)
    delayed = plan_inventory_strategy(driver, car, track, weather, inventory, 1,
                                      additional_current_stop_cost=9., control_context=queued,
                                      **options)
    assert delayed.pit_now_cost == pytest.approx(plain.pit_now_cost + 9., abs=1.e-8)
    assert delayed.wait_cost == pytest.approx(plain.wait_cost, abs=1.e-8)
    assert delayed.set_id == plain.set_id


def test_more_sets_of_the_same_compound_do_not_create_compliance():
    driver, car, track, weather, _ = pool_fixture(4)
    inventory = TireInventory.from_sets([{"id": name, "compound": "hard"} for name in "AB"])
    inventory.fit("A")
    result = plan_inventory_strategy(
        driver, car, track, weather, inventory, 1, remaining_stops=0,
        used_compounds={TireCompound.HARD},
        control_context=single_car_context(track, car, "sc", 4))
    assert result.set_id is None
    assert result.pit_now_cost == result.wait_cost == inf
    assert result.pit_now_laps == -1
    assert result.wait_laps == 3  # The illegal final dry crossing earns no distance.
    running = 0.
    driver = driver.model_copy(deep=True)
    for lap in range(1, 4):
        driver.current_tire_laps = lap - 1
        running += LapSimulator().calculate_lap_time(
            driver, car, track, TIRE_COMPOUNDS[TireCompound.HARD], weather,
            lap, 4, sample_variation=False, active_aero_enabled=False) * 1.4
    assert result.wait_partial_time == pytest.approx(running, abs=1.e-8)
    assert not result.should_pit()
