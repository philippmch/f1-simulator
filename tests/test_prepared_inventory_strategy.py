"""Untimed finite-pool preparation preserves exact decisions and custom dispatch."""

from copy import deepcopy
from math import inf

import pytest
from test_prepared_lap_evaluator import _models, _run_preimport_hook_script

from f1sim.models import ActiveAeroZone, Car, Driver, Tire, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.inventory_strategy import plan_inventory_strategy
from f1sim.simulation.lap import LapSimulator, minimum_lap_time
from f1sim.simulation.tire_inventory import TireInventory


def _inputs(wetness=0., rain=0., state="retained"):
    driver, car, track = _models()
    track.total_laps = 7
    driver.current_tire_laps = 101
    weather = Weather(track_wetness=wetness, rain_intensity=rain)
    inventory = TireInventory.from_sets([
        {"compound": "soft", "age": 18}, {"compound": "hard", "age": 7},
        {"compound": "hard"}, {"compound": "intermediate", "age": 4},
        {"compound": "wet", "age": 1},
    ])
    if state != "free":
        inventory.fit("set-1")
    if state == "refitted":
        inventory.fit("set-2", current_age=23)
        inventory.fit("set-1", current_age=9)
    if state == "force":
        inventory.mark_current_unavailable(23)
    return driver, car, track, weather, inventory


def _snapshot(inputs):
    return (tuple(model.model_dump_json() for model in inputs[:-1]),
            deepcopy(inputs[-1].__dict__))


@pytest.mark.parametrize("wetness,rain", [
    (0., 0.), (.16, .12), (.61, .72), (.3, 0.), (.04, .7),
])
@pytest.mark.parametrize("state", ["retained", "refitted", "force", "free"])
@pytest.mark.parametrize("budget", [0, 1, 3])
@pytest.mark.parametrize("pending", [False, True])
def test_prepared_inventory_decisions_are_bit_exact_and_preserve_inputs(
    monkeypatch, wetness, rain, state, budget, pending,
):
    inputs = _inputs(wetness, rain, state)
    before = _snapshot(inputs)
    options = dict(
        tire_age=0 if state == "free" else 23, remaining_stops=budget,
        remaining_dry_stops=1, remaining_damp_stops=0,
        used_compounds=() if state == "free" else ("soft", "hard"),
        force_stop=state == "force", free_fit=state == "free",
        pit_lane_factor=.6, additional_current_stop_cost=-3.125,
        current_lap_time_modifier=1.85 if state == "force" else 1.35,
        active_aero_enabled=False, physical_total_laps=17,
        weather_intervals=(0, 0, 1, 3, 6), current_traffic_gaps=(.3125, 1.375),
        tire_warmup={compound.value: 1.125 + index * .875
                     for index, compound in enumerate(TireCompound)},
        current_fit_pending=pending,
    )
    prepared = plan_inventory_strategy(*inputs, 3, **options)
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    public = plan_inventory_strategy(*inputs, 3, **options)
    assert prepared == public
    assert _snapshot(inputs) == before


def test_untimed_inventory_prepares_once_and_uses_native_evaluator(monkeypatch):
    inputs = _inputs()
    prepare = LapSimulator.prepare_deterministic_lap_time
    preparations = []
    evaluations = []

    def observe(self, driver, car, track, physical):
        evaluator = prepare(self, driver, car, track, physical)
        assert evaluator is not None
        preparations.append(physical)

        def evaluate(tire, weather, lap, age, gap, aero):
            evaluations.append((lap, age, gap, aero))
            return evaluator(tire, weather, lap, age, gap, aero)
        return evaluate

    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", observe)
    plan_inventory_strategy(*inputs, 3, tire_age=23, physical_total_laps=17,
                            used_compounds=("soft", "hard"),
                            current_traffic_gaps=(.3125, 1.375), active_aero_enabled=False)
    assert preparations == [17]
    assert len(evaluations) > 5
    assert any(lap == 3 and gap == .3125 and not aero for lap, _, gap, aero in evaluations)
    assert any(lap == 3 and gap == 1.375 and not aero for lap, _, gap, aero in evaluations)
    assert all(gap is None and aero for lap, _, gap, aero in evaluations if lap > 3)


def test_prepared_inventory_preserves_lap_floor_and_first_input_tie(monkeypatch):
    driver, car, track, weather, _ = _inputs()
    driver.skill_rating = car.base_pace = car.straight_line_speed = 1.
    track.active_aero_zones = [
        ActiveAeroZone(zone_id=index + 1, sector=1, time_gain=1.) for index in range(16)
    ]
    inventory = TireInventory.from_sets([{"compound": "soft"}, {"compound": "soft"}])
    inventory.fit("set-1")
    options = dict(remaining_stops=0, require_compound_rule=False, free_fit=True)
    native = plan_inventory_strategy(driver, car, track, weather, inventory, 3, **options)
    assert native.wait_cost == native.pit_now_cost == 5 * minimum_lap_time(track)
    assert native.set_id == "set-1"
    assert native.compound is TireCompound.SOFT
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert plan_inventory_strategy(driver, car, track, weather, inventory, 3, **options) == native


class _CustomDriver(Driver):
    pass


class _CustomCar(Car):
    def pace_delta_seconds(self, reference_lap_time):
        return super().pace_delta_seconds(reference_lap_time) + .125


class _CustomTrack(Track):
    @property
    def total_active_aero_gain(self):
        return super().total_active_aero_gain + .125


@pytest.mark.parametrize("index,model", [(0, _CustomDriver), (1, _CustomCar), (2, _CustomTrack)])
def test_untimed_inventory_model_subclasses_keep_public_dispatch(monkeypatch, index, model):
    inputs = list(_inputs())
    inputs[index] = model.model_validate(inputs[index].model_dump())
    prepare = LapSimulator.prepare_deterministic_lap_time
    preparations = 0

    def observe(self, *args):
        nonlocal preparations
        preparations += 1
        evaluator = prepare(self, *args)
        assert evaluator is None
        return evaluator

    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", observe)
    actual = plan_inventory_strategy(*inputs, 3, tire_age=23, used_compounds=("soft", "hard"))
    assert preparations == 1
    assert actual.wait_cost < inf
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert plan_inventory_strategy(*inputs, 3, tire_age=23,
                                   used_compounds=("soft", "hard")) == actual


def test_untimed_inventory_custom_calculator_keeps_age_and_current_controls(monkeypatch):
    inputs = _inputs()
    seen = []
    calculate = LapSimulator.calculate_lap_time

    def custom(self, driver, *args, **kwargs):
        seen.append((driver.current_tire_laps, kwargs))
        return calculate(self, driver, *args, **kwargs) + .125

    monkeypatch.setattr(LapSimulator, "calculate_lap_time", custom)
    assert LapSimulator().prepare_deterministic_lap_time(*inputs[:3], 17) is None
    options = dict(tire_age=23, physical_total_laps=17, used_compounds=("soft", "hard"),
                   active_aero_enabled=False, current_traffic_gaps=(.3125, 1.375))
    actual = plan_inventory_strategy(*inputs, 3, **options)
    assert seen and any(age == 23 for age, _ in seen)
    assert any(values["gap_to_car_ahead"] == .3125
               and not values["active_aero_enabled"] for _, values in seen)
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert plan_inventory_strategy(*inputs, 3, **options) == actual


class _CustomTire(Tire):
    def time_penalty_per_lap(self, *args, **kwargs):
        return super().time_penalty_per_lap(*args, **kwargs) + .125


class _CustomWeather(Weather):
    def lap_time_multiplier(self):
        return super().lap_time_multiplier() + .01


@pytest.mark.parametrize("leaf", ["tire", "weather"])
def test_prepared_inventory_preserves_custom_tire_and_weather_input_behavior(monkeypatch, leaf):
    inputs = list(_inputs())
    if leaf == "tire":
        original = TIRE_COMPOUNDS[TireCompound.SOFT]
        monkeypatch.setitem(TIRE_COMPOUNDS, TireCompound.SOFT,
                            _CustomTire.model_validate(original.model_dump()))
    else:
        inputs[3] = _CustomWeather.model_validate(inputs[3].model_dump())
    # Keep the native simulator guard intact. Custom tyres dispatch inside
    # preparation; existing untimed surface projection normalizes Weather
    # subclasses before either evaluator sees them.
    assert LapSimulator().prepare_deterministic_lap_time(*inputs[:3], 17) is not None
    prepare = LapSimulator.prepare_deterministic_lap_time
    leaf_calls = []

    def observe(self, driver, *args):
        evaluator = prepare(self, driver, *args)
        assert evaluator is not None

        def evaluate(tire, weather, lap, age, gap, aero):
            result = evaluator(tire, weather, lap, age, gap, aero)
            if isinstance(tire, _CustomTire) or isinstance(weather, _CustomWeather):
                # The public fallback writes explicit age into the copied driver.
                assert driver.current_tire_laps == age
                leaf_calls.append(age)
            return result
        return evaluate

    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", observe)
    options = dict(tire_age=23, used_compounds=("soft", "hard"), physical_total_laps=17)
    actual = plan_inventory_strategy(*inputs, 3, **options)
    if leaf == "tire":
        assert leaf_calls
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    assert plan_inventory_strategy(*inputs, 3, **options) == actual


@pytest.mark.parametrize("model_kind,target", [
    ("car", "class"), ("car", "instance"), ("car", "extra"),
    ("track", "class"), ("track", "instance"), ("track", "extra"),
])
@pytest.mark.parametrize("budget", [0, 2])
def test_inventory_stateful_fixed_model_hooks_match_public_costs_and_set_choices(
    monkeypatch, model_kind, target, budget,
):
    inputs = list(_inputs())
    index = 1 if model_kind == "car" else 2
    model = inputs[index]
    owner = type(model)
    name = "pace_delta_seconds" if model_kind == "car" else "total_active_aero_gain"
    original = getattr(owner, name)
    calls = 0

    def stateful(self, *args):
        nonlocal calls
        calls += 1
        value = (original.fget(self) if isinstance(original, property)
                 else original(self, *args))
        return value + calls * .125

    if target == "class":
        monkeypatch.setattr(owner, name,
                            property(stateful) if isinstance(original, property) else stateful)
    else:
        def bound(*args):
            return stateful(model, *args)
        if target == "instance":
            inputs[index] = model.model_copy(update={name: bound})
        else:
            object.__setattr__(model, "__pydantic_extra__", {name: bound})

    # Test the real shared boundary before comparing equal initial hook states.
    assert LapSimulator().prepare_deterministic_lap_time(*inputs[:3], 17) is None
    assert calls == 0
    options = dict(tire_age=23, remaining_stops=budget, physical_total_laps=17,
                   used_compounds=("soft", "hard"), current_lap_time_modifier=1.35,
                   active_aero_enabled=False, current_traffic_gaps=(.3125, 1.375),
                   tire_warmup={"soft": .75, "hard": 1.125}, current_fit_pending=True)
    actual = plan_inventory_strategy(*inputs, 3, **options)
    actual_calls = calls
    if target == "class" or (model_kind == "car" and target == "instance"):
        assert actual_calls > 1
    calls = 0
    monkeypatch.setattr(LapSimulator, "prepare_deterministic_lap_time", lambda *args: None)
    expected = plan_inventory_strategy(*inputs, 3, **options)
    assert actual == expected
    assert calls == actual_calls


@pytest.mark.parametrize("model_kind", ["car", "track"])
def test_preimport_stateful_model_hooks_match_public_inventory_dispatch(model_kind):
    _run_preimport_hook_script(model_kind, '''
        from test_prepared_inventory_strategy import _inputs
        from f1sim.simulation.inventory_strategy import plan_inventory_strategy

        inputs = _inputs()
        assert LapSimulator().prepare_deterministic_lap_time(*inputs[:3], 17) is None
        assert calls == 0
        options = dict(tire_age=23, remaining_stops=2, physical_total_laps=17,
                       used_compounds=("soft", "hard"), current_lap_time_modifier=1.35,
                       current_traffic_gaps=(.3125, 1.375), current_fit_pending=True)
        actual = plan_inventory_strategy(*inputs, 3, **options)
        actual_calls = calls
        assert actual_calls > 1

        calls = 0
        LapSimulator.prepare_deterministic_lap_time = lambda *args: None
        expected = plan_inventory_strategy(*inputs, 3, **options)
        assert actual == expected
        assert calls == actual_calls
    ''')
