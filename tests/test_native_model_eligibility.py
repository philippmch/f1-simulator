"""Instance eligibility preserves mapping dispatch and live nested checks."""

import gc
import os
import subprocess
import sys
import weakref
from collections.abc import Mapping
from textwrap import dedent

import pytest
from pydantic import BaseModel

from f1sim.models import ActiveAeroZone, Car, Driver, Sector, Tire, Track, Weather
from f1sim.models._native import native_model
from f1sim.models.tire import TIRE_COMPOUNDS, TireCompound
from f1sim.simulation.lap import LapSimulator


def models():
    return (Driver(id="d", name="D", team_id="t"), Car(team_id="t", team_name="T"),
            Track(id="t", name="T", country="T", total_laps=3, base_lap_time=90,
                  sectors=[Sector(number=1, base_time=30)],
                  active_aero_zones=[ActiveAeroZone(zone_id=1, sector=1)]))


@pytest.mark.parametrize("kind", ["driver", "car", "track", "tire", "weather", "sector", "zone"])
def test_ordinary_native_instances_remain_eligible(kind):
    driver, car, track = models()
    choices = {"driver": driver, "car": car, "track": track,
               "tire": TIRE_COMPOUNDS[TireCompound.MEDIUM], "weather": Weather(),
               "sector": track.sectors[0], "zone": track.active_aero_zones[0]}
    assert native_model(choices[kind])


@pytest.mark.parametrize("location", ["values", "extra"])
def test_ordinary_method_hooks_are_detected_after_each_mutation(location):
    driver, car, track = models()
    calls = []

    def hook(*args):
        calls.append(1)
        return 1.

    if location == "values":
        values = car.__dict__
    else:
        values = {}
        object.__setattr__(car, "__pydantic_extra__", values)
    assert native_model(car)
    values["pace_delta_seconds"] = hook
    assert not native_model(car)
    assert LapSimulator().prepare_deterministic_lap_time(driver, car, track, 3) is None
    assert not calls
    del values["pace_delta_seconds"]
    assert native_model(car)
    assert LapSimulator().prepare_deterministic_lap_time(driver, car, track, 3) is not None


class MembershipDict(dict):
    def __init__(self, values):
        super().__init__(values)
        self.reject = False
        self.contains_calls = []

    def __contains__(self, name):
        self.contains_calls.append(name)
        return self.reject and name == "pace_delta_seconds"

    def __iter__(self):
        raise AssertionError("Eligibility must preserve custom membership without iterating keys")


class MembershipMapping(Mapping):
    def __init__(self):
        self.reject = False
        self.contains_calls = []
        self.truthy = True

    def __getitem__(self, name):
        raise KeyError(name)

    def __contains__(self, name):
        self.contains_calls.append(name)
        return self.reject and name == "pace_delta_seconds"

    def __iter__(self):
        raise AssertionError("Eligibility must preserve custom membership without iterating keys")

    def __len__(self):
        return int(self.truthy)


@pytest.mark.parametrize("location", ["values", "extra_dict", "extra_mapping"])
def test_custom_membership_is_used_instead_of_stored_keys(location):
    _, car, _ = models()
    custom = MembershipMapping() if location == "extra_mapping" else MembershipDict({
        **car.__dict__, "pace_delta_seconds": lambda *args: 99.,
    })
    object.__setattr__(car, "__dict__" if location == "values" else "__pydantic_extra__", custom)
    # Membership deliberately hides a stored hook, then reports an override
    # independently of key iteration. This is the existing mapping contract.
    assert native_model(car)
    assert custom.contains_calls
    custom.reject = True
    assert not native_model(car)
    assert "pace_delta_seconds" in custom.contains_calls
    custom.reject = False
    assert native_model(car)


def test_falsey_extra_mapping_is_still_normalized_before_membership():
    _, car, _ = models()
    custom = MembershipMapping()
    custom.reject = True
    custom.truthy = False
    object.__setattr__(car, "__pydantic_extra__", custom)
    assert native_model(car)
    assert custom.contains_calls == []
    custom.truthy = True
    assert not native_model(car)


@pytest.mark.parametrize("nested_name,owner", [("sectors", Sector),
                                               ("active_aero_zones", ActiveAeroZone)])
@pytest.mark.parametrize("parent", ["track", "car"])
def test_nested_extensions_are_checked_even_in_unexpected_raw_keys(nested_name, owner, parent):
    class CustomNested(owner):
        pass

    driver, car, track = models()
    model = track if parent == "track" else car
    native = track.sectors[0] if owner is Sector else track.active_aero_zones[0]
    values = model.__dict__
    values[nested_name] = [native]
    assert native_model(model)
    values[nested_name].append(CustomNested(**native.model_dump()))
    assert not native_model(model)
    assert LapSimulator().prepare_deterministic_lap_time(driver, car, track, 3) is None
    values[nested_name].pop()
    assert native_model(model)


def test_nested_checks_keep_order_and_short_circuit():
    class CustomSector(Sector):
        pass

    class NestedAccess(dict):
        def get(self, name, default=None):
            accessed.append(name)
            if name == "active_aero_zones":
                raise AssertionError("Later nested groups must not run after rejection")
            return super().get(name, default)

    _, car, _ = models()
    accessed = []
    object.__setattr__(car, "__dict__", NestedAccess({
        **car.__dict__, "sectors": [CustomSector(number=1, base_time=30), object()],
    }))
    assert not native_model(car)
    assert accessed == ["sectors"]


def test_eligibility_does_not_retain_checked_instances():
    weather = Weather()
    reference = weakref.ref(weather)
    for _ in range(20):
        assert native_model(weather)
    del weather
    gc.collect()
    assert reference() is None


def test_tire_instance_hook_rejects_native_dispatch_without_invocation():
    tire = Tire(**TIRE_COMPOUNDS[TireCompound.SOFT].model_dump())
    calls = []
    object.__setattr__(tire, "time_penalty_per_lap", lambda *args: calls.append(args))
    assert not native_model(tire)
    assert not calls


@pytest.mark.parametrize("owner", [Weather, BaseModel])
@pytest.mark.parametrize("api", ["model_copy", "__copy__", "__deepcopy__"])
def test_warm_surface_cache_preserves_class_copy_dispatch(monkeypatch, owner, api):
    from f1sim.simulation import surface_projection

    weather = Weather(rain_intensity=.5, track_wetness=.1)
    cache = surface_projection._surface_snapshots
    cache.cache_clear()
    native = surface_projection.projected_surfaces(weather, 3)
    assert surface_projection.projected_surfaces(weather, 3) == native
    before = cache.cache_info()
    assert before.hits == 1
    original = getattr(owner, api)
    extra = [.1]
    calls = []

    def hook(self, *args, **kwargs):
        calls.append(kwargs.get("deep"))
        copied = original(self, *args, **kwargs)
        copied.track_wetness += extra[0]
        return copied

    monkeypatch.setattr(owner, api, hook)
    assert not native_model(weather)
    assert calls == []
    first = surface_projection.projected_surfaces(weather, 3)
    extra[0] = .2
    second = surface_projection.projected_surfaces(weather, 3)
    if api == "__copy__":
        assert first == second == native
        assert calls == []  # Public model_copy(deep=True) does not use __copy__.
    else:
        assert first[1].track_wetness == pytest.approx(.28)
        assert second[1].track_wetness == pytest.approx(.38)
        assert first != second
        assert calls
        if api == "model_copy":
            assert all(calls)
    assert cache.cache_info() == before
    assert weather.track_wetness == .1
    assert first[0] is not weather and first[0] is not second[0]
    first[0].humidity = .9
    assert weather.humidity == .5
    assert second[0].humidity == .5


@pytest.mark.parametrize("api", ["model_copy", "__copy__", "__deepcopy__"])
@pytest.mark.parametrize("location", ["values", "extra"])
def test_warm_rain_cache_preserves_instance_copy_dispatch(monkeypatch, api, location):
    from f1sim.simulation import rain_strategy

    driver, car, track = models()
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    tire = TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].model_copy(deep=True)
    rain_strategy._plan.cache_clear()

    def decide():
        return rain_strategy.plan_rain_stop(driver, car, track, weather, tire, 0, 1, 0)

    native = decide()
    assert decide() == native
    before = rain_strategy._plan.cache_info()
    assert before.hits == 1
    original = getattr(tire, api)
    extra = [.01]
    calls = []

    def hook(*args, **kwargs):
        calls.append(kwargs.get("deep"))
        copied = original(*args, **kwargs)
        copied.degradation_rate += extra[0]
        return copied

    if location == "values":
        monkeypatch.setitem(tire.__dict__, api, hook)
    else:
        monkeypatch.setattr(tire, "__pydantic_extra__", {api: hook})
    assert not native_model(tire)
    assert calls == []
    first = decide()
    extra[0] = .02
    second = decide()
    if api == "__copy__" or location == "extra":
        assert first == second == native
        assert calls == []
    else:
        assert native.wait_cost < first.wait_cost < second.wait_cost
        assert calls
        if api == "model_copy":
            assert all(calls)
    assert rain_strategy._plan.cache_info() == before
    assert tire.degradation_rate == TIRE_COMPOUNDS[TireCompound.INTERMEDIATE].degradation_rate
    assert driver.current_tire_laps == 0


@pytest.mark.parametrize("name", ["__pydantic_private__", "__pydantic_fields_set__"])
def test_warm_surface_cache_preserves_state_descriptor_dispatch(monkeypatch, name):
    from f1sim.simulation import surface_projection

    weather = Weather(rain_intensity=.5, track_wetness=.1)
    cache = surface_projection._surface_snapshots
    cache.cache_clear()
    native = surface_projection.projected_surfaces(weather, 3)
    assert surface_projection.projected_surfaces(weather, 3) == native
    before = cache.cache_info()
    descriptor = BaseModel.__dict__[name]
    calls = []

    def getter(self):
        calls.append(self)
        return descriptor.__get__(self)

    def setter(self, value):
        descriptor.__set__(self, value)

    monkeypatch.setattr(Weather, name, property(getter, setter))
    assert not native_model(weather)
    assert calls == []
    first = surface_projection.projected_surfaces(weather, 3)
    assert first == native
    assert calls  # Public deep copying still reads the replaced descriptor.
    calls.clear()
    assert surface_projection.projected_surfaces(weather, 3) == native
    assert calls
    assert cache.cache_info() == before
    assert weather.track_wetness == .1
    first[0].humidity = .9
    assert weather.humidity == .5


COPY_CACHE_REGRESSION = dedent("""
    from f1sim.models import Weather
    from f1sim.simulation.surface_projection import projected_surfaces
    weather = Weather(rain_intensity=.5, track_wetness=.1)
    native = projected_surfaces(weather, 3)
    assert projected_surfaces(weather, 3) == native
    original = Weather.model_copy
    state = [.1]
    def copy(self, *, update=None, deep=False):
        assert deep
        result = original(self, update=update, deep=deep)
        result.track_wetness += state[0]
        return result
    Weather.model_copy = copy
    first = projected_surfaces(weather, 3)
    state[0] = .2
    second = projected_surfaces(weather, 3)
    print('warm native:', [item.track_wetness for item in native])
    print('copy hook +.1:', [item.track_wetness for item in first])
    print('copy hook +.2:', [item.track_wetness for item in second])
    assert abs(first[1].track_wetness - .28) < 1e-12
    assert abs(second[1].track_wetness - .38) < 1e-12
    assert weather.track_wetness == .1
""")


def test_copy_api_warm_cache_regression_in_isolated_package():
    result = subprocess.run(
        [sys.executable, "-c", COPY_CACHE_REGRESSION], capture_output=True, text=True,
        timeout=30, env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
