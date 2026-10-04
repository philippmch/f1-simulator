"""Compact native weather clocks agree with complete held-field events."""

from copy import deepcopy
from dataclasses import replace
from math import floor

import pytest
from test_controlled_strategy_field import signature

from f1sim.models import _native
from f1sim.simulation.chronological_finish import (
    ChronologicalFinishCar,
    ChronologicalFinishContext,
    ObservedChronologicalField,
    ObservedFieldCrossing,
    _ProjectedLap,
)
from f1sim.simulation.controlled_weather_strategy import (
    _green_weather_clock_key,
    green_weather_forecast,
)
from f1sim.simulation.race_timing import DriverFinishState, RaceFinishClock, RaceFinishTimeline
from f1sim.simulation.strategy_weather_clock import StrategyWeatherClock


def green_field(now, own_pace, scheduled, off_track, retired):
    paces = {"A": own_pace, **{f"B{i}": 99. + (i * 19 % 11) * .13 for i in range(21)}}
    paces["B20"] = 70.25
    ledger = RaceFinishTimeline(scheduled, paces)
    observations = sorted(
        (lap * pace, -lap, key)
        for key, pace in paces.items() for lap in range(1, floor(now / pace) + 1)
    )
    for time, negative_lap, key in observations:
        lap = -negative_lap
        leading = lap > max(row.completed_laps for row in ledger.states.values())
        ledger.observe_crossing(key, lap, time, is_leader=leading)
    if retired:
        ledger.retire("B20", now)
    rivals = []
    for index, (key, pace) in enumerate(paces.items()):
        if key == "A" or ledger.states[key].retired:
            continue
        row = ledger.states[key]
        service = off_track and key == "B0"
        rivals.append(ChronologicalFinishCar(
            key, row.completed_laps, pace,
            now + 20. if service else row.last_crossing_time + pace,
            None if service else row.last_crossing_time, False, index, 7. if service else 0.,
        ))
    order = ("A", *(row.identifier for row in rivals if row.running_start is not None))
    return ObservedChronologicalField(ChronologicalFinishContext(
        "A", ledger, order, tuple(rivals), own_pace, 1., False, control_intervals=0,
    ), now)


@pytest.mark.parametrize("now", [1000., 7050.])
@pytest.mark.parametrize("own_pace", [80., 140.])
@pytest.mark.parametrize("scheduled", [150, 1_000_000])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("retired", [False, True])
@pytest.mark.parametrize("horizon", [2, 100])
def test_compact_clock_matches_full_grid_crossings(
    monkeypatch, now, own_pace, scheduled, off_track, retired, horizon,
):
    field = green_field(now, own_pace, scheduled, off_track, retired)
    before = signature(field)
    expected = green_weather_forecast(field, horizon, 200.)

    def unexpected_crossing(*args, **kwargs):
        pytest.fail("native green clocks must not schedule every follower crossing")

    monkeypatch.setattr(ObservedChronologicalField, "cross", unexpected_crossing)
    assert green_weather_forecast(field, horizon, 200., native=True) == expected
    assert signature(field) == before


@pytest.mark.parametrize("control", ["sc", "vsc"])
@pytest.mark.parametrize("intervals", [2, 6])
@pytest.mark.parametrize("now", [1000., 7050.])
@pytest.mark.parametrize("own_pace", [80., 100., 140.])
@pytest.mark.parametrize("off_track", [False, True])
@pytest.mark.parametrize("retired", [False, True])
def test_native_prefix_matches_full_grid_ledger_and_physical_order(
    control, intervals, now, own_pace, off_track, retired,
):
    root = green_field(now, own_pace, 150, off_track, retired)
    root.safety_car = control == "sc"
    root.modifier = 1.4 if root.safety_car else 1.2
    root.intervals_left = intervals
    before = signature(root)
    reference, native = root.fork(), root.fork()
    native._native_crossings = True
    for branch in (reference, native):
        offset = 0
        while not branch.finished:
            stopped = offset in {1, 3}
            branch.enter(20. if stopped else None)
            branch.cross(own_pace, 2. if stopped else 0.)
            offset += 1
    assert signature(native) == signature(reference)
    assert signature(root) == before


@pytest.mark.parametrize("now", [1000., 7050.])
def test_unrecorded_native_prefix_preserves_crossings_and_rebased_clocks(now):
    root = green_field(now, 140., 150, True, True)
    root.safety_car, root.modifier, root.intervals_left = True, 1.4, 6
    reference, native = root.fork(), root.fork()
    native._native_crossings, native._record_events = True, False
    while reference.projection_required and not reference.finished:
        for branch in (reference, native):
            branch.enter(20.)
            result = branch.cross(140., 2.)
            assert result.identifier == "A"
        assert native.events == []
        actual = list(signature(native))
        actual[7] = reference.events
        assert tuple(actual) == signature(reference)
    if not native.finished:
        assert green_weather_forecast(native, 3, 200., native=True) == green_weather_forecast(
            reference, 3, 200.)
        assert green_weather_forecast(native, 3, 200.) == green_weather_forecast(reference, 3, 200.)


def test_clock_key_keeps_every_reachable_paid_entry_surface_and_update_cap():
    first = StrategyWeatherClock((0., 100., 200.), 60., 100., 3, 10., 10.,
                                 update_offsets=(60., 160., 260.))
    equivalent = StrategyWeatherClock((0., 102., 201.), 62., 100., 3, 10., 10.,
                                      update_offsets=(62., 162., 262.))
    assert first != equivalent
    assert _green_weather_clock_key(first, {}) == _green_weather_clock_key(equivalent, {})
    for offset in range(3):
        for paid in range(offset + 2):
            for stopped_first in (False, True) if paid else (False,):
                assert first.updates(offset, paid, stopped_first) == equivalent.updates(
                    offset, paid, stopped_first)
    unreachable = replace(equivalent, update_offsets=(62., 130., 262.))
    assert first.updates(1, 3) != unreachable.updates(1, 3)
    assert _green_weather_clock_key(first, {}) == _green_weather_clock_key(unreachable, {})
    changed = replace(equivalent, update_offsets=(62., 120., 262.))
    assert first.updates(1, 2) != changed.updates(1, 2)
    assert _green_weather_clock_key(first, {}) != _green_weather_clock_key(changed, {})
    assert _green_weather_clock_key(first, {"wet": 1.}) is first
    long = replace(first, lap_start_offsets=tuple(i * 100. for i in range(101)))
    assert _green_weather_clock_key(long, {}) is long


@pytest.mark.parametrize("horizon", [13, 30, 53, 100])
def test_full_distance_clock_key_matches_every_physical_entry(horizon):
    offsets = tuple(100. * i for i in range(horizon))
    events = tuple(60. + 100. * i for i in range(horizon))
    first = StrategyWeatherClock(offsets, 60., 100., len(events), 10., 10.,
                                 update_offsets=events)
    equivalent = replace(
        first, lap_start_offsets=(0., *(value + 2. for value in offsets[1:])),
        first_update_after=62., update_offsets=tuple(value + 2. for value in events),
    )
    assert first != equivalent
    assert _green_weather_clock_key(first, {}) == _green_weather_clock_key(equivalent, {})
    for offset in range(horizon):
        # Eligibility observes the count before a fit; running observes it
        # after that fit. This includes every one-fit-per-entry schedule.
        for paid in range(offset + 2):
            for stopped_first in (False, True) if paid else (False,):
                assert first.updates(offset, paid, stopped_first) == equivalent.updates(
                    offset, paid, stopped_first)
    assert first.updates(horizon - 1, horizon) == first.max_updates
    assert first.updates(horizon - 1, 0) < first.max_updates
    # Move a reachable event just past an exact service exit. The keys must
    # retain this difference even though most of the observation table agrees.
    changed_events = list(equivalent.update_offsets)
    changed_events[6] += .25
    changed = replace(equivalent, update_offsets=tuple(changed_events))
    assert first.updates(6, 6) != changed.updates(6, 6)
    assert _green_weather_clock_key(first, {}) != _green_weather_clock_key(changed, {})


@pytest.mark.parametrize("shape", ["regular", "unequal_service", "first_running", "subclass"])
@pytest.mark.parametrize("horizon", [3, 53])
def test_extended_clock_key_keeps_dispatch_for_other_clock_shapes(shape, horizon):
    first = StrategyWeatherClock(
        tuple(100. * i for i in range(horizon)), 60., 100., horizon,
        10., 10., update_offsets=tuple(60. + 100. * i for i in range(horizon)),
    )
    if shape == "regular":
        clock = replace(first, update_offsets=None)
    elif shape == "unequal_service":
        clock = replace(first, current_stop_delay=20.)
    elif shape == "first_running":
        clock = replace(first, current_running_times=(100., 105.))
    else:
        class CustomClock(StrategyWeatherClock):
            def updates(self, *args, **kwargs):
                return super().updates(*args, **kwargs) + 1

        clock = CustomClock(
            first.lap_start_offsets, first.first_update_after, first.update_interval,
            first.max_updates, first.current_stop_delay, first.future_stop_delay,
            update_offsets=first.update_offsets)
    assert _green_weather_clock_key(clock, {}) is clock


@pytest.mark.parametrize("owner", ["timeline", "clock", "state"])
def test_extra_ledger_state_keeps_full_projection(monkeypatch, owner):
    field = green_field(1000., 140., 150, True, False)
    expected = green_weather_forecast(field, 3, 200.)
    target = (field.timeline if owner == "timeline" else field.timeline._clock
              if owner == "clock" else field.timeline.states["B0"])
    object.__setattr__(target, "notes", ["extension"])
    seen = []
    cross = ObservedChronologicalField.cross

    def observed(self, *args, **kwargs):
        seen.append(self.now)
        return cross(self, *args, **kwargs)

    monkeypatch.setattr(ObservedChronologicalField, "cross", observed)
    assert green_weather_forecast(field, 3, 200., native=True) == expected
    assert seen
    assert target.notes == ["extension"]


def test_native_clock_keeps_entry_phase_validation():
    field = green_field(1000., 140., 150, True, False)
    field.enter()
    before = signature(field)
    with pytest.raises(ValueError, match="cannot enter"):
        green_weather_forecast(field, 3, 200., native=True)
    assert signature(field) == before


def test_clock_subclass_keeps_its_projection_dispatch():
    calls = []

    class CustomClock(RaceFinishClock):
        def observe_leader_crossing(self, *args, **kwargs):
            calls.append(args)
            return super().observe_leader_crossing(*args, **kwargs)

    field = green_field(1000., 140., 150, True, False)
    clock = CustomClock(150)
    clock.__dict__.update(deepcopy(field.timeline._clock.__dict__))
    field.timeline._clock = clock
    before = signature(field)
    assert green_weather_forecast(field, 3, 200., native=True) == green_weather_forecast(
        field, 3, 200.)
    assert calls
    assert signature(field) == before


@pytest.mark.parametrize("owner,name", [
    (RaceFinishTimeline, "observe_crossing"),
    (RaceFinishTimeline, "_commit_crossing"),
    (RaceFinishTimeline, "_active_state"),
    (RaceFinishClock, "observe_leader_crossing"),
    (DriverFinishState, "__init__"),
    (_ProjectedLap, "__init__"),
    (ObservedFieldCrossing, "__init__"),
])
def test_replaced_ledger_dispatch_disables_native_forecasts(monkeypatch, owner, name):
    assert _native.native_physics()
    original = getattr(owner, name)

    def observed(*args, **kwargs):
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, observed)
    assert not _native.native_physics()
