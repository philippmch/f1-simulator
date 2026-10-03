"""Leading finish bounds agree with independent authoritative crossing ledgers."""

from copy import deepcopy
from itertools import product

import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.finish_strategy import (
    LeadingFinishContext,
    ReplacementOption,
    RivalFinishForecast,
    evaluate_finish_protection,
)
from f1sim.simulation.race_timing import RaceFinishClock, RaceFinishTimeline
from f1sim.simulation.weather_schedule import WeatherForecastContext


class BoundPhysics:
    """The paid alternative runs exactly at the model's future floor."""

    def __init__(self):
        self.calls = []

    def calculate_lap_time(self, driver, car, track, tire, weather, lap, total, **options):
        self.calls.append((lap, total, weather.model_copy(deep=True), dict(options)))
        return 99. if tire.compound == TireCompound.MEDIUM else 95.


def models(maximum=90):
    return (Driver(id="A", name="A", team_id="A"), Car(team_id="A", team_name="A"),
            Track(id="timed", name="timed", country="test", total_laps=maximum,
                  base_lap_time=100., pit_lane_delta=10.))


def history(now, rivals, maximum):
    """Replay legal past events without copying the projection's arithmetic."""
    records = [("A", 71, now)]
    records += [(rival.identifier, rival.completed_laps,
                 min(now - 1., rival.next_crossing_time - rival.running_pace))
                for rival in rivals]
    timeline = RaceFinishTimeline(maximum, [row[0] for row in records])
    events = [(last * lap / count, -lap, driver)
              for driver, count, last in records for lap in range(1, count + 1)]
    for time, negative_lap, driver in sorted(events):
        active = max(state.completed_laps for state in timeline.states.values()
                     if not state.retired and state.finish_time is None)
        timeline.observe_crossing(driver, -negative_lap, time, is_leader=-negative_lap > active)
    return timeline


def execute(timeline, now, rivals, maximum, fee, pace, lockstep, *, modifier=1.,
            first_crossing=None):
    """Execute all future mean crossings through the real finish controller."""
    first = now + fee + modifier * pace if first_crossing is None else first_crossing
    candidate = {lap: first + (lap - 72) * pace for lap in range(72, maximum + 1)}
    if lockstep:
        # Standard execution advances everyone once per leading lap and feeds
        # that lap's earliest free crossing to its authoritative shared clock.
        clock = RaceFinishClock(maximum)
        for lap in range(1, 72):
            clock.observe_leader_crossing(lap, now * lap / 71.)
        # The past signal can belong to an earlier rival crossing.
        if not timeline.time_limit_announced and clock.time_limit_announced:
            clock = RaceFinishClock(maximum)
            for lap in range(1, 72):
                clock.observe_leader_crossing(lap, (now - 1.) * lap / 71.)
        for lap in range(72, maximum + 1):
            times = [candidate[lap]]
            for rival in rivals:
                if rival.completed_laps == 71:
                    times.append(rival.next_crossing_time + (lap - 72) * rival.running_pace)
            clock.observe_leader_crossing(lap, min(times))
            if clock.winner_time is not None:
                return lap - 71, candidate[lap]
        raise AssertionError("missing lockstep flag")
    events = [(time, -lap, "A") for lap, time in candidate.items()]
    events += [(rival.next_crossing_time + (lap - rival.completed_laps - 1) * rival.running_pace,
                -lap, rival.identifier)
               for rival in rivals for lap in range(rival.completed_laps + 1, maximum + 1)]
    for time, negative_lap, driver in sorted(events):
        if not timeline.can_start_next_lap(driver):
            continue
        active = max(state.completed_laps for state in timeline.states.values()
                     if not state.retired and state.finish_time is None)
        timeline.observe_crossing(driver, -negative_lap, time,
                                  is_leader=(timeline.chequered_time is None
                                             and -negative_lap > active))
        state = timeline.states["A"]
        if state.finish_time is not None:
            return state.completed_laps - 71, state.finish_time
    raise AssertionError("missing individual flag")


RIVALS = (
    (), ((71, 98., 100.),), ((71, 101., 100.),), ((71, 10., 96.),),
    ((71, 98., 100.), (71, 200., 98.)), ((65, 10., 120.),),
)


CASES = tuple(row for row in product(
    (False, True), (7098., 7100., 7200.), (0., 10., 90., 500.), (75, 90), RIVALS,
) if not row[0] or all(rival[0] == 71 for rival in row[4]))


@pytest.mark.parametrize("lockstep,now,fee,maximum,rows", CASES)
def test_leading_finish_bound_matches_executed_crossing_controller(
    lockstep, now, fee, maximum, rows,
):
    rivals = tuple(RivalFinishForecast(completed, now + offset, pace, str(index))
                   for index, (completed, offset, pace) in enumerate(rows))
    past = history(now, rivals, maximum)
    context = LeadingFinishContext(7200., past.time_limit_announced, rivals, lockstep=lockstep)
    retained = execute(deepcopy(past), now, rivals, maximum, 0., 99., lockstep)
    stopped = execute(deepcopy(past), now, rivals, maximum, fee, 95., lockstep)
    driver, car, track = models(maximum)
    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, 72, Weather(),
        now, None, maximum, lap_simulator=BoundPhysics(), expected_lane_loss=fee,
        replacements=(ReplacementOption(TireCompound.SOFT),), leading_finish_context=context,
    )
    assert result.stop_laps == stopped[0]
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    if result.retained_laps is not None:
        assert result.retained_laps == retained[0]
        assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.veto == (retained[0] > stopped[0])


def test_a_rival_crossing_before_expiry_preserves_the_extra_lap():
    driver, car, track = models()
    arguments = (driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8,
                 72, Weather(), 7100., None, 90)
    options = dict(lap_simulator=BoundPhysics(), expected_lane_loss=10.,
                   replacements=(ReplacementOption(TireCompound.SOFT),))
    alone = evaluate_finish_protection(*arguments, **options,
                                       leading_finish_context=LeadingFinishContext(7200.))
    rival = evaluate_finish_protection(
        *arguments, **options, leading_finish_context=LeadingFinishContext(
            7200., rivals=(RivalFinishForecast(71, 7198., 100., "B"),)),
    )
    assert alone.retained_laps == 3 and alone.stop_laps == 2 and alone.veto
    assert rival.retained_laps == rival.stop_laps == 3 and not rival.veto


@pytest.mark.parametrize("now,fee", [(7050., 0.), (7100., 10.), (7100., 100.), (7200., 500.)])
@pytest.mark.parametrize("modifier,rows", [
    (1.2, ()), (1.4, ()), (1.2, ((71, 98., 100.),)), (1.2, ((71, 130., 96.),)),
])
def test_controlled_bound_matches_the_authoritative_shared_finish_clock(now, fee, modifier, rows):
    rivals = tuple(RivalFinishForecast(count, now + offset, pace, str(index))
                   for index, (count, offset, pace) in enumerate(rows))
    maximum = 90
    past = history(now, rivals, maximum)
    context = LeadingFinishContext(7200., past.time_limit_announced, rivals, lockstep=True)
    retained = execute(deepcopy(past), now, rivals, maximum, 0., 99., True, modifier=modifier)
    stopped = execute(deepcopy(past), now, rivals, maximum, fee, 95., True, modifier=modifier)
    driver, car, track = models(maximum)
    physics = BoundPhysics()
    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, 72, Weather(),
        now, None, maximum, lap_simulator=physics, expected_lane_loss=fee,
        replacements=(ReplacementOption(TireCompound.SOFT),), leading_finish_context=context,
        current_lap_time_modifier=modifier, active_aero_enabled=False,
    )
    assert result.stop_laps == stopped[0]
    assert result.stop_crossing_time == pytest.approx(stopped[1], abs=1.e-8)
    if result.retained_laps is not None:
        assert result.retained_laps == retained[0]
        assert result.retained_crossing_time == pytest.approx(retained[1], abs=1.e-8)
    assert result.veto == (retained[0] > stopped[0])
    assert all(not options["active_aero_enabled"] for lap, _, _, options in physics.calls
               if lap == 72)
    assert all(options["active_aero_enabled"] for lap, _, _, options in physics.calls
               if lap > 72)


@pytest.mark.parametrize("lockstep", [False, True])
def test_surface_updates_follow_the_engine_cadence_and_exclude_the_flag(lockstep):
    driver, car, track = models()
    physics = BoundPhysics()
    weather = Weather(track_wetness=.5, rain_intensity=.5)
    schedule = WeatherForecastContext.from_schedule([dict(lap=73, rain_intensity=.7)],
                                                    leading_lap=72)
    context = LeadingFinishContext(7200., rivals=(RivalFinishForecast(71, 7110., 100., "B"),),
                                   forecast_context=schedule, lockstep=lockstep)
    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.INTERMEDIATE], 8, 72, weather,
        7100., None, 90, lap_simulator=physics, expected_lane_loss=250.,
        replacements=(ReplacementOption(TireCompound.WET),), leading_finish_context=context,
    )
    outlap = physics.calls[0][2]
    expected = weather.model_copy(deep=True)
    if not lockstep:
        # Rival crosses at 7110 and 7210, then receives the flag at 7310.
        # Two updates occur before the 7350 pit exit; the flag adds none.
        for updates in range(2):
            expected = schedule.advanced(updates).project_next(expected)
    assert outlap == expected
    assert result.stop_feasible


def test_leading_forecast_hooks_cannot_mutate_live_inputs_or_shared_tires(monkeypatch):
    driver, car, track = models()
    weather = Weather()
    for compound, tire in TIRE_COMPOUNDS.items():
        monkeypatch.setitem(TIRE_COMPOUNDS, compound, tire.model_copy(deep=True))
    before = deepcopy((driver, car, track, weather, TIRE_COMPOUNDS))

    class MutatingPhysics(BoundPhysics):
        def calculate_lap_time(self, driver, car, track, tire, weather, *args, **kwargs):
            value = super().calculate_lap_time(driver, car, track, tire, weather, *args, **kwargs)
            driver.total_race_time = 9999.
            car.pit_stop_avg = 9.
            track.pit_lane_delta = 999.
            tire.initial_grip = .8
            weather.track_temperature = 45.
            return value

    evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, 72, weather,
        7100., None, 90, lap_simulator=MutatingPhysics(), expected_lane_loss=10.,
        replacements=(ReplacementOption(TireCompound.SOFT),),
        leading_finish_context=LeadingFinishContext(7200.),
    )
    assert (driver, car, track, weather, TIRE_COMPOUNDS) == before


def test_identity_keyed_physics_without_copied_driver_observations_cannot_veto():
    driver, car, track = models()
    observed_pace = {id(driver): 100.}

    class ObservedPhysics:
        def calculate_lap_time(self, driver, *args, **kwargs):
            return observed_pace[id(driver)]

    result = evaluate_finish_protection(
        driver, car, track, TIRE_COMPOUNDS[TireCompound.MEDIUM], 8, 72, Weather(),
        7100., None, 90, lap_simulator=ObservedPhysics(), expected_lane_loss=10.,
        replacements=(ReplacementOption(TireCompound.SOFT),),
        leading_finish_context=LeadingFinishContext(7200.),
    )
    assert not result.veto and not result.retained_feasible and not result.stop_feasible
    assert observed_pace == {id(driver): 100.}
