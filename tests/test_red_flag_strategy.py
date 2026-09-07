"""Free suspension tyres use remaining distance and leave paid stops intact."""

from copy import deepcopy
from itertools import product
from math import inf

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation.events import EventType, RaceEvent
from f1sim.simulation.lap import LapSimulator
from f1sim.simulation.pit_strategy import expected_stationary_time
from f1sim.simulation.race import DriverRaceState, DriverStatus, RaceSimulator

SLICKS = (TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD)


def fixture(laps=30, stress=0.7):
    state = DriverRaceState(Driver(id="A", name="A", team_id="A"),
                            Car(team_id="A", team_name="A"), 1,
                            current_tire=TIRE_COMPOUNDS[TireCompound.MEDIUM].model_copy(deep=True))
    track = Track(id="test", name="Test", country="Test", total_laps=laps,
                  base_lap_time=90, tire_stress=stress)
    return state, track


def exhaustive_cost(state, track, candidate, laps, paid_budget, wet):
    """Enumerate every lap-by-lap keep/change action, including repeated sets."""
    best = inf
    initial_used = set(state.tire_compound_history) | {candidate.value}
    if not wet and len(initial_used) < 2:
        paid_budget = max(paid_budget, 1)
    # None means keep; a compound means buy a fresh set, even if repeated.
    for actions in product((None, *SLICKS), repeat=laps - 1):
        if sum(action is not None for action in actions) > paid_budget:
            continue
        age, cost, compound = 0, 0.0, candidate
        used = initial_used.copy()
        for action in (None, *actions):
            if action is not None:
                compound, age = action, 0
                used.add(compound.value)
                cost += track.pit_lane_delta + expected_stationary_time(state.car)
            cost += LapSimulator.tire_pace_contribution(
                state.driver, state.car, track, TIRE_COMPOUNDS[compound], age)
            age += 1
        if wet or len(used) >= 2:
            best = min(best, cost)
    return best


@pytest.mark.parametrize("remaining", [1, 2, 5])
@pytest.mark.parametrize("paid_used,wet", [(0, False), (1, False), (1, True),
                                          (2, False), (3, False), (3, True)])
def test_choice_matches_exhaustive_remaining_race(remaining, paid_used, wet):
    state, track = fixture(stress=1.0)
    state.pit_stops = paid_used
    if wet:
        state.tire_compound_history.insert(0, "intermediate")
    costs = {c: exhaustive_cost(state, track, c, remaining, 3 - paid_used, wet)
             for c in SLICKS}
    choice = RaceSimulator()._choose_red_flag_tire(state, Weather(), track,
                                                  track.total_laps - remaining)
    assert costs[choice] == pytest.approx(min(costs.values()))


def test_horizon_and_wear_change_free_set_choice():
    sim = RaceSimulator()
    state, track = fixture(60, 1.0)
    state.pit_stops = 3
    state.tire_compound_history = ["soft", "medium"]
    assert sim._choose_red_flag_tire(state, Weather(), track, 59) == TireCompound.SOFT
    assert sim._choose_red_flag_tire(state, Weather(), track, 10) == TireCompound.HARD


def test_free_change_can_repeat_compound_and_consumes_no_rng():
    sim = RaceSimulator(np.random.default_rng(42))
    state, track = fixture()
    state.current_tire = TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(deep=True)
    state.tire_compound_history = ["medium", "soft"]
    state.tire_laps = 10
    before = deepcopy(sim.rng.bit_generator.state)
    sim._handle_red_flag_stop([state], Weather(), track, 29)
    assert state.tire_compound_history == ["medium", "soft", "soft"]
    assert state.tire_laps == state.driver.current_tire_laps == 0
    assert sim.rng.bit_generator.state == before


def test_free_change_preserves_paid_stops_incident_clock_and_retired_car():
    sim = RaceSimulator()
    state, track = fixture()
    state.pit_stops, state.pit_laps = 1, [8]
    state.total_time = 999.0
    state.force_pit_next_lap = True
    retired = deepcopy(state)
    retired.status = DriverStatus.DNF
    before = deepcopy(retired)
    sim._handle_red_flag_stop([state, retired], Weather(), track, 20)
    assert (state.pit_stops, state.pit_laps, state.total_time) == (1, [8], 999.0)
    assert not state.force_pit_next_lap
    assert retired == before


def test_final_lap_suspension_does_not_invent_a_stint():
    sim = RaceSimulator()
    state, track = fixture()
    state.tire_laps = 12
    state.force_pit_next_lap = True
    before = deepcopy(state)
    sim.event_manager.red_flag_active = True
    sim._handle_red_flag_stop([state], Weather(track_wetness=0.9), track, track.total_laps)
    assert state == before
    assert not sim.event_manager.red_flag_active


@pytest.mark.parametrize("stop_lap", [2, 3])
def test_full_race_free_set_runs_only_after_suspension_and_repairs_puncture(monkeypatch, stop_lap):
    sim = RaceSimulator()
    state, track = fixture(3)
    observed = []
    monkeypatch.setattr(sim, "_should_pit", lambda *a, **kw: False)
    monkeypatch.setattr(sim, "_process_overtakes", lambda *a, **kw: 0)
    def timing(**kw):
        observed.append((kw["lap_number"], kw["driver"].current_tire_laps))
        return 90.0
    monkeypatch.setattr(sim.lap_simulator, "calculate_lap_time", timing)
    def events(lap, **kw):
        if lap == stop_lap:
            return [RaceEvent(EventType.PUNCTURE, lap, ["A"], time_loss_seconds=7,
                              forces_pit_stop=True), RaceEvent(EventType.RED_FLAG, lap)]
        return []
    monkeypatch.setattr(sim.event_manager, "process_lap", events)
    result = sim.simulate_race([state.driver], {"A": state.car}, track,
                               Weather(change_probability=0), ["A"],
                               starting_tires={"A": TireCompound.MEDIUM})[0]
    assert result.pit_stops == 0
    assert result.total_time == pytest.approx(277)
    assert len(result.strategy) == (2 if stop_lap == 2 else 1)
    assert observed == [(1, 0), (2, 1), (3, 0 if stop_lap == 2 else 2)]
