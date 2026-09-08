"""In-race decisions anticipate the clock without declaring an early finish."""

import numpy as np
import pytest

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.simulation.chronological_race import ChronologicalRace, _PendingLap
from f1sim.simulation.pit_strategy import SLICKS, expected_stationary_time
from f1sim.simulation.race import (
    DriverRaceState,
    DriverStatus,
    RaceSimulator,
    TeamStrategyArchetype,
)
from f1sim.simulation.race_timing import RaceFinishTimeline


@pytest.mark.parametrize("pause,expected", [(0, 65), (600, 65), (4000, 62)])
def test_chronological_forecast_accounts_for_bounded_suspension_extension(pause, expected):
    simulator = RaceSimulator(np.random.default_rng(7))
    engine = ChronologicalRace(simulator)
    state = DriverRaceState(Driver(id="d", name="Driver", team_id="t"),
                            Car(team_id="t", team_name="Team"), 1,
                            total_time=5000 + pause, laps_completed=44)
    engine.states = {"d": state}
    engine.track = Track(id="t", name="Track", country="Test", total_laps=90,
                         base_lap_time=110)
    engine.timeline = RaceFinishTimeline(90, engine.states)
    if pause:
        engine.timeline.begin_suspension(5000)
        engine.timeline.end_suspension(5000 + pause)
    engine.running_paces = {"d": 110}
    engine.pending = {}
    planning = engine._planning_track(state, state.total_time)
    assert planning.total_laps == expected
    assert engine.timeline.final_lap == engine.track.total_laps == 90
    assert not engine.timeline.time_limit_announced


def test_announced_finish_stays_at_lapped_successors_next_crossing():
    engine = ChronologicalRace(RaceSimulator(np.random.default_rng(7)))
    engine.track = Track(id="t", name="Track", country="Test", total_laps=90,
                         base_lap_time=110)
    engine.states = {
        name: DriverRaceState(Driver(id=name, name=name, team_id="t"),
                              Car(team_id="t", team_name="Team"), position,
                              laps_completed=laps, total_time=time)
        for name, position, laps, time in [("old", 1, 64, 7200), ("new", 2, 60, 7180),
                                            ("follower", 3, 59, 7205)]
    }
    engine.states["old"].status = DriverStatus.DNF
    engine.timeline = RaceFinishTimeline(90, engine.states)
    for lap in range(1, 65):
        engine.timeline.observe_crossing("old", lap, lap * 112.5, is_leader=True)
    leader = engine.states["new"]
    engine.pending = {"new": _PendingLap(61, 7180, 7290, Weather(), False,
                                          leader.current_tire, 0, 110)}
    engine.running_paces = {"new": 110, "follower": 100}
    assert engine.timeline.time_limit_announced
    assert engine._planning_track(engine.states["follower"], 7205).total_laps == 60
    assert engine.timeline.final_lap == 65


@pytest.mark.parametrize("engine", ["standard", "chronological"])
def test_forced_replacement_matches_completed_timed_alternatives(monkeypatch, engine):
    def run(compound=None, *, legacy=False):
        driver = Driver(id="d", name="Driver", team_id="t")
        car = Car(team_id="t", team_name="Team")
        track = Track(id="t", name="Track", country="Test", total_laps=90,
                      base_lap_time=110, pit_lane_delta=22)
        simulator = RaceSimulator(np.random.default_rng(7))
        simulator._infer_team_strategy = lambda *args: TeamStrategyArchetype.BALANCED
        simulator.event_manager.process_lap = lambda *args, **kwargs: []
        simulator.event_manager._check_mechanical_failure = lambda *args, **kwargs: None
        simulator.event_manager._check_random_incident = lambda *args, **kwargs: None
        actual_lap = simulator.lap_simulator.calculate_lap_time
        fuel_distances = []

        def running(*args, **kwargs):
            fuel_distances.append(kwargs.get("total_laps", args[6] if args else None))
            return actual_lap(*args, **kwargs, sample_variation=False)

        simulator.lap_simulator.calculate_lap_time = running
        simulator.lap_simulator.calculate_pit_stop_time = expected_stationary_time
        original_pit = simulator._should_pit
        original_choice = simulator._choose_committed_dry_compound
        horizons = []

        def should_pit(state, states, planning, lap, *args, **kwargs):
            horizons.append((lap, planning.total_laps))
            if lap <= 45:
                return lap == 45
            return original_pit(state, states, planning, lap, *args, **kwargs)

        def choose(state, planning, lap, *args, **kwargs):
            if lap == 45 and compound is not None:
                return compound
            return original_choice(state, planning, lap, *args, **kwargs)

        simulator._should_pit = should_pit
        simulator._choose_committed_dry_compound = choose
        monkeypatch.setattr(Weather, "evolve", lambda self, rng: self.project_surface())
        execute = (simulator.simulate_race if engine == "standard"
                   else ChronologicalRace(simulator).run)
        with monkeypatch.context() as control:
            if legacy:
                module = ("f1sim.simulation.race" if engine == "standard"
                          else "f1sim.simulation.chronological_race")
                control.setattr(module + ".forecast_final_lap",
                                lambda scheduled, *args: scheduled)
            result, = execute([driver], {"t": car}, track, Weather(), ["d"],
                              starting_tires={"d": TireCompound.MEDIUM})
        assert set(fuel_distances) == {90}
        if legacy:
            assert dict(horizons)[45] == 90
        else:
            assert dict(horizons)[45] < 90
        assert result.race_time_limited
        return result

    selected = run()
    alternatives = {compound: run(compound) for compound in SLICKS}
    assert selected.strategy[1] == TireCompound.SOFT.value
    assert selected.laps_completed == 65
    best = min(alternatives.values(),
               key=lambda result: (-result.laps_completed, result.total_time))
    assert selected.laps_completed == best.laps_completed
    assert selected.total_time == pytest.approx(
        best.total_time, abs=1e-8,
    )
    assert selected.pit_laps == [45]
    previous = run(legacy=True)
    assert previous.laps_completed == selected.laps_completed
    assert previous.total_time - selected.total_time > 21.6
