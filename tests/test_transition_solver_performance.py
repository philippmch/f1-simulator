"""Structural bounds for the iterative transition dependency solver."""

from cProfile import Profile

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import rain_strategy


def test_each_uncached_suffix_scans_its_running_row_once(monkeypatch):
    rain_strategy._transition_plan.cache_clear()
    rain_strategy._reset_transition_cache_after_fork()
    rows = []
    completed = []
    running_row = rain_strategy._running_row
    remember = rain_strategy._remember_transition

    def count_row(*args, **kwargs):
        rows.append(args)
        return running_row(*args, **kwargs)

    def count_completed(key, value):
        completed.append(key)
        remember(key, value)

    monkeypatch.setattr(rain_strategy, "_running_row", count_row)
    monkeypatch.setattr(rain_strategy, "_remember_transition", count_completed)
    profile = Profile()
    result = profile.runcall(rain_strategy.plan_rain_transition,
        Driver(id="A", name="A", team_id="T"),
        Car(team_id="T", team_name="T"),
        Track(id="t", name="T", country="T", total_laps=20,
              base_lap_time=90, pit_lane_delta=5),
        Weather(track_wetness=.1, rain_intensity=.1),
        TIRE_COMPOUNDS[TireCompound.SOFT].model_copy(deep=True),
        5, 1, 3, used_compounds=set(),
    )
    assert result is not None
    assert len(completed) > 50  # Exercise a branching dependency graph.
    assert len(completed) == len(set(completed))
    # One row per solved suffix, one retained stint, and first-lap physics
    # adjustments for retaining and the three possible fresh slick choices.
    assert len(rows) == len(completed) + 5
    assert len(rain_strategy._transition_suffixes) <= rain_strategy._TRANSITION_SUFFIX_LIMIT
    calls = {entry.code.co_name: entry.callcount for entry in profile.getstats()
             if hasattr(entry.code, "co_name")}
    # Legality and reduced allowances are stint invariants, independent of
    # the number of future laps and fresh candidates examined by each stint.
    assert calls["legal"] <= len(completed) + 5
    assert calls["reduced"] <= 2 * len(completed) + 8
