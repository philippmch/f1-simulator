"""Structural bounds for the iterative transition dependency solver."""

from cProfile import Profile
from types import CodeType

from f1sim.models import Car, Driver, TireCompound, Track, Weather
from f1sim.models.tire import TIRE_COMPOUNDS
from f1sim.simulation import rain_strategy


def test_each_uncached_suffix_scans_its_running_row_once():
    rain_strategy._transition_plan.cache_clear()
    rain_strategy._reset_transition_cache_after_fork()
    rain_strategy._running_row.cache_clear()
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
    completed = len(rain_strategy._transition_suffixes)
    assert completed > 50  # Exercise a branching dependency graph.
    stats = profile.getstats()

    def callcount(code):
        return sum(entry.callcount for entry in stats if entry.code is code)

    # Count writes by the native function's exact code object: every solved
    # suffix is inserted once, with no duplicate completion or eviction.
    assert callcount(rain_strategy._remember_transition.__code__) == completed
    row_cache = rain_strategy._running_row.cache_info()
    rows = row_cache.hits + row_cache.misses
    # One row per solved suffix, one retained stint, and first-lap physics
    # adjustments for retaining and each independently observed-safe fresh fit.
    surface = Weather(track_wetness=.1, rain_intensity=.1)
    candidates = sum(surface.tire_mismatch(compound) != "critical" for compound in TireCompound)
    assert rows == completed + 2 + candidates
    assert completed <= rain_strategy._TRANSITION_SUFFIX_LIMIT
    transition_code = rain_strategy._transition_plan.__wrapped__.__code__
    invariants = {code.co_name: code for code in transition_code.co_consts
                  if isinstance(code, CodeType) and code.co_name in ("legal", "reduced")}
    # Legality and reduced allowances are stint invariants, independent of
    # the number of future laps and fresh candidates examined by each stint.
    assert callcount(invariants["legal"]) <= completed + 2 + candidates
    assert callcount(invariants["reduced"]) <= 2 * completed + 2 + 2 * candidates
