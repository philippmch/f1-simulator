"""Physical schedules bound the search relaxation, including nonmonotone pace."""

from itertools import product
from math import fsum, inf
from random import Random

import pytest

from f1sim.simulation.inventory_strategy import _conserved_wear_lower_bounds


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("scale", [1e-9, 1., 1e12])
def test_bound_never_exceeds_any_reachable_physical_suffix(seed, scale):
    rng = Random(seed)
    horizon = 5
    initial = (("soft", 18), ("soft", 18), ("intermediate", 34))
    critical = {compound: tuple(rng.random() < .15 for _ in range(horizon))
                for compound, _ in initial}
    # Arbitrary age and calendar-lap costs include a clipping floor and
    # improving older ages; the relaxation must not assume monotone wear.
    costs = {(lap, compound, age + elapsed): scale * max(60., rng.uniform(40., 130.))
             for compound, age in initial for elapsed in range(horizon)
             for lap in range(horizon)}
    bound = _conserved_wear_lower_bounds(
        horizon, initial, critical, lambda *key: costs[key],
    )
    actual = [inf] * horizon
    for schedule in product(range(len(initial)), repeat=horizon):
        ages = [age for _, age in initial]
        laps = []
        safe = []
        for lap, target in enumerate(schedule):
            compound = initial[target][0]
            safe.append(not critical[compound][lap])
            laps.append(costs[lap, compound, ages[target]])
            ages[target] += 1
        for offset in range(1, horizon):
            if all(safe[offset:]):
                actual[offset] = min(actual[offset], fsum(laps[offset:]))
    assert bound[horizon] == 0.
    for offset in range(1, horizon):
        assert bound[offset] <= actual[offset]


def test_reusable_sets_have_distinct_age_uses_and_keep_duplicate_multiplicity():
    horizon = 4
    critical = {"soft": (False,) * horizon}
    single = _conserved_wear_lower_bounds(
        horizon, (("soft", 0),), critical, lambda offset, compound, age: 100. + age,
    )
    duplicate = _conserved_wear_lower_bounds(
        horizon, (("soft", 0), ("soft", 0)), critical,
        lambda offset, compound, age: 100. + age,
    )
    # Three future uses cannot all get one set's age-zero price. Two physical
    # sets do supply two separate age-zero uses in the optimistic relaxation.
    assert single[1] == pytest.approx(303., rel=0., abs=1e-10)
    assert duplicate[1] == pytest.approx(301., rel=0., abs=1e-10)
    assert single[1] > duplicate[1] > 300.


def test_one_unusable_future_surface_makes_the_entire_suffix_impossible():
    bound = _conserved_wear_lower_bounds(
        4, (("soft", 0),), {"soft": (False, False, True, False)},
        lambda *args: 100.,
    )
    assert bound[1] == bound[2] == inf
    assert bound[3] <= 100.


def test_near_equal_costs_keep_bound_below_rounded_schedule_totals():
    horizon = 12
    initial = (("soft", 0), ("hard", 4))
    critical = {compound: (False,) * horizon for compound, _ in initial}

    def running(offset, compound, age):
        return 90. + (offset * 3 + age + (compound == "hard")) * 1e-14

    bound = _conserved_wear_lower_bounds(horizon, initial, critical, running)
    for schedule in product(range(2), repeat=horizon):
        ages = [age for _, age in initial]
        laps = []
        for lap, target in enumerate(schedule):
            compound = initial[target][0]
            laps.append(running(lap, compound, ages[target]))
            ages[target] += 1
        # Match the planner's reverse floating-point summation rather than
        # assuming exact real arithmetic when a branch is almost tied.
        total = 0.
        for offset in range(horizon - 1, 0, -1):
            total = laps[offset] + total
            assert bound[offset] <= total
