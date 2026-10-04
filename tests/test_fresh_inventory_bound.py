"""The fresh-stock relaxation keeps exact costs without recursive call growth."""

import sys
from functools import lru_cache
from math import inf, nextafter

import pytest

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.simulation.inventory_strategy import _fresh_inventory_completion_bound


def recursive_reference(horizon, compounds, running, eligible, advance, canonical, stop):
    """Retain the original small-horizon calculation as a rounding oracle."""
    def service(offset, clock):
        return 0. if offset == horizon else fitted(offset, canonical(offset, clock))

    @lru_cache(maxsize=None)
    def fitted(offset, clock):
        best = inf
        after = advance(clock)
        for compound in compounds:
            if not eligible(offset, compound, clock):
                continue
            total = nextafter(stop, -inf)
            for number in range(offset, horizon):
                if number > offset and not eligible(number, compound, after):
                    break
                total = nextafter(total + running(number, compound, number - offset, after), -inf)
                best = min(best, nextafter(total + service(number + 1, after), -inf))
        return best

    def bound(offset, compound, age, clock, expiry):
        best, total = service(offset, clock), 0.
        for number in range(offset, horizon):
            current_age = age + number - offset
            if (expiry >= 0 and current_age >= expiry
                    or not eligible(number, compound, clock)):
                break
            total = nextafter(total + running(number, compound, current_age, clock), -inf)
            best = min(best, nextafter(total + service(number + 1, clock), -inf))
        return best

    return bound


@pytest.mark.parametrize("horizon", [1, 4, 7])
@pytest.mark.parametrize("expiry", [-1, 0, 3])
@pytest.mark.parametrize("clocked", [False, True])
def test_stint_bound_retains_original_rounding_eligibility_and_paid_clock(horizon, expiry, clocked):
    compounds = ("slick", "rain")

    def running(offset, compound, age, clock):
        surface = (offset + clock) % 4 if clocked else offset % 4
        return 11.125 + age * .03125 + (surface * .7 if compound == "slick" else .13)

    def eligible(offset, compound, clock):
        return compound == "rain" or (offset + clock) % 4 < 3

    arguments = (horizon, compounds, running, eligible,
                 lambda clock: clock + int(clocked), lambda offset, clock: min(clock, 3), 2.1)
    expected = recursive_reference(*arguments)
    actual = _fresh_inventory_completion_bound(*arguments)
    for offset in range(horizon):
        for compound in compounds:
            for clock in (0, 2, 4):
                assert actual(offset, compound, 2, clock, expiry) == expected(
                    offset, compound, 2, clock, expiry)


def test_thousands_of_compulsory_stints_do_not_grow_the_call_stack():
    depth = []

    def running(*args):
        frame, count = sys._getframe(), 0
        while frame is not None:
            count += 1
            frame = frame.f_back
        depth.append(count)
        return 3.

    bound = _fresh_inventory_completion_bound(
        1200, ("even", "odd"), running,
        lambda offset, compound, clock: compound == ("odd" if offset % 2 else "even"),
        lambda clock: clock, lambda offset, clock: clock, 1.)
    # Each surface forces the next fresh compound; retaining the initial set
    # saves exactly one entry. This path previously exhausted recursion limits.
    assert bound(0, "even", 0, None, -1) == pytest.approx(4799., abs=1.e-8)
    assert max(depth) - min(depth) <= 2


def test_deep_relaxation_remains_cancellable():
    visits = 0

    def cancelled():
        nonlocal visits
        visits += 1
        return visits > 30

    bound = _fresh_inventory_completion_bound(
        1200, ("even", "odd"), lambda *args: 3.,
        lambda offset, compound, clock: compound == ("odd" if offset % 2 else "even"),
        lambda clock: clock, lambda offset, clock: clock, 1.)
    with cancellation_scope(cancelled), pytest.raises(SimulationCancelled):
        bound(0, "even", 0, None, -1)
