"""The fresh-stock relaxation keeps exact costs without recursive call growth."""

import sys
from functools import lru_cache
from math import inf, nextafter

import pytest

from f1sim.cancellation import SimulationCancelled, cancellation_scope
from f1sim.simulation.inventory_strategy import _fresh_inventory_completion_bound


def recursive_reference(horizon, compounds, running, eligible, advance, canonical, stop,
                        *, fitted_running=None, fitted_advance=None):
    """Retain the original small-horizon calculation as a rounding oracle."""
    def service(offset, clock):
        return 0. if offset == horizon else fitted(offset, canonical(offset, clock))

    @lru_cache(maxsize=None)
    def fitted(offset, clock):
        best = inf
        after = advance(clock)
        fitted_clock = after if fitted_advance is None else fitted_advance(after)
        for compound in compounds:
            if not eligible(offset, compound, clock):
                continue
            total = nextafter(stop, -inf)
            for number in range(offset, horizon):
                active = after if number == offset else fitted_clock
                if number > offset and not eligible(number, compound, active):
                    break
                evaluate = (fitted_running if number == offset and fitted_running is not None
                            else running)
                total = nextafter(total + evaluate(number, compound, number - offset, active), -inf)
                best = min(best, nextafter(total + service(number + 1, fitted_clock), -inf))
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
@pytest.mark.parametrize("fitting", [False, True, "delay"])
def test_stint_bound_retains_original_rounding_eligibility_and_paid_clock(
    horizon, expiry, clocked, fitting,
):
    compounds = ("slick", "rain")

    def running(offset, compound, age, clock):
        surface = (offset + clock) % 4 if clocked else offset % 4
        return 11.125 + age * .03125 + (surface * .7 if compound == "slick" else .13)

    def eligible(offset, compound, clock):
        return compound == "rain" or (offset + clock) % 4 < 3

    arguments = (horizon, compounds, running, eligible,
                 lambda clock: clock + int(clocked), lambda offset, clock: min(clock, 3), 2.1)
    fees = ({"fitted_running": lambda offset, compound, age, clock:
             running(offset, compound, age, clock) + (.3 if compound == "slick" else .7)}
            if fitting else {})
    if fitting == "delay":
        fees["fitted_advance"] = lambda clock: clock + 3
    expected = recursive_reference(*arguments, **fees)
    actual = _fresh_inventory_completion_bound(*arguments, **fees)
    for offset in range(horizon):
        for compound in compounds:
            for clock in (0, 2, 4):
                for age in (0, 2):
                    assert actual(offset, compound, age, clock, expiry) == expected(
                        offset, compound, age, clock, expiry)


def test_ready_age_zero_set_pays_no_fitting_fee_but_an_expired_set_must_pay():
    bound = _fresh_inventory_completion_bound(
        1, ("slick",), lambda *args: 3., lambda *args: True,
        lambda clock: clock, lambda offset, clock: clock, 1.,
        fitted_running=lambda *args: 10.)
    assert bound(0, "slick", 0, None, -1) == pytest.approx(3.)
    assert bound(0, "slick", 0, None, 0) == pytest.approx(11.)


def test_fitting_delay_changes_later_entries_and_preserves_the_first_outlap():
    def running(offset, compound, age, clock):
        return 10. + clock

    bound = _fresh_inventory_completion_bound(
        2, ("slick",), running, lambda *args: True,
        lambda clock: clock + 1, lambda offset, clock: clock, 2.,
        fitted_running=lambda offset, compound, age, clock:
            running(offset, compound, age, clock) + 7.,
        fitted_advance=lambda clock: clock + 10)
    # Service costs 2 + 11 + 7; the next lap sees the accumulated delay and
    # costs 21. Keeping a ready set sees neither a fitting fee nor its delay.
    assert bound(0, "slick", 0, 0, 0) == pytest.approx(41.)
    assert bound(0, "slick", 0, 0, -1) == pytest.approx(20.)


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


@pytest.mark.parametrize("horizon", [1, 4, 7])
@pytest.mark.parametrize("limit", [0, 1, 10, 1000])
def test_limited_service_work_stays_optimistic_for_every_remaining_stint(horizon, limit):
    arguments = (
        horizon, ("even", "odd"), lambda *args: 3.,
        lambda offset, compound, clock: compound == ("odd" if offset % 2 else "even"),
        lambda clock: clock, lambda offset, clock: clock, 1.)
    expected = recursive_reference(*arguments)
    actual = _fresh_inventory_completion_bound(*arguments, max_work=limit)
    for offset in range(horizon):
        compound = "odd" if offset % 2 else "even"
        for expiry in (0, -1):
            value = actual(offset, compound, 0, None, expiry)
            exact = expected(offset, compound, 0, None, expiry)
            assert 0. <= value <= exact
            if limit == 1000:
                assert value == exact


def test_unfinished_service_queries_keep_only_completed_reusable_costs():
    arguments = (
        4, ("even", "odd"), lambda *args: 3.,
        lambda offset, compound, clock: compound == ("odd" if offset % 2 else "even"),
        lambda clock: clock, lambda offset, clock: clock, 1.)
    expected = recursive_reference(*arguments)
    solved = {}
    limited = _fresh_inventory_completion_bound(*arguments, solved=solved, max_work=6)
    assert limited(0, "even", 0, None, 0) == 0.
    assert (0, None) not in solved and solved
    assert all(type(cost) is float for cost in solved.values())
    # Completed descendants remain valid even after the computing budget ends,
    # and another continuation can reuse them without consuming any more work.
    expected_tail = expected(1, "odd", 0, None, 0)
    assert limited(1, "odd", 0, None, 0) == expected_tail
    reused = _fresh_inventory_completion_bound(*arguments, solved=solved, max_work=0)
    before = dict(solved)
    assert reused(1, "odd", 0, None, 0) == expected_tail
    assert reused(0, "even", 0, None, 0) == 0.
    assert solved == before


def test_exhausted_service_budget_does_not_mask_cancellation():
    bound = _fresh_inventory_completion_bound(
        4, ("slick",), lambda *args: 3., lambda *args: True,
        lambda clock: clock, lambda offset, clock: clock, 1., max_work=0)
    with cancellation_scope(lambda: True), pytest.raises(SimulationCancelled):
        bound(0, "slick", 0, None, 0)
