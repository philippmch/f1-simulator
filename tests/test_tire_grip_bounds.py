"""Degradation cannot manufacture grip or a negative wear penalty."""

import pytest

from f1sim.models.tire import Tire, TireCompound


@pytest.mark.parametrize("initial", [0.0, 0.1, 0.3, 0.5, 0.9, 1.2])
def test_valid_tyre_grip_stays_bounded_and_wear_never_improves_pace(initial):
    tire = Tire(compound=TireCompound.INTERMEDIATE, initial_grip=initial)
    for management in (0.0, 0.8, 1.0):
        grips = [tire.grip_at_lap(age, management) for age in range(101)]
        assert grips[0] == initial
        assert all(0 <= grip <= initial for grip in grips)
        assert all(later <= earlier for earlier, later in zip(grips, grips[1:]))
        assert all(tire.time_penalty_per_lap(age, 90, management) >= 0 for age in range(101))


def test_low_grip_fresh_set_does_not_receive_a_wear_speed_bonus():
    tire = Tire(compound=TireCompound.INTERMEDIATE, initial_grip=0.3)
    assert tire.grip_at_lap(0) == 0.3
    assert tire.time_penalty_per_lap(0, 90) == 0.0
