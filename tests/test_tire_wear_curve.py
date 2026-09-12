"""Bounded grip cannot erase configured long-stint degradation or the cliff."""

import pytest

from f1sim.models.tire import TIRE_COMPOUNDS, Tire, TireCompound


@pytest.mark.parametrize("compound", list(TireCompound))
@pytest.mark.parametrize("management", [0.0, 0.8, 1.0])
def test_configured_pace_slopes_survive_grip_floor_and_cliff(compound, management):
    tire = TIRE_COMPOUNDS[compound]
    penalties = [tire.time_penalty_per_lap(age, 90, management) for age in range(121)]
    differences = [right - left for left, right in zip(penalties, penalties[1:])]
    normal = 90 * tire.degradation_rate * (2 - management) * .03
    assert penalties[0] == 0
    assert differences[:tire.cliff_threshold] == pytest.approx(
        [normal] * tire.cliff_threshold,
    )
    assert differences[tire.cliff_threshold:] == pytest.approx(
        [normal * tire.cliff_multiplier] * (120 - tire.cliff_threshold),
    )
    assert tire.grip_at_lap(100, management) == tire.grip_at_lap(120, management)
    assert penalties[120] > penalties[100]


@pytest.mark.parametrize("compound", list(TireCompound))
def test_pre_floor_wear_cost_keeps_existing_conversion(compound):
    tire = TIRE_COMPOUNDS[compound]
    for age in range(10):
        assert tire.grip_at_lap(age, .8) > .5
        legacy = 90 * (tire.initial_grip - tire.grip_at_lap(age, .8)) * .03
        assert tire.time_penalty_per_lap(age, 90, .8) == pytest.approx(legacy)


def test_custom_zero_wear_remains_zero_beyond_cliff():
    tire = Tire(compound=TireCompound.SOFT, degradation_rate=0, cliff_threshold=1)
    assert tire.time_penalty_per_lap(1000, 90, 0) == 0
    assert tire.grip_at_lap(1000, 0) == tire.initial_grip


@pytest.mark.parametrize("initial", [0, .1, .3, .5])
def test_low_initial_grip_does_not_suppress_wear_or_grant_fresh_bonus(initial):
    tire = Tire(compound=TireCompound.INTERMEDIATE, initial_grip=initial)
    assert tire.grip_at_lap(100) == initial
    assert tire.time_penalty_per_lap(0, 90) == 0
    assert tire.time_penalty_per_lap(100, 90) > tire.time_penalty_per_lap(50, 90) > 0
