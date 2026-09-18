"""Tests for deterministic conditional pit-service forecasts."""

from __future__ import annotations

from math import erfc, exp, isfinite, pi, sqrt
from types import SimpleNamespace

import numpy as np
import pytest

from f1sim.models import Car
from f1sim.simulation.pit_service import expected_remaining_service
from f1sim.simulation.pit_strategy import expected_stationary_time


def _car(mean: float = 2.5, std: float = 0.3) -> Car:
    return Car(team_id="test", team_name="Test", pit_stop_avg=mean, pit_stop_std=std)


def _independent_quadrature(car: Car, elapsed: float) -> float:
    """Numerically integrate the raw mixture without using helper primitives."""

    mean = car.pit_stop_avg
    std = car.pit_stop_std
    normal_z = (elapsed - mean) / std
    normal_tail = 0.5 * erfc(normal_z / sqrt(2.0))
    normal_density = exp(-0.5 * normal_z * normal_z) / sqrt(2.0 * pi)
    normal_numerator = (mean - elapsed) * normal_tail + std * normal_density

    nodes, weights = np.polynomial.legendre.leggauss(512)
    shifts = 5.0 + 3.0 * nodes
    shifted_z = (elapsed - mean - shifts) / std
    shifted_tail = np.fromiter(
        (0.5 * erfc(float(value) / sqrt(2.0)) for value in shifted_z),
        dtype=float,
    )
    shifted_density = np.exp(-0.5 * shifted_z * shifted_z) / sqrt(2.0 * pi)
    shifted_numerator = (mean + shifts - elapsed) * shifted_tail + std * shifted_density
    slow_tail = 0.5 * float(np.dot(weights, shifted_tail))
    slow_numerator = 0.5 * float(np.dot(weights, shifted_numerator))

    denominator = 0.95 * normal_tail + 0.05 * slow_tail
    numerator = 0.95 * normal_numerator + 0.05 * slow_numerator
    return numerator / denominator


@pytest.mark.parametrize(
    ("mean", "std", "elapsed"),
    [
        (2.2, 0.15, 1.8),
        (2.5, 0.30, 2.5),
        (3.7, 0.80, 3.2),
        (2.0, 1.00, 4.2),
    ],
)
def test_conditioned_service_matches_independent_quadrature(
    mean: float, std: float, elapsed: float
) -> None:
    car = _car(mean, std)

    assert expected_remaining_service(car, elapsed) == pytest.approx(
        _independent_quadrature(car, elapsed), rel=2.0e-9, abs=2.0e-11
    )


def test_observing_the_floor_excludes_the_floor_atom() -> None:
    car = _car()
    fresh_remaining = expected_stationary_time(car) - 1.8

    conditioned = expected_remaining_service(car, 1.8)

    assert conditioned > fresh_remaining
    assert conditioned == pytest.approx(_independent_quadrature(car, 1.8), rel=2.0e-9)


@pytest.mark.parametrize("elapsed", [20.0, 100.0, 1.0e6, 1.0e308])
def test_extreme_tail_is_positive_and_finite(elapsed: float) -> None:
    result = expected_remaining_service(_car(), elapsed)

    assert isfinite(result)
    assert result > 0.0


def test_before_floor_is_the_unconditioned_mean_minus_elapsed() -> None:
    car = _car(3.1, 0.6)

    for elapsed in (0.0, 0.7, 1.799999999):
        assert expected_remaining_service(car, elapsed) == pytest.approx(
            expected_stationary_time(car) - elapsed, rel=1.0e-14
        )


@pytest.mark.parametrize("elapsed", [-1.0, float("nan"), float("inf")])
def test_elapsed_must_be_finite_and_nonnegative(elapsed: float) -> None:
    with pytest.raises(ValueError, match="elapsed"):
        expected_remaining_service(_car(), elapsed)


def test_zero_standard_deviation_reports_impossible_survival() -> None:
    car = SimpleNamespace(pit_stop_avg=2.5, pit_stop_std=0.0)

    assert expected_remaining_service(car, 1.0) == pytest.approx(
        expected_stationary_time(car) - 1.0
    )
    assert expected_remaining_service(car, 2.0) == pytest.approx(
        expected_stationary_time(car) - 2.0
    )
    with pytest.raises(ValueError, match="impossible"):
        expected_remaining_service(car, 20.0)


def test_helper_does_not_consume_rng_or_mutate_car() -> None:
    car = _car()
    original = car.model_dump()
    rng = np.random.default_rng(1234)
    expected_next = rng.random(4)
    rng = np.random.default_rng(1234)

    expected_remaining_service(car, 2.7)

    assert np.array_equal(rng.random(4), expected_next)
    assert car.model_dump() == original
