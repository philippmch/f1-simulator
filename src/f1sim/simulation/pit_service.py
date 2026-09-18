"""Deterministic forecasts for an observed pit-stop service.

The execution model used by :class:`~f1sim.simulation.lap.LapSimulator` is

``S = max(1.8, X + B * U)``

where ``X`` is normal with the car's mean and standard deviation, ``B`` is a
Bernoulli variable with probability .05, and ``U`` is uniform on ``[2, 8]``.
``expected_remaining_service`` returns ``E[S - elapsed | S > elapsed]``.  An
observation before the 1.8 second floor does not select a tail, so the helper
uses the ordinary service mean there.  At and after the floor, the atom at the
floor is excluded by the strict ``S > elapsed`` condition.

Only deterministic arithmetic is used.  The normal and uniform-shifted normal
tail integrals have closed forms.  They are evaluated in log space, with an
endpoint asymptotic used only after ordinary double precision can no longer
represent the normal tail's logarithm accurately.
"""

from __future__ import annotations

from math import erfc, exp, expm1, isfinite, log, log1p, nextafter, pi, sqrt
from typing import Any

from .pit_strategy import expected_stationary_time

_FLOOR = 1.8
_SLOW_PROBABILITY = 0.05
_NORMAL_PROBABILITY = 1.0 - _SLOW_PROBABILITY
_SLOW_LOW = 2.0
_SLOW_HIGH = 8.0
_SLOW_WIDTH = _SLOW_HIGH - _SLOW_LOW
_SQRT_TWO = sqrt(2.0)
_LOG_SQRT_TWO_PI = 0.5 * log(2.0 * pi)
_LOG_NORMAL_PROBABILITY = log(_NORMAL_PROBABILITY)
_LOG_SLOW_PROBABILITY = log(_SLOW_PROBABILITY)
_MIN_POSITIVE = nextafter(0.0, 1.0)


def expected_remaining_service(car: Any, elapsed: float) -> float:
    """Return the conditional mean service time still outstanding.

    ``elapsed`` is the already observed stationary service in seconds.  For
    observations below the hard 1.8 second floor, every execution is still in
    progress and the result is exactly ``expected_stationary_time(car) -
    elapsed``.  At or above the floor, only executions with ``S > elapsed``
    are considered; in particular, the probability mass clipped to 1.8 is
    deliberately removed at ``elapsed == 1.8``.

    The public :class:`~f1sim.models.car.Car` model requires a positive normal
    standard deviation.  A zero standard deviation is also handled for small
    model-only fixtures using its literal deterministic-mixture semantics; a
    ``ValueError`` is raised when that distribution has no surviving service
    at the observed time.  Invalid or non-finite car parameters and elapsed
    values raise ``ValueError``.
    """

    time = _validated_elapsed(elapsed)
    try:
        mean = float(car.pit_stop_avg)
        std = float(car.pit_stop_std)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("car must provide finite pit_stop_avg and pit_stop_std") from exc
    if not isfinite(mean) or not isfinite(std) or std < 0.0:
        raise ValueError("car pit-stop mean and standard deviation must be finite, std >= 0")

    if time < _FLOOR:
        return _positive_finite(expected_stationary_time(car) - time)
    if std == 0.0:
        return _deterministic_remaining(mean, time)
    return _positive_finite(_conditioned_positive_tail(mean, std, time))


def _validated_elapsed(elapsed: float) -> float:
    try:
        time = float(elapsed)
    except (TypeError, ValueError) as exc:
        raise ValueError("elapsed must be a finite non-negative number") from exc
    if not isfinite(time) or time < 0.0:
        raise ValueError("elapsed must be a finite non-negative number")
    return time


def _positive_finite(value: float) -> float:
    """Keep the public result representable and strictly positive."""

    if isfinite(value) and value > 0.0:
        return value
    if value == 0.0:
        return _MIN_POSITIVE
    raise ValueError("pit-stop conditional mean is not finite")


def _deterministic_remaining(mean: float, elapsed: float) -> float:
    """Conditional service for the optional ``std == 0`` model fixture."""

    # The non-slow branch is a point mass after clipping to the floor.
    ordinary_service = max(_FLOOR, mean)
    tail_probability = _NORMAL_PROBABILITY if ordinary_service > elapsed else 0.0
    tail_numerator = _NORMAL_PROBABILITY * (ordinary_service - elapsed) \
        if ordinary_service > elapsed else 0.0

    # For the slow branch, the floor cannot contribute once elapsed >= floor;
    # integrate the surviving part of mean + U directly.
    lower = max(_SLOW_LOW, elapsed - mean)
    if lower < _SLOW_HIGH:
        width = _SLOW_HIGH - lower
        tail_probability += _SLOW_PROBABILITY * width / _SLOW_WIDTH
        start_excess = mean + lower - elapsed
        tail_numerator += _SLOW_PROBABILITY * (
            start_excess * width + width * width / 2.0
        ) / _SLOW_WIDTH

    if tail_probability <= 0.0:
        raise ValueError("conditional pit-stop service is impossible at elapsed")
    return _positive_finite(tail_numerator / tail_probability)


def _conditioned_positive_tail(mean: float, std: float, elapsed: float) -> float:
    """Condition the two continuous raw branches on exceeding ``elapsed``."""

    # In this regime the 1.8-second atom has no contribution.  The ordinary
    # branch is X, and the slow branch is X+U.  For a very negative upper
    # endpoint, endpoint asymptotics avoid subtracting logs around -1e300 and
    # still retain a positive finite mean residual.
    shifted_upper = (mean + _SLOW_HIGH - elapsed) / std
    if shifted_upper < -10_000.0:
        return _slow_endpoint_asymptotic(mean, std, elapsed)

    z = (elapsed - mean) / std
    log_normal_tail = _log_normal_survival(z)
    log_normal_numerator = log(std) + _log_normal_excess(z)

    shifted_tail, shifted_numerator = _shifted_normal_logs(
        mean, std, elapsed
    )
    log_denominator = _logaddexp(
        _LOG_NORMAL_PROBABILITY + log_normal_tail,
        _LOG_SLOW_PROBABILITY + shifted_tail,
    )
    log_numerator = _logaddexp(
        _LOG_NORMAL_PROBABILITY + log_normal_numerator,
        _LOG_SLOW_PROBABILITY + shifted_numerator,
    )

    if not isfinite(log_denominator) or not isfinite(log_numerator):
        return _slow_endpoint_asymptotic(mean, std, elapsed)
    log_result = log_numerator - log_denominator
    if not isfinite(log_result):
        return _slow_endpoint_asymptotic(mean, std, elapsed)
    if log_result < log(_MIN_POSITIVE):
        return _MIN_POSITIVE
    return exp(log_result)


def _shifted_normal_logs(mean: float, std: float, elapsed: float) -> tuple[float, float]:
    """Log tail and log excess numerator for ``X + Uniform(2, 8)``.

    Put ``x = (mean + u - elapsed) / std``.  The relevant antiderivatives are

    ``F(x) = x Phi(x) + phi(x)`` and
    ``G(x) = ((x^2 + 1) Phi(x) + x phi(x)) / 2``.

    Thus the tail is ``std * (F(x_high)-F(x_low)) / 6`` and its excess
    numerator is ``std^2 * (G(x_high)-G(x_low)) / 6``.
    """

    x_low = (mean + _SLOW_LOW - elapsed) / std
    x_high = (mean + _SLOW_HIGH - elapsed) / std
    log_f_difference = _log_difference(
        _log_primitive_f(x_high), _log_primitive_f(x_low)
    )
    log_g_difference = _log_difference(
        _log_primitive_g(x_high), _log_primitive_g(x_low)
    )
    log_width = log(_SLOW_WIDTH)
    return (
        log(std) - log_width + log_f_difference,
        2.0 * log(std) - log_width + log_g_difference,
    )


def _slow_endpoint_asymptotic(mean: float, std: float, elapsed: float) -> float:
    """Stable limit for a slow-branch tail beyond ordinary log precision."""

    distance = elapsed - mean - _SLOW_HIGH
    if distance <= 0.0:
        # This branch is only expected for unusual subnormal inputs.  The
        # ordinary normal residual remains a valid positive fallback.
        z = (elapsed - mean) / std
        log_q = _log_normal_survival(z)
        log_h = _log_normal_excess(z)
        if isfinite(log_q) and isfinite(log_h):
            return _positive_finite(std * exp(log_h - log_q))
        return _MIN_POSITIVE

    y = distance / std
    if isfinite(y) and y > 8.0:
        # J(y)/H(y) is the shifted-normal endpoint residual in standardized
        # units.  The endpoint dominates uniformly once y is this large.
        log_ratio = log(std) + _log_j_ratio(y) - _log_h_ratio(y)
        if isfinite(log_ratio):
            if log_ratio < log(_MIN_POSITIVE):
                return _MIN_POSITIVE
            return exp(log_ratio)

    value = std * std / distance
    return _positive_finite(value)


def _log_normal_survival(z: float) -> float:
    """Log Q(z), using Mills' expansion after ``erfc`` loses the tail."""

    if z <= 8.0:
        q = 0.5 * erfc(z / _SQRT_TWO)
        if q > 0.0:
            return log(q)
    if not isfinite(z):
        return -float("inf")
    return _log_standard_normal_density(z) + _log_mills_ratio(z)


def _log_normal_excess(z: float) -> float:
    """Log of ``phi(z) - z Q(z)`` (the standardized excess numerator)."""

    if z <= 8.0:
        q = 0.5 * erfc(z / _SQRT_TWO)
        phi = exp(_log_standard_normal_density(z))
        value = phi - z * q
        if value > 0.0 and isfinite(value):
            return log(value)
    if z > 0.0 and isfinite(z):
        return _log_standard_normal_density(z) + _log_h_ratio(z)
    # For a finite negative z the direct branch above is stable.  This is
    # reachable only when a subnormal input has made the density unusable.
    return _log_standard_normal_density(z)


def _log_standard_normal_density(z: float) -> float:
    if not isfinite(z):
        return -float("inf")
    return -0.5 * z * z - _LOG_SQRT_TWO_PI


def _log_mills_ratio(z: float) -> float:
    """Log of Q(z)/phi(z) for positive z."""

    return -log(z) + log(_asymptotic_h_series(z, power=0))


def _log_h_ratio(z: float) -> float:
    """Log of (phi(z)-zQ(z))/phi(z) for positive z."""

    return -2.0 * log(z) + log(_asymptotic_h_series(z, power=1))


def _log_j_ratio(z: float) -> float:
    """Log of G(-z)/phi(z), where G is the shifted-tail primitive."""

    inverse_square = _inverse_square(z)
    term = 1.0
    total = 1.0
    last_abs = abs(term)
    for k in range(100):
        term *= -((2.0 * k + 3.0) * (2.0 * k + 4.0)) \
            / ((2.0 * k + 2.0)) * inverse_square
        if abs(term) > last_abs or total + term <= 0.0:
            break
        total += term
        last_abs = abs(term)
        if abs(term) <= abs(total) * 1.0e-16:
            break
    return -3.0 * log(z) + log(total)


def _asymptotic_h_series(z: float, *, power: int) -> float:
    """Return a positive optimally truncated Mills-related series.

    ``power=0`` gives ``1 - 1/z^2 + 3/z^4 - ...`` and ``power=1`` gives
    ``1 - 3/z^2 + 15/z^4 - ...``.
    """

    inverse_square = _inverse_square(z)
    term = 1.0
    total = 1.0
    last_abs = abs(term)
    for k in range(100):
        multiplier = (2.0 * k + (1.0 if power == 0 else 3.0))
        term *= -multiplier * inverse_square
        if abs(term) > last_abs or total + term <= 0.0:
            break
        total += term
        last_abs = abs(term)
        if abs(term) <= abs(total) * 1.0e-16:
            break
    return total


def _inverse_square(value: float) -> float:
    if not isfinite(value) or value == 0.0:
        return 0.0
    square = value * value
    if not isfinite(square):
        return 0.0
    return 1.0 / square


def _log_primitive_f(x: float) -> float:
    """Log of ``F(x) = x Phi(x) + phi(x)``."""

    if x < -8.0:
        return _log_standard_normal_density(-x) + _log_h_ratio(-x)
    if not isfinite(x):
        return float("inf") if x > 0.0 else -float("inf")
    if x > 1.0e150:
        return log(x)
    phi = exp(_log_standard_normal_density(x))
    cdf = 0.5 * erfc(-x / _SQRT_TWO)
    value = x * cdf + phi
    return log(value)


def _log_primitive_g(x: float) -> float:
    """Log of ``G(x) = ((x^2 + 1) Phi(x) + x phi(x)) / 2``."""

    if x < -8.0:
        return _log_standard_normal_density(-x) + _log_j_ratio(-x)
    if not isfinite(x):
        return float("inf") if x > 0.0 else -float("inf")
    if x > 1.0e150:
        return 2.0 * log(x) - log(2.0)
    phi = exp(_log_standard_normal_density(x))
    cdf = 0.5 * erfc(-x / _SQRT_TWO)
    value = ((x * x + 1.0) * cdf + x * phi) / 2.0
    return log(value)


def _log_difference(log_high: float, log_low: float) -> float:
    """Log of exp(log_high) - exp(log_low), stably for close values."""

    if log_high < log_low:
        log_high, log_low = log_low, log_high
    if log_high == -float("inf"):
        return log_high
    if log_low == -float("inf"):
        return log_high
    delta = log_low - log_high
    if delta >= 0.0:
        return -float("inf") if delta == 0.0 else log_high
    return log_high + log(-expm1(delta))


def _logaddexp(left: float, right: float) -> float:
    if left == -float("inf"):
        return right
    if right == -float("inf"):
        return left
    high = max(left, right)
    return high + log1p(exp(min(left, right) - high))
