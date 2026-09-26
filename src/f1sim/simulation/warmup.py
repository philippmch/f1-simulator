"""Validation and small helpers for post-fit first-lap running costs."""

from collections.abc import Mapping
from math import isfinite
from numbers import Real

from f1sim.models import TireCompound

TIRE_WARMUP_POLICY = "post_fit_first_lap_v1"
MAX_TIRE_WARMUP_SECONDS = 60.0

_COMPOUNDS = frozenset(compound.value for compound in TireCompound)


def validate_tire_warmup(value):
    """Normalize an optional compound-to-seconds profile.

    Only positive entries are retained, so ``None``, an empty mapping and an
    all-zero mapping all select the exact legacy behavior.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("tire_warmup must be a mapping or None")
    normalized = {}
    for compound, seconds in value.items():
        if type(compound) is not str or compound not in _COMPOUNDS:
            raise ValueError(f"tire_warmup has unknown compound: {compound!r}")
        if isinstance(seconds, bool) or not isinstance(seconds, Real):
            raise ValueError("tire_warmup values must be finite real seconds")
        try:
            cost = float(seconds)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError("tire_warmup values must be finite real seconds") from exc
        if not isfinite(cost) or not 0.0 <= cost <= MAX_TIRE_WARMUP_SECONDS:
            raise ValueError("tire_warmup values must be between zero and 60 seconds")
        if cost > 0.0:
            normalized[compound] = cost
    return normalized


def parse_tire_warmup_spec(spec: str) -> dict[str, float]:
    """Parse ``compound=seconds`` pairs for command-line entry points."""
    if not isinstance(spec, str):
        raise ValueError("tire warm-up specification must be text")
    if not spec.strip():
        return {}
    values = {}
    for part in spec.split(","):
        pair = part.strip().split("=")
        if len(pair) != 2 or not pair[0].strip() or not pair[1].strip():
            raise ValueError("tire warm-up entries must be compound=seconds pairs")
        compound, raw_seconds = pair[0].strip(), pair[1].strip()
        if compound in values:
            raise ValueError(f"duplicate tire warm-up compound: {compound}")
        try:
            seconds = float(raw_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("tire warm-up values must be numeric seconds") from exc
        values[compound] = seconds
    return validate_tire_warmup(values)


def tire_warmup_seconds(profile, compound) -> float:
    """Return the normalized fitting cost for an enum or canonical compound."""
    key = getattr(compound, "value", compound)
    return profile.get(key, 0.0)
