"""Versioned random-stream policies for reproducible race trials."""

import numpy as np

RNG_POLICIES = ("shared_v1", "isolated_weather_v1")
DEFAULT_RNG_POLICY = "isolated_weather_v1"


def validate_rng_policy(value: object) -> str:
    """Reject unknown stream layouts instead of silently changing seeded races."""
    if not isinstance(value, str) or value not in RNG_POLICIES:
        raise ValueError(f"rng_policy must be one of: {', '.join(RNG_POLICIES)}")
    return value


def weather_rng_for_trial(
    seed: int, race_rng: np.random.Generator, rng_policy: str,
) -> np.random.Generator:
    """Keep legacy draws shared or derive weather from a stable WEAT namespace."""
    if validate_rng_policy(rng_policy) == "shared_v1":
        return race_rng
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0x57454154,)))
