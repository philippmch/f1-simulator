"""Versioned random-stream policies for reproducible race trials."""

import hashlib
from collections.abc import Callable

import numpy as np

RNG_POLICIES = (
    "shared_v1",
    "isolated_weather_v1",
    "isolated_weather_mechanical_v1",
)
DEFAULT_RNG_POLICY = "isolated_weather_v1"
MECHANICAL_NAMESPACE = 0x4D454348

MechanicalRngFactory = Callable[[str, int], np.random.Generator]


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


def mechanical_rng_factory_for_trial(
    seed: int, policy: str,
) -> MechanicalRngFactory | None:
    """Return stable mechanical streams for the opt-in mechanical policy.

    The factory deliberately creates a fresh generator for every request.  A
    driver's UTF-8 ID is hashed into the complete eight-word SHA-256 digest,
    preserving a fixed little-endian ``SeedSequence`` layout without a cache.
    Legacy policies return ``None`` so callers retain their existing generator.
    """
    if validate_rng_policy(policy) != "isolated_weather_mechanical_v1":
        return None

    def factory(driver_id: str, lap: int) -> np.random.Generator:
        digest = hashlib.sha256(driver_id.encode("utf-8")).digest()
        driver_words = tuple(
            int.from_bytes(digest[offset:offset + 4], "little", signed=False)
            for offset in range(0, len(digest), 4)
        )
        spawn_key = (MECHANICAL_NAMESPACE, *driver_words, int(lap))
        return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=spawn_key))

    return factory
