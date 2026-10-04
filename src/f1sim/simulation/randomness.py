"""Versioned random-stream policies for reproducible race trials."""

import hashlib
from collections.abc import Callable

import numpy as np

RNG_POLICIES = (
    "shared_v1",
    "isolated_weather_v1",
    "isolated_weather_mechanical_v1",
    "isolated_race_v1",
)
DEFAULT_RNG_POLICY = "isolated_weather_v1"
MECHANICAL_NAMESPACE = 0x4D454348

MechanicalRngFactory = Callable[[str, int], np.random.Generator]
DriverRngFactory = Callable[[str, str], np.random.Generator]

_DRIVER_NAMESPACE = 0x44525652
_DRIVER_PURPOSES = {
    "qualifying_lap": 1,
    "race_lap": 2,
    "pit_service": 3,
    "opening_choice": 4,
    "strategy": 5,
    "replacement_choice": 6,
    "overtake_attempt": 7,
    "collision_loss": 8,
    "race_events": 9,
}


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
    """Return stable mechanical streams for policies that isolate those checks.

    The factory deliberately creates a fresh generator for every request.  A
    driver's UTF-8 ID is hashed into the complete eight-word SHA-256 digest,
    preserving a fixed little-endian ``SeedSequence`` layout without a cache.
    Legacy policies return ``None`` so callers retain their existing generator.
    """
    if validate_rng_policy(policy) not in ("isolated_weather_mechanical_v1", "isolated_race_v1"):
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


def driver_rng_factory_for_trial(seed: int, policy: str) -> DriverRngFactory | None:
    """Own independent mutable streams for each native driver/purpose pair.

    Creation order never assigns streams. A repeated lookup returns the same
    generator at its current position; callers consume it only during actual
    sampling. The empty driver ID is reserved for the field-level event stream.
    Each factory belongs to one trial and retains no model objects or global
    state. Mechanical checks retain their separate driver/own-lap namespace.
    """
    if validate_rng_policy(policy) != "isolated_race_v1":
        return None
    streams = {}

    def factory(driver_id: str, purpose: str) -> np.random.Generator:
        if (not isinstance(driver_id, str)
                or (not driver_id and purpose != "race_events")):
            raise ValueError("driver stream requires an ID; only race_events uses an empty ID")
        if not isinstance(purpose, str) or purpose not in _DRIVER_PURPOSES:
            raise ValueError("unknown driver random-stream purpose")
        key = (driver_id, purpose)
        if key not in streams:
            digest = hashlib.sha256(driver_id.encode("utf-8")).digest()
            words = tuple(int.from_bytes(digest[offset:offset + 4], "little", signed=False)
                          for offset in range(0, len(digest), 4))
            streams[key] = np.random.default_rng(np.random.SeedSequence(
                seed, spawn_key=(_DRIVER_NAMESPACE, _DRIVER_PURPOSES[purpose], *words),
            ))
        return streams[key]

    return factory
