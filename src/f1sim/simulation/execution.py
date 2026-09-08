"""Shared public names for selecting race execution engines."""

RACE_ENGINES = ("standard", "chronological")


def validate_race_engine(value: str) -> str:
    """Reject unknown engines rather than silently changing execution semantics."""
    if not isinstance(value, str) or value not in RACE_ENGINES:
        raise ValueError(f"race_engine must be one of: {', '.join(RACE_ENGINES)}")
    return value
