"""Runtime provenance for interpreting saved simulation inputs."""

import hashlib
import json
import platform
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydantic

from f1sim import __version__

_RUNTIME_FIELDS = (
    "f1sim",
    "python",
    "numpy",
    "pydantic",
    "simulation_source_sha256",
)
_DIGEST_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")


@lru_cache(maxsize=1)
def _simulation_source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    paths = [root / "cancellation.py"]
    paths.extend(
        path
        for directory in ("analysis", "models", "simulation")
        for path in (root / directory).glob("*.py")
    )
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8") + b"\0")
        # Normalize checkout line endings across Windows and Unix.
        digest.update(path.read_text(encoding="utf-8").encode("utf-8") + b"\0")
    return digest.hexdigest()


def simulation_runtime() -> dict[str, str]:
    """Return independent metadata; source/version equality is not a calibration claim."""
    return {
        "f1sim": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pydantic": pydantic.__version__,
        "simulation_source_sha256": _simulation_source_digest(),
    }


def compare_saved_runtime(saved_runtime: object) -> tuple[str, tuple[str, ...]]:
    """Compare saved provenance with the installed runtime without exposing values.

    Returns ``("match", ())``, ``("mismatch", fields)``, or
    ``("unavailable", ())``. Provenance is informational and does not promise
    exact reproducibility.
    """
    if not isinstance(saved_runtime, dict):
        return "unavailable", ()

    normalized: dict[str, str] = {}
    for name in _RUNTIME_FIELDS:
        value = saved_runtime.get(name)
        if type(value) is not str or not value:
            return "unavailable", ()
        if name == "simulation_source_sha256":
            if _DIGEST_PATTERN.fullmatch(value) is None:
                return "unavailable", ()
            value = value.lower()
        elif len(value) > 128 or any(character.isspace() for character in value):
            return "unavailable", ()
        normalized[name] = value

    installed = simulation_runtime()
    differing_fields = tuple(
        name for name in _RUNTIME_FIELDS
        if normalized[name] != (
            installed[name].lower()
            if name == "simulation_source_sha256" else installed[name]
        )
    )
    return ("mismatch", differing_fields) if differing_fields else ("match", ())


def saved_runtime_status(
    path: str | Path, scenario: str | None = None,
) -> tuple[str, tuple[str, ...]]:
    """Report whether one saved scenario records this installed runtime.

    The same scenario-selection rules as offline replay apply. File, JSON,
    selection, and missing-provenance errors are represented as unavailable;
    replay/comparison validation remains responsible for reporting bad inputs.
    """
    from f1sim.analysis.replay import _select_saved_scenario

    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        selected, _ = _select_saved_scenario(saved, scenario)
    except (OSError, UnicodeError, ValueError, TypeError):
        return "unavailable", ()
    inputs = selected.get("simulation_inputs")
    if not isinstance(inputs, dict):
        return "unavailable", ()
    return compare_saved_runtime(inputs.get("runtime"))


def format_saved_runtime_status(status: tuple[str, tuple[str, ...]]) -> str:
    """Format a safe, concise installed-versus-saved provenance line."""
    state, fields = status
    if state not in ("match", "mismatch", "unavailable"):
        state, fields = "unavailable", ()
    summary = f"Runtime provenance (installed vs saved): {state}"
    if state == "mismatch":
        known_fields = tuple(name for name in _RUNTIME_FIELDS if name in fields)
        if known_fields:
            summary += f" ({', '.join(known_fields)})"
    return summary
