"""Runtime provenance for interpreting saved simulation inputs."""

import hashlib
import platform
from functools import lru_cache
from pathlib import Path

import numpy as np
import pydantic

from f1sim import __version__


@lru_cache(maxsize=1)
def _simulation_source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for directory in ("analysis", "models", "simulation"):
        for path in sorted((root / directory).glob("*.py")):
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
