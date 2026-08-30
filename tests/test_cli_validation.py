"""CLI validation must reject bad work before any live network fetch."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLI = PROJECT_ROOT / "examples" / "simulate_race.py"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--simulations", "0"], "must be a positive integer"),
        (["--seed", "-1"], "must be a non-negative integer"),
        (["--max-workers", "0"], "must be a positive integer"),
        (["--top-n", "0"], "must be a positive integer"),
        (["--simulations", "1001"], "must be at most 1000"),
        (["--max-workers", "17"], "must be at most 16"),
        (["--top-n", "23"], "must be at most 22"),
        (["--seed", "4294967296"], "must be at most 4294967295"),
        (["--scenarios", "dry,snow"], "Unknown scenario label"),
    ],
)
def test_invalid_cli_input_fails_before_live_fetch(arguments: list[str], message: str) -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI), *arguments],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert completed.returncode == 2
    assert message in completed.stderr
    assert "Fetching the current calendar" not in completed.stdout
