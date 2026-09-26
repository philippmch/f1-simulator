"""Evaluate saved normalized tyre-timing reports without network access."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from f1sim.analysis.relative_tyre_wear import evaluate_relative_tyre_wear

_MAX_INPUT_BYTES = 100_000_000


def _read_reports(paths: list[str]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    total_bytes = 0
    for raw_path in paths:
        path = Path(raw_path)
        size = path.stat().st_size
        total_bytes += size
        if total_bytes > _MAX_INPUT_BYTES:
            raise ValueError(f"Input files exceed the {_MAX_INPUT_BYTES}-byte read limit")
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(value, dict):
            reports.append(value)
        elif isinstance(value, list):
            reports.extend(value)
        else:
            raise ValueError(f"{path} must contain an archive report object or a flat list")
    return reports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        help=(
            "saved --include-laps archive JSON reports; a file may contain one report "
            "or a flat list"
        ),
    )
    args = parser.parse_args(argv)
    try:
        reports = _read_reports(args.reports)
        result = evaluate_relative_tyre_wear(reports)
        rendered = json.dumps(result, indent=2, allow_nan=False)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        parser.exit(1, f"Could not evaluate relative tyre wear: {exc}\n")
    sys.stdout.write(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
