"""Score explicit saved pace maps offline; no fitting, fetching, or simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from f1sim.analysis.race_pace_diagnostics import score_race_pace_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", type=Path,
        help=("JSON object containing starting_grid, baseline_pace_percent, "
              "candidate_pace_percent, and observed_pace_percent; omit missing labels"),
    )
    parser.add_argument("--output", type=Path, help="Write the diagnostic JSON to this file")
    args = parser.parse_args()
    try:
        record = json.loads(args.input.read_text(encoding="utf-8"))
        fields = (
            "starting_grid", "baseline_pace_percent", "candidate_pace_percent",
            "observed_pace_percent",
        )
        if not isinstance(record, dict) or any(key not in record for key in fields):
            raise ValueError(f"Input must contain {', '.join(fields)}")
        diagnostic = score_race_pace_components(**{key: record[key] for key in fields})
        text = json.dumps(diagnostic, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
