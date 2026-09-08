#!/usr/bin/env python3
"""Replay one saved Monte Carlo trial offline with the installed simulator."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from f1sim.analysis.replay import replay_saved_simulation
from f1sim.output import ConsoleOutput, Exporter


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay saved simulation inputs offline, using installed code.",
        epilog="Exact reproduction requires matching model and dependency versions. "
        "Saved runtime provenance is informational.",
    )
    parser.add_argument("path", type=Path, help="Exported statistics JSON containing inputs")
    parser.add_argument("--simulation", type=int, default=1,
                        help="One-based trial number (default: 1)")
    parser.add_argument("--export", action="store_true", help="Export a new unique replay bundle")
    parser.add_argument("--scenario", help="Exact scenario key in a dashboard or comparison export")
    parser.add_argument("--output-dir", type=Path, default=Path("output/replays"))
    args = parser.parse_args()
    try:
        results = replay_saved_simulation(args.path, args.simulation, args.scenario)
        print(f"Saved scenario: {results.track_name} | model: {results.race_engine} | "
              f"simulation: {args.simulation} | effective seed: {results.seed}")
        ConsoleOutput.print_qualifying_results(results.qualifying_results[0])
        ConsoleOutput.print_race_results(results.race_results[0])
        if args.export:
            paths = Exporter(args.output_dir).export_all(results, prefix="replay")
            for kind, path in paths.items():
                print(f"{kind}: {path}")
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
