"""Add Monte Carlo score diagnostics to a saved evaluation, entirely offline."""

import argparse
import hashlib
import json
from pathlib import Path

from f1sim.analysis.race_probability_evaluation import rescore_saved_winner_evaluation

_MAX_REPORT_BYTES = 32 * 1024 * 1024


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="JSON from evaluate_race_probabilities.py")
    args = parser.parse_args(argv)
    try:
        with args.report.open("rb") as stream:
            raw = stream.read(_MAX_REPORT_BYTES + 1)
        if len(raw) > _MAX_REPORT_BYTES:
            raise ValueError("saved report exceeds 32 MiB")
        source = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys)
        result = rescore_saved_winner_evaluation(source)
        result["rescoring"]["source_report_sha256"] = hashlib.sha256(raw).hexdigest()
        result["rescoring"]["source_report_filename"] = args.report.name
        output = json.dumps(result, indent=2, allow_nan=False)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    print(output)


if __name__ == "__main__":
    main()
