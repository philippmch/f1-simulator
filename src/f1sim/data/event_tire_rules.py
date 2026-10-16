"""Sourced event-specific race tyre constraints, independent of result feeds.

Only the listed season/event combinations apply. In particular, Monaco's 2025
three-set experiment was abandoned for 2026; it is not a circuit default.
"""

from __future__ import annotations

import copy
import re
import unicodedata
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

_MONACO_SOURCE = (
    "https://www.fia.com/system/files/documents/"
    "fia_2025_formula_1_sporting_regulations_-_issue_4_-_2025-02-26.pdf"
)
_QATAR_SOURCE = "https://press.pirelli.com/a-maximum-of-25-laps-per-tyre-set-in-qatar/"

_CIRCUITS = {
    "monaco": "monaco",
    "circuit_de_monaco": "monaco",
    "circuit_de_monte_carlo": "monaco",
    "monte_carlo": "monaco",
    "losail": "lusail",
    "lusail": "lusail",
    "losail_international_circuit": "lusail",
    "lusail_international_circuit": "lusail",
}

_RULES: dict[tuple[int, str], dict[str, dict[str, Any]]] = {
    (2025, "monaco"): {
        "minimum_sets": {
            "value": 3,
            "source_url": _MONACO_SOURCE,
            "source_title": "2025 FIA Formula One Sporting Regulations, Issue 4",
            "source_section": "Article 30.5(m), page 38",
            "published_at": "2025-02-26",
            "publication_precision": "day",
            "available_by": "2025-02-27T00:00:00Z",
            "scope": "2025 Monaco Grand Prix race; three sets of any tyre specification.",
            "limitations": [
                "A minimum set count, not three compounds or two mandatory pit-lane stops.",
                "Wet tyre use does not remove the minimum three-set requirement.",
                "Suspended races that cannot restart have separate penalty provisions.",
                "The source gives a publication date only; cutoff uses the following UTC day.",
                "This registry does not implement the full sporting regulations or sanctions.",
            ],
        },
    },
    (2025, "lusail"): {
        "max_set_laps": {
            "value": 25,
            "source_url": _QATAR_SOURCE,
            "source_title": "A maximum of 25 laps per tyre set in Qatar",
            "published_at": "2025-11-17T09:36:00Z",
            "publication_precision": "minute",
            "available_by": "2025-11-17T09:37:00Z",
            "scope": "2025 Qatar Grand Prix; a 25-lap cumulative tyre-set weekend limit.",
            "limitations": [
                "The race-only completed-lap cap is a necessary upper bound, not full compliance.",
                "Started but incomplete laps are not represented by the completed-race-lap model.",
                "Prior weekend use is not reconstructed; a fresh, unlimited set pool is assumed.",
                "Explicit per-set inventory remaining_laps may impose a stricter race limit.",
                "Safety Car and Virtual Safety Car laps count toward the weekend limit.",
                "Grid, formation and post-chequered-flag laps are excluded by the source.",
            ],
        },
    },
}


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_text).strip("_")


def _utc_cutoff(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Tyre-rule cutoff must be a timezone-aware datetime")
    return value.astimezone(timezone.utc)


def get_event_tire_rules(
    year: int,
    event: Mapping[str, Any],
    *,
    cutoff: datetime | str | None = None,
) -> tuple[dict[str, int], dict[str, dict[str, Any]]]:
    """Return detached rule/provenance maps keyed by ``minimum_sets``/``max_set_laps``.

    Match only normalized loader circuit IDs/names, never race results, country or
    a substring of a venue name. A cutoff includes only a source known to precede
    it, using a conservative bound where publication precision is limited. ``None``
    means no publication cutoff; the current clock is never consulted.
    """
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError("Tyre-rule year must be an integer season")
    instant = _utc_cutoff(cutoff)
    circuits = {
        _CIRCUITS[name]
        for key in ("circuit_id", "circuit_name")
        if (name := _normalise(event.get(key))) in _CIRCUITS
    }
    if len(circuits) != 1:
        return {}, {}
    for key in ("year", "season", "date"):
        value = str(event.get(key) or "")
        if value[:4].isdigit() and int(value[:4]) != year:
            return {}, {}
    rules: dict[str, int] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for key, source in _RULES.get((year, circuits.pop()), {}).items():
        available = datetime.fromisoformat(source["available_by"].replace("Z", "+00:00"))
        if instant is not None and available > instant:
            continue
        rules[key] = source["value"]
        provenance[key] = copy.deepcopy({k: v for k, v in source.items() if k != "value"})
    return rules, provenance
