"""Fixed practice-ranking reference frozen alongside pre-qualifying forecasts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone

from f1sim.analysis.race_probability_scores import score_winner_probabilities

PRACTICE_WINNER_REFERENCE = "practice_rank_softmax_12_v1"


def _timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Practice reference timestamps must include a timezone")
    instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if instant.tzinfo is None:
        raise ValueError("Practice reference timestamps must include a timezone")
    return instant.astimezone(timezone.utc)


def build_practice_winner_reference(
    entrant_ids, practice, *, year, target_round, recorded_at, qualifying_starts_at
):
    """Use exp(12 * normalized practice rank), without fitting to race outcomes.

    Drivers missing an observed practice time receive the fixed middle-rank
    strength .5, exactly as in the historical practice_12 reference. At least
    half the identified field must have timed practice observations.
    """
    ids = list(entrant_ids)
    if (
        len(ids) < 2
        or len(ids) > 30
        or len(set(ids)) != len(ids)
        or any(not isinstance(key, str) or not key.strip() for key in ids)
    ):
        raise ValueError("Practice reference requires a unique identified field")
    if (
        not isinstance(practice, Mapping)
        or type(year) is not int
        or type(target_round) is not int
        or target_round < 1
        or type(practice.get("year")) is not int
        or practice["year"] != year
        or type(practice.get("round")) is not int
        or practice["round"] != target_round
        or type(practice.get("session_number")) is not int
        or practice["session_number"] not in (1, 2, 3)
    ):
        raise ValueError("Practice reference must identify the forecast year, round and session")
    started = _timestamp(practice.get("practice_started_at"))
    fetched = _timestamp(practice.get("fetched_at"))
    recorded, cutoff = _timestamp(recorded_at), _timestamp(qualifying_starts_at)
    if (
        started.year != year
        or cutoff.year != year
        or not started + timedelta(hours=1) <= fetched <= recorded < cutoff
    ):
        raise ValueError("Practice reference must be completed and fetched before qualifying")
    if not isinstance(practice.get("source_url"), str) or not practice["source_url"].strip():
        raise ValueError("Practice reference requires its observed source URL")
    probabilities = practice_rank_probabilities(ids, practice.get("rows"))
    return {
        "policy": PRACTICE_WINNER_REFERENCE,
        "probabilities": probabilities,
        "no_classified_winner_probability": 0.0,
        "evidence": deepcopy(dict(practice)),
        "score": None,
    }


def practice_rank_probabilities(entrant_ids, rows):
    """Pure fixed reference for both retrospective diagnosis and live recording."""
    ids = list(entrant_ids)
    if (
        not 2 <= len(ids) <= 30
        or len(set(ids)) != len(ids)
        or any(not isinstance(key, str) or not key.strip() for key in ids)
    ):
        raise ValueError("Practice reference requires a unique identified field")
    if not isinstance(rows, list) or not math.ceil(len(ids) / 2) <= len(rows) <= len(ids):
        raise ValueError("Practice reference requires timed observations for half the field")
    ranks, positions = {}, set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Practice reference rows must be objects")
        key, rank, seconds, laps = (
            row.get(name)
            for name in (
                "driver",
                "position",
                "lap_seconds",
                "laps",
            )
        )
        if (
            not isinstance(key, str)
            or key not in ids
            or key in ranks
            or type(rank) is not int
            or not 1 <= rank <= 30
            or rank in positions
            or type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or not 30 <= seconds <= 240
            or type(laps) is not int
            or not 0 <= laps <= 200
        ):
            raise ValueError(
                "Practice reference rows require unique ranks and valid timed identities"
            )
        ranks[key] = rank
        positions.add(rank)
    strengths = {key: 1 - (ranks[key] - 1) / (len(ids) - 1) if key in ranks else 0.5 for key in ids}
    center = max(strengths.values())
    weights = {key: math.exp(12 * (value - center)) for key, value in strengths.items()}
    total = math.fsum(weights.values())
    return {key: weights[key] / total for key in ids}


def score_saved_practice_reference(
    saved,
    entrant_ids,
    *,
    year,
    target_round,
    recorded_at,
    qualifying_starts_at,
    observed_winner=None,
):
    """Recompute the fixed reference from frozen observations; never refetch practice."""
    if not isinstance(saved, Mapping) or saved.get("policy") != PRACTICE_WINNER_REFERENCE:
        raise ValueError("Unknown recorded practice reference policy")
    rebuilt = build_practice_winner_reference(
        entrant_ids,
        saved.get("evidence"),
        year=year,
        target_round=target_round,
        recorded_at=recorded_at,
        qualifying_starts_at=qualifying_starts_at,
    )
    if (
        saved.get("probabilities") != rebuilt["probabilities"]
        or type(saved.get("no_classified_winner_probability")) not in (int, float)
        or saved["no_classified_winner_probability"] != 0.0
    ):
        raise ValueError("Practice reference probabilities must match its frozen observations")
    if observed_winner is not None:
        rebuilt["score"] = score_winner_probabilities(
            rebuilt["probabilities"],
            0.0,
            observed_winner,
        )
    return rebuilt
