"""Validation and small state helpers for explicit paid pit plans.

The race engines deliberately keep their automatic plan slots separate from
these instructions.  This module owns the public input shape and the
per-driver outcome record; engine-specific code decides when a compulsory
stop takes precedence and when a service has actually committed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from copy import deepcopy

from f1sim.models.tire import TireCompound

_PLAN_KEYS = frozenset(("lap", "compound"))
_OUTCOME_KEYS = (
    "lap",
    "compound",
    "status",
    "reason",
    "actual_compound",
    "actual_set_id",
)


def _strict_lap(value, *, message: str = "pit plan laps must be integers") -> int:
    """Validate one API lap without accepting bools or numeric coercion."""
    if type(value) is not int:
        raise ValueError(message)
    if value < 2:
        raise ValueError("pit plan laps must be integers >= 2")
    return value


def _canonical_compound(value) -> str:
    if not isinstance(value, str):
        raise ValueError("pit plan compounds must be canonical compound strings")
    try:
        return TireCompound(value).value
    except ValueError as exc:
        choices = ", ".join(item.value for item in TireCompound)
        raise ValueError(
            f"pit plan compounds must be one of: {choices}",
        ) from exc


def _inventory_compounds(tire_inventory) -> dict[str, set[str]]:
    """Extract compound availability without changing the caller's models."""
    if tire_inventory is None:
        return {}
    if hasattr(tire_inventory, "sets") and hasattr(tire_inventory, "current_set_id"):
        # A live inventory is useful to direct engine callers.  Its mapping is
        # read only here; execution performs the actual availability check.
        return {
            "__live__": {
                item.compound.value
                for item in tire_inventory.sets.values()
            },
        }
    if not isinstance(tire_inventory, Mapping):
        raise ValueError("tire_inventory must be an object mapping drivers to set lists")
    compounds: dict[str, set[str]] = {}
    for driver, records in tire_inventory.items():
        if not isinstance(driver, str):
            continue
        if not isinstance(records, list):
            # The inventory validator reports the structural error.  Keep this
            # helper deterministic when it is called independently by only
            # collecting valid record containers.
            continue
        values: set[str] = set()
        for record in records:
            if not isinstance(record, Mapping):
                continue
            value = record.get("compound")
            if isinstance(value, TireCompound):
                values.add(value.value)
            elif isinstance(value, str):
                values.add(value)
        compounds[driver] = values
    return compounds


def validate_pit_plans(
    value,
    driver_ids=None,
    total_laps=None,
    tire_inventory=None,
) -> dict[str, list[dict]]:
    """Validate and copy explicit own-lap pit instructions.

    ``None`` and an empty mapping mean that every driver remains automatic.
    A listed driver with an empty list is retained in the normalized mapping,
    which is the public representation for suppressing elective stops.
    """
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("pit_plans must be an object mapping driver IDs to instruction lists")
    if total_laps is not None and type(total_laps) is not int:
        raise ValueError("total_laps must be an integer when validating pit plans")
    known = set(driver_ids) if driver_ids is not None else None
    available = _inventory_compounds(tire_inventory)
    result: dict[str, list[dict]] = {}
    for driver, records in value.items():
        if not isinstance(driver, str) or not driver.strip():
            raise ValueError("pit plan driver IDs must be nonempty strings")
        if known is not None and driver not in known:
            raise ValueError(f"Unknown pit plan driver ID: {driver}")
        if not isinstance(records, list):
            raise ValueError("pit plan drivers must map to instruction lists")
        if len(records) > 20:
            raise ValueError("pit plans allow at most 20 instructions per driver")
        normalized: list[dict] = []
        previous_lap = None
        pool = available.get(driver)
        for record in records:
            if not isinstance(record, dict) or set(record) != _PLAN_KEYS:
                raise ValueError("each pit plan instruction must contain exactly lap and compound")
            lap = _strict_lap(record["lap"])
            if total_laps is not None and lap > total_laps:
                raise ValueError("pit plan laps must not exceed total_laps")
            if previous_lap is not None and lap <= previous_lap:
                raise ValueError("pit plan laps must be strictly increasing")
            previous_lap = lap
            compound = _canonical_compound(record["compound"])
            if pool is not None and compound not in pool:
                raise ValueError(
                    f"pit plan compound {compound!r} is absent from {driver}'s tire inventory",
                )
            normalized.append({"lap": lap, "compound": compound})
        result[driver] = normalized
    return result


def parse_pit_plan_spec(text: str) -> dict[str, list[dict]]:
    """Parse ``DRIVER=18:medium,36:hard;NOR=24:hard`` shorthand."""
    if not isinstance(text, str):
        raise ValueError("pit plan specification must be a string")
    if not text.strip():
        return {}
    result: dict[str, list[dict]] = {}
    for group in re.split(r"[;\n]", text):
        group = group.strip()
        if not group:
            raise ValueError("pit plan specification contains an empty group")
        pieces = group.split("=")
        if len(pieces) != 2 or not pieces[0].strip():
            raise ValueError("pit plans expect DRIVER=lap:compound,...")
        driver = pieces[0].strip()
        if driver in result:
            raise ValueError(f"duplicate pit plan driver: {driver}")
        spec = pieces[1].strip()
        if spec == "none":
            result[driver] = []
            continue
        if not spec:
            raise ValueError("pit plan instruction list cannot be empty; use DRIVER=none")
        instructions = []
        for token in spec.split(","):
            token = token.strip()
            fields = token.split(":")
            if len(fields) != 2 or not fields[0] or not fields[1]:
                raise ValueError("pit plans expect DRIVER=lap:compound,...")
            lap_token, compound = fields
            if not lap_token.isascii() or not lap_token.isdigit():
                raise ValueError("pit plan laps must be decimal integers")
            instructions.append({"lap": int(lap_token), "compound": compound})
        result[driver] = instructions
    return validate_pit_plans(result)


def initialize_pit_plan_state(state, plan: list[dict] | None) -> None:
    """Attach independent plan and outcome copies to one race state."""
    if plan is None:
        state.pit_plan = None
        state.pit_plan_history = None
    else:
        state.pit_plan = deepcopy(plan)
        state.pit_plan_history = [
            {
                "lap": item["lap"],
                "compound": item["compound"],
                "status": None,
                "reason": None,
                "actual_compound": None,
                "actual_set_id": None,
            }
            for item in plan
        ]
    state.pit_plan_index = 0
    state.pit_plan_target = None
    state.pit_plan_target_set_id = None
    state.pit_plan_reason = None
    state.pit_plan_override_reason = None


def current_pit_plan_instruction(state, lap: int):
    """Return the next unresolved instruction when it is due on ``lap``."""
    plan = getattr(state, "pit_plan", None)
    history = getattr(state, "pit_plan_history", None)
    index = getattr(state, "pit_plan_index", 0)
    if plan is None or history is None or index >= len(plan):
        return None
    item = plan[index]
    if item["lap"] != lap or history[index]["status"] is not None:
        return None
    return item


def _finish_instruction(state, *, status, reason, actual_compound=None, actual_set_id=None):
    history = getattr(state, "pit_plan_history", None)
    index = getattr(state, "pit_plan_index", 0)
    if history is None or index >= len(history) or history[index]["status"] is not None:
        return False
    record = history[index]
    record.update(
        status=status,
        reason=reason,
        actual_compound=actual_compound,
        actual_set_id=actual_set_id,
    )
    state.pit_plan_index = index + 1
    state.pit_plan_target = None
    state.pit_plan_target_set_id = None
    state.pit_plan_reason = None
    state.pit_plan_override_reason = None
    return True


def commit_pit_plan_service(
    state, *, reason, actual_compound, actual_set_id=None, status=None,
):
    """Mark a request after a paid service has committed."""
    if status is None:
        status = "overridden" if reason in {
            "forced_repair", "critical_weather", "compound_requirement",
        } else "executed"
    return _finish_instruction(
        state,
        status=status,
        reason=reason,
        actual_compound=actual_compound,
        actual_set_id=actual_set_id,
    )


def skip_pit_plan_instruction(state, reason):
    """Mark a due request that has a definitive non-service outcome."""
    return _finish_instruction(state, status="skipped", reason=reason)


def override_pit_plan_instruction(state, reason):
    """Close a due request when a compulsory rule keeps the fitted set."""
    return _finish_instruction(state, status="overridden", reason=reason)


def finalize_pit_plan(state, reason: str):
    """Close every unresolved request at a terminal race boundary."""
    history = getattr(state, "pit_plan_history", None)
    if history is None:
        return None
    for index in range(getattr(state, "pit_plan_index", 0), len(history)):
        record = history[index]
        if record["status"] is None:
            record.update(
                status="not_reached",
                reason=reason,
                actual_compound=None,
                actual_set_id=None,
            )
    state.pit_plan_index = len(history)
    return deepcopy(history)


def outcome_keys() -> tuple[str, ...]:
    """Expose the stable output key order for serializers and tests."""
    return _OUTCOME_KEYS
