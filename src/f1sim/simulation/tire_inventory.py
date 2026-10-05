"""Explicit race tyre pools and reusable physical-set bookkeeping."""

import re
from dataclasses import dataclass, replace

from f1sim.models.tire import TireCompound
from f1sim.simulation.execution import (
    MAX_STARTING_TIRE_AGE,
    parse_starting_tire_spec,
    validate_starting_tire_ages,
    validate_starting_tires,
)


def _age(value, maximum=None):
    if type(value) is not int or value < 0 or (maximum is not None and value > maximum):
        raise ValueError("tire_inventory ages must be nonnegative integers"
                         + (f" <= {maximum}" if maximum is not None else ""))
    return value


def _remaining_laps(value):
    if value is not None and (type(value) is not int or not 0 <= value <= MAX_STARTING_TIRE_AGE):
        raise ValueError("tire_inventory remaining_laps must be an integer from 0 through 1000")
    return value


def validate_tire_inventory(value, starting_tires=None, starting_tire_ages=None,
                            driver_ids=None):
    """Copy explicit pools, validating optional roster and opening selections."""
    known = set(driver_ids) if driver_ids is not None else None
    openings = validate_starting_tires(starting_tires, known)
    ages = validate_starting_tire_ages(starting_tire_ages, openings, known)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("tire_inventory must be an object mapping drivers to set lists")
    result = {}
    for driver, records in value.items():
        if not isinstance(driver, str) or not driver.strip():
            raise ValueError("tire_inventory driver IDs must be nonempty strings")
        if known is not None and driver not in known:
            raise ValueError(f"Unknown tire_inventory driver ID: {driver}")
        if not isinstance(records, list) or not 1 <= len(records) <= 20:
            raise ValueError("tire_inventory requires 1..20 sets per listed driver")
        copied, identifiers = [], set()
        for index, record in enumerate(records):
            if (not isinstance(record, dict)
                    or set(record) - {"id", "compound", "age", "remaining_laps"}):
                raise ValueError(
                    "tire_inventory set records allow only id, compound, age and remaining_laps")
            identifier = record.get("id", f"set-{index + 1}")
            if (not isinstance(identifier, str) or not identifier.strip()
                    or identifier in identifiers):
                raise ValueError("tire_inventory set IDs must be nonempty and unique per driver")
            compound = record.get("compound")
            if not isinstance(compound, str):
                raise ValueError("tire_inventory compound must be a canonical compound string")
            try:
                compound = TireCompound(compound).value
            except ValueError as exc:
                raise ValueError(
                    "tire_inventory compound must be a canonical compound string",
                ) from exc
            age = _age(record.get("age", 0), MAX_STARTING_TIRE_AGE)
            item = {"id": identifier, "compound": compound, "age": age}
            remaining = _remaining_laps(record.get("remaining_laps"))
            if remaining is not None:
                item["remaining_laps"] = remaining
            copied.append(item)
            identifiers.add(identifier)
        if all(item.get("remaining_laps") == 0 for item in copied):
            raise ValueError("tire_inventory needs at least one set with a permitted race lap")
        if driver in openings and not any(
            item["compound"] == openings[driver] and item["age"] == ages.get(driver, 0)
            and item.get("remaining_laps") != 0
            for item in copied
        ):
            raise ValueError("tire_inventory must contain the exact opening compound and age "
                             "with at least one permitted race lap")
        result[driver] = copied
    return result


def parse_tire_inventory_spec(text):
    """Parse DRIVER=compound[@age][/remaining],... groups separated by semicolons/newlines."""
    if not isinstance(text, str):
        raise ValueError("tire_inventory specification must be a string")
    if not text.strip():
        return {}
    result = {}
    for group in re.split(r"[;\n]", text.strip()):
        parts = group.split("=")
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError("tire_inventory expects DRIVER=compound[@age][/remaining],...")
        driver = parts[0].strip()
        if driver in result:
            raise ValueError("tire_inventory driver IDs must be unique")
        records = []
        for entry in parts[1].split(","):
            pieces = entry.strip().split("/")
            if (len(pieces) > 2 or len(pieces) == 2
                    and not re.fullmatch(r"[0-9]+", pieces[1].strip())):
                raise ValueError("tire_inventory allowance requires /remaining_laps")
            compound, age = parse_starting_tire_spec(pieces[0].strip())
            item = {"compound": compound, "age": age}
            if len(pieces) == 2:
                item["remaining_laps"] = _remaining_laps(int(pieces[1].strip()))
            records.append(item)
        result[driver] = records
    return validate_tire_inventory(result)


TIRE_USAGE_POLICY = "completed_race_laps_v1"


def has_tire_usage_limits(inventory):
    return any(item.get("remaining_laps") is not None
               for records in inventory.values() for item in records)


def validate_tire_usage_snapshot(snapshot, inventory):
    """An older replay must never silently reinterpret a new usage constraint."""
    limited = has_tire_usage_limits(inventory)
    version = snapshot.get("schema_version")
    if version == 9 or version in (10, 11, 12) and (limited or "tire_usage_policy" in snapshot):
        if not limited or snapshot.get("tire_usage_policy") != TIRE_USAGE_POLICY:
            raise ValueError(
                f"Schema {version} requires usage-limited tire_inventory and tire_usage_policy",
            )
    elif limited or "tire_usage_policy" in snapshot:
        raise ValueError("Schemas 1-8 cannot contain tyre usage limits or tire_usage_policy")


@dataclass(frozen=True)
class TireSet:
    id: str
    compound: TireCompound
    age: int
    remaining_laps: int | None = None


def tire_set_slot(item, age=None):
    """Anonymous forecast identity: compound, wear and absolute wear at expiry."""
    return (item.compound.value, item.age if age is None else age,
            -1 if item.remaining_laps is None else item.age + item.remaining_laps)


def tire_slot_usable(age, expiry):
    return expiry < 0 or age < expiry


def exchange_tire_slots(pool, index, current):
    """An expired physical set remains in the ledger, but cannot be refitted."""
    returned = (current,) if tire_slot_usable(current[1], current[2]) else ()
    return tuple(sorted(pool[:index] + pool[index + 1:] + returned))


class TireInventory:
    """Mutable race ledger; removed undamaged sets remain reusable."""

    def __init__(self, sets):
        self.sets = {item.id: item for item in sets}
        for item in self.sets.values():
            _remaining_laps(item.remaining_laps)
        self.current_set_id = None
        self.unavailable_ids = set()

    @classmethod
    def from_sets(cls, canonical_records):
        records = validate_tire_inventory({"driver": canonical_records})["driver"]
        return cls(TireSet(item["id"], TireCompound(item["compound"]), item["age"],
                           item.get("remaining_laps"))
                   for item in records)

    def replacements(self):
        return tuple(item for identifier, item in self.sets.items()
                     if identifier != self.current_set_id
                     and identifier not in self.unavailable_ids and item.remaining_laps != 0)

    def current_remaining_laps(self, current_age=None):
        """Remaining allowance after completed use of the currently fitted set."""
        self._check_current_age(current_age)
        if self.current_set_id is None:
            return None
        item = self.sets[self.current_set_id]
        if item.remaining_laps is None:
            return None
        return item.remaining_laps - (0 if current_age is None else current_age - item.age)

    @staticmethod
    def _used(item, age):
        remaining = (None if item.remaining_laps is None else
                     item.remaining_laps - (age - item.age))
        return replace(item, age=age, remaining_laps=remaining)

    def _check_current_age(self, current_age):
        if current_age is not None:
            _age(current_age)
            if (self.current_set_id is not None
                    and current_age < self.sets[self.current_set_id].age):
                raise ValueError("Current tire_inventory wear cannot decrease")
            if self.current_set_id is not None:
                item = self.sets[self.current_set_id]
                if item.remaining_laps is not None and current_age - item.age > item.remaining_laps:
                    raise ValueError(
                        "Current tire_inventory use exceeds its remaining_laps allowance")

    def fit(self, set_id, current_age=None):
        # Validate the full operation before touching the outgoing set.
        if not isinstance(set_id, str) or set_id not in self.sets:
            raise ValueError("Unknown tire_inventory set ID")
        if set_id in self.unavailable_ids:
            raise ValueError("Cannot fit an unavailable tire_inventory set")
        self._check_current_age(current_age)
        available = (self.current_remaining_laps(current_age) if set_id == self.current_set_id
                     else self.sets[set_id].remaining_laps)
        if available == 0:
            raise ValueError("Cannot fit a tire_inventory set with no permitted race laps")
        if self.current_set_id is not None and current_age is not None:
            current = self.sets[self.current_set_id]
            self.sets[self.current_set_id] = self._used(current, current_age)
        self.current_set_id = set_id
        return self.sets[set_id]

    def mark_current_unavailable(self, current_age):
        _age(current_age)
        self._check_current_age(current_age)
        if self.current_set_id is None:
            raise ValueError("No current tire_inventory set to mark unavailable")
        current = self.sets[self.current_set_id]
        self.sets[self.current_set_id] = self._used(current, current_age)
        self.unavailable_ids.add(self.current_set_id)

    def snapshot(self, current_age=None):
        self._check_current_age(current_age)
        return [{"id": item.id, "compound": item.compound.value,
                 "age": current_age if identifier == self.current_set_id
                 and current_age is not None else item.age,
                 "current": identifier == self.current_set_id,
                 "unavailable": identifier in self.unavailable_ids,
                 "available": identifier != self.current_set_id
                 and identifier not in self.unavailable_ids and item.remaining_laps != 0,
                 **({"remaining_laps": self.current_remaining_laps(current_age)
                     if identifier == self.current_set_id else item.remaining_laps}
                    if item.remaining_laps is not None else {})}
                for identifier, item in self.sets.items()]
