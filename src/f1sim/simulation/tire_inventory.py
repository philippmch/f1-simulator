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
            if not isinstance(record, dict) or set(record) - {"id", "compound", "age"}:
                raise ValueError("tire_inventory set records allow only id, compound and age")
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
            copied.append({"id": identifier, "compound": compound, "age": age})
            identifiers.add(identifier)
        if driver in openings and not any(
            item["compound"] == openings[driver] and item["age"] == ages.get(driver, 0)
            for item in copied
        ):
            raise ValueError("tire_inventory must contain the exact opening compound and age")
        result[driver] = copied
    return result


def parse_tire_inventory_spec(text):
    """Parse DRIVER=compound[@age],... groups separated by semicolons/newlines."""
    if not isinstance(text, str):
        raise ValueError("tire_inventory specification must be a string")
    if not text.strip():
        return {}
    result = {}
    for group in re.split(r"[;\n]", text.strip()):
        parts = group.split("=")
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError("tire_inventory expects DRIVER=compound[@age],...")
        driver = parts[0].strip()
        if driver in result:
            raise ValueError("tire_inventory driver IDs must be unique")
        records = []
        for entry in parts[1].split(","):
            compound, age = parse_starting_tire_spec(entry.strip())
            records.append({"compound": compound, "age": age})
        result[driver] = records
    return validate_tire_inventory(result)


@dataclass(frozen=True)
class TireSet:
    id: str
    compound: TireCompound
    age: int


class TireInventory:
    """Mutable race ledger; removed undamaged sets remain reusable."""

    def __init__(self, sets):
        self.sets = {item.id: item for item in sets}
        self.current_set_id = None
        self.unavailable_ids = set()

    @classmethod
    def from_sets(cls, canonical_records):
        records = validate_tire_inventory({"driver": canonical_records})["driver"]
        return cls(TireSet(item["id"], TireCompound(item["compound"]), item["age"])
                   for item in records)

    def replacements(self):
        return tuple(item for identifier, item in self.sets.items()
                     if identifier != self.current_set_id
                     and identifier not in self.unavailable_ids)

    def _check_current_age(self, current_age):
        if current_age is not None:
            _age(current_age)
            if (self.current_set_id is not None
                    and current_age < self.sets[self.current_set_id].age):
                raise ValueError("Current tire_inventory wear cannot decrease")

    def fit(self, set_id, current_age=None):
        # Validate the full operation before touching the outgoing set.
        if not isinstance(set_id, str) or set_id not in self.sets:
            raise ValueError("Unknown tire_inventory set ID")
        if set_id in self.unavailable_ids:
            raise ValueError("Cannot fit an unavailable tire_inventory set")
        self._check_current_age(current_age)
        if self.current_set_id is not None and current_age is not None:
            current = self.sets[self.current_set_id]
            self.sets[self.current_set_id] = replace(current, age=current_age)
        self.current_set_id = set_id
        return self.sets[set_id]

    def mark_current_unavailable(self, current_age):
        _age(current_age)
        self._check_current_age(current_age)
        if self.current_set_id is None:
            raise ValueError("No current tire_inventory set to mark unavailable")
        current = self.sets[self.current_set_id]
        self.sets[self.current_set_id] = replace(current, age=current_age)
        self.unavailable_ids.add(self.current_set_id)

    def snapshot(self, current_age=None):
        self._check_current_age(current_age)
        return [{"id": item.id, "compound": item.compound.value,
                 "age": current_age if identifier == self.current_set_id
                 and current_age is not None else item.age,
                 "current": identifier == self.current_set_id,
                 "unavailable": identifier in self.unavailable_ids,
                 "available": identifier != self.current_set_id
                 and identifier not in self.unavailable_ids}
                for identifier, item in self.sets.items()]
