"""Distinguish race starting positions from simulated qualifying lap results."""

from f1sim.simulation.execution import validate_starting_grid_snapshot


def race_grid_context(snapshot):
    if not isinstance(snapshot, dict) or "starting_grid" not in snapshot:
        return ""
    try:
        drivers = [d["id"] for d in snapshot["drivers"]]
        grid = validate_starting_grid_snapshot(snapshot, drivers)
    except (KeyError, TypeError, ValueError):
        return "Race grid: invalid saved starting order."
    return ("Race starts from supplied order: " + ", ".join(grid)
            + ". Qualifying lap results remain simulated and do not set this race grid.")
