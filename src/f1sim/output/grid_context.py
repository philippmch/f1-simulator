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
    pit = snapshot.get("pit_lane_starters", [])
    return ("Race starts from supplied order: " + ", ".join(grid)
            + (". Pit-lane starters: " + ", ".join(pit)
               + " (assumed 5-second delayed release)" if pit else "")
            + ". Qualifying lap results remain simulated and do not set this race grid.")
