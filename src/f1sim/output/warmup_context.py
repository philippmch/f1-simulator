"""Label saved sensitivity assumptions without consulting current form controls."""

from f1sim.simulation.warmup import validate_tire_warmup


def warmup_context(snapshot) -> str:
    if not isinstance(snapshot, dict) or "tire_warmup" not in snapshot:
        return ""
    try:
        profile = validate_tire_warmup(snapshot["tire_warmup"])
    except (TypeError, ValueError):
        return "Post-fit cost sensitivity: invalid saved profile."
    if not profile:
        return ""
    if snapshot.get("tire_warmup_policy") != "post_fit_first_lap_v1":
        return "Post-fit cost sensitivity: unrecognized saved policy."
    values = ", ".join(f"{compound}={seconds:g} s" for compound, seconds in sorted(profile.items()))
    return (
        f"Assumed post-fit first-running-lap costs: {values}. "
        "Opening and qualifying tyres are ready. Sensitivity assumptions, not calibrated physics."
    )
