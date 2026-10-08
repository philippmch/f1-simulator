"""Describe the completed run's prescribed rainfall scenario from saved inputs."""

from f1sim.simulation.weather_schedule import validate_weather_schedule


def weather_schedule_context(snapshot) -> str:
    if not isinstance(snapshot, dict) or "weather_schedule" not in snapshot:
        return ""
    try:
        track = snapshot.get("track")
        distance = track.get("total_laps") if isinstance(track, dict) else None
        schedule = validate_weather_schedule(snapshot["weather_schedule"], total_laps=distance)
        if not schedule:
            return ""
        if (type(snapshot.get("schema_version")) is not int
                or snapshot["schema_version"] not in (8, 9, 10, 11, 12, 13, 14)):
            return "Prescribed race rainfall: unrecognized saved schema."
    except (TypeError, ValueError):
        return "Prescribed race rainfall: invalid saved schedule."
    steps = []
    for step in schedule:
        condition = (step["condition"].replace("_", " ") if "condition" in step
                     else "condition unchanged")
        steps.append(f"lap {step['lap']}: rain {step['rain_intensity'] * 100:g}%, {condition}")
    return (
        "Prescribed race rainfall (known to strategy; shared leading laps): "
        + "; ".join(steps)
        + ". Surface water continues to evolve; random atmosphere changes are disabled."
    )
