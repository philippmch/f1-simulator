"""Racing-clock finish deadline; suspension wall time is not modeled."""

RACING_TIME_LIMIT_SECONDS = 7200.0


def announced_final_lap(current_final_lap: int, completed_lap: int, leader_time: float) -> int:
    """After two hours expire, finish the following lap or the scheduled last lap."""
    if leader_time >= RACING_TIME_LIMIT_SECONDS:
        return min(current_final_lap, completed_lap + 1)
    return current_final_lap
