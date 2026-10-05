"""Frozen, season-specific common dry race-clock calibration.

Fit on Canada's 2026 green, zero-rainfall laps. This is a conditional timing
correction, not a driver ranking or a claim about pre-event winner accuracy.
"""

DRY_RACE_CLOCK_POLICY = "canadian_field_clock_2026_v1"
DRY_RACE_CLOCK_FIT_ROUND = 5
DRY_RACE_CLOCK_ADJUSTMENT = 0.003905707495222259


def dry_race_pace_adjustment_for_event(year: int, round_number: int | None) -> float:
    """Use the fixed calibration only after its training event in 2026.

Unidentified rounds and other seasons retain the native clock. Calibration
does not fetch data or use any performance from the event being forecast.
"""
    if (type(year) is int and year == 2026 and type(round_number) is int
            and round_number > DRY_RACE_CLOCK_FIT_ROUND):
        return DRY_RACE_CLOCK_ADJUSTMENT
    return 0.0
