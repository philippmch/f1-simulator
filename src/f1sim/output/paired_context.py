"""Shared wording for paired-comparison coverage reports."""


FINISHED_TIME_NOTE = (
    "Elapsed time includes only pairs where the driver finished both races over the same "
    "positive number of laps, including equal-distance lapped finishes. Retirements, unequal "
    "distances and missing times are excluded. Negative changes mean faster. This conditional "
    "subset is not an overall strategy ranking; compare points, retirements and distance too. "
    "SE measures sampling uncertainty, and zero SE does not establish certainty."
)


def finished_race_time_text(stats: dict | None) -> str:
    """Format the optional equal-distance finishing subset from paired analysis."""
    if not stats:
        return "Elapsed time: Not recorded."
    pairs, excluded = stats["paired_races"], stats["excluded_pairs"]
    coverage = f"{pairs} same-distance finish pairs ({excluded} excluded from time subset)"
    if not pairs:
        return f"Elapsed time: {coverage}; no comparable elapsed times."
    error = stats["seconds_difference_standard_error"]
    uncertainty = f"SE {error:.3f} s" if error is not None else "SE needs at least 2 time pairs"
    return (
        f"Elapsed time: {coverage}; mean {stats['reference_mean_seconds']:.3f} -> "
        f"{stats['variant_mean_seconds']:.3f} s, change "
        f"{stats['mean_seconds_difference']:+.3f} s ({uncertainty}); "
        f"faster/equal/slower={stats['faster_races']}/{stats['equal_time_races']}/"
        f"{stats['slower_races']}"
    )


def paired_coverage_text(comparison: dict) -> str:
    available = comparison["available_seed_pairs"]
    trial_unit = "trial" if available == 1 else "trials"
    mismatches = comparison["qualifying_mismatches"]
    mismatch_unit = "qualifying mismatch" if mismatches == 1 else "qualifying mismatches"
    return (
        f'Overlapping recorded seeds {comparison["seed_from"]}–{comparison["seed_to"]} '
        f'({available} {trial_unit}); {mismatches} {mismatch_unit} excluded. '
        'A qualifying mismatch means the qualifying record was missing, invalid, or different.'
    )


def paired_exclusion_detail(comparison: dict, excluded: int) -> str:
    mismatches = comparison["qualifying_mismatches"]
    driver_exclusions = max(excluded - mismatches, 0)
    mismatch_unit = "qualifying mismatch" if mismatches == 1 else "qualifying mismatches"
    driver_unit = (
        "missing/invalid driver observation"
        if driver_exclusions == 1 else "missing/invalid driver observations"
    )
    return (
        f'of these, {mismatches} {mismatch_unit} and '
        f'{driver_exclusions} {driver_unit}'
    )
