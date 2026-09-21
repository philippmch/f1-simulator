"""Shared wording for paired-comparison coverage reports."""


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
