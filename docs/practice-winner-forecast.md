# Winner probabilities after current practice

Dry forecasts after usable current practice and before the first qualifying
session now retain the native simulated driver win probabilities. The earlier
point allocation had been validated with the older qualifying model. With the
practice-informed model, it discards useful driver separation and increases
winner forecast error in the current-season replay.

The dashboard/API, CLI and newly recorded forecasts use the same selection.
It requires the validated 2026 practice policy, matching driver ratings, a
known qualifying cutoff, zero rain/wetness/weather-change probability, no
qualifying or race weather schedule override, and no supplied starting grid.
Sprint qualifying is part of the cutoff. Missing practice, custom qualifying,
wet or changing weather and forecasts after qualifying retain the earlier
policy. Explicit saved allocations and old forecast records retain their
original interpretation.

This changes reported probabilities. Every simulated winner, constructor win
probability, lap, podium, retirement and strategy outcome remains unchanged.
The native sampling intervals continue to describe simulation sampling error,
not uncertainty about the real race.

## Measured result

[Sealed evidence](../evidence/practice-native-winner-2026.json) covers all 16
completed 2026 events, with 22 entrants in each, 400 native chronological
trials per event, fixed dry weather and the same seeds and inputs for both
reporting policies. All original 100-trial prefixes were reproduced exactly.
Two earlier research inputs had omitted drivers without qualifying rows:
Verstappen, Sainz and Stroll at round 1, and Bearman and Stroll at round 14.
Those fields were rebuilt before this comparison; no exclusion removes them.

Equal-event mean unhalved winner Brier falls from **0.72949 to 0.68327**,
**6.3% lower**. Ten races improve, four worsen and two are unchanged. The
paired 10,000-resample bootstrap interval for mean improvement is
**[0.00099, 0.08998]**. Removing the three largest gains still leaves a
**0.01591 improvement**. The lower interval bound is small; the evidence
should not be described as overwhelming.

These are already-inspected retrospective outcomes from revised feeds, with
entrant identities supplied retrospectively. Fixed dry forecasts were scored
against the observed winners; authenticated historical weather forecasts were
not supplied. This is a supported reporting improvement within its stated
context, not an independent test or proof of future calibration. Sixteen races
and this gain do **not** satisfy the broader
[winner-model acceptance protocol](predictive-model-protocol.md).

Run `python examples/verify_practice_winner_policy.py` to check the seal,
qualifying parameter digest, complete rosters, causal point/history cutoffs,
real deterministic qualifying laps, production policy selection, all winner
scores and aggregate sensitivities offline. It recomputes reporting and scores;
it does not rerun the 400 race trials or use these observations in live loading.
