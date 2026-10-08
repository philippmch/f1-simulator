# Race forecasts after Grand Prix qualifying

The primary forecast window is after Grand Prix qualifying, before the race,
using the published race starting grid. Sprint qualifying does not open this
window. The race grid incorporates the publisher's grid penalties; the model
does not apply the penalties again from the notes.

```powershell
python examples/record_race_forecast.py --stage post_qualifying --race 17 --scenario dry --trials 400 --seed 1701007 --parallel --workers 8 --output output/forecasts/2026-round-17-after-qualifying.json
python examples/score_recorded_forecast.py output/forecasts/2026-round-17-after-qualifying.json
```

The current calendar gives Singapore GP qualifying as 10 October 2026 at
13:00 UTC and the race as 11 October at 12:00 UTC. Recording becomes eligible
at 14:00 UTC on 10 October, subject to a complete published grid being
available. These are fetched calendar times, not a guarantee that a session
ends on schedule. No forecast for that window has been recorded yet.

## Recorded inputs and scoring

Schema 4 records require dated GP qualifying and race times, a fresh complete
Formula1.com starting grid, and no target or later race result. Current GP
qualifying observations are allowed. Later qualifying and race observations
are rejected. The recording must start after the GP qualifying start plus one
hour and finish before the scheduled race start. An unavailable or ambiguous
published grid rejects recording instead of certifying a simulated grid.

The record freezes the full driver field, actual race order, identified pit
starters, source URL and fetch time, model inputs, assumptions, native winner
counts and any reporting allocation. It also freezes three fixed grid-rank
references with scales 6, 12 and 18. Each uses
`exp(scale * (1 - (rank - 1) / (field_size - 1)))`, normalized across the same
full field. The references are not fitted to that race's outcome. All three
are reported, so a comparison cannot select a weak reference after results.

Scoring reconstructs the references from saved grid evidence. It does not
fetch an updated grid, rerun a race or refit point allocations. It reports
each winner loss and each driver's contribution to the model's loss relative
to each grid reference. The qualifying trials stored with the simulation are
diagnostic: post-qualifying records never score already observed qualifying
as a successful prediction. The original pre-qualifying records remain a
separate forecast stage and retain their original bytes and interpretation.

The writer refuses to overwrite an existing file. Local timestamps and a
content seal detect accidental changes; independent dated publication before
the race is still required to authenticate a prospective timing claim.

## Pit-lane starting assumption

The official grid parser resolves numeric slots, explicit `PL` rows and
identified pit-lane instructions. A pit starter remains a separate input at
the ordered tail, rather than an ordinary numbered grid starter. Incomplete
fields, conflicting identities, unknown teams and unresolved instructions
fail validation. Script payloads cannot add instructions to the visible grid.

Both race engines use a fixed **five-second delayed release** for pit starters.
The chronological engine keeps them off track until release and preserves
their published queue order. The start does not count as a paid pit stop,
fit a tyre or increment a lap. Replays and strategy variants preserve the
same start context under simulation input schema 15.

The delay is an explicit whole-lap approximation, not a duration specified by
the FIA or a calibrated circuit-specific launch model. The actual rule opens
the pit exit after the on-track field passes it. Grid spacing, pit-exit
location, launch incidents and changes to the queue on race day are not
resolved by this approximation. See articles B5.3.2 and B5.7.3 of the
[FIA sporting regulations, issue 09](https://api.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf).

## Evaluation boundary

The 2026 race outcomes have already been inspected. Replays using revised
published grids and qualifying are development evidence, not an untouched
test or a record made before those races. Fixed dry conditions are also a
conditional assumption, not an authenticated historical weather forecast.

This forecast stage retains the substantial-improvement standard: winner
Brier at least 10% below the best fixed reference, log loss at least 5% lower,
positive improvement after removing the three largest gains, and a positive
lower bound in a paired 10,000-resample bootstrap. Broad historical evidence
requires at least 40 complete events in two seasons and checks on positions,
podiums and retirements. Future recorded forecasts must then demonstrate
skill with probabilities and references published before their races.

Better qualifying or conditional lap-time estimates alone do not satisfy the
winner-forecast gate. Failed candidates remain in the research evidence and
are not deployed as accuracy improvements.

## Current diagnostic

The [sealed 16-event diagnostic](../evidence/post-qualifying-winner-diagnostic-2026.json)
uses complete 22-driver fields, observed GP qualifying, published grids,
separate pit starters, fixed dry conditions and 100 chronological trials per
event. Reproduce its scores with `python examples/verify_post_qualifying_diagnostic.py`.

| Forecast or candidate | Mean winner Brier, lower is better |
| --- | ---: |
| Native model with the published grid | 0.49351 |
| Existing teammate-point reporting on those same counts | 0.61338 |
| Fixed grid scale 6 | 0.70633 |
| Fixed grid scale 12 | 0.55676 |
| Fixed grid scale 18 | 0.49050 |
| 2023-fitted current-qualifying pace correction | 0.58491 |
| That correction with earlier-race residual uncertainty | 0.58865 |
| 2023-fitted earlier green race pace weighted by circuit | 0.49727 |

The native model is approximately tied with the strongest grid reference.
None of the three pace candidates met the winner gate; none changes the live
race-pace model or reporting policy. The point comparison is a reporting
comparison using the same complete-grid trials, not a reconstruction of the
old app's fallback on unsupported pit-lane grids. These scores do not establish
future predictive skill or a substantial reference-beating improvement.

One concrete failure is Monza: the native model recorded zero Antonelli wins
in its 100-trial development replay from P19. The tyre supplier reports an
early red flag with free tyre changes, followed by a lap-29 VSC pit stop for
Antonelli while Russell stayed on his earlier hard set. The Formula 1 race
report also describes repeated passes and deteriorating late-race tyres.
These observations motivate controlled checks of recovery, regrouping and
strategy; they do not by themselves establish a bug or permit using the
observed event schedule in a forecast. [Pirelli race analysis](https://press.pirelli.com/it/trionfo-italiano-di-kimi-nel-gran-premio-pirelli-a-monza/),
[Formula 1 race report](https://www.formula1.com/en/latest/article/antonelli-beats-russell-to-italian-gp-win-with-stunning-comeback-drive.15WtFEBT5JEe4drdeO88t2).
