# Race forecasts after Grand Prix qualifying

The primary forecast window is after Grand Prix qualifying, before the race,
using the published race starting grid. Sprint qualifying does not open this
window. The race grid incorporates the publisher's grid penalties; the model
does not apply the penalties again from the notes.

New post-qualifying records report the simulation's driver win frequencies
directly. Live dashboard and CLI runs also do this when their supplied grid
matches verified published evidence after qualifying and before the race.
Earlier points still inform the live driver and car inputs, but do not
redistribute the resulting team wins afterward. Existing saved forecasts
continue to score the policy and probabilities stored when they were created.

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

Diagnostic format 2 stores identical literal model fields once and references
a shared earlier-point history. The verifier expands those fields and checks
each reconstructed allocation against its original digest. Derived score
diagnostics are recomputed rather than duplicated. The format records the
previous content seal; this deliberate compaction changes the development
evidence seal, while retaining every input, trial count and published loss.
Prospectively recorded forecast files remain unchanged.

| Forecast or candidate | Mean winner Brier, lower is better |
| --- | ---: |
| Native model with the published grid | 0.49351 |
| Native model after the numerical pit-cost tie correction | 0.48489 |
| Native model after the post-qualifying input correction | 0.42728 |
| Existing teammate-point reporting on those same counts | 0.61338 |
| Fixed grid scale 6 | 0.70633 |
| Fixed grid scale 12 | 0.55676 |
| Fixed grid scale 18 | 0.49050 |
| 2023-fitted current-qualifying pace correction | 0.58491 |
| That correction with earlier-race residual uncertainty | 0.58865 |
| 2023-fitted earlier green race pace weighted by circuit | 0.49727 |

Before the input correction, the native model was approximately tied with the
strongest grid reference. The corrected 2026 development score is 12.9% lower
than that reference, but its paired interval still includes zero.
None of the three pace candidates met the winner gate; none changes the live
race-pace model or reporting policy. The point comparison is a reporting
comparison using the same complete-grid trials, not a reconstruction of the
old app's fallback on unsupported pit-lane grids. These scores do not establish
future predictive skill or a substantial reference-beating improvement.

The [paired reporting decision](../evidence/post-qualifying-reporting-decision-2026.json)
compares both policies on the same corrected trials. Native reporting yields
Brier 0.48489 versus 0.61222 with teammate-point redistribution, a 20.8% lower
development loss. This changes the reported forecast rather than improving
the simulated pace by 20.8%. The paired interval includes zero, and the
native score is only 1.1% below the best grid reference; substantial future
reference-beating skill remains unproven. These measurements use fixed dry
conditions and do not establish wet-weather forecast accuracy.

## Post-qualifying input correction

Two input problems weakened the observed qualifying and recent-form signals.
Constructor points occupied roughly one unit in the team blend while timing
residuals occupied hundredths, so the advertised timing weights had little
effect. Also, the provider's 2025 and 2026 race rows retained fastest-lap times
while omitting average speed; those usable observations silently contributed
no race-form signal.

After a sufficiently complete GP qualifying observation, the builder places
each timing bucket on a common field scale with maximum absolute value 0.5,
then applies the existing 0.5/0.3/0.2 weights and constructor anchor. Recent
race form uses inverse lap seconds when the entire event lacks positive speed
observations. An event with supplied speeds keeps that unit for every driver;
the builder never mixes inverse seconds with km/h. Pre-qualifying input
behavior is retained.

The [sealed paired receipt](../evidence/post-qualifying-input-correction.json)
contains 16 complete 2026 contexts and 24 complete 2025 sensitivity contexts,
each with 100 trials and the same grid and seeds before and after correction.
Its literal model defaults and input differences retain both sets of models.
Recompute the counts and references offline with
`python examples/verify_post_qualifying_input_correction.py`.

| Development check | Previous model | Corrected model | Strongest grid reference |
| --- | ---: | ---: | ---: |
| 2026, 16 events | 0.48489 | 0.42728 | 0.49050 |
| 2025, 24 sensitivity events | 0.64232 | 0.58372 | 0.51894 |
| Combined 40 events | 0.57935 | 0.52114 | 0.50757 |

The combined loss is 10.0% lower than the previous model. The paired gain is
0.05821 with a 95% bootstrap interval of 0.00680 to 0.11582, and stays positive
at 0.02230 after removing the three largest gains. This supports improving
the existing input model; it does not establish the full reference-beating
forecast goal. The combined corrected model still trails the strongest grid
reference, and the 2026 reference comparison remains uncertain.

The 2025 checks use current simulator rules and fixed dry weather. They test
the direction of the input correction across a second season, rather than
reconstructing the old rules or authenticating historical forecasts. Fastest
laps still reflect tyre choices, traffic and race programs. This correction
does not import historical driver or team seeds into current-season runtime,
and it does not establish wet-weather or future winner accuracy.

Three further controlled changes failed the full winner gate: a 2023-fitted
passing correction (0.52283), persistent uncertainty around unchanged native
mean pace (0.57050), and a completed-Sprint pace update on the five applicable
weekends (0.56549). The other eleven Sprint-candidate forecasts retained the
corrected baseline counts. None of these physics changes is deployed.

The numerical correction refuses a paid stop when its projected saving is
within one billionth of a second of the existing decision threshold. A Monza
trace had accepted a stop for only 9.09e-13 seconds of saving. The
[paired correction evidence](../evidence/post-qualifying-pit-tie-2026.json)
retains all 16 input contexts and seeds: mean native Brier falls by 1.75%.
The paired 95% bootstrap interval for the gain is -0.00465 to 0.02009, so
this sample does not establish a reliable forecast improvement. The numerical
correctness fix still falls short of the 10% reference improvement required
above. The same verifier checks its saved win counts.

One concrete failure is Monza: the native model recorded zero Antonelli wins
in its 100-trial development replay from P19. The tyre supplier reports an
early red flag with free tyre changes, followed by a lap-29 VSC pit stop for
Antonelli while Russell stayed on his earlier hard set. The Formula 1 race
report also describes repeated passes and deteriorating late-race tyres.
These observations motivate controlled checks of recovery, regrouping and
strategy; they do not by themselves establish a bug or permit using the
observed event schedule in a forecast. [Pirelli race analysis](https://press.pirelli.com/it/trionfo-italiano-di-kimi-nel-gran-premio-pirelli-a-monza/),
[Formula 1 race report](https://www.formula1.com/en/latest/article/antonelli-beats-russell-to-italian-gp-win-with-stunning-comeback-drive.15WtFEBT5JEe4drdeO88t2).
