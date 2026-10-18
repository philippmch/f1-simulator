# App quality checkpoint — 18 October 2026

The app is useful for conditional simulation and strategy experiments. Its
experimental confirmed-grid forecast has a small historical advantage over
the grid reference; reliable predictive superiority remains unproven. Native
simulation remains the default.

The [confirmed-grid forecast evidence](learned-afterqual-forecast.md) retains
40 after-qualifying races from 2025–2026. On the cohort with five declared input
repairs, the experimental forecast selects **28/40 winners (70%)**, compared
with **27/40 (67.5%)** for the first driver on the official grid and
**25/40 (62.5%)** for native simulation. Winner probability Brier loss is
**0.44001**, versus **0.50757** for the fixed grid reference and **0.47572** for
native simulation: 13.3% and 7.5% lower, respectively. Lower is better.

That is one additional correct winner over the grid on an already examined
sample. The native-relative uncertainty interval includes zero, and removing
the three largest probability gains reverses that comparison. These are
historical development results under a fixed dry assumption and current-era
physics, not authenticated forecasts issued before those races or independent
validation. The learned estimate reweights whole simulated outcomes; it does
not repair the native car/driver pace decomposition or strategy decisions.

Select **Confirmed-grid forecast (experimental)** for its supported official
grid workflow. **Native simulation** retains official-grid races and simulated
qualifying followed by a race, with custom weather and strategy controls. The
[forecast guide](learned-afterqual-forecast.md) explains scope, saved evidence
and offline verification.

The 18 October [qualifying-source repair](qualifying-classification.md) separates
disqualifications and valid lap evidence from qualifying rank and race grid. A
four-source historical replay preserves all 224 eligible events and changes 13
feature matrices. Retraining on those corrected inputs did not establish an
additional winner-forecast gain, so the packaged model artifact is unchanged.

The [native rain-strategy optimization](strategy-runtime.md) measured 17.35%
faster total runtime in the frozen damp workload and 5.22% in changing rain,
with dry runtime effectively unchanged. Complete outputs and cache statistics
match across all nine paired workloads; separate complete state/RNG comparisons
match for one seed in each scenario. This improves simulation speed.

The 14 October qualifying-first screen used fixed 70% current qualifying timing
and 30% recent team fastest-lap form. Across the same 40 development contexts,
winner Brier fell from 0.50366 to 0.48491, versus 0.50757 for the strongest grid
reference. That 4.5% reference gain is not robust: removing the three largest
improvements reverses it, and 2026 worsens. The candidate is not deployed.
Fastest-lap form can outweigh small qualifying gaps despite its nominally
smaller weight. The audit also found incorrect historical distances at Jeddah
and Imola; those original conditional results do not represent authenticated
historical races. Current-season circuit distances pass the separate audit.

Race setup now exposes an explicit official grid or simulated qualifying choice.
Official mode rejects unavailable grids and conflicting custom qualifying
weather; simulated mode races each trial from its own qualifying result.

## Completed improvements

Qualifying now uses a fixed correction from the three most recent earlier
eligible team-Q1 observations. It retains native differences between teammates
and changes qualifying laps only. Default current-season loading and winner
holdouts use the correction; the separate component diagnostic retains an
explicit uncalibrated baseline. Source rounds and fallback are recorded, saved
inputs retain the coefficient, and earlier snapshots load with zero correction.
Recent qualifying weight zero disables that qualifying correction. It leaves
race pace, tyre wear, mechanical reliability and driver skill unchanged; the
separate common dry race-clock correction below affects only race timing.

The [dry race-clock correction](dry-race-clock.md) reduces equal-event conditional
lap-time MAE from **3.8941 to 3.5949 seconds (7.7%)**, across 9,079 eligible laps
in ten events. Every event improves, including the one-lap age sensitivity. The
Canadian coefficient is fixed and only applies to fully dry modeled race laps
in 2026 rounds after its training race. The original six-event 10% gate failed;
a separately declared four-event expansion passed its 5% gates without refitting.
Actual native physics reproduces the measured correction. This conditions on
observed compounds and ages and does not establish better winner forecasts.

[Forecast recording and scoring](recorded-forecasts.md) now saves explicit
weather, frozen inputs, counts, probabilities, historical references and source
provenance before qualifying. It refuses to overwrite a file and never reruns
the model when scoring. Two 100-trial fixed-dry Singapore forecasts preserve
the previous and corrected clock before that event. Fresh results collected on
14 October score winner Brier at 0.59260 and 0.59120. The separately recorded
teammate estimate scores 0.54275. Public GitHub file hashes and dated push runs
verify publication before qualifying; the original files remain unchanged.
This is one pre-qualifying conditional-dry event, and supplies no evidence for
the requested after-qualifying grid-reference improvement. The
[score checkpoint and offline verifier](recorded-forecasts.md#singapore-scores-collected-on-14-october)
retain that distinction.

Native clocked rain planning now merges histories after the compound-use rule
is satisfied and scans safe forced stints when ordinary stops run out. Critical
weather still permits every compulsory replacement. On the frozen native
Singapore workload, repeated alternating processes reduced mean three-trial
runtime from **123.60 to 96.56 seconds**, or **21.9%**. All 12 full outcomes
matched exactly. Another 108 synthetic complete outputs matched across both
engines, dry and changing weather, physical stock and opening modes. See the
[benchmark method](strategy-model.md#completed-compound-history-and-safe-stints).

Compulsory [full-wet resumptions](wet-resumptions.md) now preserve physical stock,
prior wear, actual use, custom plans and replay in both engines. Race setup now
puts circuit, weather, trial count and Run first; **More race settings** exposes
optional controls. A visible race budget includes scenario and automatic-plan
comparisons. Validation opens hidden settings and focuses the invalid input.
Recorded tyre stock is readable rather than displayed as raw JSON.

Winner evaluation now freezes earlier constructor points and prior race-win
frequencies, records unavailable history and compares each reference with the
model on identical scored events. Offline rescoring verifies the recorded
formulas and evidence without fetching data or adding trials.

## Predictive evidence

The earlier qualifying-only snapshot covers 16 completed races, with 100
chronological trials per race, seed 42 and assumed fixed dry rainfall. Every
forecast uses earlier performance evidence and its own simulated qualifying.
The paired calibration check retains those exact inputs, event seeds and trial
counts, changes only the qualifying correction and runs 100 fresh trials for
each of the 15 non-cold-start events. The unchanged opening forecast is reused
explicitly. Comparisons use equal event weights; lower Brier loss is better.

| Reference | Common events | Calibrated model mean | Reference mean | Model minus reference |
|---|---:|---:|---:|---:|
| Equal chance | 16 | 0.78279 | 0.95381 | −0.17103 |
| Earlier constructor points | 15 | 0.76829 | 0.81275 | −0.04446 |
| Earlier race-win frequency | 15 | 0.76829 | 0.88555 | −0.11726 |

The original model scored **0.83125** over the same 16 events; the calibrated
loss is **5.8% lower**. The first race has no historical reference. Correcting
estimated finite-trial score bias changes the calibrated constructor comparison
to **−0.051825**. This does not establish significance or future predictive
skill. The earlier native model's round-16 sampled zero illustrates the large
event-level misses that aggregate scores can hide. Revised historical data,
known target entrant identities, assumed weather and static venue physics limit
this retrospective check. See [the original evaluation method](race-probability-evaluation.md)
and [paired calibration evidence](pace-evaluation.md#qualifying-only-calibration-validation).
Eight event scores improved and seven worsened; the opening score was unchanged.
These winner figures describe the earlier qualifying-only revision, before
the dry race-clock change.

The protocol fixed the already inspected Q1 candidate before collecting and
scoring additional Q2/Q3 labels. On 15 common events, Q2 normalized pace MAE
fell **11.0%** (230 observations) and Q3 **2.9%** (144 observations), beating
native laps and previous Q1. Position ranking worsened: Q2 rank MAE rose from
2.3217 to 2.3957 places and Q3 from 1.5833 to 1.8472. Teammate separation remains
too small. These correlated retrospective sessions justify the narrow deployed
qualifying correction, without establishing prospective pole or winner skill.
See the [protocol](model-improvement-protocol.md) and [pace validation](pace-evaluation.md).

Rechecking seven saved normalized tyre archives gives 6,096 eligible laps. Every
pooled compound-contrast range, including the driver-trend sensitivity fit,
crosses zero. This descriptive evidence does not justify tuning the live wear
coefficients. Mechanical reliability also remains an assumption distinct from
observed finish rates. See [tyre evidence](relative-tyre-wear.md).

The latest [experiment protocol](predictive-quality-protocol.md) records rejected
teammate-gap, persistent uncertainty, common qualifying-to-race package and
constructor-pooling candidates. The selected package change worsened eight-event
validation winner loss by **22.1%**; the constructor forecast head improved only
**1.3%**, below its 5% gate. Neither is deployed. The dry timing change does not
resolve teammate separation or winner calibration.
The dry clock's separate eight-event, 100-trial deployment regression check was
near-neutral: winner Brier loss **0.845200 to 0.841725**. It passed the declared
maximum 5% regression guard without establishing a winner improvement.

The [teammate race-point forecast](teammate-forecasts.md) retains each simulated
constructor win probability and allocates it using strictly earlier driver
race points with a fixed 25-point prior. Its seven-event 100-trial check improved
winner Brier loss 4.1% and failed the 5% gate. A single verified resolution at
400 trials passed: **0.863825 to 0.817194**, or **5.4% lower**, versus a
0.854928 constructor reference. The production implementation reproduces every
score and point-history cutoff. Four events improve and three worsen; much of
the gain comes from Belgium, and the remaining six together worsen 1.4% without
that event. This small retrospective sample establishes neither future skill
nor a race-position or strategy-counterfactual improvement. Native win counts
remain separate from reported estimates in the API, dashboard and exports.

The separate [fastest-lap](race-form-accuracy-protocol.md),
[completed-race](completed-race-form-protocol.md) and
[pit-service](pit-service-accuracy-protocol.md) policies failed their fixed
acceptance gates and were removed. Their implementation checks and partial
metric gains do not override those failures.

## Engineering and practical limits

The teammate forecast implementation passed a fresh source-bound **9,662-test**
native Windows Python 3.11 suite, with two skips and the existing TestClient
warning, plus **239** focused Python 3.12 checks. Real workers and second-trial
replay preserved the native race outcomes in both engines. The 22-driver live
API, packaged wheel and complete browser checks passed, including the calibrated
win display, separate raw counts and exports. No allocator override was used.
The earlier sporadic Windows native crash remains unresolved; these successful
checks do not establish a fix.

With calibration enabled by default, the source-bound implementation passed
**9,634 Python tests**, with two skips and the existing TestClient deprecation
warning, on native Windows Python 3.11. Another **504** focused checks passed on
native Python 3.12, including default loading, dry/wet and qualifying parity,
real worker grids, second-trial replay and exact planner decisions.
Published revisions run the full Python 3.11/3.12 Windows/Linux and browser CI
checks. The full browser suite passed with production HTML and fresh native
simulation fixtures, including
keyboard access, hidden-input recovery, 320/390/1440-pixel layouts, exports and
cancellation. The real API calendar, ratings and race run all returned 200 with
22 drivers and 11 constructors. The ten-trial dry race exported the fixed dry
race-clock coefficient and nonzero qualifying corrections, and executed through
real workers, using ordinary native physics
and no allocator override. Local source-bound runs retained unchanged files;
the final edits add documentation of these completed checks.

Speed remains a material limit despite the isolated 21.9% gain. At the earlier
checkpoint, with two workers and other validation running,
ten evolving-weather Singapore trials took **433.49 seconds** in the race API;
calendar and ratings took 0.22 and 5.05 seconds. This is not an isolated latency
benchmark and should not be extrapolated to other machines or worker counts.
A separately profiled single trial spent approximately 101 of 123.5 simulation
seconds in recursive rain-strategy solving, including nested work. Instrumented
times are slower than normal execution. The newer isolated comparison is
reported separately above; it is not a before/after API latency measurement.

Earlier sporadic Windows access violations remain unexplained. A fresh native
Python 3.11 browser-fixture run again ended with `0xC0000005` in rain-transition
planning. Its diagnostic is retained; a new native Python 3.12 fixture and browser
run passed without an allocator override. Successful native
tests, live runs and CI are useful evidence, but do not establish that this fault
is fixed. The large dashboard document and rules shared across two race engines
also make future changes costly; smaller internal modules should accompany a
specific behavior change with regression evidence.

## Next acceptance criteria

1. Validate on later unseen events using forecasts recorded before qualifying
   and explicit weather assumptions. Report ranking, pace and winner failures
   separately; this sample cannot establish prospective accuracy.
2. Improve teammate separation and position ranking only with a new frozen
   development/validation split. Do not retune against the additional Q2/Q3
   labels used to accept this candidate.
3. Retain native Windows reproduction evidence and investigate any recurring
   crash without masking it with an allocator workaround.

Local source-bound receipts, derived reports and profiling artifacts are under
the ignored `output/quality-milestone-2026-10-05/`,
`output/model-accuracy-practicality-2026-10-05/` and
`output/predictive-quality-2026-10-05/` directories. Reproduction commands
and snapshot identities are recorded in the linked evaluation documents; raw
live feed caches are not committed.
