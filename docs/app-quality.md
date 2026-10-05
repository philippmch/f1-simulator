# App quality checkpoint — 5 October 2026

Continued development makes sense as a simulation and strategy experiment tool.
The current checks support rule consistency, reproducibility and a working user
workflow. The forecasts have not established an advantage over a simple
constructor-points reference. Pace calibration and weather-planning speed should
take priority over further rule expansion.

## Completed milestone

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

The fresh current-season snapshot covers 16 completed races, with 100
chronological trials per race, seed 42 and assumed fixed dry rainfall. Every
forecast uses earlier performance evidence and its own simulated qualifying.
The comparisons below use equal event weights; lower Brier loss is better.

| Reference | Common events | Model mean | Reference mean | Model minus reference |
|---|---:|---:|---:|---:|
| Equal chance | 16 | 0.83125 | 0.95381 | −0.12256 |
| Earlier constructor points | 15 | 0.81999 | 0.81275 | +0.00723 |
| Earlier race-win frequency | 15 | 0.81999 | 0.88555 | −0.06556 |

The first race has no historical reference. Correcting the estimated finite-trial
score bias changes the constructor comparison to **+0.000875**. This does not
establish a predictive advantage or statistical significance. The model assigned
the observed winner at round 16 zero wins in 100 trials; that is a sampled zero,
not proof of impossibility. Revised historical data, known target entrant
identities, assumed weather and static venue physics limit this retrospective
check. See [the evaluation method and snapshot](race-probability-evaluation.md).

On the same 15 qualifying folds and 319 driver observations, native ranking error
was **3.1097 places**, versus **3.1850** for previous Q1. Relative pace error was
**0.6477 percentage points**, versus **0.5162**. The earlier-team-Q1 candidate
scored 2.9592 places and 0.4680 percentage points, but improved driver ranking in
only seven of 16 events, tied three and worsened six. It remains an experiment;
no live rating transformation was promoted. See [pace evaluation](pace-evaluation.md).

Rechecking seven saved normalized tyre archives gives 6,096 eligible laps. Every
pooled compound-contrast range, including the driver-trend sensitivity fit,
crosses zero. This descriptive evidence does not justify tuning the live wear
coefficients. Mechanical reliability also remains an assumption distinct from
observed finish rates. See [tyre evidence](relative-tyre-wear.md).

## Engineering and practical limits

The exact implementation passed **9,506 Python tests**, with two skips and the
existing TestClient deprecation warning, on native Windows Python 3.11. The 108
winner-evaluation and scoring tests also passed on Python 3.12. The full browser
suite passed with production HTML and native simulation fixtures, including
keyboard access, hidden-input recovery, 320/390/1440-pixel layouts, exports and
cancellation. The real API calendar, ratings and race run all returned 200 with
22 drivers and 11 constructors, using ordinary native physics and no allocator
override.

Speed remains a material limit. With two workers and other validation running,
ten evolving-weather Singapore trials took **433.49 seconds** in the race API;
calendar and ratings took 0.22 and 5.05 seconds. This is not an isolated latency
benchmark and should not be extrapolated to other machines or worker counts.
A separately profiled single trial spent approximately 101 of 123.5 simulation
seconds in recursive rain-strategy solving, including nested work. Instrumented
times are slower than normal execution. No speed improvement is claimed here.

Earlier sporadic Windows access violations remain unexplained. Successful native
tests, live runs and CI are useful evidence, but do not establish that this fault
is fixed. The large dashboard document and rules shared across two race engines
also make future changes costly; smaller internal modules should accompany a
specific behavior change with regression evidence.

## Next acceptance criteria

1. Freeze a pace/rating candidate and validation protocol before additional
   outcomes. Compare it with the historical references on the same events,
   report per-event failures and weather assumptions, and require evidence beyond
   this already inspected sample before changing live ratings.
2. Reduce recursive rain-planning work while preserving legal choices, selected
   plans and seeded outcomes. Use independent strategy checks plus an isolated
   repeatable timing comparison before claiming a speed gain.
3. Retain native Windows reproduction evidence and investigate any recurring
   crash without masking it with an allocator workaround.

Local source-bound receipts, derived reports and profiling artifacts are under
the ignored `output/quality-milestone-2026-10-05/` directory. Reproduction commands
and snapshot identities are recorded in the linked evaluation documents; raw
live feed caches are not committed.
