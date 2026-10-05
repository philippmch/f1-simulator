# Conditional dry race-clock correction

The current-season loader uses one frozen common race-clock correction for
**2026 rounds after Canada (round 5)**. It adds
`reference_lap * 0.003905707495222259` to the race pace terms before weather
scaling and the numerical lap-time floor. It applies only when modeled rainfall
and surface wetness are both exactly zero. It does not change qualifying laps,
driver/car ratings, relative teammate pace or tyre degradation curves.

The coefficient was fitted once to 895 eligible Canadian laps: the median
`(observed_seconds - field_median_prediction) / reference_lap`, bounded to
plus/minus 5%. It is a common timing bias, not a causal fuel, tyre or traffic
estimate. Later observations never refit it. Other seasons and unidentified or
earlier rounds receive zero. Directly constructed tracks and old snapshots
default to zero; saved inputs explicitly preserve the applied coefficient.

## Measured result

The implemented native prepared-lap evaluator was checked against **9,079**
eligible observations from ten later events. Equal-event mean absolute error
fell from **3.8941 to 3.5949 seconds (7.7%)**. Every event improved. A uniform
one-lap tyre-age shift gave **3.8597 to 3.5634 seconds**. The actual implementation
matches the frozen offset calculation to within 0.000000000001 seconds.

The original six-event test improved by 7.2%, missing its declared 10% gate.
That failure remains recorded. A separately declared expansion collected four
previously unused archives without changing the Canadian coefficient. Its
5% gates required improvement on both the new sample and the combined sample,
including the age sensitivity, with no new event worsening by more than 10%.
The four new events improved from **2.9492 to 2.6873 seconds (8.9%)**. These
expanded gates passed. This is modest retrospective timing evidence after
several rejected model experiments, not independent confirmation.

| Event | Eligible laps | Native MAE, seconds | Corrected MAE, seconds |
|---|---:|---:|---:|
| Monaco | 1,093 | 2.4660 | 2.2086 |
| Barcelona | 987 | 2.1843 | 1.9447 |
| Austria | 1,108 | 2.8636 | 2.5942 |
| Britain | 817 | 3.5966 | 3.2451 |
| Belgium | 671 | 3.7269 | 3.3128 |
| Hungary | 1,211 | 6.5869 | 6.2823 |
| Netherlands | 690 | 4.2829 | 4.0017 |
| Italy | 863 | 5.4615 | 5.1413 |
| Spain | 978 | 6.2254 | 5.8661 |
| Azerbaijan | 661 | 1.5469 | 1.3523 |

The executable report is authoritative for the exact counts and per-event
values. Errors remain large; this correction alone does not make the simulation
a reliable real-world forecast.

The deployment regression guard used the eight existing round 9–16 forecasts,
with identical frozen inputs, event seeds and 100 new trials each. Mean winner
Brier loss was **0.845200 before and 0.841725 after**, passing the predeclared
maximum 5% regression guard. This near-neutral difference is not evidence of a
winner improvement: the sample was already inspected, and 100-trial sampling
noise is material. No coefficient or forecast head was changed after scoring.

## What this measures

Version-1 normalized official timing evidence supplies green laps with no
reported rain, unchanged slick compounds and no pit exposure. Unknown prior
wear is excluded. Stint age is known prior wear plus laps since the earliest
observed lap in that driver's stint. The age-plus-one calculation checks the
timing boundary. Zero reported rainfall does not establish a fully dry surface.

For each observation, the benchmark uses its actual compound, stint age and
race lap, assumed dry weather, static venue physics and the complete frozen
pre-target driver/car roster. It predicts the field median, without inventing
a mapping from archive car numbers to permanent driver IDs. Traffic, pace
management, driver variation and surface conditions remain confounders.
Observed compounds and ages condition this component check; a pre-event model
has not predicted them. Thousands of laps do not represent thousands of
independent events. **The 7.7% result is not an improvement in winner accuracy.**

## Reproduction

No network or fitting is involved in the offline evaluator. Repeat `--archive`
for the ten normalized archive files; the output contains input SHA-256 values,
per-event errors, age sensitivity and equal-event means.

```powershell
python examples/evaluate_race_clock.py --baseline output/model-accuracy-practicality-2026-10-05/calibrated-winner-evaluation.json --archive output/relative-wear-british-2026-09-27.json
```

The frozen baseline digest is
`d288b57978d42863d3ba181460dbb2f75cf339d27fcf473abf75f9cdf5dcfa0e`;
the Canadian training archive digest is
`0739f9ca1fae961692215cf24d6403beba0528dcdf45baa361eb0443530c109b`.
Local receipts, the coefficient lock, both gate results, new archives and the
actual-physics report are preserved under the ignored
`output/predictive-quality-2026-10-05/` directory. Earlier normalized archives
remain in `output/`. The repository keeps the evaluator and method; raw live
feed caches are not committed.

See the [complete experiment protocol](predictive-quality-protocol.md) and
[forecasts recorded before qualifying](recorded-forecasts.md) for the route to
later evidence of prospective skill.
