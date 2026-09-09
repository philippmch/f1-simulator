# Mechanical reliability and observed retirement evidence

Updated 2026-09-09.

Live runs use a nominal mechanical reliability of 0.95 for each car and component.
This is a model assumption. It is not estimated from the fraction of cars that
finish. [Jolpica's status documentation](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/status.md)
explains that generic `Retired` records can include accidents; non-starts and
disqualifications also do not identify hardware failures. Turning all these
outcomes into mechanical failures, in addition to the simulator's separate
accident process, attributes unsupported causes to the observed results.

The loader retains the observed information in `DriverStats`:

| Field | Meaning |
|---|---|
| `team_finish_rate` | Finished/lapped result records divided by sampled team result records; `null` when no records are available |
| `team_result_count` | Denominator, including non-starts and other non-finishes |
| `team_finished_count` | Numerator of the observed finish rate |
| `team_reliability` | Nominal mechanical survival input used to create the car |
| `team_reliability_source` | `model_prior` for live loading, or `provided` for ordinary explicitly supplied statistics |

Each result belongs to its recorded constructor, including before a driver
transfers teams. The legacy personal `dnf_rate`, `sample_size`,
`current_season_starts` and `classified_finishes` fields remain descriptive
result-row summaries; their denominator can include non-starts. They are not
mechanical-failure labels or mechanical exposure estimates.

The loader does not fit component reliability from isolated status labels in
an otherwise incomplete cause feed. Identifying some engine failures would not
establish what caused the other generic retirements. Such calibration needs
cause coverage, actual exposure and treatment of competing retirement causes.
The prior is deliberately common across teams until that evidence exists.

Explicit `DriverStats.team_reliability` values still create configured cars,
subject to the existing conversion range of 0.7 to 1.0. Direct `Car` models
support the full 0–1 range and independent component values. Saved simulation
inputs retain their recorded car values for offline replay; replay still uses
the installed model implementation, so results can change across code versions.
The dashboard identifies live model priors in its car-rating table, and the live
CLI states this assumption. Ratings JSON includes each car's `reliability_source`.

## From nominal survival to lap failure probability

The mechanical process combines the configured car and component inputs:

`R = 0.55 × car reliability + 0.30 × mean component reliability + 0.15 × weakest component reliability`

These are blended inputs, not independent full-race survival events multiplied
together. To retain the existing preference for later failures, lap `l` of the
original scheduled distance `N` receives weight `w(l) = 0.85 + 0.35 × l/N`.
The sum is `W = 0.85N + 0.35(N + 1)/2`. The per-lap probability is

`p(l) = 1 − exp(log(R) × w(l)/W × stress × heat)`.

With neutral stress and temperature, the product of all `1 − p(l)` is exactly
`R`, regardless of race distance. A reliability of 0.7 therefore represents
30% nominal mechanical attrition, rather than the roughly 26% produced by
dividing 0.3 across many independent laps. Endpoint reliabilities of 1 and 0
give probabilities of 0 and 1 respectively.

The existing stress factor is `0.9 + 0.25 × track tire stress`, so tyre stress
0.4 is neutral. Heat adds the existing cooling-dependent factor from 45°C
upward. These factors multiply hazard: at constant stress and heat the full-race
survival is `R ** (stress × heat)`. They remain uncalibrated model assumptions.

Only laps actually run receive exposure. Shortened or lapped races retain the
original fuel/race distance as the hazard denominator, and fewer laps mean less
exposure. Chronological execution checks mechanical risk once per own lap, with
its existing pending-lap conditions. Race incidents can still end a car's race
first; actual simulated mechanical-DNF frequency is therefore affected by those
competing events and need not equal the nominal probability.

Analytic regression tests check full and partial survival across race lengths,
the endpoints, stress and heat responses, and unchanged random-draw ownership.
This verifies that the model implements its stated parameters; it does not
validate the 0.95 prior against real hardware failures.
