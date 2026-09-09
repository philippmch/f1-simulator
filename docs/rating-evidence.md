# How pace evidence becomes ratings

The live loader separates team pace from a driver's performance relative to
their teammates. These are model ratings derived from current-season results,
not measured intrinsic driver ability or car performance. The same assembly is
used by the [held-out qualifying evaluator](pace-evaluation.md), which supplies
only earlier-round performance inputs.

## Qualifying comparisons

The provider reports [Q1, Q2 and Q3 separately](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/qualifying.md).
Comparing one driver's Q3 time with another's Q1 can confuse a change in track
conditions with a pace difference. Rating calculations therefore retain the
session identity:

- **Team pace:** for each round, use Q1 if valid times cover at least 80% of
  the active roster. Otherwise try Q2, then Q3, with the same coverage rule.
  Each team's median is compared with the field median from that one session.
  If no session has sufficient coverage, that round supplies no qualifying
  team residual. A team can contribute its available driver's time without
  inventing a missing teammate's time.
- **Driver pace:** for each constructor in each round, use the latest session
  shared by at least two resolved teammates. Compare their times with their
  median in that session. If one teammate has only Q1 and the other reaches Q3,
  their comparison uses Q1. Without a shared session there is no teammate
  residual for that group.

Historical rows retain their recorded constructor. A transfer does not make a
driver's former teammate a member of their current team. Only drivers resolved
to the requested roster enter these comparisons.

Session times must be finite and positive. Generic `bestTime` or `time` fields
do not identify a session and are excluded from qualifying rating residuals.
They may still establish that a qualifying result exists for form-window
coverage. Likewise, `qualifying_samples` counts selected qualifying rows; it
does not count usable same-session teammate comparisons. The evaluator's
observed labels and previous-Q1 baseline always require Q1 explicitly.

These comparisons cannot remove traffic, tyre choice, run timing or weather
changes within a session. They also cannot establish whether a driver used
the car's full available pace.

## Constructor points and missing evidence

Constructor points provide the existing team prior: points divided by the
largest constructor total, or 0.5 for every team when no points are available.
Target qualifying, recent race pace and recent qualifying each supply a signed
fractional residual relative to their event reference. Faster is positive;
matching the reference gives zero. Each bucket contributes its median residual
multiplied by the requested weight:

```text
team score = constructor prior
           + target qualifying residual × track weight
           + recent race residual × form weight
           + recent qualifying residual × qualifying weight
```

A missing bucket adds no residual and does not rescale the constructor prior.
In particular, removing neutral qualifying evidence cannot promote a team
above a stronger constructor. A zero weight disables the corresponding pace
signal. The field's scores retain the existing linear conversion to ratings
between 0.76 and 1.0; an entirely tied field uses 0.86.

The constructor prior remains the dominant team signal at default weights.
These corrections do not tune its strength, the rating spread or the lap-time
coefficients. Constructor points also reflect finishing reliability and race
outcomes, while the race pace input is fastest-lap average speed rather than
an estimate of sustained stint pace. Both are limitations for future
calibration, not evidence that the ratings isolate true car performance.

## Verification

Controlled tests check that uniform shifts between sessions and advancement
alone do not create driver or team gaps, while real same-session differences
remain usable. They also cover incomplete evidence, session fallback, missing
teammates, transfers, disabled weights and preservation of the former team
formula when every team has the same evidence availability.

The qualifying evaluator measures outcomes separately from those correctness
invariants. Its dated results and limitations are recorded in
[the evaluation report](pace-evaluation.md#current-season-snapshot-9-september-2026).
Improved input comparability does not by itself establish better prediction.
