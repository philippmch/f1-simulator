# Accuracy and practicality protocol — 5 October 2026

The objective is to improve the existing model's measured accuracy and runtime.
The starting implementation is `840249b`. These choices are recorded before
collecting or scoring the additional qualifying-session observations below.

## Runtime

Use the native chronological engine, ordinary Python allocation and an isolated
saved-input workload. Freeze a fresh 22-driver Singapore input snapshot, then
run seeds 42–44 sequentially in one process. Report the first cold run and later
runs separately. Compare complete race, qualifying, weather and event output
digests before interpreting speed differences. Cover dry, rain transitions,
prescribed weather and finite stock with reproducible synthetic workloads too.

Reduce repeated rain-planning work without changing the legal action set, paid
weather clock, compound-use rule, fitting fees, extensions or cancellation.
Independent enumeration and the existing generic path must agree with native
costs and choices. A runtime gain alone does not establish model accuracy.

## Pace accuracy

Keep the existing earlier-team-Q1 candidate fixed: medians from the three most
recent earlier eligible events, combined with the native within-team prediction.
Do not change its window, weights or transformation after seeing new scores.
The already inspected Q1 results are development evidence.

Collect Q2 as the primary additional validation and Q3 as confirmation. Build
each target forecast using performance data strictly before its round. Target
Q1 times, Q2/Q3 times and later results must not enter predictions. Score only
shared observed entrants and report coverage, session-specific exclusions and
weather assumptions. This is retrospective validation across different sessions
of known events, not independent prospective race validation.

The primary metric is normalized driver pace mean absolute error, comparing
candidate, native and previous-Q1 reference on identical observations. Report
driver and team ranking errors, per-event deltas and teammate gaps as well.
Require at least eight scored events in each session and improvement over both
references on the primary metric before considering a live pace change. Report
any ranking regression explicitly. Failure is not a reason to retune this
candidate using the validation labels.

A live change also needs replay, worker and extension compatibility checks, plus
a paired race-winner evaluation against the frozen historical references.
Report those winner scores without claiming prospective skill or changing
mechanical reliability and tyre wear from this experiment.

## Deployment decision

The fixed candidate passed the additional pace and paired-winner checks. Deploy
it as a qualifying-only correction, preserving the native race pace, driver
skill, cars, reliability and wear model. Q2 and Q3 validate qualifying spacing;
they do not justify transferring that correction to race laps. The existing
weather multiplier and lap floor still apply. Keep native inputs for cold starts,
invalid supplemental history, incompatible physics or out-of-bounds adjustments.
Setting the recent qualifying weight to zero disables the live correction.

The [quality checkpoint](app-quality.md), [pace validation](pace-evaluation.md#qualifying-only-calibration-validation)
and [runtime measurements](strategy-model.md#completed-compound-history-and-safe-stints)
record the results, including the ranking regressions and retrospective limits.
