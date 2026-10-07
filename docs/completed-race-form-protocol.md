# Completed-race form experiment — 7 October 2026

Starting revision: `ab89ea2`. The fastest-time ingestion repair failed its fixed
winner gate at both 100 and 400 trials and was removed. Current race form has
no signal when the provider omits average fastest-lap speed. Test one new source
of form: comparable completed-race pace, with no fitted weights or multipliers.

## Fixed policy

For each strictly earlier race, require a uniquely identified classified winner
with a finite positive total elapsed time in the published `Time.millis` field
and a positive completed lap count. Use only classified finishers who completed
that same lap count and whose elapsed time is no smaller than the winner's.
Their pace is completed laps divided by elapsed seconds. This comparison uses
equal race distance and includes tyre, pit, traffic and strategy effects; it
is observed race form, not an estimate of intrinsic car or driver speed.

Use this metric only when at least two active drivers provide comparable values.
Otherwise retain the legacy published average-speed metric if at least two
active drivers supply it. Never mix metrics within one event. Missing, lapped,
retired, malformed or incomparable observations remain unavailable. Do not infer
elapsed times from gaps or invent race distances. Keep existing normalization,
form window, weights, skill scales, qualifying calibration and physics intact.

## Evaluation

Use rounds **3–9 for development** and **10–16 for validation**, with the same
in-memory provider snapshot for both model variants. Model inputs use only
earlier performance and constructor standings. Target identities and static
venue profiles remain the existing holdout assumptions. Keep target scoring
observations separate from model assembly and seal inputs and implementation.
These previously inspected season outcomes make the evidence retrospective.

Use 100 actual native chronological trials per model/event, existing event seeds
derived from seed 42, fixed dry rainfall and automatic strategy. Require lower
development mean winner Brier loss before validation is opened. Lock the one
policy; do not search variants or change its rules after validation.

Validation requires seven events, at least **5%** lower equal-event winner Brier
loss and no more than **5%** worse qualifying or race mean-position MAE. Report
all event changes, finite-trial diagnostics and the identical-cohort constructor
reference. If simulation noise leaves a near-boundary result, permit only one
predeclared resolution run to 400 total trials, preserving every 100-trial
prefix and retaining the same gates, source, models, events and weather. Record
any original gate failure and both results. This does not add independent races.

An accepted policy must run in default current-season loading, the API and
recorded forecasts; pass input, worker, replay, browser and release checks; and
publish a new sealed pre-event forecast. Keep existing records immutable. Group
delivery into substantial commits and push directly to main. A failed policy
remains undeployed and cannot count as achieving the accuracy goal.

## Development result

Across seven events and 100 fresh trials per model/event, equal-event winner
Brier loss fell from 0.695800 to 0.656800 (5.6%). Adjusted loss was 0.688716 to
0.649841. Qualifying mean-position error improved from 2.1577 to 2.1558 places,
and race error from 3.8249 to 3.7684 places. The frozen development run passed
with unchanged sources and inputs. Do not tune this policy when opening the
round 10–16 validation.

## Validation result and rejection

The seven validation events completed with the sealed inputs and implementation
unchanged. Winner Brier loss increased from **0.833229 to 0.838314** (0.6%);
finite-ensemble-adjusted loss increased from 0.825339 to 0.830216. Qualifying
mean-position MAE increased from 2.6597 to 2.6664 places, and race MAE from
3.3663 to 3.4281 places (1.8%). This fails the fixed 5% winner-improvement gate.
The result is not near that boundary, so no sampling-resolution run is used.

Reject and remove this policy. Preserve its frozen sources, inputs, per-event
scores and native run receipts under the ignored
`output/completed-race-form-2026-10-07/` directory. The published default model
and existing forecasts remain unchanged. No accuracy gain is claimed from
either rejected race-form experiment.
