# Pit-service accuracy experiment — 7 October 2026

Starting revision: `ab89ea2`. Both race-form policies failed their declared
winner gates and have been removed. This is a new component hypothesis, not a
revision of those policies: the current model assigns every live constructor
the same 2.5-second ordinary stationary stop mean and 0.3-second standard
deviation, despite available official stationary timing observations.

## Fixed policy and source

Use only the current UTC season's official `PitStopSeries` stationary
`PitStopTime`, with the corresponding race `DriverList` identifying the
constructor. Never interpret `PitLaneTime`, OpenF1's deprecated `pit_duration`,
or a fastest-stop award as an ordinary stationary stop. Resolve races by the
current calendar's exact race date and require finalised, complete, matching
race-session metadata. Missing or ambiguous source identity means no estimate.

For a target race, use at most the three strictly earlier calendar races.
Keep only finite ordinary stop times in [1.8, 5.0] seconds; longer stops remain
scoring observations but cannot distort the ordinary-service fit. Deduplicate
driver-number/lap identities and reject conflicts. Require four ordinary
observations across at least two earlier races for each constructor. Otherwise
retain the existing prior.

Shrink the ordinary mean and variance toward the existing normal prior with
12 equivalent observations, using the first and second moments. Keep the
existing 1.8-second floor, 5% slow-stop branch and uniform 2–8-second delay.
Preserve all other model inputs, physics, weather, seeds and strategy rules.
Do not fit the prior strength, observation limits or failure-branch parameters
after inspecting scores.

## Frozen evaluation

Use current-calendar rounds 3–9 for development and 10–16 for validation.
Source responses stay in memory; retain only normalized observations, source
digests, cutoffs, input models and receipts. Predict each event before scoring
its own stationary observations. Do not use later events to fill a missing
earlier archive or resolve an earlier driver's constructor.

Score the actual native stationary-service distribution against every finite
positive published stationary stop, including long services. Use empirical
CRPS from 10,000 native draws per constructor with the same random stream for
both variants; report mean absolute error of expected service separately.
Equal-weight event means prevent events with many stops dominating the result.
This measures stationary-service accuracy, not full pit-lane loss, repairs,
optimal strategies or overall winner skill. Race outcomes used by previous
experiments remain retrospective; this component evidence cannot establish
prospective accuracy.

Require lower development CRPS before opening validation. Validation needs
seven events, at least 5% lower CRPS, and at least 5% lower expected-service
absolute error. Keep all event changes and excluded/unknown observations
visible. Do not retry another parameter set after a failure.

If the component passes, compare the real native 100-trial chronological race
forecasts on the same frozen validation inputs and seeds with only car pit
mean/std changed. Winner Brier and race/qualifying mean-position MAE must not
increase more than 5%. This is a deployment regression guard, not evidence of
improved winner prediction. Passing only the pit component does not establish
that the broader predictive-accuracy goal is finished.

An accepted change must reach default loading, API ratings and recorded inputs,
preserve older snapshot replay, pass worker/browser/package checks and publish
a new pre-event forecast without replacing existing records. Group any shipped
work into substantial commits and push directly to main.

## Results and rejection

The source collection resolved all 16 completed current-calendar races and
retained 452 normalized stationary observations. The seven development events
lowered mean CRPS from 2.428584 to 2.345927 and expected-service MAE from
2.523736 to 2.486075 seconds. The development CRPS gate passed without changing
the policy.

On the seven validation events, CRPS fell from **1.743198 to 1.637866** (6.0%),
and improved at every event. Expected-service MAE fell from **1.845729 to
1.806688 seconds** (2.1%), with regressions at rounds 11 and 14. The latter
misses the fixed 5% gate, so the combined validation failed. Reject this policy;
do not change its prior strength, bounds, sample rules or gate after the result.
No default model change or race-regression run is justified by this experiment.

The native Python 3.12 distribution checks used no allocator override and
retained unchanged executable sources. Frozen observations, source hashes,
per-event scores and receipts remain in the ignored
`output/pit-service-accuracy-2026-10-07/` directory. These results establish no
new shipped accuracy gain.
