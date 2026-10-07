# Teammate forecast experiment — 7 October 2026

Starting revision: `ab89ea2`. Rejected race-form and pit-service experiments
remain undeployed. A diagnostic of the frozen native forecasts separates the
constructor's chance of winning from the allocation between its drivers. At
several events the native model assigned over 80% to the correct constructor,
but assigned the eventual winner less than half of that constructor's wins.
Earlier qualifying residuals alone do not represent all race performance.

## One fixed hypothesis

Preserve every native constructor win probability and the no-classified-winner
probability. Within a constructor, allocate its win probability according to
each modeled driver's strictly earlier current-season Grand Prix points,
with a prior of 25 points per modeled teammate. These are reported race
points, not constructor points, qualifying times, simulated wins or sprint
points. No fitted blending weight or alternative allocation rule is searched.

Count only uniquely resolved earlier race entries with an explicit matching
constructor and finite, nonnegative reported points. Each modeled teammate
must have at least two eligible entries. If a teammate's points are missing or
conflicting, or the constructor has insufficient coverage, retain its native
driver allocations. Do not transfer a driver's old-constructor points to a
new constructor. Preserve cold starts and the native simulation itself.

The 25-point prior is one Grand Prix win per seat under the current scoring
system. Freeze it before collecting this experiment's point history; do not
change it in response to the development or validation results. This forecast
head is distinct from the rejected constructor-pooling head: it changes no
constructor probability and introduces no external constructor forecast.

## Evaluation and gates

Use the same frozen native chronological race outcomes, inputs, seeds and
fixed dry weather already recorded at 100 trials per event. Use rounds 3–9
for development and 10–16 for validation. Require lower equal-event winner
Brier loss in development before opening validation. Validation needs seven
events, at least 5% lower equal-event winner Brier loss, and lower loss than
the recorded earlier-constructor reference on the identical cohort.

Report every event's change, raw native counts separately from calibrated
probabilities, source hashes and strictly prior point history. Compute the
finite-ensemble Brier correction for this fixed linear allocation using its
actual transformation; do not apply the native formula to changed counts.
This introduces no new independent races. The season outcomes have already
informed earlier experiments and the motivating diagnostic, so this remains
retrospective evidence, not independent or prospective validation.

One sampling-resolution run to 400 total trials may be declared before adding
trials if a near-boundary result warrants it. Preserve the original 100-trial
prefixes, policy, cohorts, sources and gates. No policy retuning or repeated
budget increases after viewing validation.

An accepted head must be used in the actual default forecast workflow, the
API/dashboard and newly recorded forecasts; changing only an evaluator is
insufficient. Keep native win counts identifiable and distinguish Monte Carlo
sampling uncertainty from model uncertainty. Preserve the old recorded
forecasts and saved input behavior. Verify real native execution, workers,
replay, browser exports, the wheel and the full checks. Publish a new sealed
pre-event forecast, group related changes into substantial commits and push
directly to main. Failure leaves the native forecast in production.

## Development result

The one fixed policy reduced seven-event mean winner Brier loss from 0.695800
to 0.636460 (8.5%). Finite-ensemble-adjusted loss was 0.688716 to 0.634235;
the identical-cohort constructor reference scored 0.785054. Four event scores
improved and three worsened. The original native outcomes were reused, not
rerun or relabeled. Fresh point-history source hashes match the earlier
frozen provider snapshot exactly. The development lock records the complete
policy source and point-history hashes before validation is opened.

## Initial validation and one sampling resolution

The 100-trial validation reduced mean winner Brier loss from 0.833229 to
0.798712 (4.1%), below the fixed 5% gate. Adjusted loss was 0.825339 to
0.795336; the constructor reference scored 0.854928. Three event scores
improved and four worsened. Record the initial gate as failed.

Resolve the near-boundary result once at **400 native trials per event**,
with the same seven validation events, points, 25-point prior and gates.
The existing earlier experiment already ran native seeds 100–399 from these
exact baseline inputs. A separate verification confirmed identical complete
inputs, event seeds, original 100-trial outputs and unchanged physics sources
for every event. Reuse those verified 400-trial ensembles instead of adding
duplicate simulation work. Only the fixed new allocation and its scores are
computed; no additional real races or policy changes are introduced.

## Resolution result

The fixed 400-trial resolution passed: mean winner Brier loss fell from
**0.863825 to 0.817194** (5.4%), below the identical-cohort constructor
reference's 0.854928. Adjusted loss fell from 0.861840 to 0.816348. Four event
scores improved and three worsened. The policy and original 100-trial prefixes
were unchanged. No further sampling resolution is allowed for this candidate.

| Round | Native | Point allocation |
|---|---:|---:|
| 10 | 0.852575 | 0.453809 |
| 11 | 1.128900 | 1.125212 |
| 12 | 0.888375 | 0.929584 |
| 13 | 0.737575 | 0.844160 |
| 14 | 0.826913 | 0.772618 |
| 15 | 0.681888 | 0.793495 |
| 16 | 0.930550 | 0.801476 |

Belgium drives much of the gain. Removing it makes the remaining six-event
mean 1.4% worse, so this small retrospective sample does not establish a
broad or statistically significant improvement. The declared gates pass on
the complete frozen cohort; this sensitivity and the original 100-trial
failure remain part of the evidence. No claim is made about future accuracy,
race-position accuracy or the accuracy of strategy counterfactuals.

The production helper reproduced all seven sealed scores and finite-trial
corrections. Fresh current-season loading reconstructed the same strictly prior
point history and allocations. Detailed source identities, original native
run receipts, reuse verification, per-event scores and production validation
are retained under the ignored `output/teammate-forecast-2026-10-07/` directory.
