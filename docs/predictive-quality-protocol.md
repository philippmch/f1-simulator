# Predictive quality protocol — 5 October 2026

Starting revision: `325eaae`. The previous qualifying correction improved
relative pace but worsened Q2/Q3 ranking and retained almost no teammate gap.
The objective is better driver ordering, teammate separation and race-winner
forecasts, with reproducible evidence and the production model actually using
the accepted change.

## Development and validation boundary

Use current-season rounds 2–8 for development and rounds 9–16 for the locked
validation. These are chronological retrospective partitions of revised data.
Earlier work inspected aggregate scores and some individual outcomes across
this season. This split therefore tests a new policy without tuning on its
validation scores; it cannot be called independent or prospective validation.

Each forecast may use only performance strictly before its target round.
Validation events can supply history for subsequent forecasts, as they would
in a real rolling forecast. Known target identities and static venue physics
remain the existing holdout assumptions. Target timing and race results never
enter a forecast. Record source rounds and the weather assumption.

Select at most one candidate using development Q1 scores. After selection,
freeze the policy and its source digest before calculating validation scores.
Do not search new coefficients, history windows or variants in response to
validation failures. A failed candidate remains undeployed with its results
recorded. Any later experiment requires a separate development boundary.

## Candidate family

Retain the existing three-event team-Q1 median and replace its compressed
native teammate residual with a shrunk historical teammate residual:

1. Within each eligible earlier event, use Q1 times from at least two drivers
   entered by the same constructor. Normalize each driver's difference from
   that team's median by the event's field Q1 median.
2. For each target driver, take the median of those residuals over the existing
   three-event history window. Match both historical driver and constructor;
   do not transfer a residual from a different constructor.
3. Blend this estimate with the native residual using `n / (n + k)`, where `n`
   is the number of usable earlier events. The predefined candidate values for
   `k` are **1, 3 and 6**. Missing driver evidence retains the native residual.
4. Center the target team's resulting residuals to preserve its predicted
   median. Retain existing cold-start, invalid-history, custom-physics and
   coefficient-bound fallbacks. Apply the result to qualifying only.

Choose the candidate with the lowest equal-event development Q1 rank MAE.
Break ties using relative pace MAE, then the larger `k`. Require development
rank and pace improvement over the deployed team-only policy before selection.
This family changes driver separation, not race pace, tyre wear or reliability.

The first family failed the development rank gate for all three priors; no
candidate validation scores were calculated. Its Q1-only gap signs were worse
than the native signal, which also uses recent race and qualifying evidence.
Continue development with the native within-team residual multiplied by **2,
4 or 8**, retaining the same team correction and centering rule. Select once
using the same development Q1 rank/pace gates, breaking a final tie in favor of
the smaller multiplier. Do not use
validation scores to select this multiplier. Preserve the rejected family in
the audit results and leave it out of production.

Development found better rank MAE at each multiplier, with relative pace MAE
between 0.03% and 1.69% worse. For this family, permit at most the same **2%**
pace regression already specified for validation, while requiring strictly
better ranking. This makes the development gate reflect the ordering objective
and the declared validation tradeoff. Record this amendment before scoring any
candidate on rounds 9–16; do not relax any validation gate after seeing it.

## Acceptance

Report Q1, Q2 and Q3 on identical observed-driver cohorts for each variant,
including the deployed policy, uncalibrated model and previous-Q1 reference.
Report event-weighted and observation-weighted rank/pace errors, teammate gap
errors, sign agreement, coverage and per-event failures.

Before deployment, require eight scored validation events in Q1/Q2, at least
six in Q3, and:

- lower Q1 and Q2 rank MAE than the deployed policy;
- lower pooled Q2/Q3 rank and teammate-gap MAE;
- no more than 2% regression in Q3 rank or in any session's relative pace MAE;
- at least 5% lower validation race-winner Brier loss than the deployed model,
  on the same frozen inputs, event seeds and 100-trial budgets;
- lower validation winner loss than the earlier-constructor-points reference.

Run actual native race simulations for the paired winner comparison; do not
replace sampled outcomes with a post-hoc score adjustment. Report finite-trial
uncertainty and event-level regressions. These gates establish improvement on
this retrospective sample, not future predictive skill or significance.

Production acceptance also requires no target/future leakage, bounded saved
coefficients, old-snapshot compatibility, exact replay, worker parity, custom
physics fallback, current-season loading and appropriate Python/browser/API
checks. Group related changes into substantial commits and push to `main`.

## Rejected ordering candidate

The selected multiplier 8 failed the locked validation. Q1 rank MAE rose from
3.0471 to 3.2235, Q2 from 2.3689 to 2.5410, and Q3 from 2.1795 to 2.5897 places.
Pooled Q2/Q3 teammate-gap MAE also rose from 0.3221 to 0.3417 percentage points.
Do not deploy this change or retune its multiplier against these scores.

## Pace uncertainty experiment

Use a separate rolling estimation boundary: each target estimates uncertainty
only from errors in earlier event forecasts, never from its own outcome. The
ordering experiment above is rejected and its production code removed. Keep
the deployed mean pace and qualifying correction.

Before scoring winner forecasts, fix the uncertainty policy as follows:

- Build team residuals from normalized Q1 team medians. Earlier team forecast
  errors compare each observation with the median of its three preceding
  eligible team observations.
- Build driver residuals in the latest session shared by at least two same-team
  entrants. Earlier driver errors compare these with the median of three prior
  same-driver, same-constructor observations.
- Use `1.4826 * median(abs(error))` for a robust zero-centered error scale. Pool
  earlier errors as the sparse-data reference and blend each team's or driver's
  variance with that reference using `n / (n + 5)`. Require at least five pooled
  errors. Bound standard deviations at 2% of the reference lap; otherwise keep
  zero uncertainty and record unavailable evidence.
- Sample one shared team effect and centered within-team driver effects per
  trial, retaining those effects through qualifying and every race lap. Use
  independent identifier-based streams so physics random draws stay unchanged.
  Clip effects at four standard deviations and retain the lap-time floors.
- Record the policy, scales, source rounds and sampled-input replay contract.
  This transfers relative qualifying uncertainty to race pace as an explicit
  model assumption; it does not establish measured race-lap uncertainty.

Evaluate all scoreable rounds 5–16 with the existing frozen mean inputs, seeds,
100 trials and weather assumptions. Require at least twelve scored events,
at least 5% lower Brier loss than the deployed fixed-pace model, and lower loss
than the earlier-constructor-points reference. Report all event-level changes
and the finite-trial score correction. Do not vary the scale formula, pooling,
clipping or history window in response to these winner scores. This season
has already informed earlier experiments, so describe this as retrospective
rolling evidence and record a genuinely pre-event forecast before claiming
prospective skill.

The fixed uncertainty experiment failed its gates: mean winner Brier loss over
rounds 5–16 rose from **0.786867 to 0.845933** (7.5% worse), also exceeding the
constructor reference at 0.835691. Native execution used 100 fresh trials for
each of twelve events, unchanged mean inputs and no allocator override. Keep
automatic uncertainty estimation disabled; do not retune it against these scores.

## Shared team pace experiment

Fix one structural candidate before its winner evaluation: move the team's
median deployed qualifying correction into the shared car pace term and retain
each driver's remaining qualifying-only correction. This leaves deterministic
qualifying predictions unchanged to floating-point tolerance and applies the
observed team package correction to race laps. Set uncertainty scales to zero
for this comparison, and retain native driver skill, wear and reliability.

Use rounds 2–8 as development and 9–16 as validation. Do not search transfer
coefficients: the candidate uses the same relative car pace term in both lap
calculations. Require lower development winner Brier loss than the deployed
model before scoring validation. Validation requires eight events, at least 5%
lower winner Brier loss than the deployed model and lower loss than the earlier
constructor reference. Report per-event failures, finite-trial effects and the
unchanged qualifying check. Known season outcomes and previous experiments
still make this retrospective evidence; do not call it independent validation.

The shared-package assumption is different from transferring qualifying
forecast-error variance to race laps. These experiments are scored separately.

For the shared-package experiment, fix one qualifying driver candidate before
development scoring: estimate same-driver, same-constructor residuals from the
latest qualifying session shared by teammates in each of the three selected
earlier events. Use their median with weight `n / (n + 3)`, retaining the native
residual otherwise and recentering the target team. This removes race-form
evidence from the qualifying driver estimate. It differs from the rejected
Q1-only candidate. Do not search its prior or window. Include this driver change
only if development Q1 ranking improves with at most 2% pace regression; if
included, the original Q1/Q2/Q3 validation rank, gap and pace gates also apply.

That shared-session driver candidate also failed validation: Q1/Q2/Q3 rank
MAE rose to 3.1765/2.3934/2.4359 places, and pooled teammate-gap MAE rose to
0.3360 percentage points. It is removed from production. Retain the deployed
qualifying driver estimate in the package experiment.

Before evaluating its development winner scores, define two package variants:
the shared team mean alone and that same mean with the fixed earlier-error
uncertainty formula. The latter aligns the uncertain car's mean with the
historical team forecast whose errors estimate its variance; the earlier
failed uncertainty experiment retained the original constructor-derived race
mean. Do not tune the variance formula or a transfer coefficient. Choose the
lower-loss variant on development rounds 2–8, requiring improvement over the
deployed model. Freeze it before evaluating rounds 9–16. Apply the same winner
validation gates and record all failures.

Also freeze final qualifying positions for scoring. Compare the candidate's
trial-mean grid positions with qualifying-only runs of the frozen deployed
inputs using identical trial seeds. Rank observations within the modeled roster
when the historical feed has an incomplete entrant list. Report that coverage
and each event's position error; do not use these labels to fit the car mean or
variance. This checks the actual full qualifying forecast rather than treating
observed Q2/Q3 entrants as a forecasted complete field.

Development selected the shared mean alone: mean winner loss was **0.620029**
versus **0.680400** for the deployed model. The joint uncertainty variant scored
**0.791829** and worsened full-grid position error. Remove its production code
before validation, leaving the selected mean policy and coefficients unchanged.
Preserve the development package and source lock, and seal the cleaned candidate
package separately before any validation score. Use the candidate's actual
default loading in regression checks; enable it in the published version only
if the fixed validation gates pass.

The shared-mean candidate failed: validation winner loss rose from **0.845200**
to **1.031625**, versus **0.858470** for constructor shares. Its grid predictions
were unchanged. Remove the candidate's production code and retain the deployed
qualifying-only model. Its successful development result did not transfer to
the later races.

## Forecast calibration experiment

Test a forecast head separately from race-physics changes. Preserve the native
simulation outcomes and counts. Before target scoring, combine their empirical
winner probabilities with that target's frozen, earlier-constructor-points
reference: `p = (1 - w) * p_simulation + w * p_constructor`. The no-classified-winner
category receives the same simulation weight; its historical reference is zero.
If the historical reference is unavailable, retain the simulation forecast.
This changes the emitted forecast probabilities, not the loss formula.

Use a separate development boundary, **rounds 2–7**, with **rounds 8–16** for
validation. Select one weight from the predefined **0.25, 0.5, 0.75** using lowest
development mean winner Brier loss, breaking ties toward the smaller weight.
Require development improvement before selection. Freeze the chosen weight,
input digests and forecast formula before scoring validation. Validation requires
nine events, at least **5%** lower loss than the deployed simulation forecast,
and lower loss than the earlier-constructor reference. Do not choose a runner-up
or change weights after viewing validation.

Use the existing actual 100-trial native forecasts and their frozen historical
references on identical cohorts and seeds. The race inputs and simulation engine
are unchanged; this forecast-head experiment requires no new simulated outcomes.
Report raw simulation counts separately from calibrated probabilities, each
event's failures, and the finite-trial bias: the simulation score-bias estimate
is multiplied by `(1 - w)^2`. Monte Carlo ranges describe only sampling of the
simulation component conditional on the historical reference. The revised-data
season has already informed several experiments; neither this changed split
nor the new forecast head constitutes independent validation. Publish the
result as retrospective evidence and record a new pre-event forecast for the
first prospective check.

The selected constructor weight 0.25 lowered validation loss by only **1.3%**
(0.827422 to 0.816714), falling short of the fixed 5% gate. After finite-trial
bias correction the difference was smaller. Do not promote it or select another
weight against these validation outcomes. Retain the simulation forecasts.

## Conditional green-lap clock experiment

Use the existing version-1 official archive observations to test absolute race
lap times separately from winner probabilities. Fit at most one common relative
race-clock offset on the Canadian event (round 5). Evaluate it on the six later
archived events: Britain, Belgium, Hungary, Italy, Spain and Azerbaijan. Neither
this existing archive nor this season is independent evidence. Target compound,
observed stint age and green/zero-rainfall eligibility condition this component
test; they are not predicted by a pre-event forecast. Zero rainfall alone cannot
establish a fully dry surface, and traffic and driver intent remain confounders.

Compare each eligible observation with the median deterministic lap prediction
over the complete frozen pre-target model roster, using the archived compound,
stint lap age (including known prior wear), race lap and static venue profile.
The field-median prediction does not assert a mapping from archive race numbers
to permanent driver identities. Use fixed dry model weather and the normal fuel
and wear curves. Skip unknown prior wear; do not invent ages for it.

Freeze the median Canadian `(observed - predicted) / reference_lap` offset before
calculating any later-event error, bounding it at +/- 5%. Do not tune separate
venue, driver, tyre, fuel or weather coefficients. A production change needs at
least six validation events, at least 10% lower equal-event absolute lap-time
MAE, and no event whose MAE worsens by more than 10%. Repeat scoring with a
uniform one-lap age shift as a timing-boundary sensitivity check. Keep the
conditional scope and per-event misses visible. If it fails, remove it; a
diagnostic or improved documentation alone is not an accuracy improvement.

The clock candidate reduced equal-event conditional lap-time MAE from **4.5240
to 4.2000 seconds** (7.2%) and improved every event. With the one-lap shift, MAE
fell from 4.4884 to 4.1680 seconds. Both missed the predeclared 10% aggregate gate.
That six-event test did not justify deployment. It also exposes substantial absolute
timing error that the relative qualifying scores did not measure.

Retain the rejected coefficient for one separately preregistered expansion:
collect the previously unused Monaco, Barcelona, Austria and Netherlands timing
archives. Freeze the same Canadian coefficient; do not refit it on either the
six earlier validation results or these new observations. Before collection,
require **four new events**, at least **5%** lower equal-event MAE in that new
sample and the combined ten-event sample, and no new event with more than 10%
regression. Require those same aggregate/regression gates under the one-lap age
shift. This expanded evidence is required for a smaller timing improvement;
the original six-event 10% gate remains recorded as failed. This expansion is
still a conditional retrospective clock check, not proof of winner accuracy or
independent future skill. A dry-only correction must preserve qualifying,
wet-weather behavior, old snapshots, native prepared-lap parity and workers.

The four additional events improved by 8.9% without a coefficient refit; the
combined ten improved by 7.7%, including the one-lap age sensitivity. Every event
improved. This passes the separately declared expanded 5% gates; the original
10% gate remains failed. The production correction is limited to zero rainfall
and zero modeled surface wetness, in 2026 rounds strictly after Canada. The
actual prepared physics must reproduce the frozen offset result. Before
publication, also run a winner-regression check on the eight existing round
9–16 forecasts using the same frozen inputs, seeds and 100 trials each. Require
mean winner loss not to rise by more than 5%; this is a deployment regression
guard, not a new winner-improvement test or an independent sample.

The eight-event guard passed: winner Brier loss was 0.845200 before and 0.841725
after. The difference is effectively neutral and does not establish better
winner forecasting. Native actual physics reproduced the ten-event clock MAE;
full tests, saved zero-correction outcomes, workers, replay, browser and live API
checks passed. The sporadic Windows native crash remains unresolved.

## Future evidence

Provide a way to save a forecast before an event and score that same immutable
forecast afterwards, recording timestamps, inputs, weather and source digests.
Do not label a forecast prospective if timing cannot be proved. Later outcomes
are needed to establish prospective skill; do not manufacture that evidence
from today's completed races.
