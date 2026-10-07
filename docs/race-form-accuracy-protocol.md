# Race-form accuracy repair — 7 October 2026

Starting revision: `ab89ea2`. Fresh current-season result evidence contains
352 driver results across 16 completed events. Average fastest-lap speed is
absent in all 352 rows, while fastest-lap times are available in 339. The current race-form adapter
requires speed, so it silently discards usable race pace. This experiment
repairs missing signal ingestion without fitting weights or pace multipliers.

## Fixed candidate

Choose one metric for each earlier race: retain published average speed when
at least two active-driver rows supply a finite positive value; otherwise use
the reciprocal of a finite positive fastest-lap time when at least two rows
supply it. Within one race never mix kilometres per hour and inverse seconds.
Rows missing the selected metric remain unavailable. Existing same-event and
same-constructor normalization cancels the common distance/time scale.
Keep the form window, model weights, rating scales, qualifying calibration,
race engine, weather assumptions and tyre/reliability parameters unchanged.

This still uses fastest laps, which are affected by tyres, fuel, traffic and
driver intent. Repairing data availability is not proof that this is the best
pace signal. Promote it only if the actual forecast improves under the fixed
comparison below; report misses as well as gains.

## Comparison and gates

Collect one current-season source snapshot in memory. Construct both models
with the same earlier results, qualifying, constructor standings, target
identities and static track profiles. Keep target performance out of model
inputs. Save derived model inputs and scoring observations separately, together
with source hashes and cutoff metadata. Do not persist raw live feed caches.

Use rounds 2–8 for development, then lock the implementation before scoring
rounds 9–16. This is a structural repair with one candidate, not a parameter
search. These season outcomes informed previous experiments, so the evidence
is retrospective and cannot be called independent or prospective.

Use 100 native chronological trials per variant and event, seed 42 with the
existing event-seed derivation, fixed dry rainfall and automatic strategy.
Compare actual simulated winner distributions, qualifying mean positions and
race mean finishing positions on identical modeled-driver cohorts. Lower Brier
loss and rank MAE are better. Development requires lower winner loss than the
existing model before validation is opened. Validation requires eight events,
at least 5% lower equal-event winner Brier loss, and no more than 5% worse pooled
qualifying or race position MAE. Also report the frozen constructor-points
reference on the same events, and finite-trial winner-score corrections.

Do not change the metric rule, weights or gates in response to validation
failures. If the repair fails, retain the finding as a documented limitation
and continue accuracy work with a separately frozen hypothesis.

An accepted repair must reach the default current-season loader, APIs and
recorded forecasts, pass unit/worker/replay/browser checks, and produce a new
forecast published before the upcoming event. Preserve the two existing
Singapore records. Commit related work in substantial pieces and push to main.

## Development result

The frozen seven-event native run used 100 fresh trials for each model and
event. Equal-event winner Brier loss fell from 0.678743 to 0.626257 (7.7%).
Finite-trial adjusted loss fell from 0.671833 to 0.619798. Mean qualifying
position error rose from 2.3457 to 2.3510 places, and race position error from
3.8574 to 3.8855 places. Canada, Barcelona and Austria had worse winner scores;
the other four improved. The development winner gate passed. Source and input
hashes were unchanged. Lock that source and policy before opening validation;
these development numbers alone do not justify deployment.

## Initial validation and Monte Carlo resolution

The eight-event 100-trial validation lowered winner Brier loss from 0.841725
to 0.806800 (4.1%), missing the 5% gate. The adjusted values were 0.833939 and
0.799066. Qualifying position error was 2.5129 to 2.5191 places; race position
error was 3.4641 to 3.4155 places. Five winner scores improved and three worsened.
Keep the original gate result recorded as failed.

The shortfall from the gate is 0.007161 Brier units. The plugin standard-error
upper bound for the mean paired difference, using the individual score standard
errors and Cauchy-Schwarz, is approximately 0.05654. It is an estimated sampling
diagnostic, not a confidence interval or proof of improvement. Resolve this
finite-trial uncertainty once, with **400 trials per model and event** across
all eight validation events. Retain each original 100-trial prefix and add
seeds 100–399. Freeze that budget before running any additional trials.

Do not change the policy, inputs, event selection, weights, weather, loss or
5% winner/5% position gates. No more budget increases for this candidate after
the 400-trial result. Report both gate results and all event changes. This
increases simulation resolution; it does not add independent real races.

The 400-trial check also failed the winner gate. Mean Brier loss was 0.869341
before and 0.861634 after (0.9% lower); adjusted loss was 0.867383 to 0.859687.
Qualifying position error was 2.5087 to 2.5158 places; race position error was
3.4679 to 3.4233 places. The fastest-time fallback was removed from production.
Its passing implementation tests and improved data availability do not override
the accuracy gate. The private candidate forecast remains unpublished.
