# Held-out race-winner probabilities

This diagnostic tests the simulator's winner probabilities against completed
current-season races. It builds each target's models from earlier evidence,
simulates qualifying and the race, and compares the forecast with the observed
winner. It does not fit coefficients or change live simulation settings.

```powershell
python examples/evaluate_race_probabilities.py --race 14 --trials 100 --seed 42
python examples/evaluate_race_probabilities.py --all --trials 100 --seed 42
```

Choose a completed race or explicitly request all completed targets. Race
simulation is more expensive than the deterministic qualifying diagnostic;
start with a small trial count. The report records the trial budget, engine,
random-stream policy, per-event seed, weather assumption, and input provenance.
The same event uses the same seed whether evaluated alone or with other events.
Increasing the trial count preserves the earlier trial seeds. The default is
100 trials per event and seed 0; both the per-event and total selected-event
budgets are capped at 10,000 trials. Runs execute serially by default.

For larger checks, distribute each event's trials across worker processes and
show progress without changing the report's JSON output:

```powershell
python examples/evaluate_race_probabilities.py --all --trials 100 --seed 42 --parallel --workers 4 --progress
```

Events still execute in round order. Within an event, changing the worker count
preserves trial seeds and result order; it does not change the physics or widen
the modeled uncertainty. `--workers` requires `--parallel` and accepts 1–61,
a portable bound for supported Windows and Unix process pools. Omitting it lets
the runner choose from the available CPUs, capped by the event's trial count.
Process startup and separate worker caches can outweigh the benefit for small
runs. The report records the requested execution settings.

`--progress` writes collection phases, per-event trial counts, and whole-run
counts to stderr. Repeated trial updates are limited to once every two seconds,
with event boundaries and final counts always shown. Stdout contains only the
complete JSON report. Targets excluded for insufficient input coverage consume
no trials; their skipped budget is distinguished from executed trials.

Press Ctrl-C to request cooperative cancellation. The evaluator checks between
collection steps, and the simulator checks while races and strategy searches
run. Active provider requests remain subject to their HTTP timeout. Worker
processes finish cancellation cleanup before the command exits with status 130;
cancelled runs emit no partial JSON report. Collection and simulation failures
also leave stdout empty. Full fetch provenance, including any HTTP retries,
is captured before simulation and included in successful reports.

Python callers can pass `parallel`, `max_workers`, `progress_callback`, and
`cancel_requested` to `evaluate_race_probabilities`. The progress callback runs
in the calling process and receives a separate dictionary for each update:
`phase`, `events_total`, `events_completed`, `collection_events_completed`,
`trials_requested`, `trials_total`, and `trials_completed`, plus current event
identity and trial counts when available. Counts are unknown until collection
resolves them. `trials_total` counts executable trials after coverage exclusions.
Callback errors propagate and stop the run; cancellation raises
`SimulationCancelled` rather than returning a partial report.

`MonteCarloRunner.run` also accepts a parent-process `progress_callback` with
two integer arguments: completed trials and total trials. It first receives
zero, then a monotonically increasing count as completed results are collected.
Parallel completion notifications can arrive before earlier seeded trials;
returned results always remain in seed order.

Chronological execution is the new-run default. Use `--engine standard` to
select the synchronous model or `--engine chronological` explicitly,
`--scenario light_rain` or `--scenario heavy_rain` for wet assumptions, and
`--form-races 0` to disable recent form inputs. Live collection defaults to a
120-second total fetch budget; `--fetch-budget` accepts 1–300 seconds.
The evaluator collects and validates inputs for every selected event before
starting simulations. A later collection failure therefore spends no simulation
trials, and Monte Carlo computation cannot use up the time available to fetch
another event's standings. Collection still shares one bounded fetch deadline.

## What the forecast knows

The shared holdout builder uses the same historical evidence rules as
[qualifying pace evaluation](pace-evaluation.md): target qualifying supplies
entrant identities and team assignments, while eligible form rounds and
constructor standings precede the target. Venue physics comes from the static
configuration. Target lap times, starting grid, finishing results, and later
performance are excluded from model inputs. Entrants without a usable target
Q1 time remain in the race forecast.

Driver order is canonicalized by identifier before simulation, so the order of
the target qualifying feed cannot assign different random draws to entrants.
Each trial simulates its own qualifying session. This is a retrospective
pre-qualifying-style forecast conditional on the actual entrant roster, using
today's revised feeds and model code; it does not reconstruct a forecast that
was published before the event.

The default weather assumption is dry with fixed rainfall. Wet scenarios also
hold rainfall fixed while surface wetness evolves. These are specified model
conditions, not observations of the target race's weather. Tyres and strategy
use the simulator's default automatic policies and unlimited race tyre sets.

## Outcomes and coverage

The evaluator requires near-complete overlap between target qualifying and race
result identities, then exactly one classified position-one result resolving
to a forecast entrant. Missing lower-place rows do not automatically invalidate
an otherwise covered event with a known winner. Unscoreable targets retain an
explicit reason and coverage counts. Collection failures and conflicting
identities fail the evaluation instead of producing a partial report.

The overlap gate requires at least 80% of the larger qualifying/result entrant
count. It can therefore admit an incomplete qualifying roster. Read the reported
coverage counts: probabilities and the equal-chance baseline are conditional on
the modeled entrants, and a winner outside that roster makes the event unscored.

Each simulation trial contributes to exactly one outcome: a classified winner
from the full entrant roster, or `no_classified_winner`. The latter is a separate
category, not discarded or redistributed across drivers. Classification follows
the same rule as the simulator's win statistics, including eligible classified
retirements. It is distinct from the race-distance report's finished-winner
definition.

Probabilities and intervals in this report use fractions from zero to one.
Each probability is its outcome count divided by the complete trial count.
Individual 95% Wilson intervals describe Monte Carlo sampling uncertainty
conditional on the model and inputs. They are not simultaneous bounds or
confidence intervals for real-race predictive accuracy. A zero estimate means
that the outcome was not seen in these trials.

## Score and baseline

For each scored event, the multiclass Brier loss is

\[
  \sum_c (p_c-y_c)^2,
\]

where categories include every entrant and the no-classified-winner outcome,
and the observed winner has `y = 1`. All other categories have `y = 0`.
This uses the unhalved multiclass definition, ranging from zero to two; lower
is better. See the [scikit-learn scoring definition](https://github.com/scikit-learn/scikit-learn/blob/main/doc/modules/model_evaluation.rst#brier-score-loss).
The implementation calculates the sum directly and does not require sklearn.

The fixed baseline assigns each entrant probability `1 / entrant_count` and
assigns zero to no classified winner. It is a minimal equal-chance reference,
not a strong expert forecast. Score deltas are model minus baseline, so negative
values favor the model. Aggregates weight scored events equally and retain
scored and excluded event counts. A small set of outcomes cannot establish
calibration or future superiority; driver categories within one event are not
independent observations.

Scores use the empirical Monte Carlo probabilities. Finite trial counts add
sampling error to those probabilities and their squared-error scores. Compare
results with their recorded budgets and assumptions. The following diagnostic
quantifies finite-trial score noise without fitting a probability correction.

### References informed by earlier performance

Each new fold also freezes two simple historical references before the target is
simulated or scored. Their cutoff is `target_round - 1`, using the same earlier
constructor standings and modeled entrant identities as the holdout assembly:

- `constructor_points_share_v1` divides each represented constructor's share of
  earlier points equally among its modeled drivers. It normalizes over represented
  constructors, so the reference is conditional on the modeled roster. Standings
  points can include sprint points.
- `prior_race_wins_share_v1` assigns each modeled driver their unsmoothed share of
  resolved earlier race winners in that roster. Unresolved winners and winners
  outside the modeled roster are recorded as excluded evidence. A zero share
  means no recorded earlier win, not that a future win is impossible.

Both references assign zero probability to no classified winner. Missing prior
standings, missing modeled constructors, zero total represented points or no
resolved earlier winners make the corresponding reference unavailable: its
probabilities and score are `null`. In particular, the opening event has no
usable history. We do not substitute an equal-chance forecast for missing data.
These fixed formulas are not fitted to target outcomes. They are more informative
comparisons than equal chance, but neither is an expert forecast accounting for
the current grid, weather, strategy or changes in competitiveness.

Per-fold `baselines` preserve the policies, cutoffs, earlier evidence,
probabilities, availability reasons and recomputed scores. Top-level
`baseline_comparisons` reports model and reference mean Brier scores on their
**identical scored events**, with equal event weighting. The selected, scored and
unpaired event counts make missing history visible. Model-minus-reference deltas
are negative when the model scores better. The adjusted delta uses the same
common events for which the model's finite-trial correction is available and
reports that count separately; deterministic references need no correction.

Offline rescoring rebuilds these references from recorded evidence and rejects
inconsistent probabilities, policies, cutoffs or future-round evidence. This
checks internal consistency, not the authenticity of the original provider data.
Older saved reports without historical references remain supported; rescoring
does not invent missing evidence.

### Finite-trial score diagnostics

Every scored event now also includes `score.mc_adjustment`. It reports the
estimated finite-trial bias of the empirical Brier score, an adjusted score and
its delta from the same exact equal-chance baseline. This follows the
[finite-ensemble Brier correction in Ferro's *Fair scores for ensemble forecasts*,
sections 2.2–2.3](https://empslocal.ex.ac.uk/people/staff/ferro/Publications/ferro2013.pdf).
The correction applies when trials are independent samples from a fixed winner
distribution. It removes the expected excess score due to estimating that
distribution with finitely many trials; it does not improve the forecasts.

For `N > 1`, with empirical category frequencies `p_hat[c]`, the reported
correction and adjusted score are:

```text
estimated_empirical_score_bias = (1 - sum(p_hat[c]^2)) / (N - 1)
adjusted_brier_score = empirical_brier_score - estimated_empirical_score_bias
```

Every category contributes, including `no_classified_winner`. For example,
six wins by A, three by B and one no-winner outcome in ten trials, scored against
an observed A win, give an empirical loss of 0.26, a correction of 0.06 and an
adjusted loss of 0.20. The two-driver equal-chance baseline remains exactly 0.50;
the adjusted delta is −0.30. A deterministic baseline needs no simulation-budget
correction.

The implementation evaluates the equivalent mean pair loss directly from integer
counts, avoiding cancellation near zero. With `y` the fixed observed category,
the pair kernel is `h(X,Z) = I(X=Z) - I(X=y) - I(Z=y) + 1`. It is zero when either
trial gives `y`, two when both give the same other category, and one otherwise.
The resulting adjusted loss lies between zero and two. Two different trial
budgets estimate the same underlying loss in expectation under the stated
sampling assumption; one observed estimate can still be far from that loss.

`mc_standard_error` is a conditional plug-in estimate for the adjusted loss.
For a category distribution `p`, let `zeta1` be the variance of
`p[X] - I(X=y)` and `zeta2` the variance of `h(X,Z)` for independent draws. The
variance of the average pair loss is:

```text
4 * (N - 2) / (N * (N - 1)) * zeta1
+ 2 / (N * (N - 1)) * zeta2
```

The diagnostic substitutes the observed category frequencies for `p` and reports
the square root, recording method `multinomial_plugin_u_statistic_v1`. The second
term retains sampling variation when the first-order term is zero. This plug-in
estimate is not itself unbiased and supplies no confidence interval or
significance test. It can miss outcomes not observed in the trials. With only one
observed category, the standard error is explicitly unavailable rather than
reported as zero. With only one trial, the score adjustment is unavailable too;
the empirical score remains recorded. Reasons distinguish these cases.

Aggregates retain the original empirical scores and add the mean adjusted loss,
mean estimated finite-trial bias and mean adjusted baseline delta. Each adjusted
event receives equal weight and its contributing count is recorded as
`adjusted_events`; events with one trial contribute to empirical summaries only.
No aggregate standard error assumes independence across reused events.

Python callers with complete winner counts can use `score_winner_counts` from
`f1sim.analysis.race_probability_scores`. Counts must be nonnegative integers
with at least one total trial; booleans, floating-point counts, unknown observed
drivers and invalid identifiers are rejected. The probability-only
`score_winner_probabilities` function retains its existing result.

Existing saved evaluations can be rescored entirely offline:

```powershell
python examples/rescore_race_probabilities.py output/race-probabilities.json > output/race-probabilities-rescored.json
```

The command checks the declared trial count, roster, category counts and matching
empirical probabilities before emitting any JSON. It preserves the recorded
forecasts, simulation inputs and provenance, and adds the source file's SHA-256
digest. It neither downloads current feeds nor runs new trials. Saved exclusions
stay excluded; this check cannot establish the truth of original outcome labels
or the validity of a saved physical model. Duplicate event rounds, duplicate JSON
keys and inconsistent evidence fail. The input file remains unchanged. Python
callers can use `rescore_saved_winner_evaluation` from
`f1sim.analysis.race_probability_evaluation` for the same count checks.

These diagnostics concern Monte Carlo sampling with fixed inputs and a fixed
observed outcome. They do not quantify model error, parameter uncertainty,
weather uncertainty, real-world calibration or the uncertainty of future races.
Choosing or tuning a model after inspecting adjusted scores still requires fresh
evaluation evidence.

Only the current UTC season is supported. Data is fetched for each invocation
without a persistent provider-feed cache. Saved reports contain derived
evaluation evidence and model inputs, not a reusable raw-feed cache.

## Initial whole-season snapshot, 26 September 2026

The standard engine, default dry scenario, three-round form window, and seed 42
completed 100 trials for each of 15 available races (1,500 trials total). All
15 events had a scoreable observed winner. Reproduce the collection with:

```powershell
python examples/evaluate_race_probabilities.py --all --trials 100 --seed 42 --fetch-budget 180 --engine standard
```

Mean Brier loss was **0.8174**, against **0.9538** for the equal-chance baseline;
the mean event delta was **−0.1363**. These are empirical probabilities at the
stated trial budget, not an uncertainty-adjusted score or significance result.

| Round | Modeled entrants | Observed winner | Estimated winner probability | Model Brier loss |
|---|---:|---|---:|---:|
| 1 | 19 | RUS | 0.03 | 1.0050 |
| 2 | 22 | ANT | 0.36 | 0.7706 |
| 3 | 22 | ANT | 0.39 | 0.6390 |
| 4 | 22 | ANT | 0.55 | 0.3356 |
| 5 | 22 | ANT | 0.53 | 0.3624 |
| 6 | 22 | ANT | 0.41 | 0.4768 |
| 7 | 22 | HAM | 0.02 | 1.4262 |
| 8 | 22 | RUS | 0.43 | 0.5324 |
| 9 | 22 | LEC | 0.03 | 1.4580 |
| 10 | 22 | ANT | 0.19 | 1.1136 |
| 11 | 22 | NOR | 0.00 | 1.4184 |
| 12 | 22 | NOR | 0.03 | 1.3212 |
| 13 | 22 | ANT | 0.46 | 0.3952 |
| 14 | 20 | ANT | 0.45 | 0.4520 |
| 15 | 22 | RUS | 0.44 | 0.5550 |

Every event had 22 result entrants. The qualifying feeds for rounds 1 and 14
therefore supplied incomplete modeled rosters, admitted under the documented
coverage gate. No trial produced a no-classified-winner outcome. Zero estimates
mean unobserved in 100 trials: for example, the round-11 winner's individual
95% Wilson sampling interval is approximately 0–0.037, not proof of impossibility.

The lower aggregate loss coexists with substantial event-level misses. This
small retrospective sample uses an assumed dry scenario for every target and
overlapping training windows. It neither establishes calibrated real-world
probabilities nor justifies fitting a correction to these 15 outcomes. No
ratings, strategy settings, or probability coefficients were changed from
these scores.

The run used Python 3.11.9, NumPy 2.4.6, and Pydantic 2.13.5. Saved input
snapshots recorded simulation source fingerprint
`7a1986db791a42ea73936d3ea1c7c846d665c79e6b30990f7a8cc23efa942a15`.
Current provider revisions and runtime changes may alter a later run.

## Chronological whole-season check, 4 October 2026

A fresh run at revision `b794938` completed 100 trials for each of the same
15 available completed races. It used the chronological engine, default dry
assumption, three-round form window and seed 42: 1,500 trials in total. All
events had a scoreable observed winner. Each saved input snapshot matches that
revision's simulation source fingerprint
`8164c6492b3d7fb7ca8aa39ba9b0b900a79a0aaadfce7a81694f592e95acaec2`.

Mean multiclass Brier loss was **0.8026**, versus **0.9538** for the equal-chance
baseline; the mean event delta was **−0.1512**. Rounds 1 and 14 again had
incomplete modeled rosters of 19 and 20 entrants respectively. No trial produced
a no-classified-winner outcome.

The aggregate still hides substantial misses: the observed winners in rounds
7, 9, 11 and 12 received probabilities of 0.05, 0.05, 0.01 and 0.02. The lower
aggregate loss is not a calibration result. These are retrospective outcomes
under assumed dry conditions, and no new completed race was available beyond
round 15. The September standard-engine snapshot used earlier code; comparing
its score with this run does not isolate the engine's effect. No live ratings
or probability coefficients were fitted from this check.

Offline rescoring of that same 15-event snapshot gives a mean estimated
finite-trial score bias of **0.0065865**, an adjusted mean Brier loss of
**0.7960135**, and an adjusted mean delta of **−0.1577505** against the unchanged
equal-chance baseline. All original winner counts, forecasts, saved model inputs
and source URLs were retained. The original empirical mean remains 0.8026.
These numbers use the same 1,500 already-recorded trials and observed outcomes;
the lower adjusted loss supplies no new evidence of predictive improvement.

### 5 October snapshot with historical references

The new chronological run scored all 16 completed targets, with 100 trials each
(1,600 total), seed 42, four process workers, the default three-event form window
and fixed dry rainfall. Earlier evidence was collected before simulation;
neither baseline formula was fitted or selected from the resulting scores.

```powershell
python examples/evaluate_race_probabilities.py --all --trials 100 --seed 42 --engine chronological --parallel --workers 4 --progress --fetch-budget 180
```

| Comparison | Common scored events | Model mean Brier | Reference mean Brier | Model minus reference |
|---|---:|---:|---:|---:|
| Equal chance | 16 | 0.831250 | 0.953813 | −0.122563 |
| Earlier constructor points | 15 | 0.819987 | 0.812755 | +0.007232 |
| Earlier race-win frequency | 15 | 0.819987 | 0.885549 | −0.065563 |

Both historical references are unavailable at the opening event. The corresponding
finite-trial-adjusted mean deltas on the same 15 events are **+0.000875** against
constructor points and **−0.071920** against earlier race wins. The adjusted
all-event model mean is 0.824697. These diagnostics do not establish statistical
significance or an advantage over constructor points.

Rounds 1 and 14 model only 19 and 20 entrants; the other rounds model 22. The
observed winners at rounds 1, 7, 9, 11, 12 and 16 received empirical probabilities
of 0.03, 0.05, 0.05, 0.01, 0.02 and 0.00 respectively. The round 16 zero is zero
wins in 100 trials, with a conditional 95% Wilson upper bound of 0.03699.
The fixed dry assumption does not reproduce that event's rain-affected race
conditions, described in the [FIA race report](https://www.fia.com/news/f1-verstappen-wins-dramatic-bahrain-grand-prix-malaysia-ahead-antonelli-and-hamilton).
This mismatch limits interpretation; it does not establish which model change
would improve the forecast. No live coefficients were changed from these scores.

The derived report is locally saved as
`output/quality-milestone-2026-10-05/live-evaluation.json`, SHA-256
`991c9e218ca356fed1f51500a8530cc32f4621f8ffd8235941ddd0be69902616`.
Its simulation source fingerprint is
`0be91808485992d123f2b4a41d7df42b85abdb57afc16a25c6f6c3dd24830bed`,
with Python 3.11.9, NumPy 2.4.6 and Pydantic 2.13.5. The fetch timestamp is
`2026-10-05T10:29:15.750863Z`. Python sources remained unchanged throughout the
run; concurrent dashboard and documentation edits did not alter simulation.
Offline rescoring preserved the original report bytes, counts, model inputs and
historical references and reproduced every aggregate score without new trials.
