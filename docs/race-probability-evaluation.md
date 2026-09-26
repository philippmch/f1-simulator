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

Use `--engine chronological` to select the alternative execution engine,
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
results with their recorded budgets and assumptions; this diagnostic supplies
neither a significance test nor a fitted probability correction.

Only the current UTC season is supported. Data is fetched for each invocation
without a persistent provider-feed cache. Saved reports contain derived
evaluation evidence and model inputs, not a reusable raw-feed cache.

## Initial whole-season snapshot, 26 September 2026

The standard engine, default dry scenario, three-round form window, and seed 42
completed 100 trials for each of 15 available races (1,500 trials total). All
15 events had a scoreable observed winner. Reproduce the collection with:

```powershell
python examples/evaluate_race_probabilities.py --all --trials 100 --seed 42 --fetch-budget 180
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
