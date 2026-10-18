# Experimental confirmed-grid forecasts

The learned forecast uses the official race grid as its strongest starting
signal, with bounded adjustments from qualifying times and earlier team and
driver form. It adjusts the probabilities of complete simulated race outcomes.
It does not change physical pace ratings, simulated races or strategy decisions.

Select **Confirmed-grid forecast (experimental)** in the dashboard's Forecast
control. It selects the official grid, qualifying-led pace, one fixed dry
scenario and at least 400 trials. Custom tyres, pit plans, weather, control
schedules, warmup assumptions and strategy comparisons are unsupported. A
contradictory request fails before simulations. Simulated qualifying followed
by a race remains available through Native simulation and the Race grid control.

The forecast requires completed Grand Prix qualifying, a complete fresh
published grid and no target or later race results. Sprint qualifying alone
does not open this window. The run must finish before the scheduled race start
to be recorded. Missing evidence or unsupported native winner strata cause an
explicit failure, rather than an invented probability or a silent fallback.

```powershell
python examples/simulate_race.py --race 18 --race-grid official --weather-mode fixed_rainfall --simulations 400 --forecast-policy learned_afterqual_v1 --export
python examples/record_race_forecast.py --stage post_qualifying --race 18 --scenario dry --trials 400 --parallel --workers 8 --forecast-policy learned_afterqual_v1 --output output/forecasts/after-qualifying-learned.json
python examples/score_recorded_forecast.py output/forecasts/after-qualifying-learned.json
```

Use the applicable upcoming current-season race. The dry assumption is specified
by the user; it is not a forecast of the weather. As in other recorded
forecasts, fixed rainfall pins atmospheric change probability to zero. Changed
temperatures, humidity, rain or initial surface wetness are not admitted by this
policy.

## What changed and why

The native calculation combines team Q1 medians, separately scaled team-form
signals and a smaller teammate-relative driver adjustment. Team averages can
therefore overwhelm a driver's individual qualifying evidence. In the retained
Canada and Abu Dhabi 2025 inputs, it rates the pole sitter about 0.2 seconds per
lap slower than McLaren and selects the wrong winner. The learned model
restores the grid's priority in those cases. This repairs the reported forecast;
the underlying pace decomposition remains an identified limitation. The
[retained input archive](../evidence/postqual-pace-policy-inputs-2026-10-16.json.gz)
contains the qualifying and native model inputs for these comparisons.

The model is a fixed 100-tree grid residual with a separate nonfinish head.
It was selected on 2021–2023 data and evaluated by training each historical
forecast only on earlier completed races. It uses 16 qualifying/grid/history
features and three nonfinish-history inputs. No model is fitted during an app
request. The packaged artifact trains through 2026 round 15; its creation time,
training boundary, source seals and exact parameters accompany every estimate.
A target must follow all its training events. This frozen serving artifact is
distinct from the expanding per-event retraining protocol in the historical
study; its training age is retained rather than hidden.

When the learned favourite is already among the native favourites, the native
distribution is retained. Otherwise the fixed reverse-KL projection constrains
winner-probability ratios to the learned favourite and reweights whole native
paths. Native zero support stays zero; no-classified-winner mass is preserved.
The analytic model's all-nonfinish allocation has a different classification
convention, so its driver scores are used only as classification-score ratios.
Exact ties use official grid order and then driver ID, with the no-classified-
winner category last. This same primary-pick rule is saved and used in reports.

The conditional identity rule can switch abruptly when native favourites are
close. A stable primary winner does not imply stable probability estimates.
The confirmation therefore retains the same-trial 32/128/400 distributions and
separate multinomial resampling diagnostics, including changes in that trigger.

## Estimates, counts and saved evidence

Win, classified podium, mean ordinal position and operational nonfinish
estimates use the **same path weights**. Nonfinish means status other than
finished; a late nonfinish can still be classified. Integer wins, raw race
outcomes and their Wilson sampling intervals remain separate. The learned
estimates are point estimates without calibrated confidence intervals.

JSON exports retain the artifact, ordered feature snapshot, raw feature-source
evidence, complete minimal joint outcomes, weights, support diagnostics and
effective sample size. HTML and a separate forecast CSV show all four estimates.
The complete native driver/car/track/weather/start context, seed and trial count
are bound before simulation and checked before attaching or loading the forecast.

Schema 5 records retain the raw winner counts and the separate learned estimate.
Loading reconstructs features from saved qualifying/history and reconstructs
inference, projection and marginals offline. Scoring uses the saved probabilities
and compares them with all three frozen grid references; it does not fetch new
model inputs or rerun simulations. Seals verify content consistency, not
independent publication. Replay of a single native race remains a native trial,
not a reconstruction of an ensemble's weighted estimate.

Optional [primary qualifying classification metadata](qualifying-classification.md)
now separates disqualified or untimed entrants from valid timing evidence.
Target and earlier form features use the same validity rules while saved source
contexts retain the original provider fields. Legacy contexts reproduce their
original features. This source repair leaves the frozen model artifact and the
historical forecast evidence below unchanged.

## Historical development evidence

The confirmation retains all 40 events from 2025-2026, with each historical
model trained only on earlier completed races. It also evaluates a second
complete cohort with five declared input repairs: Jeddah and Imola distance,
Monaco and Qatar tyre rules, and Miami qualifying times. The cohorts share
35 events and are not independent samples.

| Cohort, 400 trials/event | Native correct winners | Learned | Grid first | Native winner Brier | Learned | Grid probability reference |
|---|---:|---:|---:|---:|---:|---:|
| Original 40 | 25 | 28 | 27 | 0.465015 | 0.438966 | 0.507567 |
| Corrected 40 | 25 | 28 | 27 | 0.475725 | 0.440013 | 0.507567 |

The correction improves winner Brier by 5.6% against the original native model
and 7.5% with corrected inputs, or 13.3% against the grid probability reference.
Both seasons improve against native. In the corrected cohort,
mean-position MAE improves by 0.00316 places, podium Brier by 0.000179 and
nonfinish Brier by 0.000030. In the original cohort, position MAE worsens by
0.00050 places and podium Brier by 0.000164; nonfinish Brier improves by
0.000044. Both cohorts pass the small materiality limits frozen before
execution. This does not establish improvement in every measure or season.

All complete forecast and bootstrap checkpoints preceded scoring. Eleven
model-event contexts retain genuine complete 400-trial joint outcomes. The
other 34 use archived 400-trial aggregates only because their correction is
mathematically the identity; missing joint paths are never synthesized. Six
regenerated original contexts reproduce their archived 400-trial winner,
position, podium and nonfinish counts exactly. All 45 contexts retain complete
same-trial 32/128 prefixes. The continuation ran 7,312 additional native races.
Independent audits reconstruct the scores, projection and production inference.

Both complete cohorts retain 28 correct picks at 32, 128 and 400 trials.
All 1,000 saved multinomial count draws for each of the 45 contexts retain
its primary winner pick, with no unsupported strata. These draws resample
native winner counts; they are not additional race simulations or confidence
bounds for real-world skill.

Probability stability is weaker than pick stability. Five contexts switch
between identity and correction in the resampling check. Japan 2025 is identity
in 73.6% of draws; corrected Qatar is identity in 19.7%. Rare identity draws
in Canada change Russell's probability by as much as 28.25 percentage points.
Miami changes from identity at 32 trials to correction at 128/400, moving
Verstappen's probability from 34.375% to 62.577%/61.049%. This is why the mode
remains opt-in and experimental, with native simulation the default.

The corrected cohort's paired native-relative winner-Brier gain has a 95%
event-bootstrap interval of -0.00206 to 0.08488; the original cohort's is
-0.01824 to 0.07794. Removing the three largest gains reverses both mean gains.
One extra correct winner over grid does not establish future superiority.
These are development results on already examined races, not untouched
validation. Historical source cutoffs are logical, not authenticated publication
vintages. Inputs retain current-era physics and fixed-dry assumptions; the five
repairs do not reconstruct every historical regulation or actual weather
condition. The existing substantial-improvement standard in
[post-qualifying forecasts](post-qualifying-forecasts.md) remains separate from
these descriptive gains.

The [portable evidence archive](../evidence/historical-learned-afterqual-2026-10-16.json.gz)
retains frozen per-event models, features, minimal native outcomes or explicit
identity aggregates, observed labels, saved resampling counts and source seals.
Recompute inference, weights, both complete-cohort scores and stability
diagnostics offline, without feeds, fitting or simulations:

```powershell
python examples/verify_historical_learned_afterqual.py
```
