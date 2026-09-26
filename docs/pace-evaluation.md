# Held-out current-season qualifying pace

The evaluator asks whether the current rating model orders qualifying pace
better than a simple previous-Q1 baseline when target performance is withheld.
It does not use the live simulator's current-strength snapshot for a past target.
The [race-winner probability evaluator](race-probability-evaluation.md) shares
these historical input rules and scores simulated race outcomes separately.

```powershell
python examples/evaluate_qualifying_pace.py --race 13
python examples/evaluate_qualifying_pace.py --race "Italian Grand Prix"
python examples/evaluate_qualifying_pace.py
python examples/evaluate_qualifying_pace.py --components
python examples/evaluate_qualifying_pace.py --form-races 5 --scenario light_rain
```

Only the current UTC season is supported. Omitting the race evaluates all
completed events with available result and qualifying feeds. Each invocation
fetches current provider data and prints one JSON document. The default total
fetch budget is 120 seconds; `--fetch-budget` accepts 1–300 seconds. Requests use
the live loader's rate limits, timeouts and response-size limits. The evaluator
does not retain a provider-feed cache. Collection failure prints an error on
standard error and emits no partial JSON report.

## What a fold knows

For target round `k`, the evaluator assembles fresh models using:

- Driver and constructor identities from target qualifying. Names, codes and
  team assignments are known inputs; target positions and lap times are not
  copied into the models. Entrants with no usable Q1 time remain in the roster.
- The latest eligible form rounds strictly before `k`. A round needs result and
  qualifying evidence for at least 80% of the target roster, as in live form
  loading. `form_rounds` records the chosen window; `--form-races 0` disables it.
- Constructor standings after round `k − 1`, with season and round validated in
  both response wrappers. Round 1 has no preceding standings and makes no
  round-zero request. The [provider's standings endpoint](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/constructorStandings.md)
  supplies the championship aggregate, including preceding sprint points;
  the evaluator does not reconstruct it by summing race results.
- Static venue configuration. A completed target's fastest lap and any already
  cached live track calibration are ignored.

The evaluator uses the same [rating assembly as live runs](rating-evidence.md),
including comparisons within qualifying sessions. Recent race and qualifying
weights are 0.3 and 0.2, and target-qualifying weight is zero. Constructor points
use the same prior. Form-window rounds and standings cutoff
are reported separately because the standings include more than the form window.
Live model caches, selected form-round provenance and current-roster selection
are not replaced by these evaluation snapshots.

Qualifying predictions use the existing lap model without random variation or
mistakes, choosing the fastest fresh compound under the selected fixed weather
scenario. The default is dry. This is an assumed scenario, not observed target
weather; rain, traffic, track evolution and different run timing can affect the
real Q1 measurements. Q1 is used because it precedes elimination and offers a
common session across the field. Q2/Q3 times never substitute for missing Q1.

## Metrics and denominators

Each fold lists predicted times, observed Q1 times and previous-Q1 times for
each entrant, with missing observations represented by `null`. Target qualifying
and result identities must have near-complete overlap; counts are retained in
the report. This checks internal coverage, not completeness of the provider.
Conflicting driver identities or constructor aliases fail collection.

| Metric | Definition |
|---|---|
| `rank_mae` | Mean absolute rank error; lower is better. Tied times receive average ranks. |
| `relative_pace_mae_pct` | Mean absolute difference between predicted and observed time divided by their respective cohort medians, expressed in percentage points. Lower is better. |
| `pairwise_concordance` | Fraction of differently timed observed pairs ordered correctly; higher is better. Predicted ties receive half credit and observed ties are excluded. |

Metrics require at least two usable Q1 observations. Otherwise the fold is
marked `insufficient_q1_times`, with undefined scores left `null`. Undefined
pairwise concordance also remains `null` when every observed time is tied.

The baseline carries Q1 times from the latest eligible earlier round. Baseline
and model are compared on exactly the same drivers with usable times in both
target and preceding Q1. These are `paired_comparison` scores; the fold's main
`model` scores can cover more drivers. Normalization and ranking are recomputed
on each scoring cohort, so paired and whole-field scores can differ. This avoids
penalizing one method for entrants that the other method could not score.

Aggregate rank and pace errors are weighted by scored driver observations;
pairwise concordance is weighted by comparable pairs. The report retains these
denominators and the number of scored folds. Driver observations repeat across
races and are not independent samples for a confidence interval. No statistical
significance, predicted winning odds or optimal race strategy is claimed.

## Separating team and driver evidence

Use `--components` to compare three fixed model variants on the same held-out
entrants: constructor points alone with neutral drivers, the current team model
with neutral drivers, and the full current model. These three variants use the actual
noise-free qualifying lap simulator and the same weather and compound choices.
The diagnostic does not fit weights or change the live simulation.

The neutral driver uses skill 0.92 (the rating formula's zero-residual
intercept), consistency and tyre management 0.968, wet-skill modifier 0.9936,
and overtaking skill 0.90. The latter attributes are fixed diagnostic
assumptions based on the loader's default dispersion, so driver evidence
cannot re-enter through wet-weather pace. This is a residual ablation, not
the loader's no-evidence skill fallback of 0.90. Constructor-only cars use
the existing rating builder with all pace-evidence weights set to zero;
constructor normalization is not reimplemented in the evaluator.

Alongside field-wide scores, it measures team-median pace and gaps between
teammates. Team medians use only drivers with usable target Q1 times, with the
same drivers included for every variant. A team with one observed driver can
contribute a team median but cannot supply a teammate comparison. Teams with
more than two observed drivers contribute every distinct teammate pair.
Team ranking and relative-pace scores require at least two observed teams and
normalize across those team medians. A single observed team can still provide
usable teammate-gap evidence.

Teammate gap error compares signed predicted and observed gaps, each divided
by its own scoring cohort's median lap time, in percentage points. The report
also gives mean absolute predicted and observed gaps: small errors alone can
hide a model that predicts almost no separation. Ordering accuracy excludes
observed ties and gives half credit to predicted ties. Missing comparisons
remain undefined, rather than becoming zero error.

The previous-Q1 baseline is compared on the common cohort with both target and
previous Q1 observations. Its cohort can be smaller than the main component
evaluation. Aggregate team errors are weighted by team observations, teammate
gap errors by teammate pairs, and ordering accuracy by non-tied observed pairs.
These are repeated observations across events, not independent samples.

Component differences show where the current predictions come from; they do
not establish intrinsic driver ability or causal car performance. In particular,
an overall field score can improve while teammate predictions deteriorate.
Changes chosen after inspecting these results need fresh validation before
claiming better predictive accuracy.

### Experimental forecast from earlier team Q1 times

The component report also includes `recent_team_q1`, a separate experimental
forecast. It replaces the native model's team spacing with the median observed
team Q1 residual from the last three earlier scored events. Each earlier
event's team median is divided by that event's field Q1 median before combining
events, so different circuit lap lengths are not compared as raw seconds.
Historical team assignments remain attached to their original observations.
Scored history requires at least two usable Q1 times as well as the evaluator's
entrant/result identity coverage check. This does not guarantee a complete Q1
field: the report retains historical coverage, and sparse Q1 labels can make
the estimated field median less representative.

The history window is fixed at three events, independently of `--form-races`,
which still controls the native rating model. It selects earlier scoreable
evaluation targets, not the native model's form-eligibility window. A team
missing from that window
uses its native model contribution. The report records each team's evidence
rounds and fallback, so a prediction without historical observations is not
presented as measured pace. No-history forecasts preserve native times exactly.

The forecast retains the native model's differences between teammates. It
constructs predictions for the complete target roster before selecting usable
target Q1 labels for scoring. Target times, missing target labels, and later
events cannot influence that prediction map. Selecting one target with `--race`
uses the same earlier evidence as evaluating that target in an all-event run.

This is a transformation of noise-free model predictions, not another native
car-rating variant. Its prediction rows retain native car and driver ratings
for reference; those ratings alone do not produce the transformed times.
It does not modify live qualifying, race pace, wet performance, or constructor
ratings. The method was chosen after inspecting this season's errors and
requires prospective validation before claiming future predictive improvement.

Each component fold also reports `paired_event_comparison`, which compares
`recent_team_q1` with `full_model` for that event. It intersects driver IDs
whose two variants both have a prediction and an observed Q1 time, then scores
both variants on that exact cohort and its shared Q1 labels and team assignments.
The report gives the matched driver, team, and teammate-pair counts alongside
candidate and reference errors for driver rank and relative pace, team-median
rank and relative pace, and teammate-gap error. An error can remain `null` when
its cohort is too small to define that metric.

The aggregate event comparison summarizes defined event deltas, where each
delta is candidate error minus reference error. Negative values favor
`recent_team_q1`; positive values favor `full_model`. Every event receives equal
weight in the mean and median summaries, regardless of its driver, team, or
teammate-pair counts. The event count and improved, tied, and worsened counts
are reported separately for each metric because a metric may be undefined in
some events. This event summary complements the existing observation-weighted
aggregate scores. Events reuse drivers and are not independent samples; these
descriptive deltas do not establish statistical significance or future
predictive improvement. Direction counts use the exact numerical sign of each
delta, with exactly zero counted as tied and no practical-significance
threshold. Tiny normalization differences therefore count as improved or
worsened; inspect the delta magnitudes before treating those counts as
meaningful.

## Current-season snapshot, 9 September 2026

The initial evaluator at revision `30fed16`, using the default dry scenario
and three-round form window, scored 282 driver
observations across 13 completed events. The paired comparison covers 258
observations across 12 events; round 1 has no earlier-Q1 baseline.

| Paired metric | Initial model (`30fed16`) | Corrected evidence | Previous Q1 |
|---|---:|---:|---:|
| Mean absolute rank error (places) | 3.178 | 3.202 | 3.070 |
| Relative pace error (percentage points) | 0.6592 | 0.6587 | 0.4952 |
| Pairwise concordance | 79.766% | 79.313% | 79.841% |

The corrected model uses [shared-session comparisons and a constructor prior
unaffected by missing pace buckets](rating-evidence.md). Both model versions
were evaluated against the same provider responses held temporarily in memory
on 9 September. Form rounds, standings, entrants, target labels and baseline
scores were verified identical. No raw feeds were persisted.

The initial model had lower rank error than the baseline in 5 of the 12 paired
events; the corrected model did so in 6. Neither improves on the baseline in
aggregate. The corrections remove demonstrated comparison artifacts, while
aggregate rank error is slightly worse and relative pace error slightly better.
No weights or lap coefficients were fitted to these outcomes. The assumed dry
weather and differing Q1 conditions limit what can be inferred about the pace
equations or future predictive performance.

Coverage also matters: the Australian qualifying feed supplied 19 entrants
against 22 result entrants, within the declared 80% overlap threshold. Miami
had one entrant without a usable Q1 time. These measurements describe the
available scoring cohorts, not a complete field at every event. The report
records each cohort and the 19 provider URLs used for this collection. Provider
revisions can change a subsequent run.

## Component snapshot, 19 September 2026

The unchanged pace model at revision `164be5e`, evaluated with the component
diagnostic in dry weather and a three-round form window, scored 302 driver
observations across 14 events. The table uses the common previous-Q1 cohort:
278 driver observations across 13 events, 143 team observations and 135
teammate pairs. All variants used the same provider responses, held in memory;
no weights or lap coefficients were fitted. The existing full-model and
baseline scores matched the earlier collection that day exactly.

| Paired metric | Constructor prior | Team form | Full model | Previous Q1 |
|---|---:|---:|---:|---:|
| Driver rank MAE (places) | 2.996 | 3.018 | 3.122 | 3.086 |
| Relative pace error (percentage points) | 0.6618 | 0.6614 | 0.6608 | 0.5017 |
| Team rank MAE (places) | 1.308 | 1.287 | 1.343 | 1.371 |
| Team relative pace error (percentage points) | 0.6168 | 0.6164 | 0.6157 | 0.4346 |
| Teammate gap error (percentage points) | 0.4033 | 0.4033 | 0.3974 | 0.5987 |
| Mean absolute predicted teammate gap (percentage points) | 0.0000 | 0.0000 | 0.0310 | 0.4184 |
| Teammate ordering accuracy | 50.00% | 50.00% | 62.22% | 48.15% |

The mean absolute observed teammate gap was 0.4033 percentage points. The
neutral-driver variants predict tied teammates and receive half credit for
ordering; their gap error is therefore a useful zero-gap reference. The full
model improves slightly on that reference and orders teammates better than
previous Q1 here, but its predicted gaps are much smaller than observed.

Constructor points dominate the team predictions: adding team form changes
relative pace error very little. The full model orders team medians slightly
better than previous Q1 but estimates their separation less accurately.
Adding driver evidence worsens overall rank error relative to the neutral
variants while slightly improving pace error. These mixed outcomes identify
calibration questions; they do not justify removing driver differences or
scaling all pace gaps by one factor. Q1 conditions and run quality remain
uncontrolled, and later completed rounds are needed to validate changes
chosen after inspecting this snapshot.

## Earlier-team-Q1 experiment, 20 September 2026

A fixed three-event team-history forecast was tested using the derived data
collected on 19 September, with the native model at revision `e875202`.
Rounds 1–3 were designated warmup before inspecting the candidate scores.
The primary comparison below covers rounds 4–14: 237 matched driver
observations, 121 team observations, and 116 teammate pairs. Lower errors
are better; pace and gap errors are in percentage points.

| Paired metric | Native model | Earlier team Q1 | Previous Q1 |
|---|---:|---:|---:|
| Driver rank MAE (places) | 3.1814 | 3.0295 | 3.1561 |
| Relative pace MAE | 0.6803 | 0.4586 | 0.5134 |
| Team rank MAE (places) | 1.4215 | 1.5207 | 1.4050 |
| Team relative pace MAE | 0.6318 | 0.4142 | 0.4445 |
| Teammate gap MAE | 0.4141 | 0.4141 | 0.6032 |

The candidate improved overall driver ranking and pace spacing in this
sample, while team-median ranking worsened. Teammate gaps were essentially
unchanged because the forecast retains native within-team differences.
Relative pace error was lower than the native model in 10 of the 11 primary
events and lower than previous Q1 in 7; the improvement was not universal.
This is a promising qualifying forecast, not evidence for changing global
car ratings or wet/race pace. Q1 run quality and weather remain uncontrolled,
and choosing this hypothesis after seeing the dataset limits the strength
of its apparent improvement. There was no coefficient search or fitting.

The evaluator includes this forecast so later events can test the same fixed
method. It reports all eligible events by default; the dated table above is
the explicitly selected primary subset, not the all-event aggregate.

## Interpretation limits

This is a round holdout using today's revised provider data and today's model
code. It does not reconstruct exactly what was published before an event:
later data corrections or retrospective penalties may already be reflected in
the historical standings. It is conditional on the target qualifying entrants
and their team assignments, rather than predicting who would enter. Changes to
the model after seeing these results require fresh validation before claiming
predictive improvement.

The live CLI and dashboard continue to use the latest current-season form,
live standings, available target qualifying and completed-target track pace.
Those snapshots answer a different current-strength question and are intentionally
not used as held-out predictions here. This evaluator adds no historical race
replay mode, parameter fitting or automatic tuning.
