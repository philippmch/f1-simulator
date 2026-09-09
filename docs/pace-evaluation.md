# Held-out current-season qualifying pace

The evaluator asks whether the current rating model orders qualifying pace
better than a simple previous-Q1 baseline when target performance is withheld.
It does not use the live simulator's current-strength snapshot for a past target.

```powershell
python examples/evaluate_qualifying_pace.py --race 13
python examples/evaluate_qualifying_pace.py --race "Italian Grand Prix"
python examples/evaluate_qualifying_pace.py
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
