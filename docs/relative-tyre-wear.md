# Relative tyre-wear evidence

Lap timing can help compare compound trends, but it does not directly measure
tyre degradation. Fuel burn, traffic, energy use, temperature and driver intent
also change lap time. This diagnostic estimates descriptive relative trends;
it does not fit or change the simulator's tyre coefficients.

## Collect and evaluate evidence

Collect each completed current-season race using the existing explicit network
diagnostic, then run the evaluator offline:

```powershell
python examples/check_tyre_evidence.py --meeting Canadian --include-laps > output/canadian-evidence.json
python examples/check_tyre_evidence.py --meeting Italian --include-laps > output/italian-evidence.json
python examples/check_tyre_evidence.py --meeting Spanish --include-laps > output/spanish-evidence.json
python examples/evaluate_relative_tyre_wear.py output/canadian-evidence.json output/italian-evidence.json output/spanish-evidence.json
```

The collector keeps raw provider feeds in memory. The saved files contain
derived lap observations and source provenance. The evaluator accepts those
reports individually or as a flat JSON list and never downloads data. It
requires the current normalizer version and rejects duplicate event identities.
Regenerate older unversioned reports with the collector; adding a version field
by hand does not establish that the correction safeguards were applied.

Results are JSON on standard output, suitable for redirecting to a report file.
Input errors fail before a partial report is printed. Events without sufficient
compound coverage or an identifiable comparison have an explicit unavailable
result rather than a fabricated zero slope.

## What the comparison can identify

The linear model accounts for a separate starting pace for each driver stint
and a separate shared effect for every event lap. Against those effects, it
compares the lap-time slope on soft and medium tyres with the slope on hard
tyres. A positive soft-minus-hard estimate means soft lap times increase more
quickly than hard lap times within this model, in seconds per additional lap.
It does not mean the absolute soft degradation rate equals that estimate.

Within a stint, tyre age and race lap differ by a constant. The stint effect
absorbs that constant, including prior wear and the uncertainty of locating
the fitting within a lap. Consequently, a uniform one-lap shift of a stint's
age cannot provide an independent sensitivity check for this linear model.
The common slope across all compounds is absorbed by the event-lap effects;
absolute wear cannot be recovered by choosing hard as the reference.

The report must distinguish an estimable contrast from a rank-deficient
comparison. Missing compound overlap or insufficient variation must not
produce a confident zero estimate. Uncertainty reflects dependence within
the observed drivers, not uncertainty about the physical model or evidence
from an independent race.

Each event receives equal total weight in the pooled fit, so an event with more
eligible laps does not automatically receive more weight. Event-specific fits
remain visible. An event with incomplete compound coverage can still contribute
to a pooled comparison if it has within-event compound variation; the pooled
fit must itself meet the coverage and rank requirements. Events without enough
within-event variation after accounting for fixed effects are explicitly listed
as pooled exclusions. For example, all drivers switching compounds together
cannot distinguish compound effects from shared race-lap conditions. Such an
event must not inflate pooled driver counts or uncertainty calculations. A second fit
adds a separate linear trend for each driver within
an event, testing sensitivity to individual pace evolution. It can lose the
variation needed to identify a compound contrast, which is reported explicitly.

## Sampling uncertainty

The covariance calculation groups all stints from the same driver and event
into one cluster. For weighted, nuisance-residualized compound regressors `Z`
and weighted regression residuals `e`, it uses:

```text
B = inverse(Z' Z)
s[g] = sum of Z[i] * e[i] within driver-event cluster g
covariance = G/(G-1) * (N-1)/(N-rank(full design))
             * B * sum(s[g] s[g]') * B
```

This is the CR1 cluster sandwich estimate, with degrees of freedom including
the stint and lap effects. The correction follows the conventional
[cluster covariance formula](https://www.statsmodels.org/dev/_modules/statsmodels/stats/sandwich_covariance.html#cov_cluster);
the evaluator implements it using NumPy without adding a statistics dependency.
Reported ranges use the approximate normal multiplier
1.96. Ranges are unavailable with fewer than ten driver-event clusters, fewer
than five driver-event clusters for any compound, or no residual degrees of
freedom. These are reporting guards, not guarantees that a normal approximation
is accurate. The ranges describe sampling
uncertainty conditional on this model and these events; they do not measure
uncertainty about unobserved confounders or performance at another circuit.

## Interpretation

Compare event-specific results and the sensitivity to omitting each event.
Omitting an event changes the estimation sample; it is not a prediction of
that omitted event. A pooled result can conceal different track conditions,
unequal compound coverage and different driver choices.

Eligible timing observations establish known green running, zero reported
rainfall, unchanged slick compound and no recorded pit exposure. Zero rainfall
does not establish a dry track, and eligibility does not establish freedom
from traffic, deliberate pace management or thermal effects. These remaining
confounders prevent a causal interpretation of the relative slopes.

Use corrected evidence explicitly. Do not mix an earlier and corrected
normalization of the same event, even when their source-feed hashes match.
The source feeds may be unchanged while the normalizer's interpretation has
improved. Saved derived observations support offline analysis; the diagnostic
does not download or persist raw provider feeds.

Fresh collection on 26 September 2026 produced 895 eligible Canadian laps,
863 Italian laps and 978 Spanish laps under normalizer version 1. The same
feed hashes appeared in older unversioned reports containing 934, 908 and
978 eligible laps respectively. The difference reflects corrected-stint
exclusion handling, not new timing observations. Earlier descriptive coverage
counts should not be treated as the input to the current evaluator.

## First descriptive comparison

The version 1 reports above contain 2,736 eligible laps. Applying the evaluator
produced these relative slopes, in seconds per additional lap:

| Sample | Soft minus hard | Medium minus hard | Soft minus medium |
|---|---:|---:|---:|
| Canada | 0.0246 | 0.0056 | 0.0190 |
| Italy | 0.1049 | 0.0090 | 0.0959 |
| Spain | 0.0535 | 0.0289 | 0.0246 |
| Equal-total-event-weight pooled fit | 0.0422 | 0.0115 | 0.0307 |

Canada has only two drivers with eligible hard-tyre observations, so its ranges
are withheld. The pooled approximate ranges are 0.0086 to 0.0759 for soft minus
hard, -0.0025 to 0.0255 for medium minus hard, and -0.0018 to 0.0633 for soft
minus medium. These conditional ranges do not establish physical wear rates.

Adding driver-specific lap trends changes the pooled soft-minus-hard estimate
to 0.0255 and medium-minus-hard to -0.0076; all three adjusted ranges include
zero. Omitting one event moves the unadjusted soft-minus-hard estimate between
0.0327 and 0.0877. This sensitivity, sparse compound coverage and unresolved
confounding do not support changing the simulator's preset wear coefficients.
The value of this first run is a reproducible baseline for additional evidence,
not a calibration result.
