# Direct outcome prediction: frozen evaluation protocol

Started 2026-10-07 against main `42599bb`, before collecting the new
confirmation and test outcomes. This work is exclusively for predictive
performance; race-engine maintenance and unrelated UI work are out of scope.

## Data and separation

- Source: Jolpica's published Grand Prix results and qualifying, 2008–2025.
- Fit: 2008–2018. Development and model selection: 2019–2021.
- Confirmation: 2022–2023. Final test: 2024–2025.
- Final-test predictions and parameters must be frozen before test labels are
  scored. Failed confirmation or test checks remain in the evidence.
- Every event is predicted before its qualifying performance is available.
  The actual participating driver/constructor identities and circuit identity
  are supplied as an explicit retrospective roster assumption. No target grid,
  qualifying time, race result, points, retirement or later event enters its
  features. Update performance history only after saving that event's forecast.
- Earlier events in a confirmation/test season may update the rolling history,
  exactly as they would in live use. They do not refit coefficients or select
  hyperparameters. Forecasts use today's revised feed, not authenticated
  contemporaneous snapshots.
- The existing 2026 cohort is already inspected and is a secondary deployment
  check, never an independent final test or a tuning set.

## Models and references

Learn regularized conditional winner probabilities and race-ranking strength
from earlier qualifying, constructor performance, teammate differences, wins,
points and observed all-cause retirement history. Development can compare
feature subsets, regularization, fixed history windows and probabilistic
ranking/calibration heads. It cannot inspect confirmation/test scores while
choosing them. Separate winner and full-ranking objectives are permitted;
their probabilities must be identified rather than silently presented as
native physical simulation counts.

References are uniform, strictly earlier season driver/constructor point
shares, earlier race-win shares, and a fixed recent-qualifying ranking model.
Before collecting confirmation, the reference formulas were fixed at 25
prior points per driver, 0.5 prior race wins per driver, three earlier
qualifying events, one prior-strength observation, and qualifying softmax
scales 6 and 12. The sharper scale-12 reference was added after development
inspection to make the winner/podium comparison more demanding. Both remain
in the evidence; winner acceptance uses the best of every fixed reference.
Position/podium acceptance retains the originally declared scale-6 reference
and also reports scale 12. No confirmation or test score informed this choice.
Report every reference on the identical events, including opening races with
explicit smoothing. Rank/position and podium references use the same
pre-qualifying information boundary. Report unhalved multiclass winner Brier,
winner log loss, driver-averaged podium Brier, expected-position MAE and
retirement Brier, with equal event weighting, per-season scores and coverage.

## Acceptance: a substantial, broad predictive gain

The selected model must meet all of the following on the final 2024–2025 test:

1. At least 40 fully covered races, including both full seasons and all scored
   opening events; publish all exclusions and reasons.
2. Winner Brier at least 10% below the best fixed historical winner reference,
   and winner log loss at least 5% below that reference's log loss.
3. Winner Brier improves in each test season and still improves after removing
   the three events with the largest individual gains.
4. A paired 10,000-resample event bootstrap's 95% interval for mean winner
   Brier improvement excludes zero. Also report a season-block sensitivity;
   two test seasons alone cannot establish robustness to all future regimes.
5. Podium Brier and expected-position MAE each improve at least 5% against the
   fixed recent-qualifying reference; neither worsens in either test season.
6. Retirement Brier does not worsen by more than 2% against the smoothed
   earlier retirement-frequency reference.
7. Reproduce the accepted predictions through the production implementation
   with causal-history checks, then compare to the existing deployed model on
   the already inspected 2026 events. A historical model that regresses the
   live-season winner predictions is not ready to become the default.

Confirmation uses the same directional/per-season checks and reference
definitions before the final test is opened. This protocol does not allow
repeated final-test parameter selection, deleting difficult races, changing
the acceptance threshold after seeing outcomes, or claiming prospective skill
from retrospective validation. A failed final test requires a different
untouched evaluation cohort or genuinely later pre-recorded forecasts.

Source documentation: [Jolpica endpoints and pagination](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md),
[Grand Prix results](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/results.md),
[qualifying](https://github.com/jolpica/jolpica-f1/blob/main/docs/endpoints/qualifying.md).
