# Practice-informed qualifying forecasts

This is a qualifying accuracy improvement. It does **not** satisfy the
[race-winner model acceptance bar](predictive-model-protocol.md). Race-winner
quality remains an open milestone.

Live 2026 runs use the latest usable completed practice held before the first
qualifying session, including sprint qualifying. The model combines that
practice with earlier current-season Q1 results, teammate differences and an
earlier-points interaction. It predicts relative Q1 pace and adjusts the
practice session's lap-time clock using a fixed earlier-year correction.

Only `Driver.qualifying_pace_adjustment` changes. Race pace, cars, tyre wear,
reliability and strategy inputs retain their supplied values. A different
qualifying grid can still change race outcomes; the replay below measures that
effect rather than assuming it is beneficial.

## Live evidence and fallbacks

Session timestamps must identify a completed one-hour practice before the
earliest qualifying. FP2 after Friday sprint qualifying is excluded. Unknown
times do not authorize fetching a session. Classifications come freshly from
the official Formula 1 results pages, within the loader's existing HTTP budget.
The event's current-year navigation link is usable before a winner/date table
exists. Driver codes and constructors must match the modeled roster; reserve
drivers do not become simulation entrants. Coverage must reach 80%, or 50%
for sprint FP1. A cancelled or incomplete session may use an earlier eligible
practice.

Without usable practice, the existing earlier-team-Q1 forecast remains in
place. Custom physics, excessive corrections and a physical lap floor that
would flatten the predicted gaps also retain the supplied inputs. Setting
`quali_weight=0` disables qualifying calibration. The loader's provenance
records the policy, session, source URL, coverage and fallback reason.

The packaged JSON contains fitted coefficients, not an archived grid or feed.
Inference starts with empty historical driver/team state each season and uses
only the current UTC season. Parameters are scoped to 2026; another year uses
the existing qualifying policy until a new parameter set is validated.

## Training and evaluation boundary

The relative-time head is event-balanced ridge regression with regularization
0.001, 3/6/12-event features and 0.85 annual training weights. Feature scaling
comes from training rows. Development selection used year-ahead 2019–2023
forecasts; the production coefficients fit 2014–2025. The clock correction is
the median earlier Q1/practice field-time ratio by practice session, fitted to
247 earlier events. No 2026 qualifying labels train either deployed component.

Earlier experiments had already inspected 2024–2026 observations. These results
are retrospective checks on revised data, **not an untouched test or evidence
of future calibration**. The policy and evidence disclose that limitation.
Entrant identities are retrospectively known. Practice supplies information an
earlier weekend forecast does not have; improvement after practice must not be
advertised as improvement before practice.

The original direct winner head failed confirmation. Its revision achieved only
1.8% lower winner Brier on the 48 final 2024–2025 events, missing the 10% bar;
its event bootstrap included no improvement and removal of the three largest
gains reversed the result. Later revisions were therefore development research,
not additional independent tests. None is deployed as a successful winner head.

## Measured qualifying result

[Sealed evidence](../evidence/practice-qualifying-2026.json) includes normalized
prefix history, practice, actual physical inputs, forecasts and separate labels.
It scores all 16 completed current-season events on matched driver cohorts,
using native dry deterministic qualifying laps. The first-event Q2/Q3 native
forecast is reconstructed from the recorded equal cold-start lap; the previous
validator omitted that fold. These are fastest-lap pace comparisons, not
Monte Carlo full-session elimination probabilities or final grid forecasts.

Each race has equal weight. Absolute error is in seconds; relative pace error
is percent of the session field median; rank error is positions within that
observed session cohort. Lower values are better.

| Session | Existing absolute error | Practice model | Existing relative error | Practice model | Existing rank error | Practice model |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 | 1.947 s | 0.465 s | 0.492% | 0.397% | 3.327 | 2.545 |
| Q2 | 1.511 s | 0.491 s | 0.367% | 0.291% | 2.628 | 1.902 |
| Q3 | 1.333 s | 0.825 s | 0.310% | 0.263% | 1.918 | 1.555 |

Q1 absolute error falls 76%, relative error 19%, and rank error 24%. Relative
error improves on 12/16 Q1, 15/16 Q2 and 12/16 Q3 events. Its paired event
bootstrap interval is positive in each session, and remains positive after
removing the three largest improvements. These sensitivity checks do not undo
the retrospective selection limitation.

The direct-practice reference is useful: its Q1 absolute/relative/rank errors
are 0.681 s / 0.565% / 2.439. The model has lower timing and pace error, but
**slightly worse Q1 ranking than ordering practice times directly**. Q2/Q3
timing and relative errors also improve over that reference. The model should
not be described as winning every comparison.

## Race-winner result and remaining work

Seven native chronological replays use the same physical race inputs, seeds,
400-trial budgets, dry weather, automatic strategy and earlier point allocation
as the sealed baseline. Only qualifying adjustments differ.

Mean winner Brier changes from **0.8172 to 0.8025**, approximately 1.8% lower.
Four events improve and three worsen. The adjusted paired event bootstrap
interval is **[-0.171, 0.205]**; removing the three largest gains leaves a
**0.165 regression**. Hungary, the Netherlands and Bahrain in Malaysia regress. This is not
convincing winner improvement, even though the aggregate does not regress.
The qualifying change is promoted for its measured qualifying improvement;
the overall winner-quality milestone remains unmet.

The subsequent [practice-era winner reporting comparison](practice-winner-forecast.md)
uses complete 22-driver fields and finds that retaining the native simulated
probabilities improves over redistributing them to earlier teammate points.
That scoped reporting correction reduces winner Brier by 6.3% over 16
400-trial events; the broader winner milestone still remains open.

Before calling the app a strong race predictor, it needs a race-performance
signal that improves winner probabilities beyond these unstable results and
prospectively sealed forecasts with appropriate information cutoffs. Merely
running more simulations, adding tests or choosing another variant after
seeing these outcomes will not establish that skill.

## Reproduce

Run `python examples/verify_practice_qualifying.py`. It checks the evidence seal
and parameter digest, rebuilds each forecast from prefix history and practice,
calculates real native qualifying laps, and reproduces all three sessions'
scores without network access. It never substitutes these observations into
live runs. The evidence also records every race-winner regression.

Validation includes the full Python 3.12 suite (9,673 passes and two skips at
collection time), the subsequently added unfinished-event navigation test,
current-loader/calibration checks, causal target-label poisoning, immutable
race-pace inputs, repeated calibration, and loading parameters from a built
wheel. Future race accuracy is not established by those software checks.
