# Forecasts recorded before qualifying

Save a current-season forecast before qualifying and score that same file after
the race. This preserves the probabilities that were actually predicted; the
scoring command never runs simulations or updates the model.

```powershell
python examples/record_race_forecast.py --race 17 --scenario dry --trials 100 --seed 42 --parallel --workers 4 --output output/forecasts/2026-round-17-dry.json
python examples/score_recorded_forecast.py output/forecasts/2026-round-17-dry.json
```

Use an upcoming round or race name. Select an explicit weather scenario:
`dry`, `light_rain` or `heavy_rain`. Rainfall stays fixed and surface wetness
evolves. These are conditional forecasts; the selected scenario is not a
prediction of the race's weather. Each trial simulates qualifying and the race
with automatic strategy. The default budget is 100 trials and seed 42. Larger
budgets reduce Monte Carlo sampling error, not model bias. Ctrl-C cancels
collection or simulations without publishing an incomplete forecast.

The calendar must contain a qualifying date and UTC time. Recording must start
and finish before that time, and target or later performance must be absent from
the available feeds. Historical results and qualifying are kept behind the
target's training cutoff. The record contains driver identities, derived model
inputs, runtime/source provenance, the assumed weather, timestamps, winner
counts and probabilities, qualifying pole counts and mean positions, and frozen
historical references. Raw live feed caches are not committed.

New records use schema 2 with a separate `winner_estimate` from the
[teammate point allocation](teammate-forecasts.md). The original
`winner_forecast` still contains native simulation counts and probabilities.
The estimate freezes its strictly earlier point history, constructor mapping
and allocation. Scoring returns calibrated `winner_score` and separate
`native_winner_score`, with the appropriate finite-ensemble corrections.
Schema-1 files continue to score their original native probabilities.

The writer refuses to overwrite an existing record. A SHA-256 seal covers its
contents, and loading verifies counts, probabilities, timing, training boundaries
and the saved simulation models. The seal detects accidental alteration; it does
not authenticate the author's clock or prevent deliberate re-sealing. Archive or
publish the original file with an independently dated receipt before qualifying
to support a prospective timing claim. A local timestamp alone is insufficient.

Before results are available, scoring returns `unscored` with a reason. Once
the race has a uniquely identified classified winner and enough result coverage,
it reports winner Brier loss, finite-trial diagnostics, the historical reference
losses, pole loss where observed, qualifying position error and coverage. Known
entrants outside the saved roster are removed from the observed grid ordering;
missing qualifying rows remain visible in coverage. A changed race date or
circuit invalidates scoring against the original event. Keep the record and score
as separate files. Collection and scoring currently use the current season.

The [retrospective evaluation](race-probability-evaluation.md) remains useful for
development. Recorded forecasts are the route to later evidence from events
whose outcomes were unavailable when the prediction was made. One such event
cannot establish calibration or sustained predictive skill.

## First published records

Two fixed-dry Singapore forecasts, each with 100 trials and seed 42, are retained
as original files: [the previous clock](../forecasts/2026-round-17-dry.json) and
[the corrected clock](../forecasts/2026-round-17-dry-clock.json). The latter was
completed at **2026-10-05 17:18:31 UTC**, before the calendar's qualifying start
of **2026-10-10 13:00 UTC**. Inputs contain only performance rounds 1–16. Their
content seals are respectively
`a8e53346895db8d90d943de6069dcfbd00004da73f54aedecc4fd7455c1ed8db`
and `121d4255f2e2fa93ab95057b193326c8fb5f880b65920d52023c0b1f4252e48d`.

These were separate live collections. Inspection confirmed identical driver,
car, venue and weather inputs aside from the clock coefficient, and identical
qualifying forecasts. Runtime/source provenance differs. Both files retain
their original local-clock disclaimer. The dated GitHub publication of the
containing commit is separate evidence of pre-event availability. Scores remain
unavailable until the event has a classified result; no future accuracy is
claimed today. Run the scoring command against each original path afterwards.
