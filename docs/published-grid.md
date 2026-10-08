# Published race starting order

Forecasts after qualifying should use the starting order already available.
Live dashboard and CLI runs now fetch the current year's official Formula 1
starting-grid page after Grand Prix qualifying has completed. Before that
session, they continue to start the race from simulated qualifying.

This is an improvement in the information used by an after-qualifying forecast.
It does not improve forecasts made before qualifying, and it does not establish
that race-winner predictions meet the model's acceptance bar.

## Controls and evidence

The dashboard's Race grid control defaults to Published when available. The
CLI defaults to `--race-grid auto`; `--race-grid simulated` retains the
previous behavior. The API accepts `race_grid_mode: "auto"` or `"simulated"`
and an optional complete `starting_grid` list of modeled driver IDs. A supplied
grid conflicts with the explicitly simulated mode. Custom qualifying weather
disables automatic published-grid lookup so that the hypothetical qualifying
conditions still determine the race grid; an explicitly supplied API grid
continues to take precedence.

The loader uses fresh GETs within its existing budget. Session completion
requires a known Grand Prix qualifying timestamp plus one hour. Sprint
qualifying does not authorize a Sunday grid. The current-year results index
must identify the event, and every driver code, constructor and starting slot
must resolve exactly once. Fetch provenance includes the URL, timestamp, order
and whether lookup succeeded. An unavailable or incomplete grid retains the
simulated grid and records a fallback reason. There is no persisted grid cache
or older-season live fallback.

Pit-lane starts are not yet represented faithfully by the engines. A grid with
a pit-lane slot or a published pit-lane-start note is therefore unsupported and
retains simulated qualifying. Numerically listing a pit-lane starter at the
back of the table is insufficient evidence of an ordinary grid start.

## Simulation and replay

`MonteCarloRunner(starting_grid=[...])` sets only the race's starting order.
Qualifying laps still run with the original random streams, and their tables
remain simulated. Console, dashboard and HTML reports explicitly distinguish
those lap results from the supplied race order. Weather and strategy variants
share the same grid, as do saved-input tyre and engine comparisons.

Saved inputs use schema 14 when a grid is supplied. Replay requires the complete
order and rejects attempts to downgrade that snapshot to an older schema.
Existing schemas 1–13 retain their previous interpretation. Different grids
are incompatible for paired strategy comparisons. Weather, race-control,
finite tyre inventory and pit-plan assumptions can coexist with schema 14.

## Accuracy checkpoint

The earlier seven-event exploratory grid replay reported 19.1% lower mean
winner Brier, but ignored pit-lane-start notes. It is **not a performance claim
for this shipped policy**. A fresh check of the seven source pages accepts
Belgium, Azerbaijan and Bahrain in Malaysia and rejects Hungary, the
Netherlands, Italy and Spain because they contain pit-lane-start notes.

Those observations are already inspected retrospective data. They cannot serve
as a new independent test. The overall race-winner milestone remains unmet;
the qualifying improvement and passing software checks do not change that.

Validation covers both real engines, worker/serial equivalence, exact seeded
replay, strict roster validation, old-schema rejection, all existing scenario
overlays, consistent strategy/reference grids, fresh HTTP headers, the Grand
Prix qualifying cutoff, unsupported pit-lane starts and unavailable sources.
