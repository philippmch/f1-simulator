# Race-win estimates from simulated constructors and earlier race points

Live forecasts retain the simulator's chance of each constructor winning.
Within a constructor, each driver's chance uses that driver's earlier
current-season Grand Prix points, with 25 prior points per modeled teammate.
The default CLI, race API, dashboard and pre-event records use this fixed
`teammate_race_points_v1` policy. The race engine continues to produce the
same native trials, win counts, positions, events and strategy decisions.

For a constructor with drivers `a` and `b`, the reported estimate is

`P(a wins) = P(constructor wins) × (25 + earlier_points_a) / (50 + earlier_points_a + earlier_points_b)`.

The general formula includes every modeled teammate. Each needs two distinct
earlier Grand Prix entries with explicit matching constructor and usable
reported points. Missing or conflicting points and insufficient coverage keep
that constructor's native driver probabilities. A transferred driver's previous
constructor points are not reassigned to the new constructor. Target and later
rows are excluded before building the allocation; sprint points are not used.

The fixed [experiment protocol](teammate-forecast-protocol.md) records every
gate and event-level result. On seven validation events, the 100-trial check
improved Brier loss by 4.1% and failed its 5% gate. A single resolution using
verified existing native ensembles at 400 trials improved loss by **5.4%**,
from 0.863825 to 0.817194, and beat the 0.854928 constructor reference. The
production implementation reproduces those scores and source cutoffs.

This is a forecast-head improvement on a small, repeatedly inspected season,
using assumed dry weather and automatic strategy. Much of the gain comes from
Belgium; three event scores worsen and the other six events together worsen
1.4% when Belgium is removed. It does not prove future skill, better finishing
positions or more accurate strategy counterfactuals. Native simulated wins
remain useful for examining what actually occurred in a strategy scenario.

## Results, exports and replay

`DriverStatistics.wins` and `win_rate` remain native counts and percentages.
`SimulationResults.get_win_probabilities()` returns reported estimates when
frozen point history is supplied; otherwise it retains the native rates.
`get_winner_forecast()` exposes the policy, allocation history, estimates,
native counts and native probabilities. Win sampling ranges condition on
that fixed history and scale the constructor's Wilson interval; they do not
include uncertainty in historical form or in the allocation model.

The API and statistics JSON include `winner_forecast` and an
`estimated_win_rate` percentage alongside each driver's raw `win_rate`.
The dashboard ranks its win charts by the estimate and shows raw simulated
wins in the sampling table. HTML reports and CLI output identify the two
quantities. Probability ranges and finite-ensemble score corrections use
the actual fixed linear allocation, rather than treating estimates as
new simulation counts.

The point allocation is forecast metadata, outside the physical simulation
input schema. Saved statistics replay using the recorded allocation without
fetching history, while older snapshots retain their native forecast behavior.
Native worker and second-trial replay checks cover both engines. The policy
is optional for directly constructed runners with custom or synthetic inputs.

New [pre-event records](recorded-forecasts.md) use schema 2, retaining the native
`winner_forecast` and a separate frozen `winner_estimate`. Scoring reports the
estimate's loss as `winner_score` and native loss as `native_winner_score`.
It rebuilds the saved allocation to verify its probabilities and never refits
it from later results. Existing schema-1 records remain immutable and continue
to score their original native probabilities.

## Frozen validation evidence

The [normalized validation record](../evidence/teammate-points-2026-10-07.json)
contains the seven native ensembles' counts, prior point history, constructor
references, input snapshots, seeds and scores. It contains derived observations
and models, not raw live-feed responses. Its content seal detects accidental
changes; it does not independently authenticate the simulation execution.

Recompute the published loss offline with the installed production helper:

```python
import json
from pathlib import Path
from statistics import mean
from f1sim.analysis.race_probability_scores import summarize_winner_counts
from f1sim.analysis.teammate_forecast import build_teammate_allocation, score_teammate_forecast

evidence = json.loads(Path("evidence/teammate-points-2026-10-07.json").read_text())
scores = []
for event in evidence["events"]:
    native = summarize_winner_counts(event["native_win_counts"], event["no_classified_winner_count"])
    allocation = build_teammate_allocation(event["entrant_teams"], event["earlier_race_points"],
                                           cutoff_round=event["cutoff_round"])
    scores.append(score_teammate_forecast(native, allocation, event["observed_winner"])["brier_score"])
print(mean(scores))  # 0.8171935237932408
```

Each event also has `simulation_inputs` and `metadata` suitable for the existing
offline replay contract when saved as its own file. Exact native reproduction
still depends on matching the model and dependency versions.
