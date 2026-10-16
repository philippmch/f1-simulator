# Qualifying-led race pace

The dashboard, simulation CLI and post-qualifying forecast recorder now default
to `qualifying_led`. The Race pace control also offers `season_points` for a
matched comparison with the previous model. This choice is independent of the
Race grid control: use official qualifying and its published penalties, or
simulate qualifying and start each race from that trial's order.

## What changes and why

The previous car-pace calculation adds a cumulative constructor-points term
with implicit weight 1 to current qualifying and recent timing. That term can
outweigh the weekend signal even with the nominal qualifying weight at 0.5.
It carries earlier results, reliability and starting circumstances into pace
as well as the secondary form terms.

`qualifying_led` removes that points anchor after accepted, near-complete
target Grand Prix qualifying, for teams with usable positively weighted timing.
The existing current-qualifying / recent-race / recent-qualifying weights remain
0.5 / 0.3 / 0.2. These weights act on normalized signals; they are not literal
percentages of the resulting race pace. Driver traits, standings points, other
car properties and race physics retain their existing calculations. Missing,
incomplete or untimed qualifying retains the previous fallback. Before target
qualifying, the two policies produce identical inputs.

The low-level `CurrentSeasonDataLoader.get_weighted_driver_stats` API retains
`season_points` as its compatibility default. Applications select their policy
explicitly. `/api/run`, `/api/ratings` and new post-qualifying records save the
selection. Exported simulation inputs freeze the actual drivers and cars;
replaying an existing export does not rebuild it with a newer policy.

```powershell
python examples/simulate_race.py --race 18 --race-grid official --postqual-pace-policy qualifying_led
python examples/simulate_race.py --race 18 --race-grid simulated --postqual-pace-policy season_points
python examples/record_race_forecast.py --stage post_qualifying --race 18 --scenario dry --trials 400 --postqual-pace-policy qualifying_led --output output/new-record.json
```

Live official-grid runs require a complete published grid. The forecast
recorder also requires the actual post-qualifying, pre-race window. Historical
development uses the separate offline tool below and does not wait for a race.

If the provider returns a target GP classification with every Q1/Q2/Q3 time
missing, the loader now tries the official qualifying table once after the
session. It supplements times only after matching the event, full field,
drivers, teams, car numbers and qualifying positions. Ordinary eliminated
drivers with missing Q2/Q3 times do not trigger this lookup. Accepted and failed
lookups appear in data provenance. This repairs a real missing-data case in
2025 Miami; source completeness alone does not prove a forecast improvement.

Verified official-grid scenarios also retain native simulation probabilities
after the race starts. Previously the reporting policy silently switched to a
teammate-points allocation at that deadline, even when the simulated race
results and inputs were identical. Recording a prospective forecast still
requires the pre-race window.

## Measured historical change

The [sealed sensitivity](../evidence/historical-qualifying-led-2026-10-16.json)
contains 40 matched events: all 24 rounds of 2025 and rounds 1–16 of 2026,
with 400 trials for each policy and event. Lower scores are better.

| Metric, all 40 events | Season-points model | Qualifying-led | Grid scale 18 |
| --- | ---: | ---: | ---: |
| Winner Brier | 0.52074 | 0.46501 | 0.50757 |
| Correct favourite / 40 | 24 | 25 | 27 |
| Finishing-position MAE | 3.29966 | 3.34940 | — |
| Podium Brier | 0.06057 | 0.06131 | — |
| Non-finish Brier | 0.14293 | 0.14317 | — |
| Winner log loss | Infinite (one zero-win observation) | 0.99595 | 1.29382 |

Winner Brier is **10.7% lower than the previous model and 8.4% lower than the
strongest fixed grid reference** on these events. The policy also improves
winner Brier in each year: 0.56859 → 0.50074 in 2025 and 0.44898 → 0.41142 in
2026. Finishing-position MAE worsens by 0.04974 places across all events.
Non-finish truth includes non-starts and disqualifications.

**Winner picks still lose to the grid: 25/40 versus 27/40.** All five
qualifying-led choices that differ from the first grid starter are wrong on
this cohort. Canada and Abu Dhabi 2025 are the two races where the grid is
correct and the model is wrong. Better probability scores do not establish
better winner selection. The offline verifier reports both measures and the
individual disagreements; the original evidence and rejection remain intact.

Concrete changes in those same 400-trial comparisons illustrate the mechanism:

| Event | Previous → qualifying-led favourite | Observed winner's probability |
| --- | --- | ---: |
| 2025 Suzuka | Norris (grid 2) → Verstappen (grid 1) | Verstappen 22% → 39.75% |
| 2025 Monza | Norris (grid 2) → Verstappen (grid 1) | Verstappen 24% → 49.5% |
| 2026 Madrid, regression | Antonelli (grid 2) → Norris (grid 1) | Antonelli 45.25% → 29.75% |

At Monza, removing the cumulative anchor raises Red Bull's relative car pace
from 0.88165 to 0.94264 while McLaren stays at 1. Norris's simulated win chance
falls from 39.25% to 28.5%, correcting the previous wrong favourite. Madrid
shows the opposite outcome: closing McLaren's pace deficit moves the forecast
away from the eventual winner. These trace the rating change to the simulated
forecast; they do not independently measure intrinsic race pace, and coupled
strategy decisions can also mediate the change.

The paired 95% interval for the winner gain over grid18 is
**−0.02547 to +0.10445**, crossing zero. Removing the three largest gains
leaves a positive 0.01346 gain. The original broad promotion gate remains
**failed**: the point gain is below its 10% grid threshold, uncertainty includes
no gain, and the other outcome metrics do not all improve. Its original
rejection and source identities remain in the evidence. The selectable policy
implements the requested qualifying-first behavior and a measured historical
probability-score improvement; it does not certify robust or prospective superiority.

These are already-inspected, fixed-dry development races using current simulator
rules on 2025 as well as 2026. The original contexts also used incorrect planned
distances for 2025 Jeddah (57 instead of 50 laps) and Imola (57 instead of 63).
The sources were reconstructed after the events, so round cutoffs establish
which sessions enter the calculation, not the authentic publication time of
every source revision. These limits apply to both sides of the matched scores.

## Reproduce without network access

Two offline checks reconstruct the archived scores from integer counts, and
rebuild both policies through the current production input builder:

```powershell
python examples/verify_historical_qualifying_led.py
python examples/verify_postqual_pace_inputs.py
```

The second check covers all 80 model inputs to an absolute numeric tolerance of
1e-12, all 40 pre-qualifying fallbacks and 832 strategy-inference identities.
The scores above are the archived 400-trial sensitivity, not a claim that this
input verification ran another 32,000 races.

Run fresh simulations from the bundled historical inputs with:

```powershell
python examples/backtest_qualifying_led.py --list
python examples/backtest_qualifying_led.py --year 2026 --round 16 --trials 400 --workers 8 --policy both --output output/r16-pace-comparison.json
python examples/backtest_qualifying_led.py --year 2025 --trials 400 --workers 8 --policy both --output output/2025-pace-comparison.json
```

Omit `--round` to run the selected year. The runner denies network access,
rebuilds production car and driver inputs, retains the confirmed grid and seed,
and saves complete joint race and qualifying trials, including pit evidence.
It checkpoints forecasts before opening the separate outcome labels, then
scores winner, positions, podiums, non-finishes and all three grid references.
An interrupted run leaves an unscored `.partial` file. Existing outputs and
checkpoints are never overwritten. Raw infinite log loss remains explicit;
the scorer does not conceal zero winner probabilities with an epsilon.

This makes historical development and diagnosis reproducible now. A new race
is needed only for evidence that probabilities recorded before its outcome
have predictive skill.
