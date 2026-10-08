# Why the winner forecast loses to a simple reference

The current dry, practice-informed race forecast and a fixed practice-ranking
reference have approximately equal retrospective winner performance. Across
the same 16 complete 2026 fields, with 400 native trials per event, unhalved
Brier is **0.683266** for the simulator and **0.681886** for the reference.
Correcting the simulator's finite-ensemble score bias gives **0.681567**.
The paired event bootstrap for raw model minus reference is **[-0.17532,
0.16762]**. This is insufficient evidence to distinguish their overall skill.

These are already inspected, revised observations with known entrant identities
and fixed dry race assumptions. They are development evidence, not forecasts
published before those events or an independent test.

## Reproduce the losses

The [native evidence](../evidence/practice-native-winner-2026.json) and
[practice reference observations](../evidence/practice-winner-reference-2026.json)
are sealed separately. The reference uses the same latest practice session as
the native qualifying model, the complete modeled field and the fixed historical
`practice_12` transformation. Its observations never update the native inputs.

```powershell
python examples/analyze_forecast_errors.py --output output/forecast-errors.json
```

This recomputes both probability scores, attributes each event's loss to the
winner and other outcomes, and traces the winner's practice rank, modeled
qualifying pace rank and modeled clean race pace rank. The command refuses to
overwrite a saved analysis. Its pace comparison uses a fresh medium on lap 1,
without traffic or random variation; this isolates the starting pace inputs,
not the entire race's expected finishing order.

The model beats the reference on seven events and loses on nine. Its three
largest losses against the reference are:

| Race | Winner | Practice rank | Modeled qualifying pace rank | Modeled clean race pace rank | Native win probability | Reference win probability | Excess Brier loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hungary, round 11 | Norris | 1 | 3 | 5 | 8.5% | 43.5% | 0.70563 |
| Malaysia, round 16 | Verstappen | 2 | 3 | 7 | 5.0% | 24.6% | 0.39341 |
| Netherlands, round 12 | Norris | 2 | 2 | 5 | 11.25% | 24.6% | 0.31936 |

There are also substantial gains: the simulator beats the reference by 0.79014
in Italy and 0.63070 in Miami. Replacing the simulation with the reference
would discard useful information on those events.

## The mechanisms to investigate

**Current practice updates qualifying but not race pace.** The qualifying
adjustment is deliberately confined to qualifying. Race pace still comes from
earlier season car and driver ratings. Constructor points dominate the car
rating's raw scale, with smaller signed qualifying/race residuals added before
normalization. A good current practice observation can therefore raise a
driver's qualifying forecast while the race model still treats their team as
too slow. In Hungary, the initial clean-lap model puts Norris 0.274 seconds
behind the fastest driver, despite his first-place practice observation.
Practice fastest laps alone do not establish race pace; race-oriented practice
stints are the relevant next input to test.

**Repeated lap noise does not model uncertain weekend ability.** Independent
lap fluctuations average out over a race. They cannot represent a persistent
error in a team's expected pace. The controlled uncertainty experiment below
tests that missing dimension without changing qualifying or the mean pace.

**A dry scenario is conditional on its weather assumption.** The Netherlands
and Malaysia had observed rain in the separate sensor diagnostic. Scoring dry
trials against those outcomes mixes a pace/model error with a scenario error.
This cannot be repaired by copying the observed weather into a retrospective
forecast. Operational pre-event weather information needs its own verification.

**Probability-score attribution is not a causal explanation.** Over the whole
cohort, the model's loss on the actual winner is 0.02355 lower than the
reference's, but its loss on other outcomes is 0.02493 higher. These nearly
cancel. The table identifies concentrated misses; it does not prove that every
miss is caused by pace or that the simulator is universally overconfident.

## Controlled experiments

The first experiment changes only persistent race-pace uncertainty. Team and
driver scales are the existing 2023-only residual estimates, 0.36657% and
0.23120%; transferring those scales to the unchanged native mean is a stated
hypothesis. Qualifying positions remain identical in every paired 100-trial
replay. Across the first seven fixed rounds, winner Brier increases from
**0.65223 to 0.66929**, a **2.6% regression**. It stops at the declared first
stage gate and is not deployed.

A separate hindsight sensitivity replaces relative race pace with the estimated
pace of the observed race's green laps, controlling for lap number, compound
and tyre age. It changes no qualifying, weather or race-engine logic. These are
two deliberately selected failures, not a valid pre-event candidate:

| Race | Winner | Original win chance, 100 trials | With observed relative pace | Original Brier | With observed relative pace |
| --- | --- | ---: | ---: | ---: | ---: |
| Barcelona, round 7 | Hamilton | 4% | 41% | 1.2126 | 0.4646 |
| Hungary, round 11 | Norris | 7% | 34% | 1.1676 | 0.6008 |

This is evidence that race-pace inputs materially affect these failures, with
the engine and qualifying held constant. Target race observations and
unobserved fuel effects make the hindsight correction unsuitable for live
prediction. Its percentages are sensitivity results, not improved forecasting
accuracy.

The third experiment tests a conservative update from comparable long practice
runs completed before first qualifying. It fits two bounded coefficients on
2023 data only: team pace and within-team pace corrections, shrunk for missing
drivers and short runs. This changes actual race pace rather than relabeling
winner probabilities. The fixed coefficients are 0.34324075 and 0.06897690.
The initial pace checks improve field and front-three gap errors in both 2024
and 2025; those years remain inspected development data. However, the paired
native replay of all 16 current-season fields increases winner Brier from
**0.68254 to 0.68715**, a **0.7% regression** at 100 trials. It improves the
Hungary miss but substantially worsens Italy. It is not deployed. This provides
a concrete failure to investigate: converting practice long-run measurements
directly into race pace can hurt winner forecasts even when conditional pace
error improves. Unknown practice fuel loads, run programs, incomplete coverage
and transfer of coefficients from the earlier regulations remain confounders.

[Sealed experiment receipts](../evidence/forecast-error-experiments-2026.json)
preserve each input digest, seed, complete-field win count and paired qualifying
mean position. The analysis command verifies these receipts and recomputes the
scores offline. It does not rerun races, claim that the sensitivity experiments
were pre-event forecasts, or promote the rejected parameter changes.

Production winner prediction accuracy is unchanged by this diagnostic work.
The shipped changes preserve the reference and its causal recording cutoff;
the experiments identify the mean race-pace input as a priority and rule out
the tested shortcuts. The next pace candidate needs a defensible treatment of
practice programs and fuel uncertainty, plus winner checks rather than only
conditional lap-time checks.

## Future forecasts and the improvement loop

New [recorded forecasts](recorded-forecasts.md) with usable current practice
freeze both the native forecast and the simple reference before the first
qualifying session, including sprint qualifying. Later scoring uses the saved
practice observations and produces the same per-driver loss attribution.
Existing sealed forecasts retain their original interpretation.

Use these losses to select a specific mechanism, run a paired experiment that
holds the other inputs and randomness fixed, and fit any coefficients on earlier
data. A reduction in conditional pace error alone is insufficient: the resulting
winner probabilities must improve too. Future recorded predictions decide
whether the resulting model beats the reference; the broader
[winner-model protocol](predictive-model-protocol.md) remains unmet.
