# Winner-model revision after the failed confirmation

Recorded on 2026-10-07 before any 2024 or 2025 observation was collected or scored.

The first [direct-outcome candidate](predictive-model-protocol.md) failed its
2022–2023 winner checks. After correcting the provider's `Lapped` finisher
status, winner Brier was 0.605225 against the earlier-wins reference's
0.587105, a 3.1% regression. Its podium and expected-position errors improved
14.6% and 10.4%, respectively, but that does not override the winner failure.
The original forecasts, scores and parameter lock are retained.

The revised experiment adds strictly earlier, smoothed **season** winner and
point shares and logarithmic earlier-performance features. The first model
used short rolling histories, which lost evidence of sustained dominance.
Fit remains 2008–2018. All inspected 2019–2023 seasons now constitute the
development/selection set; 2022–2023 will not be presented as independent
confirmation for this revision. Coefficients, history rules, ranking and
retirement heads are frozen before collecting or scoring the untouched final
2024–2025 seasons. No test-driven retry is allowed on that final cohort.

The acceptance thresholds, references, complete-roster assumptions, equal
event weighting, per-season checks, bootstrap, removal of the three largest
gains and live-season deployment guard remain exactly those in the original
protocol. The revised experiment must pass the full final-test bar; an
improvement on the now-inspected development races is insufficient.

Outcome: the revision failed the winner bar on its 48 final events. Winner
Brier was 0.822656 against 0.837825 for the best fixed reference, only 1.8%
better. Its paired event bootstrap included zero gain and excluding the three
largest gains reversed the result. Later work treats these seasons as already
inspected research data. See the [qualifying improvement and remaining winner
limits](practice-qualifying.md); the original winner goal is not declared complete.
