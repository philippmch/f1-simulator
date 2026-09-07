# Race classification and retirement

The simulator tracks operational status separately from classification. A car
can retire and still receive a classified position, points, and a podium result.
It also remains a DNF in reliability statistics.

Race, qualifying and Monte Carlo inputs require unique driver IDs. A supplied
starting grid must also contain each ID at most once. Duplicate IDs raise a
`ValueError` before simulation state or random draws are consumed, rather than
silently overwriting an entrant or classifying the same driver twice. IDs remain
case-sensitive. Existing partial-grid behavior is unchanged: race entries without
a matching driver or car are skipped, and an unusable grid yields no results.

Qualifying retains an entrant without a matching car as eliminated in Q1 with
no recorded lap. Unavailable qualifying times are `null` in API responses,
blank in CSV exports and shown as "No time" in console output. Nonfinite times
from legacy result objects receive the same treatment. This keeps incomplete
fields serializable without dropping entrants or inventing a lap time.

The console qualifying table displays Q1, Q2 and Q3 separately. Its `Best`
column is the fastest lap across all sessions, not the lap that determines
every grid position. There is no cross-session gap to pole: an eliminated
driver can have a faster earlier-session lap while correctly starting behind
drivers who advanced further.

The distance threshold follows B2.5.5 of the [FIA 2026 Sporting Regulations,
Issue 08, 5 August 2026](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf):
cars below 90% of the winner's completed laps, rounded down to whole laps, are
not classified. Both a 60-lap and a 61-lap winner therefore require 54 completed
laps. Retirement ordering uses completed distance, then the order at the last
completed crossing.

## Discrete-lap convention

An incident sampled on lap N occurs during that lap. A retirement therefore
records N−1 completed laps, retains the preceding crossing's clock and order,
and cannot set a fastest lap on the failed lap. A valid earlier fastest lap is
retained. The model does not estimate a partial-lap retirement timestamp.
Red-flag bunching does not rewind completed distance.

Results expose `laps_completed` and `classified`. The dashboard labels retired
cars as classified or not classified, displays completed laps, and uses `NC`
for unclassified positions. Eligible late retirements count toward points,
podiums, and top-N probabilities while still counting toward DNF probability.
Raw ordinal position distributions retain every car, including unclassified
retirements. Legacy result objects without classification metadata treat only
finished cars as eligible.

The CLI race table also shows completed laps and `NC`, and labels classified
retirements explicitly. Race CSV exports append `laps_completed` and `classified`
after the existing columns. The latter uses `true` or `false`; unknown legacy lap
counts remain blank. CSV `position` retains the raw ordinal rank for analysis.
Statistics JSON includes `probability_intervals` with the same 95% Wilson
sampling ranges and per-driver trial counts as the dashboard. These ranges
describe Monte Carlo sampling noise, not accuracy against a real race outcome.

## Limits

This is a synchronous lap simulation: surviving cars complete the scheduled
distance. It does not yet model lapped-car finishing, time-limit endings,
abandonment classification, or shortened-race points and green-lap requirements.
If every car retires, there is no modeled winner and no classification or points.
The simulation records the final retirement lap and stops; subsequent scheduled
laps do not evolve weather or generate race-control events. An empty usable grid
likewise produces no laps or events, after resetting state from the previous run.
Zero-lap retirements are always unclassified. These conventions must not be
interpreted as a complete implementation of FIA race-ending regulations.

Deterministic tests cover rounded thresholds, crossing order, failed-lap clocks
and fastest laps, red flags, all-car retirement, classified retirement awards,
legacy results, serialization, and desktop/mobile dashboard presentation.
