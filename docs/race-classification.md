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

Both engines report `gap_to_leader = 0` for a retired car, including a classified
retirement. Its `total_time` still records its last completed crossing; that
partial race clock is not a finishing time gap. Finished cars retain their
elapsed gap to the winner, and displays use known lap deficits where available.
API responses and CSV exports preserve these same result fields.

The CLI race table also shows completed laps and `NC`, and labels classified
retirements explicitly. Race CSV exports append `laps_completed` and `classified`
after the existing columns. The latter uses `true` or `false`; unknown legacy lap
counts remain blank. CSV `position` retains the raw ordinal rank for analysis.
Statistics JSON includes `probability_intervals` with the same 95% Wilson
sampling ranges and per-driver trial counts as the dashboard. These ranges
describe Monte Carlo sampling noise, not accuracy against a real race outcome.

## Time-limited finishes and points

The leader's modeled racing clock is checked after each completed lap. Once it
reaches two hours, the following lap becomes the final lap, capped by the
scheduled distance. This follows B2.5.3(a) of the [2026 Sporting Regulations,
Issue 09](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf).
Pit decisions and free restart tyre choices anticipate a shorter horizon from
the leader's elapsed clock and observed running pace, then obey the actual
final-lap announcement once it occurs. This estimate never declares the finish;
actual laps and weather-strategy forecasts retain the original scheduled fuel
distance. No further
weather update, tyre fitting or incident is generated after the finish.

Following a red-flag suspension, the first counted resumption lap is behind the
safety car. It contributes distance and can announce timed expiry, but cannot
earn green-lap credit for points. The following green lap has the normal restart
restriction on Overtake Mode. See [resumption timing and limits](strategy-model.md#red-flag-suspension-timing).

Results expose `race_time_limited` and `points_awarded`. The dashboard and CLI
identify a time-limited finish and show actual points; CSV appends both fields.
Classification uses the winner's completed distance. Points use the original
scheduled distance, with reduced schedules below 25%, 50% and 75%, and require
two consecutive complete green laps, following A2.2.1 of the [2026 General
Provisions, Issue 03](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_a_general_provisions_-_iss_03_-_2026-06-25.pdf).
In this lap model, a completed leading lap that encounters SC, VSC or a red
flag breaks that consecutive-lap sequence. A scheduled SC/VSC announcement
occurs after its chosen crossing and preserves that completed lap's green
credit, including at the finish. Automatic and forced interventions sampled
during a lap still invalidate it. The chronological engine retains control
exposure across the entire own lap, including paid service and later signals:
a green track-entry snapshot or subsequent clearance cannot erase that
exposure if a lapped car later inherits the lead. Earlier valid pairs remain
valid. A classified retirement can earn points, while a scoring
finish probability counts positive awards rather than merely a top-ten place.
Legacy result objects without an explicit points award retain the previous
classification-based full-points fallback.

Consecutive green credit follows completed leader lap numbers, independently
of the shared control clock. If A completes lap one and retires before lap two,
a lapped successor completing its own lap one has not established a two-lap
pair. Consecutive laps one and two may be completed by different leaders, and
a pair already completed before a retirement remains valid.

## Recorded scoring evidence

Each native race records a shared, immutable `race_points_context` on its result
rows: `scheduled_laps`, `winner_laps` (null without a finishing winner),
`has_two_green_laps`, and policy `race_distance_points_2026_v1`. This is evidence
of the lap model's scoring decision. Explicit abandonment scenarios instead
use the recorded historical finish described in [abandoned races](race-abandonment.md).
Reusing a simulator replaces the context without changing old rows;
an empty usable grid records a no-winner outcome on the simulator itself.

Monte Carlo results retain one context per observed trial, including an empty
race. JSON statistics and comparison exports include `race_scoring_contexts`
and `race_scoring_statistics`. Full, reduced and zero-point counts use only
consistent recorded evidence; coverage counts observed races, not the requested
trial count. `zero_points_race_rate` is a fraction over races with known scoring
evidence, and is null when that denominator is zero.

The dashboard shows the representative race's scoring explanation and coverage
across recorded trials. API rows expose nullable `points_reason` and
`points_explanation`; reasons distinguish full or reduced schedules,
insufficient laps, missing green pairs, no winner, unclassified cars, and
positions outside the applicable points schedule. Console and HTML reports
display scoring evidence separately from finish probabilities. CSV appends
`race_points_policy`, `scheduled_laps`, `winner_laps`, `has_two_green_laps` and
`points_reason` when any trial has known scoring evidence; unknown trials have
blank cells. CSV files containing only legacy rows retain their existing columns.

Missing or malformed contexts, inconsistent awards, and mismatched winner
distances remain unknown. Legacy numeric points keep their previous fallback;
they do not establish full-points eligibility or explain an explicit zero award.

## Limits

The default chronological engine executes individual crossings and lapped-car
finishes. The optional standard engine is a synchronous lap simulation:
surviving cars complete the same lap count. Both implement explicit
abandonment scenarios using historical crossings; they do not predict whether
a suspension can resume. Resuming red flags preserve completed crossings and include field
collection plus a fixed pause in elapsed time. The shared finish clock extends
the two-hour threshold by accumulated suspension time, capped at one hour;
the already announced final lap remains fixed. See [suspension timing and limits](strategy-model.md#red-flag-suspension-timing).
Pit planning assumes observed pace continues, with current control on the upcoming lap and
green running thereafter. Future stops, traffic changes, incidents and weather
can make its estimated finish distance wrong; it is recalculated each lap.
The [chronological crossing design](chronological-race-design.md) describes the
finish controller and its execution conventions and limits.
If every car retires, there is no modeled winner and no classification or points.
The simulation records the final retirement incidents and stops. It does not
deploy new SC, VSC or red flags once that lap has no survivors, including forced
signals scheduled for the retirement lap. Subsequent scheduled laps do not
evolve weather or generate race-control events. An empty usable grid
likewise produces no laps or events, after resetting state from the previous run.
Zero-lap retirements are always unclassified. These conventions must not be
interpreted as a complete implementation of FIA race-ending regulations.

Deterministic tests cover rounded thresholds, crossing order, failed-lap clocks
and fastest laps, red flags, all-car retirement, classified retirement awards,
legacy results, serialization, and desktop/mobile dashboard presentation.
