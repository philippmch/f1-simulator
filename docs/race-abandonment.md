# Abandoned races and historical countback

An explicit [race-control schedule](control-schedules.md) can request a red flag
with `action="abandon"`. Both engines collect the field and finish the configured
suspension wait, then return a historical classification. They do not resume,
fit new tyres, or evolve weather after the decision. The supplied action is an
experimental assumption, not a prediction that weather or an incident prevents resumption.

```powershell
python examples/simulate_race.py --scenarios dry --no-parallel `
  --control-schedule 12:red:abandon
python examples/check_abandonment.py
```

## Crossing convention

[FIA sporting regulations B5.14.2](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf)
use the penultimate lap before the lap during which suspension was signalled
when a race cannot resume. A scheduled request occurs **after** a completed
leading crossing. After own lap N, the leader is entering N+1, so this model's
countback is N-1. The shared control ordinal and the leader's own lap can differ
after a lapped successor inherits the lead; both are recorded separately.

The original leading crossing at countback lap K becomes the historical finish
signal. Each other car finishes its current own lap at its first completion
at or after that clock. A car already retired retains its last completed
distance. Later retirements cannot turn an earlier historical finisher into a DNF.
The chronological engine therefore keeps lapped finishers on their own distances;
the standard engine retains its synchronous completion convention.

The returned pit stops, fastest lap, strategy, physical set ledger and pit-plan
outcomes describe this historical finish. Later running, stops and refits earn
no result credit. An unresolved instruction is `not_reached/race_abandoned`.
Later collection and decision waiting are recorded as race-wide suspension time,
but are excluded from these historical crossing clocks. A suspension already
completed before the historical finish remains in its crossing times.

After leading lap one there is no completed countback lap. Every row has
`status="no_result"`, zero laps, no classification and no points. This is not
a retirement, victory or podium. A positive-distance countback uses the existing
90% classification threshold and distance-based points schedules. Two consecutive
complete green leader laps must exist in the retained history; a pair completed
only after the historical finish cannot qualify the race for points.

## Dry-compound penalties

[Sporting rule B6.3.6](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf)
adds 30 seconds when a suspended race cannot restart and a driver has not met
the dry-compound requirement; intermediate or wet use waives it.

The simulator checks each driver's actual completed tyre use through suspension,
including running after the countback finish. A fitted set that never completed
a running lap does not satisfy its existing usage model. Two used dry compounds,
or a used intermediate/wet compound, avoid the penalty. Zero-distance results
and cars with no completed tyre use receive no penalty.

The penalty is added to historical elapsed time before positions, gaps and points
are recalculated. It can change the winner. A later compound change can therefore
avoid the penalty while its paid stop and stint remain outside the countback
result. Per-driver `abandonment_tire_rule` records the compounds considered,
penalty, reason and policy, separately from the historical strategy ledger.

## Evidence and replay

Result rows share immutable `race_abandonment` metadata: announcement ordinal,
signal leader lap, countback lap, historical finish clock, signal clock,
decision clock and policy `after_crossing_penultimate_lap_2026_v1`.
`get_race_abandonment_context()` requires consistent field-wide metadata,
tyre-rule records and scoring. Ordinary races, missing evidence and malformed
records return no verified abandonment context.

Statistics and comparison JSON include `race_abandonment_contexts`,
`abandonment_statistics` and `abandonment_tire_rules`. Race CSV appends
`race_abandonment` and `abandonment_tire_rule` JSON columns when such evidence
exists. Reports and the dashboard explain countback and penalties; no-result
rows use neutral labels. Scheduled red outcomes use the existing global control
history, with an additional CSV `action` column. SC/VSC-only CSV keeps its earlier columns.

Red-flag inputs use schema 12 and `observed_control_schedule_v2`, or schema 13
and `observed_control_schedule_v3` if the same schedule also contains a
[compulsory full-wet resumption](wet-resumptions.md). Replay retains
tyre inventories, usage limits, fixed/window plans, warmup, prescribed rainfall,
qualifying weather and random-stream policy. Native tests compare process workers
and replay with serial execution, and compare countback results against independently
observed production crossings in the matching resumption scenario.

## Limits

The simulation records lap crossings, not partial sectors or exact pit-lane
Control Line geometry. Suspension collection and the fixed default wait remain
approximations. It does not infer abandonment from weather recovery, enforce an
event-specific mandatory dry-specification subset, or implement special full-wet
initial SC starts, standing grids, unlapping or additional director-ordered procedures.
Automatic and forced red flags continue to request resumption. An explicit red
request at an already finished scheduled or timed crossing is suppressed as
`race_finished`. These conventions model the supplied scenario and are not
a complete implementation of FIA race-ending rules.
