# Chronological race crossings

The current production race loop advances every surviving driver once per
leader lap. This gives surviving cars equal completed distances. A controlled
ten-lap race with constant 90-second and 110-second cars currently ends at
900/1100 seconds and 10/10 laps. With chronological crossings, the slower car
should take the flag at its next crossing after the winner: 990 seconds, lap 9.
Its tenth lap must never start.

This document tracks the engine change needed to remove that limitation.
It is not a claim that the production engine already supports lapping.

## Experimental execution

`simulation/chronological_race.py` provides `ChronologicalRace(simulator).run(...)`
and `simulate_chronological_race(...)` for explicit Python experiments. They use
the existing models and return `RaceResult` objects. The normal CLI, web and
Monte Carlo entry points still use the production engine.

The experimental engine schedules individual crossings and pit exits on an
absolute timeline. A persistent constructor queue accounts for staggered box
arrivals. A circular physical order constrains crossings: a faster provisional
clock requires a passing outcome before the car can cross its predecessor.
Race classification separately accounts for completed distance, so a retired
car can outrank a finisher who completed fewer laps.

Independent constant-pace tests cover two- and four-car fields, equal clocks,
and lapped cars tied with the winner's flag. With 90/110/150/200-second cars in a
ten-lap race, free-running results are 900/990/900/1000 seconds over 10/9/6/5 laps.
Those tests also count actual lap calls, original fuel denominators and tyre age.
The display adapters render known lap deficits as `+1 lap` or `+2 laps`; unknown
legacy distances retain time gaps.

Pit planning estimates the flag time from the leading pending crossing and
stored free-running pace, then maps that time to the car's own remaining laps.
Committed pit delay affects the pending crossing; past service and blocked time
do not become the forecast's recurring lap pace. The forecast uses no random
draws and does not alter the actual finish boundary. It assumes continued pace
and the current neutralization modifier, without predicting future incidents,
weather changes or stops. Initial laps without an observed pace retain the
scheduled horizon.

This is not yet a replacement for the full production model. Running currently
uses clean-air pace; dirty-air gaps, Overtake Mode energy and detection snapshots,
restart passing boosts and a dedicated blue-flag yielding model remain to be
integrated. Safety-car bunching is also pending: current control snapshots apply
to newly started laps without moving or rewinding already running cars. These
are migration requirements, not intentional simplifications of the final app.

## Finish boundary

`RaceFinishClock` in `simulation/race_timing.py` owns the original scheduled
distance, the announced final lap and the leader's chequered clock. Production
race simulation and isolated opening-strategy projections use it. Leader
crossings advance the leading distance even when the leading driver's identity
changes. Time must be finite and monotonic. A retirement can hand the lead to
a car on an earlier lap; that explicit transition must preserve the racing
clock while allowing the leading distance to change. Invalid observations must
leave the clock unchanged.

`RaceFinishTimeline` adds per-driver crossing records and terminal states. It
accepts chronological observations; the caller resolves exact timestamp ties
with the leader first. Once the winner takes the flag, other cars remain racing
until their own next crossing or retirement. A car cannot start another lap
after its own finish. Retirement must not manufacture a crossing or a winner.
This layer is used by the experimental scheduler but not the production loop.

One experimental transition remains deliberately unsupported: a same-distance
or lapped successor taking over after a two-hour final-lap announcement. The
timeline rejects that transition without changing its state. Recomputing the
announced lap from the successor's own distance would silently change the finish
rule; freezing the old driver's lap number is not sufficient either. Define
the finish signal and classification together when integrating the scheduler,
including the case where a retired car has completed more laps than a survivor.
The current synchronous production loop cannot encounter this distance reset.

## Remaining integration

The experimental scheduler distinguishes a lap's immutable starting conditions
from consequences committed during that lap. Production migration still needs
the complete strategy, race-control and battle behavior on this timeline.
Tyre history, service draws, energy, incidents and fastest laps cannot be
reconstructed by trimming final results. The production loop's shallow
`replace(state)` snapshots share mutable driver state and cannot serve as
speculative transactions.

Elapsed crossing time must be monotonic and distinct from relative racing gaps.
The current safety-car bunching code replaces trailing cars' `total_time` values
to close gaps. Pit merges and blocked-car reconciliation also assume equal lap
counts. These operations need explicit physical ordering and lap deficits before
they can be used with asynchronous crossings.

Pit arrivals require a persistent per-team service queue on the absolute
timeline. Decisions use expected service; execution samples service once.
Only laps actually started may reserve the box, change tyres or consume random
draws. A trailing car can still pit or suffer an incident after the winner takes
the flag and before its own finish. The winner's crossing therefore cannot be
used as a global cutoff for all remaining car events.

Race-control countdowns and surface evolution cannot advance once per car.
They need a shared race timeline, while mechanical exposure, tyre age, fuel and
lap completion follow each individual car. Define the resolution of control
transitions during pending laps explicitly and apply it consistently to pace,
passing and event probabilities.

Classification must order cars by actual completed distance and crossing order.
API, CSV, console and dashboard output then need lap deficits alongside time
gaps. The existing classification threshold and reduced points still use the
winner's actual distance and the original scheduled distance respectively.

## Acceptance scenarios for the scheduler

- The 90/110-second fixture finishes at 900/990 seconds and 10/9 laps, with no
  slow-car lap-10 physics, pit service, event or random draw.
- Equal-clock crossings have deterministic ordering; the winner takes the flag
  before any trailing crossing tied at that clock is adjudicated.
- A trailing car can retire between the winner's finish and its own crossing.
- Staggered teammate pit arrivals preserve queue delay across different laps.
- Neutralization closes relative gaps without rewinding absolute time.
- Passing uses physical adjacency rather than confusing race rank with track
  position; a lapped car does not gain a race position by unlapping itself.
- Fuel uses each car's own lap count and the original fuel schedule; time-limited
  strategy planning uses the announced finish horizon.
- Repeated races reset all crossing, finish and queue state. Existing strategy,
  neutralization, retirement and output contracts remain covered by tests.
