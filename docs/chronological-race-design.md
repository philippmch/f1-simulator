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
the existing models and return `RaceResult` objects. The standard engine remains
the default. Select chronological execution through `--race-engine chronological`
in the CLI, **Race model: Lap-aware (experimental)** in the dashboard, or
`race_engine="chronological"` on `MonteCarloRunner` and the API request.
Both sequential and process-pool execution preserve per-run seeds and freshly
simulated qualifying. API summaries, JSON exports, export history and HTML
reports record the selected engine. This exposes the experiment; it does not
establish empirical calibration or change the default.

The experimental engine schedules individual crossings and pit exits on an
absolute timeline. A persistent constructor queue accounts for staggered box
arrivals. A circular physical order constrains crossings: a faster provisional
clock requires a passing outcome or a compliant blue-flag yield before the car
can cross its predecessor.
Race classification separately accounts for completed distance, so a retired
car can outrank a finisher who completed fewer laps.

Independent constant-pace tests cover two- and four-car fields, equal clocks,
and lapped cars tied with the winner's flag. With 90/110/150/200-second cars in a
ten-lap race, free-running results are 900/990/900/1000 seconds over 10/9/6/5 laps.
Those tests also count actual lap calls, original fuel denominators and tyre age.
The display adapters render known lap deficits as `+1 lap` or `+2 laps`; unknown
legacy distances retain time gaps.

Pit planning estimates the flag time from the leading pending crossing, stored
free-running pace and the time-limit deadline, then maps that time to the car's
own remaining laps. Completed suspension time extends the deadline up to the
existing one-hour cap. The estimate includes the lap following clock expiry.
Committed pit delay affects the pending crossing; past service and blocked time
do not become the forecast's recurring lap pace. The forecast uses no random
draws and does not alter the actual finish boundary. It assumes continued pace
with current control on the upcoming lap and green running thereafter, without
predicting future incidents, weather changes or stops. Initial laps without an observed pace retain the
scheduled horizon.

Chronological pit decisions receive an immutable `StrategyTrafficSnapshot`.
The gap ahead comes from the circular physical predecessor; the space behind
comes from the successor's pending crossing, including lapped traffic. These
inputs do not use race rank or old completed-crossing clocks. Production callers
without a snapshot retain their existing strategy behavior.

The rejoin forecast uses expected service, known team queue delay and the current
pit-lane factor. It preserves rivals' committed running and pit delays, then
projects their observed free pace under current control conditions. It prices
the difference in one lap's dirty air between rejoining and staying out; the
planner applies the existing weather multiplier. SC/VSC contribute no green
traffic penalty. Known pit exits can create rejoin traffic, while terminal cars
are excluded. Forecasting neither mutates live state nor consumes random draws.
This is a free-running forecast: future stops, incidents, battle delays and
weather changes remain unknown, and the estimate never determines actual order.

Red flags now hold the field for a shared restart. Already-running laps finish
their committed work once, with passing disabled during collection. Completed
cars wait; cars still receiving paid service wait at the closed pit exit and
run their existing lap only after release. Paid service is not sampled or
charged again. The restart preserves completed distances and known on-track
order, including passes made before the signal but before a line crossing.
Cars in service retain their last recorded rank slots. All survivors receive
one free tyre choice using the shared restart weather.

The common restart occurs after collection plus `red_flag_pause_seconds`, which
defaults to 600 seconds and can be configured on `ChronologicalRace` or
`simulate_chronological_race`. This is a ten-minute notice assumption, not a
calibrated estimate of incident clearance. Zero is available for controlled
experiments. The `suspensions` trace records signal time, restart time and order.
Absolute crossing and pit-exit times include the wait; previous crossings are
never rewritten. The lap containing the wait retains its original start time,
so waiting cannot manufacture a fastest lap. Strategy forecasts continue to use
free-running pace rather than treating a suspension as recurring lap time.

The finish clock adds completed suspension intervals, including collection, to
the two-hour threshold, with a maximum extension of one hour. An already
announced final lap stays fixed. This follows the timing framework in
[FIA sporting regulations B2.5.3, B5.14.2 and B5.15.2](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf).
Collection remains a lap-resolution approximation: it does not recalculate
partially driven sectors at reduced speed. The model also omits the detailed
restart formation procedure, abandonment and results countback. Production
dispatch still uses its existing instantaneous red-flag abstraction.

Running physics uses physical gaps for dirty air and Overtake Mode detection.
The gap is estimated from the preceding on-track car's progress through its
pending lap, independently of race rank and completed distance. Weather, control,
mode eligibility and restart conditions are captured for each started lap.
A paid stop samples running physics once at its actual pit exit, using the
rejoin gap; it cannot deploy Overtake Mode on that lap. Passing retains the
original detection decision, and energy recharges once per completed own lap.

Full safety-car catch-up closes gaps through future running time, preserving
every previous crossing and pit exit. Followers use the queue leader's neutralized
pace and approach a one-second gap, bounded below by their own free-running pace.
The target is the mean of the existing 0.8–1.2-second bunching model, not an
empirically calibrated value. A slower follower cannot exceed its free pace to
join the queue. VSC applies its running-time modifier without this compression.
These are lap-resolution approximations; changes do not rewrite pending laps'
starting control snapshots.
An unrun paid lap held at a closed pit exit is an exception: its tyre and control
snapshot is replaced at the shared red-flag restart before its first physics call.

When a car catches its physical predecessor on a higher own lap, the predecessor
yields without sampling a defensive battle. This models compliant blue-flag
behavior at the scheduler's encounter point. Same-lap attacks and attempts to
unlap still use the ordinary passing model. Either car's neutralized starting
snapshot prevents the yield. Pit-lane cars are absent from the on-track order.
The [2026 FIA driving standards, section K](https://api.fia.com/sites/default/files/2026_f1_driving_standards_guidelines.pdf)
describe yielding at the first opportunity, with allowance for the next straight.
The engine has no sector geometry, so it does not model that wait, noncompliance
or penalties, or add an uncalibrated time loss for yielding.

Production dispatch remains unchanged while entry-point integration and
end-to-end contracts are validated for chronological execution. Detailed restart
formation and abandonment remain model limitations, rather than features of the
existing production loop that have not yet been migrated.
The existing minor-contact time losses and personal spin/puncture/crash outcomes
are already reused; a richer damage-severity model would improve both engines
rather than close a migration gap.

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

A timed final-lap announcement now belongs to the next authoritative leading
crossing. If the leader retires, a same-distance or lapped successor takes the
flag at that crossing; it does not receive a fresh countdown or have to reach
the retired driver's personal lap number. `final_lap` retains the original
announced distance for reference, while each driver's finish ledger retains its
actual completed distance. Suspensions do not cancel an existing announcement.
Pit planning uses the current leader's next crossing once that announcement is
latched. Merely matching another active car's already-completed distance cannot
make a trailing car the leader; the crossing must advance the active lead.

For this rare handoff, the model places the timed flag recipient first and orders
the remaining cars by completed distance and crossing time. Thus an earlier
retiree can retain more completed laps than the winner and still rank ahead of
another finisher. Classification eligibility and reduced points continue to use
the winner's actual distance. This is an explicit interpretation of B2.5.3 and
B2.5.5, not a claim that an official precedent for this combination was found.
The synchronous production loop cannot encounter this distance reset.

An executable regression has A complete four laps at 7200 seconds, announce the
flag, then retire at 7250. B receives it on its second lap at 7300, and C finishes
its second lap at 7600. No B lap-three physics or pit decision runs. The result
is marked time-limited even when the original announcement matched the scheduled
distance cap; A retains all four completed laps.

## Remaining integration

The experimental scheduler distinguishes a lap's immutable starting conditions
from consequences committed during that lap. Production migration still needs
the complete strategy, race-control and battle behavior on this timeline.
Tyre history, service draws, energy, incidents and fastest laps cannot be
reconstructed by trimming final results. The production loop's shallow
`replace(state)` snapshots share mutable driver state and cannot serve as
speculative transactions.

Elapsed crossing time must be monotonic and distinct from relative racing gaps.
The production safety-car bunching code replaces trailing cars' `total_time`
values to close gaps. The experimental engine instead uses the bounded future
catch-up described above. Its pit merges and blocked-car reconciliation use
physical ordering independently of completed distance; production migration
must retain those distinctions.

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

For an offline comparison using the same frozen inputs, run
`python examples/compare_race_engines.py saved_statistics.json --simulations 100 --export`.
Use `--scenario NAME` when the saved file contains multiple scenarios. Both engines
retain the saved roster, cars, track, weather policy, starting-tyre overrides and
base seed. Only execution-model selection changes; the source file is untouched.
Automatic opening choices and later strategies remain active and can differ as
the models evolve. Identical seed ranges do not guarantee identical future random
events. The comparison therefore describes model sensitivity rather than a
controlled estimate of one isolated mechanism or proof of real-race accuracy.

Comparison exports include recorded winning distance, time-limited races,
lapped finishers, driver outcomes, sampling intervals and actual tyre sequences.
Each engine's trials can be replayed from the combined JSON with its engine name
as the replay scenario. Reproduction uses installed code and dependencies.

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
  strategy planning forecasts clock expiry and obeys the announced finish.
- Repeated races reset all crossing, finish and queue state. Existing strategy,
  neutralization, retirement and output contracts remain covered by tests.
