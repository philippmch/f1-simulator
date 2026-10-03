# Chronological race crossings

The default chronological engine advances individual cars on their actual
crossing clocks. In a controlled ten-lap race with constant 90-second and
110-second cars, the slower car takes the flag at its next crossing after the
winner: 990 seconds, lap 9. Its tenth lap never starts. The optional standard
race loop gives surviving cars equal completed distances: 900/1100 seconds and
10/10 laps for the same fixture.

This document describes the implemented chronological engine, its execution
conventions and its remaining model limits.

## Execution

`simulation/chronological_race.py` provides `ChronologicalRace(simulator).run(...)`
and `simulate_chronological_race(...)` for direct Python use. They use the
existing models and return `RaceResult` objects. Chronological execution is the
default for new `MonteCarloRunner` runs, CLI simulations, API requests and the
dashboard's **Lap-aware** selection. Race-probability evaluation uses the same
new-run default. Either engine can be selected explicitly through
`--race-engine chronological` or `--race-engine standard`, and the corresponding
`race_engine` value on Python and API calls.
The race-probability evaluation CLI uses `--engine` for the same choice.
Both sequential and process-pool execution preserve per-run seeds and freshly
simulated qualifying. API summaries, JSON exports, export history and HTML
reports record the selected engine. Individual crossing execution does not
establish empirical calibration or complete implementation of sporting rules.

The lower-level `RaceSimulator.simulate_race(...)` method retains its standard
execution semantics. Direct chronological callers use the helpers above.

The chronological engine schedules individual crossings and pit exits on an
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
Retired cars report a zero time gap, consistent with standard execution. Their
last completed crossing time and distance remain available independently;
subtracting that partial clock from the winner's finish would produce a
misleading negative gap. This also applies to classified retirements.

Pit planning estimates the flag time from the leading pending crossing, stored
free-running pace and the time-limit deadline, then maps that time to the car's
own remaining laps. Completed suspension time extends the deadline up to the
existing one-hour cap. The estimate includes the lap following clock expiry.
Committed pit delay affects the pending crossing. While service is unfinished,
the expected exit and any pending first-lap fitting cost set the projected
outlap crossing. The fitting cost is added once, after current control scaling;
it does not delay pit exit. An on-track pending crossing already contains it.
Past service, fitting costs and blocked time do not become the forecast's
recurring lap pace. The forecast uses no random draws and does not alter the
actual finish boundary. It holds observed free pace through the remaining leading
control intervals, followed by green running. It does not predict future
incidents, weather changes or stops. Initial laps without an observed pace retain
the scheduled horizon.

Chronological pit decisions receive an immutable `StrategyTrafficSnapshot`.
The gap ahead comes from the circular physical predecessor; the space behind
comes from the successor's pending crossing, including lapped traffic. These
inputs do not use race rank or old completed-crossing clocks. Production callers
without a snapshot retain their existing strategy behavior.

The rejoin forecast uses expected service, known team queue delay and the current
pit-lane factor. It preserves rivals' committed running and pit delays, including
a pending fitting cost in the first projected crossing, then projects their
observed free pace under current control conditions. It prices
the difference in one lap's dirty air between rejoining and staying out; the
planner applies the existing weather multiplier. SC/VSC contribute no green
traffic penalty. Known pit exits can create rejoin traffic, while terminal cars
are excluded. Forecasting neither mutates live state nor consumes random draws.
This is a free-running forecast: future stops, incidents, battle delays and
weather changes remain unknown, and the estimate never determines actual order.

Weather-cost planning also maps the estimated leading crossings onto each
car's future lap starts. The immutable `weather_intervals` tuple records
cumulative surface updates from the current snapshot, starting at zero; repeated
counts and skipped intervals are allowed. Rain-stint selection, compound
transitions and the reactive weather-stop cost check all use the same tuple,
including their cached future stints. The standard engine retains its ordinary
one-update-per-lap projection. Pending physics and actual shared weather updates
are unaffected by these forecasts.

When more than one SC/VSC interval remains and committed field observations
are usable, a private crossing projection supplies the candidate's own starts,
the projected flag, and explicit leading weather-update times. It retains SC
physical order, old no-passing restrictions and known fitting delays without
advancing the live engine or sampling future service. The flag supplies no
weather update. Unresolved repairs, rival tyre instructions or pace observations
keep the simpler held-pace forecast. This corrects timing and entry surfaces;
the existing cost planners still price later running and paid stops as green.

When the projected leader is another car, weather-cost planning also advances
that clock through the candidate's own planned pit delays. A stop's compound is
chosen using the pre-service surface, but its outlap and subsequent running use
the delayed surface. Current queue, expected service and lane time contribute
physical delay; later paid stops use expected green pit loss. Free restart fits
add no delay. Traffic cost adjustments remain separate from elapsed time.
This clock is shared by rain, weather-stop and finite-pool costs, with the same
timing carried into damp fallback stint comparisons. The candidate's observed
free pace and the external leading clock stay fixed within a forecast.

The candidate leader keeps the existing own-lap projection: its pit time cannot
create leading crossings while it is stationary. Initial decisions without an
observed pace and unchanging surfaces also retain the existing path. Future rival
decisions, random weather and changed free pace remain unknown; these cost
comparisons do not change the actual finish boundary or fit tyres twice.

In a controlled example, a 90-second leader next crosses at time 180 while a
follower commits a stop at 170. Expected service plus a 22-second lane loss puts
rejoin at 194.751. With wetness 0.44 and persistent rainfall 0.6 at the decision,
the outlap uses projected wetness 0.472 rather than 0.44. The shared lap model
prices the fresh intermediate outlap about 0.64 seconds slower. Under drying
conditions, the correction can instead lower the outlap cost. These are model
consistency checks, not measured real-world gains.

Constant-pace tests compare every projected future surface with actual lap-start
snapshots across faster, equal-pace and lapped cars, including timed finishes.
With a 90-second leader and a 180-second follower under fixed rainfall, the
follower's second-lap wetness is 0.288. Its next-lap forecast now reaches 0.47232,
matching the two shared updates, instead of the former one-update value 0.3904.
The estimate uses expected service for a leader still in the pits and preserves
the finish forecast's assumptions about future pace, stops and control.

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
[FIA sporting regulations B2.5.3, B5.14.2 and B5.15.2](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_09_-_2026-10-01.pdf).
Collection remains a lap-resolution approximation: it does not recalculate
partially driven sectors at reduced speed. The model also omits the detailed
restart formation procedure, abandonment and results countback. The standard
engine also uses collection plus a shared pause, but collects at the end of a
synchronous lap; see [standard suspension timing](strategy-model.md#red-flag-suspension-timing).

Running physics uses physical gaps for dirty air and Overtake Mode detection.
The gap is estimated from the preceding on-track car's progress through its
pending lap, independently of race rank and completed distance. Weather, control,
mode eligibility and restart conditions are captured when track running starts.
A paid stop samples running physics once at its actual pit exit, using the
current weather, control and rejoin gap; it cannot deploy Overtake Mode on that lap.
Weather evolution and SC/VSC deployment or clearance during service therefore
affect the upcoming running. Lane loss, queue and service remain charged from
the committed stop, and the fitted set is retained. Changing conditions at exit
does not grant another tyre change. SC catch-up uses the queue observed at track
entry, including a safety car deployed while the car was in service. Passing retains the
original detection decision, and energy recharges once per completed own lap.
Recharge uses the running conditions captured for that own lap, so a control
deployment detected at its completion affects subsequent running rather than
retroactively changing its energy gain. The standard engine uses the same
completed-lap convention.

Full safety-car catch-up closes gaps through future running time, preserving
every previous crossing and pit exit. Followers use the queue leader's neutralized
pace and approach a one-second gap, bounded below by their own free-running pace.
The target is the mean of the existing 0.8–1.2-second bunching model, not an
empirically calibrated value. A slower follower cannot exceed its free pace to
join the queue. VSC applies its running-time modifier without this compression.
These are lap-resolution approximations; changes do not rewrite pending laps'
starting control snapshots.
An active SC, VSC or red flag immediately prohibits on-track passing and
blue-flag yielding, including encounters between cars that started under green.
Already completed passes retain their physical order. A lap started under
neutralization keeps its passing restriction through that crossing even if the
signal clears meanwhile; sector-level restart timing is not modeled.
An unrun paid lap held at a closed pit exit waits for the shared red-flag restart.
Its free restart tyre fitting and current running conditions are applied before
its first physics call; paid service is not repeated. Forecasts for cars still
in service use observed current control with expected remaining service, while
cars already running retain their committed first-lap timing. Forecasts neither
read future sampled service nor anticipate future control changes.

Run `python examples/check_pit_exit_conditions.py` for five controlled full-race
checks: drying, SC deployment/clearance and VSC deployment/clearance. Two synthetic
cars use mean lap physics and no incidents. One opening stop has deliberately
extended service so a leading weather/control update occurs before its exit.
JSON records entry conditions, observed exit conditions, the running snapshot,
paid loss and completed running count. This exposes stale entry snapshots without
claiming realistic service duration or optimizing that scripted stop.

When a car catches its physical predecessor on a higher own lap, the predecessor
yields without sampling a defensive battle. This models compliant blue-flag
behavior at the scheduler's encounter point. Same-lap attacks and attempts to
unlap still use the ordinary passing model. Either car's neutralized starting
snapshot prevents the yield. Pit-lane cars are absent from the on-track order.
The [2026 FIA driving standards, section K](https://api.fia.com/sites/default/files/2026_f1_driving_standards_guidelines.pdf)
describe yielding at the first opportunity, with allowance for the next straight.
The engine has no sector geometry, so it does not model that wait, noncompliance
or penalties, or add an uncalibrated time loss for yielding.

Overtaking reports count an attempt only when this encounter reaches the passing
model. The automatic compliant yield above is excluded; ordinary same-lap
attacks and attempts to unlap are included. Counts are kept on the attacker's
result and cover passing-call outcomes, not every collision. Reports pool rates
per attempt and show available driver-race row coverage, as described in
[overtaking-counter reporting](strategy-model.md#overtaking-counter-reporting).
At a catch, passing probability is evaluated at modeled zero gap, while
Overtake Mode eligibility separately uses the gap observed at lap entry; close
pairs that never catch therefore produce no chronological attempt. Its
opportunity trigger differs from the standard engine's proximity gate, so
attempt rates across engines are not directly comparable.

Chronological execution is integrated with the runner, process workers, CLI,
API, dashboard, exports and replay. Detailed
restart formation and abandonment remain model limitations, rather than
features of the standard loop that have not yet been migrated.
The existing minor-contact time losses and personal spin/puncture/crash outcomes
are already reused; a richer damage-severity model would improve both engines
rather than close a migration gap.

## Finish boundary

`RaceFinishClock` in `simulation/race_timing.py` owns the original scheduled
distance, the announced final lap and the leader's chequered clock. Standard
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
This layer is used by the chronological scheduler; the standard loop uses the
shared leader finish clock without individual crossing records.

Race control resolves the completed leading interval before recording its
crossing and possible chequered flag. Final-interval SC, VSC and red flags still
count in the event ledger and break the green-lap sequence used for points.
Their passing restriction remains active for trailing cars. Pending running
times retain their starting conditions, so those cars finish at their actual
next crossings without instantaneous gap compression. A final red flag does
not open a suspension, fit tyres or generate restart state. If the last running
car retires, no new leading interval or control deployment is generated.

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
The synchronous standard loop cannot encounter this distance reset.

An executable regression has A complete four laps at 7200 seconds, announce the
flag, then retire at 7250. B receives it on its second lap at 7300, and C finishes
its second lap at 7600. No B lap-three physics or pit decision runs. The result
is marked time-limited even when the original announcement matched the scheduled
distance cap; A retains all four completed laps.

## Execution and compatibility

The default scheduler invokes strategy, race-control and battle behavior
on its individual-car timeline. It distinguishes a lap's immutable starting
conditions from consequences committed during that lap. Tyre history, service
draws, energy, incidents and fastest laps are recorded during execution rather
than reconstructed by trimming final results. The standard loop's shallow
`replace(state)` snapshots share mutable driver state and cannot serve as
speculative transactions.

New-run Python and API defaults share `DEFAULT_RACE_ENGINE`; the CLI and
dashboard also select chronological execution by default. An omitted API
`race_engine` field therefore changes behavior for existing clients. Send
`"standard"` explicitly to retain synchronous execution. Saved replay uses its
recorded engine and continues rejecting a missing engine. Legacy five-item
worker tuples, manually constructed result objects and display metadata without
an engine retain their standard interpretation. Engine comparisons retain their
explicit standard-first order.

Wet and finite-inventory batches can take substantially longer than Standard.
The dashboard waits for the simulation response and has no automatic run
timeout; its five-second health check is separate. Manual cancellation remains
available, and the server retains capacity until the cancelled worker drains.
Count limits still bound the admitted workload. External proxy deadlines are
not controlled by this application. Choosing this default prioritizes individual
car execution; it does not establish empirical pace or weather calibration.

Elapsed crossing time must be monotonic and distinct from relative racing gaps.
Both engines now use bounded future running for full-SC catch-up. The standard
engine resolves a frozen post-pit queue once per shared lap and collects the
field at completed lap crossings before a common red-flag restart. The
chronological engine's pit merges and blocked-car reconciliation use physical
ordering independently of completed distance. See [standard queue timing and
limits](strategy-model.md#safety-car-queues-and-elapsed-time).

Pit arrivals require a persistent per-team service queue on the absolute
timeline. Decisions use expected service; execution samples service once.
Only laps actually started may reserve the box, change tyres or consume random
draws. A trailing car can still pit or suffer an incident after the winner takes
the flag and before its own finish. The winner's crossing therefore cannot be
used as a global cutoff for all remaining car events.

The chronological engine records each executed stop's arrival, service phase,
car parameters and committed lane loss. Forecasts use those records as an
observation at the current absolute time: a service still in progress uses the
conditional expected remaining duration, while a queued commitment uses the
stationary expected service. A sampled start or end time that lies in the
future is never used to reveal its duration. Records are appended in team
reservation order, so simultaneous arrivals retain FIFO release ordering. A
completed red-flag collection clears the old service records because every
remaining car is then released or fitted from the shared restart state. Small
synthetic fixtures without lifecycle records may continue to provide their
explicit expected exit or box-release fallback; real execution always uses the
observed records.

For the default service distribution, a teammate still in service after six
seconds now contributes about 2.26 seconds of expected additional waiting.
Previously its reservation had already expired and the planner priced zero
queue delay. Actual waiting still uses the sampled completion time; the
forecast does not know whether that stop will finish at eight seconds or later.

Race-control countdowns and surface evolution use shared leading intervals,
while mechanical exposure, tyre age, fuel and lap completion follow each
individual car. Pending laps retain their starting conditions until the
documented control transitions reconcile them. Pace, passing and incident
exposure share those boundaries.

Classification orders cars by actual completed distance and crossing order,
with the timed flag recipient first. API and CSV expose completed distance;
console, HTML and dashboard displays show known lap deficits alongside time
gaps. The classification threshold and reduced points use the winner's actual
distance and the original scheduled distance respectively.

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
