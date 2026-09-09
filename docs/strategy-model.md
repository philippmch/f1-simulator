# Strategy model and limits

The simulator combines a dry-race cost optimizer with seeded, reactive weather
strategy. It optimizes the modeled remaining dry race within the allowed stop
budget, including a required compound-correction stop; it does not claim to find
the fastest strategy for a real race.

All team styles can evaluate up to three paid stops in dry running, using fewer
when further tyre gains do not cover pit loss. Previous paid stops, including
weather stops, count toward this budget; a required compound-correction stop
remains available after it is exhausted. Three is a bounded search limit, not
a requirement to make three stops or a claim that longer races never need more.
Dry red-flag projections use the same budget.

Dry planning scales tyre costs with the same current weather multiplier as
simulated laps, including cloudy conditions and the car's wet-performance
contribution. It holds that multiplier constant over the projected stint;
this is not a forecast of future weather. Expected service and pit-lane loss
are not weather-scaled. A current SC/VSC running multiplier applies in addition
to weather scaling for this lap only, with later laps assumed green.

Before a timed finish is announced, in-race strategy estimates its distance
from the leader's elapsed time and most recent running pace. Pit service,
incident loss and time spent waiting behind another car are excluded from the
recurring pace estimate. Current control affects the upcoming lap, with later
laps assumed green. The estimate includes the lap following clock expiry and
never exceeds the scheduled distance. Without a pace observation it retains
that schedule. Both engines use this estimate for pit decisions and free tyre
choices; chronological races also map the leader's estimated finish time to
each car's own remaining laps and include completed suspension extensions.

Chronological forecasts resolve equal completed distances using physical
on-track order, ahead of cars still in the pits, rather than stale crossing
times. A pitting car a full lap ahead retains distance priority. A pending stop
uses its expected exit when planning the remaining race; the actual finish
controller continues to use executed crossings.

The forecast is recalculated as the race develops. It does not change the
actual finish controller, predict future stops or weather, or guarantee a
globally optimal timed strategy. After a timed announcement, the next leading
crossing remains authoritative even if a lapped driver inherits the lead.
Original scheduled fuel distance remains separate from all strategy horizons.

Pit-rejoin traffic uses the same projected finish time, even before the timed
finish is announced. Each rival remains traffic until its own projected final
crossing; lapped cars can therefore remain after the leader takes the flag.
Exact crossing/exit ties retain the scheduler's distance and ordering rules.
This forecast retains current pace/control assumptions and does not predict
later weather, incidents or elective stops.

Explicit starting-tyre overrides are available per driver in the dashboard,
CLI, API and Monte Carlo runner. They replace only the opening choice; unlisted
drivers retain automatic selection and all later pit decisions remain active.
An unsuitable starting set can therefore be replaced before lap one. Overrides
do not change qualifying, grant a compound-use exemption, or specify a fixed
pit schedule. They are included in saved-input replay. Unknown driver codes and
invalid compounds are rejected instead of silently falling back to automatic.

The optional `fixed_rainfall` weather mode helps isolate strategy behavior under
unchanging rainfall. It prevents random condition transitions while retaining
surface-water response, incidents and adaptive pit decisions. The usual evolving
mode remains the default. Both modes are scenario assumptions, not forecasts;
saved-input tyre comparisons retain whichever weather behavior was exported.

Automatic starting-compound selection on a clearly dry track compares isolated
runs of the existing pit policy for each slick, using noise-free lap pace and
expected service time. These runs follow the race clock, retain the original
fuel distance, and include later paid stops and the distinct-compound rule.
They compare completed distance first and elapsed time second. This avoids
choosing an opening tyre for a scheduled distance that the time limit will cut
short, or rewarding a slower policy merely because it completes fewer laps.
Existing strategy weights choose among equivalent best outcomes, with a
one-billionth-of-a-second tolerance for elapsed-time ties. One-lap races and
cases without a legal projected finish retain the original weighted choice.
Explicit starting-tyre overrides and rain sets selected by the surface/rainfall
crossover retain priority.

The dashboard's Statistics tab and saved HTML reports show recorded tyre
sequence frequencies for each driver across the full run. Expand a driver to
see every sequence, its count and share, and the finished and retired counts.
JSON statistics, scenario comparisons and dashboard downloads retain the same
`strategy_statistics` data. Shares use that driver's observed races with a
recorded sequence, with missing records counted separately; the requested
simulation count is not used as a substitute for observations.

Sequences preserve fitting order and repeated compounds, including free
red-flag changes. A retirement can truncate a sequence, and a fitted set may
not complete a lap. Sequence length is therefore not the paid-stop count.
These are descriptive frequencies from the simulated policy, not controlled
comparisons of which strategy is fastest.

For a fixed-input comparison, `examples/compare_starting_tyres.py` runs one
driver's requested opening choices from an exported simulation snapshot. Each
variant preserves the roster, cars, track, weather, race engine, other drivers'
overrides and base seed range. `automatic` clears only the target driver's saved
override. The trial count can differ from the source run. All later strategy
and weather responses remain enabled, including immediate replacement of an
unsuitable opening set.

The comparison reports outcome rates and points per race, with a 95% Wilson
interval for each win rate. These intervals measure Monte Carlo sampling
uncertainty within the model. Matching seeds preserve qualifying, but changes
in race decisions can consume random draws differently, so incidents and later
events are not held fixed between variants. New runs use an independent weather
stream, so strategy choices cannot alter rainfall simply by consuming different
race draws. With matching seeds and weather inputs, the recorded weather
histories share the same prefix by weather-update interval. They are not aligned
by elapsed seconds, and retirements or timed finishes can shorten the history.
This is a comparison of opening
policies under the saved model, not a ranking of complete pit schedules or a
claim about real-race optimality. Exported variants contain their own inputs
and can be replayed using the installed simulator implementation.

The saved `rng_policy` makes this behavior explicit. `isolated_weather_v1`
retains the usual seeded qualifying/race generator and derives a separate
weather generator from `SeedSequence(seed, spawn_key=(0x57454154,))`. That fixed
namespace separates weather from race draws without advancing the race stream.
`shared_v1` retains the former single generator. New snapshots use schema 2 and
require an explicit policy, so older installations reject the unsupported
schema. Replay and comparison still accept schema 1 and infer `shared_v1` when
its policy field is absent; unknown policies are rejected.
Comparisons inherit the saved policy unless `--independent-weather` is supplied
(or the Python `rng_policy` override). This changes all variants together and
records the new policy for replay. Direct `RaceSimulator` callers retain their
shared generator unless they supply a separate `weather_rng`.

Combined JSON exports retain each scenario's observed driver counts, rate
intervals, points per observed race and paid-stop statistics. The offline HTML
comparison report places scenarios in their supplied order, with expandable
driver tables showing win, podium and retirement intervals. It does not rank
choices or treat an interval for one rate as an interval for a difference.
Absent drivers and zero observed trials display as not recorded; paid-stop
averages use their own recorded-race counts and exclude free tyre changes.
The dashboard's Scenarios tab can download this report for the completed run.
Its weather context includes the condition label and per-lap change probability,
since scenarios with identical initial rain and wetness can still evolve or run
at different speeds under the model. These settings are not a weather forecast.
The dashboard scenario chart and matrix also display the backend's Wilson
intervals and observed trial counts. Driver groups in the chart can be expanded
independently. Matrix highlights refer only to point estimates. Missing rates
are excluded from sorting averages and remain blank in matrix CSV exports;
older results without interval metadata retain their estimates with an explicit
sampling-range-unavailable label.

With no rainfall, the dry policy projection is deterministic and needs one
private run per slick. With sustained rainfall that could change the surface,
it averages the same eight private reaction seeds used below. Cached scores
include physical inputs, tyre configuration, strategy settings and the race
time limit; driver and team names do not affect them. This compares the
existing single-car policy, not every possible timed pit schedule. It assumes
constant weather condition and rainfall, no traffic or future interruptions,
and unlimited tyre inventory. A multi-car race can have a different horizon
as its leader and traffic determine the finish.
Precautionary intermediates must also pass the same mismatch check used during
the race. A rainy condition label with a sufficiently dry surface and low
rainfall does not fit intermediates that would immediately require a paid
replacement before lap one. Explicit starting-tyre overrides remain available.

When intermediates are only a precaution, the selector compares them with all
three slick compounds using isolated, traffic-free runs of the existing pit
policy. These runs cover the full race, use mean lap pace and expected service
time, and advance surface wetness after each lap under constant rainfall and
weather condition. They include later paid stops and the compound-use rule.
Each candidate uses the same eight fixed private reaction seeds; the selector
compares their average completed distance first and average total time second,
so a slower, shorter time-limited race cannot win merely by ending sooner.
It favours the existing intermediate choice in a tie. The real race's random generator and input objects are untouched.

This is an approximate comparison of the current policy under sustained
conditions, not a global wet-strategy optimizer or a forecast of changing rain,
traffic, or incidents. Close choices can depend on the sampled reaction paths.
Results are cached with a bounded capacity, including the driver, car, circuit,
weather, tyre configuration, and strategy settings. Direct selector calls
without driver/car context retain the original precautionary choice.

Aggressive, balanced and conservative profiles influence opening slick choices
and close timing decisions. The fallback weather strategy retains its style-based
stop budgets and also
responds to traffic, track position, circuit overtaking difficulty and surface
water. Pit loss includes the circuit's pit-lane
delta and sampled stationary service time; safety-car and virtual-safety-car
running reduce the relative pit-lane loss.

Weather-driven stops take precedence over normal stop budgets. This allows a
driver to react when conditions change after the planned stops are exhausted.
The dry-compound check counts distinct slick compounds rather than stops; using
an intermediate or wet compound exempts that driver's modeled dry-use rule.

A fitted set earns compound-use credit when it runs on track in a simulated
lap. Replacing it before any running does not count an extra compound or grant
a wet-tyre exemption; unused fittings are removed from stint history. When
considering staying out, the planner credits the current set's forthcoming lap.
An immediate replacement must instead use only compounds that have already
run. This also prevents a free red-flag fitting from granting credit if it is
replaced again before the restart lap. A distinct free set that can complete
the requirement by running does not force another paid stop or extend the
elective stop budget. In a two-lap race, the mandatory change waits until lap
two so the opening set gets a lap of running.
The unconditional compound-correction safeguard acts on the final lap, since
the replacement runs that lap after its stop. Earlier dry stops follow the cost
comparison, allowing a faster current set to remain on until the last legal
change when that minimizes total time.
One-lap simulations retain a single starting stint because this lap-level
model cannot run two sets within one lap.

The dashboard Statistics view summarizes paid pit stops across all observed
races for each driver: the average and the shares with zero, one, two, or at
least three stops. Retirements remain in these observations, so early failures
can lower the average. Free red-flag changes are excluded. The API and statistics
JSON export include the observed race count, mean, and exact stop-count
frequencies; drivers without race observations have no inferred stop statistics.

Race results expose the actual paid pit laps, shown below each dashboard stop
count and included as `pit_laps` in the API and downloaded scenario JSON.
The race CSV appends a `pit_laps` column containing a JSON array, such as
`[17, 34]`. An empty array means no paid stops; legacy results with unavailable
timing use null in JSON and a blank CSV cell. Free red-flag tyre changes remain
in compound history but do not add a paid pit lap. A stop performed before a
retirement on the same lap remains part of that driver's stop history.

Each paid stop also records `pit_stop_details`: the driver's own lap number,
outgoing and incoming compounds, completed laps on the outgoing set, condition,
rain intensity, surface wetness, race control, and lane/service/queue loss in
seconds. The three cost components sum to the modeled total; they are not exact
arrival timestamps and exclude subsequent on-track traffic. Recording does not
change race decisions or consume random draws. Free red-flag fittings have no
paid-stop record; an opening paid correction has tyre age zero.

The dashboard shows these observations for its selected trial. Statistics and
comparison JSON retain per-trial, per-driver records; the bundle's pit-stops CSV
has one row per paid stop with one-based trial numbers. Legacy JSON uses null for
unknown details and an empty list for a known zero stops; neither produces CSV
stop rows. These records describe executed stops, not inferred decision reasons.

Aggregate `pit_loss_statistics` reports each driver's mean total, lane, service
and queue loss per race with complete details, plus queued-stop counts and the
share of those races containing a queue delay. The denominator includes known
zero-stop races and retirements. Unknown, incomplete or inconsistent detail rows
are excluded entirely and counted separately. With no complete observations,
means and queue share are null. Dashboard and comparison reports retain this
observation count; these descriptive costs do not establish a strategy's causal
effect on race results and exclude later on-track traffic and free fittings.

Fresh weather tyre selection shares the slick-mismatch crossover: above 0.2
surface wetness or 0.4 rain intensity, a stop fits intermediates; above 0.7
surface wetness, it fits full wets. These values are normalized model parameters,
not measured millimetres of water. Starts and red-flag restarts use the same
selection, with an additional precautionary intermediate bias for rainy starts.
Already-fitted rain tyres have wider drying windows before they trigger another
stop, which avoids repeatedly switching sets near the crossover.

Qualifying compares all fresh compounds using its one-lap model with random
variation and mistakes disabled, then runs normal sampled attempts on the
fastest set. Equal projected times preserve compound enumeration order. This
avoids paying a rain-tyre penalty on a still-dry surface solely because rainfall
has begun. Race stops retain their precautionary rainfall thresholds because
the next racing laps evolve surface wetness. Qualifying's lap model applies
the same driver/car weather multiplier and flat
tyre-weather mismatch penalty as race laps, while retaining qualifying's own
base pace, fresh-tyre grip and push-level variation. A driver with stronger wet
skill can therefore improve a wet qualifying lap, and slicks no longer escape
their mismatch penalty on a wet surface. Each attempt assumes a fresh set;
weather remains fixed across Q1, Q2 and Q3. Qualifying does not model tyre
inventory, track evolution, traffic or a changing-weather session strategy.

During a red-flag suspension, dry tyre selection compares all fresh slicks over
the remaining race, including any later paid stops that the stop budget permits.
It uses the same tyre pace and wear model as ordinary dry planning. The free set
must run at least one lap before another stop; future service and pit-lane time
are priced as green running. A suspension after lap N leaves `total_laps - N`
racing laps, starting with lap N+1. Rain-tyre crossover decisions retain priority.
The race applies its usual between-lap weather update before selecting the free
set, so that choice uses the conditions in which racing resumes. This avoids
fitting a set for the completed lap's weather and then paying to replace it on
the restart. It adds no weather update or modeled suspension duration.

The free change does not consume a paid stop or pit-plan slot. A new distinct
slick can satisfy the compound-use requirement; repeating a slick remains an
option when the projected schedule includes a legal later correction. The
existing mandatory-correction stop exception remains available if needed.
Fresh tyres resolve the modeled puncture's pending stop, while incident time
already lost stays on the clock. Retired cars are not serviced. A suspension
after the final lap adds no tyre stint or compound-use credit.

Changing wheels and tyres during a suspension is permitted by B5.14.4(a)(vii) of
the [FIA 2026 Sporting Regulations, Issue 08](https://www.fia.com/system/files/documents/fia_2026_f1_regulations_-_section_b_sporting_-_iss_08_-_2026-08-05_7.pdf).
The model assumes a free fresh set is available and does not model the full
suspension work procedure, tyre inventory, or elapsed suspension duration.

A conservative driver outside the top six can switch to the balanced profile
after 40% of the scheduled race when following in modeled dirty air. The default
traffic window is below two seconds; `conservative_switch_gap` can narrow that
window and acts as a maximum gap, not a minimum. The switch persists for later
decisions and compound choices. A large gap to the car ahead does not trigger it.

In the reactive fallback strategy, a non-conservative driver can return to the
earlier-stop plan when following closely after the actual race midpoint, rather
than after a fixed lap number. Wet-condition fallback selection retains priority.
Neither traffic-based profile nor fallback-plan switches are triggered during
safety-car, virtual-safety-car, red-flag or restart laps: their compressed gaps
do not by themselves establish a reason for a lasting strategy change. These
are model heuristics, not forecasts of how long a driver will remain blocked.

For an ordinary stop on a clearly dry track, the optimizer compares pitting now
with driving at least one more lap on the current set. It searches remaining
stint lengths and eligible slick compounds through the finish, allowing unused
stops to be skipped. Terminal schedules must satisfy the distinct-compound rule.
An earlier stop may repeat a compound when enough laps and stops remain to run
a different one later. This matters when a current SC/VSC changes the best
order of stints; the requirement applies to the completed race, not each stop.
The projection uses current tyre age and the shared lap model's pace and wear,
including driver management, circuit stress and the car's degradation factor.
The selected compound is carried into the actual stop.

Projected service time includes the execution model's minimum service duration
and slow-stop probability. Current safety-car or virtual-safety-car discounts
reduce pit-lane loss. The current lap's tyre costs also use the race timing
model's active running multiplier, for both staying out and every eligible
fresh compound. Future running and stops are projected as green; this does not
predict how long a neutralization will last. A large gap behind
does not by itself make a stop worthwhile. From lap two onward, the dry optimizer
can choose an early stop, including during an early SC/VSC, when its projected
benefit justifies the cost. Elective dry stops before the first racing lap are
suppressed because stops occur at the start of a modeled lap; urgent weather
changes and mandatory safeguards retain priority. Damp/wet fallback heuristics
retain their opening five-lap guard. Beneficial dry stops can also occur in the
final five laps. Team style can shift
a near tie by at most 0.1 seconds per decision. This tolerance is a model
assumption, not an empirical fit.

Teammates share one modeled pit box. For stops on the same lap, service follows
arrival order and a car waits only while that constructor's box is occupied.
The wait is added once to pit loss; safety-car discounts affect pit-lane loss,
not stationary service or queue time. Different constructors have independent
boxes. This represents the consecutive service described in F1's
[double-stack glossary entry](https://www.formula1.com/en/latest/article/f1-glossary-a-e.1MFONigMlQSbSQtpP7YCy2).

Lap-start race clocks approximate relative pit-box arrivals. Before committing
to a dry stop, a driver is charged the expected queue from an earlier-arriving
teammate already committed on that lap. Planning uses expected service; actual
execution uses sampled service, so the decision does not know a future slow-stop
outcome. In the standard engine, reservations reset each lap and race.
The chronological engine carries reservations across individual car lap starts
and resets them for each race; cars on different laps can share the box.
Its planning reservations and projections of pitting rivals use expected
service separately from the sampled execution queue. Once service has visibly
completed, a stale expected reservation cannot keep the box occupied. Estimates
do not infer the remaining duration of an ongoing unusually slow stop.
Neither engine models pit-lane congestion, crew setup time or unsafe releases.

The recorded time for a stop lap includes the actual pit-lane, stationary and
queue losses, so its clean running pace alone cannot earn a fastest lap. Those
losses enter total race time only once. SC/VSC modifiers slow the running portion
of the lap; stationary service and queue time are not multiplied by them. The
model attributes the entire stop loss to the lap on which service occurs, rather
than splitting pit entry and exit across sector timing lines.

After the pit batch, dirty-air pace uses one frozen view of the field with
actual pit losses applied. A car can emerge into traffic, and a following car
can gain clean air when the car ahead stops. Strategy decisions and Overtake
Mode detection retain their pre-stop information. Final position changes still
use the completed lap clocks, including actual service and on-track running.

During clearly dry green running, the optimizer also compares the first lap's
dirty-air cost when stopping with that when staying out. It projects the driver's
expected lane, service and queue loss into the current field, assuming other
cars stay out. The shared lap model contributes between zero and 0.5 seconds,
depending on a gap below two seconds; this can move a close timing decision in
either direction. It does not predict a queue of slower cars over multiple laps,
passing opportunities, or rivals' stop decisions. The correction is disabled
under neutralisation. Expected gaps approximate a nonlinear cost at the mean
service time; they are not an average over every possible service outcome.

Clean air's relevance to an undercut is described in Formula 1's
[pit-strategy analysis](https://www.formula1.com/en/latest/article/jolyon-palmers-analysis-singapore-and-the-art-of-undercutting.1NgVyVsZnHTDEA9wi0s5lW).
The simulator's two-second range and half-second maximum are existing model
parameters, not values calibrated from that source. Because pit service occurs
before lap pace in this engine, the post-stop gap represents the running lap;
there is no separate in-lap/out-lap sector simulation.

Passing probability also uses the tyre pace difference between the cars in
dry, damp and wet conditions. The shared lap model includes compound,
current tyre age, driver management, circuit stress and car degradation. Tyre
pace receives the same driver/car weather multiplier as actual lap running,
plus the flat penalty for a compound that mismatches surface conditions. This
lets suitable rain tyres help an attack against slicks on a wet surface, and
lets worn or overheating rain tyres weaken an attack. Because
the maneuver is resolved after running the lap, this comparison uses the
current end-of-lap tyre ages. A fresh set fitted for that lap has age one.

The defender's tyre cost minus the attacker's cost is converted to a base-pace
equivalent using the car model's 3%-of-reference-lap scale, then bounded to one
pace unit in either direction before entering the existing passing curve.
Equal tyre contributions preserve the old probability. This is a bounded
heuristic, not a fitted relationship between lap-time advantage and passing
success. Circuit difficulty, proximity, driver skill and Overtake Mode still
affect the maneuver, and a tyre advantage cannot bypass the gap restriction.
Green-running opportunities use this probability curve on every circuit;
there is no separate difficulty cutoff that disables attempts. At the maximum
difficulty of one, the existing unboosted success probability is zero, while
contact and mode/restart effects still follow their normal rules. These are
model limits rather than calibrated circuit-specific passing rates.
The wet passing difficulty multiplier and wet Overtake Mode restriction remain
in effect. This extends the existing lap-time heuristic to rain tyres; it does
not model aquaplaning, a racing line that dries separately, or measured wet-grip
passing probabilities.

Restart passing uses a two-second attempt window, compared with 1.5 seconds
in normal running. The proximity factor decreases across the corresponding
window, so an eligible restart attempt between 1.5 and two seconds can succeed.
The outer gate still rejects larger gaps. These windows are model parameters;
the wider passing opportunity does not override Overtake Mode eligibility.

Battles are resolved from the front toward the back of the physical queue.
After a successful pass, the next attacker faces its new immediate neighbour;
it cannot skip a car by using the pre-pass order. Passing changes positions,
while the existing clock reconciliation charges blocked running without
removing elapsed race time.

Critical weather and damage stops retain priority. A noncritical weather
mismatch does not by itself justify a stop. Before the existing reaction draw,
the simulator estimates whether tyre gains over the remaining race can cover
expected pit-lane, stationary and queue loss. Reactive wet/damp pit-window
proposals, including SC/VSC opportunities, pass this same cost veto after the
window proposes stopping. A large gap behind alone does not make a stop free.
The check also prevents a newly fitted rain set from being replaced solely to
meet a planned lap. Legacy callers without weather retain their existing
behavior because there is no surface state to project. Surface wetness follows the same
deterministic rainfall and drying response used in race evolution, assuming
current rainfall persists. The projection consumes no random draws and does
not predict changes in weather condition or future race interruptions.

The comparison deliberately favours stopping, but every projected refit now pays
for lane travel and expected service. The first replacement follows the actual
fresh-weather crossover: a stop that would fit wets cannot claim the pace of
fresh intermediates. Later refits may use any noncritical compound, with no
inventory, stop-budget or compound-use restriction. Each set must run at least
one lap, ages normally, and cannot continue into a critical mismatch. This
expanded set of future options gives an optimistic cost for stopping now.

The alternative retains the existing set while it is noncritical. If it would
become critical before the finish, this waiting policy pays full lane and
expected service loss to fit an appropriate fresh set before that lap, repeating
only when another change becomes necessary. At a slick transition it chooses
the cheapest such safe-retention policy among the three slick compounds. This
provides a feasible waiting alternative without claiming optimal stop timing.
If rivals remain, the waiting policy receives maximum dirty air throughout,
while the optimistic stop-now plan receives clear air; a lone car receives no
fictional traffic relief. The calculation uses noise-free lap times,
including the pace floor. Only this lap receives the current SC/VSC running and
lane factors; queue delay applies only to the current stop. Future laps and
stops assume green running. Results are cached with bounded capacity.

If even the optimistic paid-refit plan is slower than this waiting policy,
the car stays out and re-evaluates next lap. A currently critical set and
unresolved compound-use requirements retain priority. A forecast need to change
tyres later no longer bypasses today's cost comparison. Passing the filter can
still permit a losing stop because its future options and traffic relief may be
unavailable. This remains a conditional cost filter, not an optimal wet-race
schedule or a guarantee about unpredictable weather.

When a clearly dry stop is already committed but has no optimizer proposal,
the simulator compares eligible fresh slicks over the entire remaining race.
This covers punctures, mandatory stops and returns from rain tyres. The current
paid stop consumes a budget slot before later stops are considered. Its lane
and service loss are common to all compound choices; future paid stops are
included in the comparison. The new set runs the current lap, including any
SC/VSC running multiplier, before a later stop is allowed. A repeated compound
is eligible only if the remaining projected schedule can still satisfy the
dry-use rule. The choice does not sample randomness or alter the race state.

For a damp fallback stop whose compound has not already been selected, the
simulator compares noise-free lap costs over the next stint for each eligible
fresh slick. The projection shares the actual lap model's compound pace, wear,
driver tyre management, circuit stress, car performance and degradation factor,
including its lap-time floor. Fuel uses the original scheduled race distance
even when the planning horizon is shorter. Surface conditions evolve from the
current weather without forecasting random weather changes; current race-control
and aero restrictions apply to the first lap, with future laps assuming green
running. When a new distinct slick is required, the comparison is
restricted to unused compounds. An archetype's preferred compound can override
the fastest only within 0.05 seconds per projected lap. This tolerance represents
a bounded strategy preference; it is a model assumption, not an empirical fit.

In fallback strategy, the current planned stop is consumed before determining
the next stint's target.
The horizon ends at the next remaining future plan entry that the ordinary stop
budget allows, or at the finish when none remains. Pit service occurs before that
lap's pace calculation, so a final stint includes the lap on which the stop
happens. Weather stops still consume stop budgets and fallback plan slots. On
returning to clearly dry slick running, the optimizer reassesses the remaining
race with the actual tyre age, compound history and stops remaining.

The current tyre model uses an absolute grip floor of 0.5, capped at the set's
initial grip. A custom set starting below 0.5 therefore cannot gain grip or
receive a negative wear penalty from the floor. Default compounds start above
that floor. At driver tyre-management rating 0.8, their unscaled curves reach it
at these completed tyre ages:

| Compound | Age at grip floor | Configured cliff age |
|---|---:|---:|
| Soft | 19 | 20 |
| Medium | 28 | 30 |
| Hard | 46 | 45 |
| Intermediate | 17 | 35 |
| Wet | 20 | 40 |

For most defaults, the floor is reached before the nominal cliff, so additional
age no longer increases the unscaled wear contribution. This helps explain the
rain planner's no-stop choices in the equal-performance diagnostic. It is a
model limitation, not observed tyre durability. Changing the curve or its floor
requires an explicit modeling/calibration decision; the low-initial-grip bound
fix does not change default tyre curves.

When the current rain compound remains the fresh-set choice throughout the
projected remaining surface conditions, rain strategy compares stopping now
with waiting at least one lap and making optimal later same-compound stops.
The projection uses noise-free lap physics, current tyre age, circuit stress,
car degradation and the remaining paid-stop budget. Fresh sets run on their
fitting lap. Only the current stop receives known queue/rejoin costs and the
current SC/VSC lane discount; future stops assume green running. Only the first
running lap receives current control and Active Aero restrictions. The original
fuel distance remains separate from a shortened planning horizon.

This planner can choose a worthwhile stop outside calendar windows, or wait
when a later stop is cheaper. Ties favor staying out. It assumes current rainfall
persists and replans after each lap; it does not forecast random weather changes,
future incidents, future traffic, or tyre inventory. It compares clean-air pace
for both actions, with only the immediate rejoin adjustment. This is an optimum
within the same-compound projection and budget, not a claim of globally optimal
wet-race strategy. If the projected fresh compound changes, the reactive
compound-transition strategy remains in use.

In that reactive fallback, wet/damp planned windows are consumed once. After the selected plan is exhausted,
it cannot fall through to another generic late-race window. When no explicit
plan exists, the generic schedule offers at most two stops (or one when the
ordinary budget is one). The higher wet stop allowance still permits reactive
weather changes and SC/VSC opportunities; it does not repeat the second window.
This prevents fresh intermediates being replaced again on consecutive laps
solely because the car remains inside the same calendar window. Compound-changing
fallback timing remains a heuristic.

The fallback compound comparison assumes the stop has already been chosen;
the dry optimizer additionally includes pit loss. The dry optimizer omits common
fuel and car pace terms only when all projected laps stay above the lap-time
floor. If clipping is possible, it compares full noise-free lap times using the
same physics as race execution, including the 95%-of-reference-lap floor.
This prevents crediting a fresh set with pace gains that execution would clip
away. Such clipping is possible with custom high-aero configurations; this is
a consistency correction, not a calibration to a real circuit.

The full-lap projection retains the original scheduled fuel distance even when
the planning horizon is shortened by the race clock or a car being lapped.
Current aero eligibility and SC/VSC running factors apply only to the current
lap; future laps assume green running. Cached costs in this path are absolute
lap times, whereas the ordinary fast path reports tyre-relative costs. Costs
from those two bases should not be compared across different model inputs.
Neither planner forecasts random future weather changes or incidents, prices
traffic beyond the immediate rejoin lap, limits the inventory of
tyre sets, or jointly schedules both teammates' future stops. Those remain separate opportunities
to improve strategy realism. Cost tables are bounded in-memory calculations;
they do not persist provider data or consume simulation random draws.
Static circuit profiles and car/circuit pace terms also use bounded, process-local
caches keyed by their numerical inputs. Editing sector weights, passing
opportunities, car ratings or reference pace produces a new calculation; no
mutable model objects or sampled lap outcomes are retained by these caches.

Run `python examples/check_stint_choices.py` for a deterministic synthetic
comparison of fallback stint choices against actual lap calculations over short,
medium and long stints. The diagnostic makes no network requests and compares
fresh compounds at the same stop, without traffic or incidents.
It also checks damp fallback choices on a custom high-aero circuit, including a
shortened planning horizon with the original fuel distance, against actual
noise-free lap totals and the allowed team-style tolerance.

Run `python examples/check_pit_timing.py` to compare the chosen strategy with
every permitted one-stop lap and unused compound in controlled synthetic
30-lap full races. This also checks pit execution and tyre ageing, not just
the optimizer's own cost calculation.

Run `python examples/check_dry_pit_schedules.py` for a broader bounded dry
comparison in both engines (`--engine standard` or `--engine chronological`
selects one). Each synthetic eight-lap race starts on medium and exhaustively
executes all 1,092 legal schedules of one to three stops after lap one, including
soft/medium/hard replacement sequences and repeated fresh sets of the same
compound. The initial medium set must be used and at least one different slick
must be fitted. Stop counts have 14, 168 and 910 alternatives respectively.

The balanced policy is compared with the fastest executed schedule using mean
pace, expected service, fixed dry weather and no incidents or traffic. Normal
lane loss, cheap stops with high wear, and a deliberately long 600-second lap
reference exercise one-, two- and three-stop optima. Both engines currently
match these references. The JSON output includes inputs, completed distance,
paid laps, tyre sequence, total time and the gap to the best bounded alternative.
The diagnostic makes no network requests or default file writes. Its synthetic
inputs are not calibrated venues, and it does not search beyond three stops,
different opening sets or future weather changes.

Run `python examples/check_restart_choices.py` to compare selected free restart
sets with forced soft, medium and hard alternatives in controlled 60-lap races.
The diagnostic covers a five-lap sprint, a long final stint, and high wear with
a later paid stop available. Each alternative uses the actual race engine and
remaining pit strategy with mean pace and expected stationary service.

Sampling ranges in the dashboard measure Monte Carlo noise under these
assumptions. They do not validate the strategy model against real race outcomes.

Driver points projections divide total awarded points by that driver's observed
race count, including retirements, and rank the resulting means. Requested trial
counts never dilute a partial result. Constructor projections sum the listed
drivers' individual observed means; this is a lineup projection, not an average
over a reconstructed union of team race records. Drivers with no observations
and teams containing an unobserved listed driver are omitted. A recorded zero
remains zero. Normal runs with every entrant observed in every trial are unchanged.

Event rates use the event ledger's recorded trial count when available. Legacy
aggregates without that count retain their nominal requested count; exports and
dashboard responses expose the denominator as `event_rate_trials`. Console and
dashboard event summaries show it alongside the rates. An empty mechanical
failure sample has no component-share calibration delta (`None` in Python) and
produces no tuning suggestions or reliability adjustments. An absence of observed
failures cannot establish the relative proportions of failure components.

The dashboard statistics and HTML reports include mean winning distance,
lapped finishers, time-limited races and races without a winner. Statistics JSON,
scenario comparison JSON and dashboard scenario responses expose the same
`race_distance_statistics` object. Rates are fractions from zero to one.
Race counts use recorded raw races, not the requested simulation count. Mean
winning distance and finish outcomes also appear together in offline comparison
reports, with their own denominators, so scenarios with shortened races or lapped
finishers can be interpreted alongside points and paid-stop counts. Each driver's
comparison also lists actual tyre sequences, with finished and retired counts,
shares among recorded sequences, and missing records shown separately. Free
fittings remain part of sequences without becoming paid stops. These frequencies
describe model behavior and do not establish the best strategy. Mean
winning distance includes only winners with a positive recorded lap count;
lapped-finisher rates include only finished cars whose own distance and winner's
distance are both known. Classified retirements do not count as finishers, and
the actual finished P1 supplies winner distance even if a retired car completed
more laps. Empty distance denominators return null and display as “Not recorded”.

Run `python examples/check_rain_pit_timing.py` for exhaustive short wet-race
comparisons in both Standard and Lap-aware engines. Each selected strategy is
compared with every schedule of up to four paid stops after the opening lap:
99 schedules for eight laps and 562 for twelve laps. The synthetic cases cover
intermediates, wets, and no-stop, one-stop and two-stop optima using actual tyre
ageing, fuel and pit execution. They use a single car, fixed rainfall/surface,
noise-free laps and expected service, with incidents disabled. Long reference
lap times and cheap lanes deliberately exercise worthwhile fresh-tyre stops;
these are model checks, not calibrated venues or observed race comparisons.
