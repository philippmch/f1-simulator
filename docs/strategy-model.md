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

Automatic starting-compound selection on a clearly dry track also compares the
full projected race for each driver and car. The opening set is free and must
run before a later paid stop; future stops include service cost and the
distinct-compound requirement. Existing strategy weights choose between
minimum-cost opening compounds, allowing different orders of equally fast
stints. A numerical tolerance of one billionth of a second treats floating-point
ties consistently. If no legal projected plan exists, the original weighted
choice remains available. Explicit starting-tyre overrides and rain sets selected
by the surface/rainfall crossover retain priority. This projection shares the dry
optimizer's limits on weather, traffic, inventory and future interruptions.
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
outcome. Reservations reset each lap and race. This is a same-lap box model,
not a simulation of pit-lane congestion, crew setup time or unsafe releases.

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

The alternative retains the existing set to the finish. If rivals remain, it
receives maximum dirty air while replacement sets receive clear air; a lone car
receives no fictional traffic relief. The calculation uses noise-free lap times,
including the pace floor. Only this lap receives the current SC/VSC running and
lane factors; queue delay applies only to the current stop. Future laps and
stops assume green running. Results are cached with bounded capacity.

If even the optimistic paid-refit plan is slower than retaining the old set,
the car stays out and re-evaluates next lap. Projected critical old-set conditions
and unresolved compound-use requirements retain priority. Passing the filter can
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
simulator compares tyre contribution over the
next stint for each eligible fresh slick. The projection shares the actual lap
model's compound pace, wear, driver tyre management, circuit stress and car
degradation factor. When a new distinct slick is required, the comparison is
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
the dry optimizer additionally includes pit loss. Both omit common fuel and
car pace terms that cancel between the compared dry schedules. Neither forecasts
future weather or incidents, prices traffic beyond the immediate rejoin lap, limits the inventory of
tyre sets, or jointly schedules both teammates' future stops. Those remain separate opportunities
to improve strategy realism. Cost tables are bounded in-memory calculations;
they do not persist provider data or consume simulation random draws.

Run `python examples/check_stint_choices.py` for a deterministic synthetic
comparison of fallback stint choices against actual lap calculations over short,
medium and long stints. The diagnostic makes no network requests and compares
fresh compounds at the same stop, without traffic or incidents.

Run `python examples/check_pit_timing.py` to compare the chosen strategy with
every permitted one-stop lap and unused compound in controlled synthetic
30-lap full races. This also checks pit execution and tyre ageing, not just
the optimizer's own cost calculation.

Run `python examples/check_restart_choices.py` to compare selected free restart
sets with forced soft, medium and hard alternatives in controlled 60-lap races.
The diagnostic covers a five-lap sprint, a long final stint, and high wear with
a later paid stop available. Each alternative uses the actual race engine and
remaining pit strategy with mean pace and expected stationary service.

Sampling ranges in the dashboard measure Monte Carlo noise under these
assumptions. They do not validate the strategy model against real race outcomes.

Run `python examples/check_rain_pit_timing.py` for exhaustive short wet-race
comparisons in both Standard and Lap-aware engines. Each selected strategy is
compared with every schedule of up to four paid stops after the opening lap:
99 schedules for eight laps and 562 for twelve laps. The synthetic cases cover
intermediates, wets, and no-stop, one-stop and two-stop optima using actual tyre
ageing, fuel and pit execution. They use a single car, fixed rainfall/surface,
noise-free laps and expected service, with incidents disabled. Long reference
lap times and cheap lanes deliberately exercise worthwhile fresh-tyre stops;
these are model checks, not calibrated venues or observed race comparisons.
