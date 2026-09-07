# Strategy model and limits

The simulator combines a dry-race cost optimizer with seeded, reactive weather
strategy. It optimizes the modeled remaining dry race within the allowed stop
budget, including a required compound-correction stop; it does not claim to find
the fastest strategy for a real race.

Aggressive, balanced and conservative profiles influence opening slick choices,
stop budgets and close timing decisions. The fallback weather strategy also
responds to traffic, track position, circuit overtaking difficulty and surface
water. Pit loss includes the circuit's pit-lane
delta and sampled stationary service time; safety-car and virtual-safety-car
running reduce the relative pit-lane loss.

Weather-driven stops take precedence over normal stop budgets. This allows a
driver to react when conditions change after the planned stops are exhausted.
The dry-compound check counts distinct slick compounds rather than stops; using
an intermediate or wet compound exempts that driver's modeled dry-use rule.

Fresh weather tyre selection shares the slick-mismatch crossover: above 0.2
surface wetness or 0.4 rain intensity, a stop fits intermediates; above 0.7
surface wetness, it fits full wets. These values are normalized model parameters,
not measured millimetres of water. Starts and red-flag restarts use the same
selection, with an additional precautionary intermediate bias for rainy starts.
Already-fitted rain tyres have wider drying windows before they trigger another
stop, which avoids repeatedly switching sets near the crossover.

During a red-flag suspension, dry tyre selection compares all fresh slicks over
the remaining race, including any later paid stops that the stop budget permits.
It uses the same tyre pace and wear model as ordinary dry planning. The free set
must run at least one lap before another stop; future service and pit-lane time
are priced as green running. A suspension after lap N leaves `total_laps - N`
racing laps, starting with lap N+1. Rain-tyre crossover decisions retain priority.

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
The projection uses current tyre age and the shared lap model's pace and wear,
including driver management, circuit stress and the car's degradation factor.
The selected compound is carried into the actual stop.

Projected service time includes the execution model's minimum service duration
and slow-stop probability. Current safety-car or virtual-safety-car discounts
reduce pit-lane loss; future stops are priced as green stops. A large gap behind
does not by itself make a stop worthwhile. The initial five-lap guard remains,
but a beneficial dry stop can occur in the final five laps. Team style can shift
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

Weather and damage stops retain priority. For a forced or fallback stop whose
compound has not already been selected, the simulator compares tyre contribution over the
next stint for each eligible fresh slick. The projection shares the actual lap
model's compound pace, wear, driver tyre management, circuit stress and car
degradation factor. When a new distinct slick is required, the comparison is
restricted to unused compounds. An archetype's preferred compound can override
the fastest only within 0.05 seconds per projected lap. This tolerance represents
a bounded strategy preference; it is a model assumption, not an empirical fit.

The current planned stop is consumed before determining the next stint's target.
The horizon ends at the next remaining future plan entry that the ordinary stop
budget allows, or at the finish when none remains. Pit service occurs before that
lap's pace calculation, so a final stint includes the lap on which the stop
happens. Weather stops still consume stop budgets and fallback plan slots. On
returning to clearly dry slick running, the optimizer reassesses the remaining
race with the actual tyre age, compound history and stops remaining.

The fallback compound comparison assumes the stop has already been chosen;
the dry optimizer additionally includes pit loss. Both omit common fuel and
car pace terms that cancel between the compared dry schedules. Neither forecasts
future weather or incidents, prices traffic beyond the immediate rejoin lap, limits the inventory of
tyre sets, or jointly schedules both teammates' future stops. Those remain separate opportunities
to improve strategy realism. Cost tables are bounded in-memory calculations;
they do not persist provider data or consume simulation random draws.

Run `python examples/check_stint_choices.py` for a deterministic synthetic
comparison of selected compounds against actual lap calculations over short,
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
