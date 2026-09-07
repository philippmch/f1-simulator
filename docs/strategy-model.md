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

A conservative driver who meets the mid-race traffic trigger switches to the
balanced profile for subsequent decisions, including later compound choices.

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
future weather or incidents, prices future traffic, limits the inventory of
tyre sets, or coordinates team pit stops. Those remain separate opportunities
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

Sampling ranges in the dashboard measure Monte Carlo noise under these
assumptions. They do not validate the strategy model against real race outcomes.
