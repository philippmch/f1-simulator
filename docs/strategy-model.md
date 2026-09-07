# Strategy model and limits

The simulator uses seeded, reactive strategy heuristics. It does not search all
possible pit schedules or claim to find the fastest strategy for a real race.

Aggressive, balanced and conservative profiles influence opening slick choices,
planned pit windows, stop probabilities and the next stint's compound. Decisions
also respond to traffic, track position, circuit overtaking difficulty, surface
water and safety-car opportunities. Pit loss includes the circuit's pit-lane
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

At a dry stop, the simulator compares the projected tyre contribution over the
next stint for each eligible fresh slick. The projection shares the actual lap
model's compound pace, wear, driver tyre management, circuit stress and car
degradation factor. When a new distinct slick is required, the comparison is
restricted to unused compounds. An archetype's preferred compound can override
the fastest only within 0.05 seconds per projected lap. This tolerance represents
a bounded strategy preference; it is a model assumption, not an empirical fit.

The current planned stop is consumed before determining the next stint's target.
The horizon ends at the next remaining future plan entry that the ordinary stop
budget allows, or at the finish when none remains. Pit service occurs before that
lap's pace calculation, so a final
stint includes the lap on which the stop happens. Weather stops still consume
plan slots in the same way as normal stops; the model does not rebuild an entire
pit schedule after an unscheduled stop.

This comparison assumes the stop has already been chosen. Pit loss, fuel and
other compound-independent effects do not change which fresh slick is fastest
over that fixed horizon. It does not optimize stop timing, a finite inventory of
tyre sets or coordinated team pit stops, and it does not forecast future weather.
Those remain separate opportunities to improve strategy realism.

Run `python examples/check_stint_choices.py` for a deterministic synthetic
comparison of selected compounds against actual lap calculations over short,
medium and long stints. The diagnostic makes no network requests and compares
fresh compounds at the same stop, without traffic or incidents.

Sampling ranges in the dashboard measure Monte Carlo noise under these
assumptions. They do not validate the strategy model against real race outcomes.
