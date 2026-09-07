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

The next slick compound is selected from stint length and strategy preferences,
with a strong hard-compound bias on high-stress circuits. This remains a
heuristic: it does not compare projected total race times for every compound,
account for a finite inventory of tyre sets, or optimize coordinated team pit
stops. Those are separate opportunities to improve strategy realism.

Sampling ranges in the dashboard measure Monte Carlo noise under these
assumptions. They do not validate the strategy model against real race outcomes.
