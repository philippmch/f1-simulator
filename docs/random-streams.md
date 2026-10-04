# Random draws in strategy comparisons

An extra pit stop used to advance the shared race generator, which could shift
later lap noise for an unrelated car even without a physical interaction.
The optional `isolated_race_v1` policy separates native draws by driver and
purpose. A service call cannot advance that driver's pace stream or another
driver's service stream. This makes paired strategy comparisons easier to
interpret while retaining the same pace, service, incident and hazard formulas.

Select **Driver and purpose streams** under **Match random draws** in the
dashboard, or pass `rng_policy="isolated_race_v1"` to `MonteCarloRunner` and
the saved-input comparison or strategy-selection functions. Saved-input CLI
commands accept the same override:

```powershell
python examples/compare_pit_plans.py output/saved_statistics.json --driver VER --plans plans.json --rng-policy isolated_race_v1 --simulations 100 --export
python examples/compare_starting_tyres.py output/saved_statistics.json --driver VER --compounds soft,hard --rng-policy isolated_race_v1 --export
```

All variants use the selected policy, and exports record it for offline replay.
Comparisons inherit the saved policy when no override is supplied. Replays and
process workers reconstruct streams from each trial seed; they do not retain
generators from a previous trial. The default remains `isolated_weather_v1`.
The older `shared_v1`, `isolated_weather_v1` and
`isolated_weather_mechanical_v1` layouts retain their existing seeded behavior.
Changing policy intentionally changes sampled qualifying and race outcomes.
Replay still requires compatible installed simulator code and numerical libraries.

## Draw ownership

| Draws | Stream under `isolated_race_v1` |
| --- | --- |
| Qualifying variation and conditional mistakes | Driver / qualifying lap |
| Sampled running-lap variation | Driver / race lap |
| Stationary pit service and service errors | Driver / pit service |
| Stochastic opening choice | Driver / opening choice |
| Stochastic automatic pit decisions | Driver / strategy |
| Stochastic replacement fallback | Driver / replacement choice |
| Passing success or contact | Attacking driver / overtake attempt |
| Time losses after passing contact | Attacking driver / collision loss |
| Random race incidents, control deployment and randomized field gaps | One field event stream |
| Mechanical hazard and conditional component | Driver / own lap, using the existing mechanical layout |
| Weather evolution | Existing separate weather stream |

Deterministic lap and qualifying projections do not request these driver
streams. Expected pit-service forecasts do not sample service. Native races
still call the public pit-service sampler, including custom hooks, with the
driver's service generator temporarily supplied and restored after the call.
Custom implementations that create their own randomness are outside this policy.
Direct simulator constructors retain their existing defaults; callers can
supply the factories and weather generator from `simulation.randomness`.

## What remains coupled

These driver/purpose generators are stateful. Matching seeds and IDs match
the prefix of draws for each purpose. They are not keyed to own lap or elapsed
time. If a driver samples an extra running lap, attempts another pass, or
receives an extra service error, later draws for that same purpose can shift.
Interrupted laps and conditional qualifying mistakes also affect consumption.
Mechanical checks are the exception: they retain their driver/own-lap keys.

Strategies still change tyres, pace, physical order, service queues, weather
exposure, eligibility for battles and risk inputs. Field incidents and control
events share one field stream; different car processing order or conditional
event calls can change later field draws. Identical weather draws align by
leading update interval, rather than elapsed seconds. Prescribed rainfall
disables random atmosphere changes and retains the other selected streams.
Whole race outcomes and failures need not match. This policy does not prove
lower sampling error, isolate every causal effect or validate the model against
real races.

## Versioned seed layout

Each driver ID is UTF-8 encoded and SHA-256 hashed. Its entire digest is split
into eight little-endian unsigned 32-bit words. Driver/purpose generators use:

```python
np.random.default_rng(np.random.SeedSequence(
    trial_seed, spawn_key=(0x44525652, purpose_number, *driver_words),
))
```

The fixed purpose numbers are qualifying lap 1, race lap 2, pit service 3,
opening choice 4, strategy 5, replacement choice 6, overtake attempt 7,
collision loss 8 and field events 9. Field events hash an empty ID. Lookups
reuse that trial's generator for the same key; creation order does not assign
streams. Weather retains `spawn_key=(0x57454154,)`. Mechanical draws retain
`spawn_key=(0x4d454348, *driver_words, own_lap)` and a fresh generator per check.
Python's process-specific hash is never used.

This applies NumPy's deterministic `SeedSequence` mixing of seed and stream
identifiers. NumPy describes the resulting streams as independent with very
high probability, rather than an absolute guarantee. See its
[parallel random generation documentation](https://numpy.org/doc/stable/reference/random/parallel.html).

Regression coverage includes the fixed stream layout, reordered qualifying
inputs, deterministic forecasts, custom sampler restoration, extra native
stops in both race engines, finite inventories, process workers, saved replay,
comparison overrides, dashboard controls and exported reports.

The compatibility check at this checkpoint compared 72 declared synthetic
cases against the preceding implementation: both engines, all three older
policies, seeds 17/37/91, dry and heavy-rain inputs, and automatic or finite-pool
custom plans. All full race, qualifying, event and weather records matched.
No cases or retirements were excluded. This checks that cohort's replay
compatibility; it does not establish real-world accuracy.
