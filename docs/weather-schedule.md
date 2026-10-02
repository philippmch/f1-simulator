# Prescribed race rainfall

An optional `weather_schedule` supplies known rainfall changes for a controlled
strategy experiment. Both race engines and the automatic tyre policy use the
same sequence. This makes it possible to compare strategies through a
reproducible rain-and-drying scenario using the existing surface and tyre model.
It is a scenario supplied by the user, not a prediction of real weather.

```python
schedule = [
    {"lap": 3, "rain_intensity": 0.8, "condition": "heavy_rain"},
    {"lap": 6, "rain_intensity": 0.0, "condition": "cloudy"},
]

runner = MonteCarloRunner(drivers, cars, track, weather,
                         weather_schedule=schedule, seed=42)
results = runner.run(100, parallel=False)
```

The runner, `RaceSimulator.simulate_race` and `ChronologicalRace.run` accept
`weather_schedule`. In the dashboard, configure the optional rainfall schedule
beside the starting weather. The CLI accepts repeated steps:

```console
python examples/simulate_race.py --race 1 --seed 42 --export \
  --rainfall-step 3=0.8:heavy_rain --rainfall-step 6=0:cloudy
```

The `/api/run` JSON field uses the same list of objects as Python. Leave the
schedule empty to retain the selected evolving or fixed-rainfall behavior.
With a nonempty schedule, prescribed atmospheric changes take precedence over
that setting: rainfall and condition stay fixed between and after the entries,
and random atmosphere changes are disabled. Incidents, service variation and
race control retain their ordinary behavior.

## Surface response and the shared clock

The supplied initial `Weather` is used unchanged on lap one. Entering a scheduled
leading lap first changes rainfall and the optional condition label, then applies
one ordinary surface update. A step never replaces existing standing water.
Rain can therefore start before the surface gets very wet, and water can persist
after the rain stops. Temperature and humidity retain their initial values.
Omitting `condition` preserves its previous label; tyre pace and mismatch use
rainfall and surface water rather than the label alone.

`lap` is the shared leading-lap ordinal. In a chronological race, a lapped car
can see several updates between its own laps, or repeat the same snapshot. Its
own lap number does not advance the schedule. Leading intervals retain that
ordinal through leader changes and retirements. A red-flag restart consumes the
single next shared snapshot. Suspension time does not create additional surface
updates, and no update is charged after the leading chequered crossing.

Automatic opening choices, paid tyre changes, finite-set choices and free
red-flag refits include known future changes in their existing bounded searches.
Chronological projections map future track-entry times, including expected pit
losses, to the shared weather clock. A dry start cannot use a constant-dry shortcut
that overlooks a later scheduled rain step.

Explicit opening sets retain the existing first-running-lap rule for elective
stops. Compulsory repair or critical-weather replacements can still precede
that lap. Known later rain does not turn an opening override into a pit plan.

These calculations retain the existing conditional strategy assumptions:
observed free pace supplies an external leading clock, future leader changes,
battles and interventions are not predicted, and service projections use
expectations rather than unseen sampled durations. The model still does not
resolve within-lap rain or tyre temperature. A known weather scenario does not
establish globally optimal timed-race strategy or empirical accuracy. See
[strategy model and limits](strategy-model.md).

## Validation and saved comparisons

Each entry requires `lap` and `rain_intensity`. Laps must be strictly increasing,
unique integers from 2 through the original scheduled distance. Rainfall must be
a finite number from 0 through 1. Booleans, quoted numbers, unknown fields and
unordered entries are rejected. Optional conditions use `dry`, `cloudy`,
`light_rain` or `heavy_rain`. The upper lap limit is the original distance even
when a time limit later ends the race early; unconsumed steps have no effect.
Inputs are copied so later caller edits cannot change a completed experiment.

Saved input schema 8 records a nonempty schedule, together with any starting
tyres, ages, physical sets, pit plans, post-fit costs and qualifying-session
weather. Replay, engine and tyre comparisons, pit-plan selection and weighted
rival selection preserve it. Seed-paired comparisons require the same canonical
schedule because a different known scenario changes the experiment.

Run summaries, saved reports and dashboard downloads describe the completed
run's saved schedule. Changing the controls afterward does not change that
context. Qualifying retains its fixed initial or separately configured session
weather and does not consume the race schedule.

Omitting the schedule or passing `[]` retains the existing saved schemas,
worker behavior and seeded default simulation outputs. Earlier input schemas
remain replayable and cannot claim the new schedule field.
