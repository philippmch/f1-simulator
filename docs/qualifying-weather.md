# Qualifying-session weather

Qualifying can use different conditions in Q1, Q2 and Q3 while the race keeps
its own initial weather. This supports scenarios such as a wet opening session
followed by dry qualifying and a dry race. Every trial still simulates qualifying
from the supplied drivers and cars; it does not replay an observed grid.

The optional `qualifying_weather` object uses exactly `Q1`, `Q2` and `Q3` as
keys. An omitted session uses the run's initial race weather. An explicit
session contains ordinary `Weather` fields, with the model's usual defaults for
fields you omit. Those fields do not inherit values from race weather. An empty
object or omitted option retains the existing shared-weather behavior.

Each session holds its conditions fixed while comparing fresh compounds and
sampling flying laps. A later session can therefore choose a different compound
and produce different lap times. Classification still follows Q3, then Q2,
then Q1; an earlier faster lap does not move an eliminated driver ahead of a
later-session qualifier. Session weather consumes no weather-transition draws
and does not evolve the race surface before the start.

## Dashboard and API

In the qualifying weather controls, leave a session on **Use race weather**
or select its condition and set rainfall and surface water. These numeric
values are normalized fractions from zero to one, not rainfall measurements
or water depth. Weather stays fixed within each session. Completed results
show the saved effective conditions, so editing the controls afterward does
not change the displayed run context.

The HTTP `/api/run` request accepts the same object:

```json
{
  "scenarios": "dry,light_rain",
  "qualifying_weather": {
    "Q1": {"condition": "heavy_rain", "rain_intensity": 0.8, "track_wetness": 0.8},
    "Q3": {"condition": "dry", "rain_intensity": 0, "track_wetness": 0}
  }
}
```

In this sweep, both scenarios use the explicit wet Q1 and dry Q3. Q2 inherits
each scenario's race weather. The setting also remains in automatic-strategy
comparisons and candidate/rival pit-plan selection; it is not an extra weather
forecast supplied only to the target driver.

## Python and CLI

Python callers can use weather-field dictionaries or `Weather` objects:

```python
runner = MonteCarloRunner(
    drivers, cars, track, Weather(change_probability=0), seed=42,
    qualifying_weather={
        "Q1": Weather(condition="heavy_rain", rain_intensity=0.8, track_wetness=0.8),
        "Q3": Weather(change_probability=0),
    },
)
results = runner.run(100, parallel=False)
```

Direct `QualifyingSimulator.simulate_qualifying` calls accept the same optional
keyword. Worker processes receive isolated serializable values. Changing an
original override object after constructing a runner does not change its saved
settings; invalid later edits to the runner's own settings are rejected on run.

The live CLI accepts a JSON object string. In PowerShell:

```powershell
python examples/simulate_race.py --race 1 --simulations 100 --scenarios dry --qualifying-weather '{"Q1":{"condition":"heavy_rain","rain_intensity":0.8,"track_wetness":0.8},"Q3":{"condition":"dry","rain_intensity":0,"track_wetness":0}}' --export
```

Unknown session keys, unknown weather fields, invalid conditions, out-of-range
values, booleans in numeric fields, quoted numbers and nonfinite values are
rejected before simulations or live data fetching start. Session keys are case
sensitive. Weather fields retain the ordinary model ranges, including
temperature and humidity, even though qualifying does not model tyre heating.

## Saved runs and comparisons

A nonempty override object uses simulation-input schema 7. The saved values are
complete weather-field dictionaries for explicitly overridden sessions, and
the saved race weather supplies omitted sessions. Schema 7 can also retain
starting tyre ages, finite race pools, custom pit plans and optional post-fit
cost sensitivity. Runs without overrides retain their existing schema version.
Replay continues to accept schemas 1 through 6; those versions cannot contain
qualifying-weather settings. Older installations reject schema 7 rather than
silently simulating a different grid.

Saved starting-tyre, race-engine and pit-plan alternatives keep these conditions.
Paired comparisons require identical effective weather in all three sessions,
in addition to the existing model, runtime, stream-policy and qualifying checks.
An explicit session equal to the race weather can match an inherited session;
different session conditions cannot count as matching inputs merely because
the sampled grids happen to be equal. JSON exports preserve the full settings,
and console and HTML reports label the effective session conditions.

These are controlled scenario inputs, not weather forecasts or calibrated
qualifying strategies. Weather and track grip do not evolve within a session;
traffic, tyre allocation, heat cycles and thermal preparation remain outside
the qualifying model. Weather labels govern the descriptive condition, while
numeric rainfall and surface water determine the existing pace and mismatch
responses. A different simulated grid can change later race decisions and
random-event exposure, so the option does not isolate a causal weather effect
on the final result.
