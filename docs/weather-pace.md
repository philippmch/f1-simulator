# Continuous weather pace

Rainfall and surface water determine current weather pace. Changing only the
descriptive label between dry, cloudy, light rain and heavy rain cannot change
race or qualifying lap time, the current opening candidates, or opening-choice
random draws. Labels still govern future weather transitions and appear in
reports; evolving scenarios with different labels can therefore diverge later.

Previously, an intermediate on the same surface (water 0.6, rainfall 0.2) could
produce a 93.42-second or 108.37-second mean racing lap simply by changing its
label from dry to heavy rain in a 90-second reference fixture. Tyre mismatch
thresholds also introduced instantaneous jumps of 2.5 to 8 seconds. The shared
race, qualifying and strategy physics now use continuous responses.

## Pace assumptions

Both numeric inputs are normalized fractions from zero to one. Define exposure
`s = max(track_wetness, 0.7 * rain_intensity)`, bounded to that interval. The
baseline weather multiplier is `1 + 0.2 * s`. Driver wet skill contributes
`1 + (1 - wet_skill_modifier) * 0.02 * s`; the existing car contribution is
`1 + (1 - wet_performance) * 0.06 * s`. These factors multiply together. Dry
conditions have no wet-skill advantage, and skill develops gradually as exposure
increases. Rain contributes before the surface reaches equilibrium; accumulated
water still matters after rainfall stops.

Tyre mismatch adds seconds outside the weather multiplier, using straight-line
segments between these `(surface water, added seconds)` anchors:

| Compound | Anchors |
|---|---|
| Soft, medium, hard | (0, 0), (0.2, 0), (0.5, 7.5), (1, 30) |
| Intermediate | (0, 7), (0.15, 0), (0.8, 0), (1, 10) |
| Full wet | (0, 14), (0.3, 0), (1, 0) |

The dry and flooded endpoints and zero-penalty intervals are retained from the
previous model. Interpolation removes fixed jumps at their boundaries; slope
changes remain. Compound choices are discrete, so a small physical change can
still change the best tyre when projected strategies cross. The separate
survivability thresholds, fresh-rain tyre thresholds, paid-stop budgets and
surface-response cadence remain as described in the [strategy model](strategy-model.md).

These coefficients, normalized inputs and interpolation are model assumptions,
not a fit to telemetry or water depth. [Pirelli's tyre descriptions](https://www.pirelli.com/tires/en-us/motorsport/car/formula-1)
support distinguishing intermediates on wet and drying surfaces from full wets
for heavier water; they do not supply these numerical anchors. Temperature,
warm-up, aquaplaning physics and visibility are not resolved by these pace
curves. [OpenF1 rainfall observations](weather-calibration.md#observed-reference)
are binary and cannot calibrate intensity or standing water. Saved inputs replay
with installed simulator code, so earlier wet-race results may change after
this physics correction.

## Reproduction

```powershell
python examples/check_weather_pace.py
pytest -q tests/test_weather_pace_continuity.py tests/test_weather_pace_diagnostic.py
python examples/check_weather_transitions.py
```

The pace diagnostic compares all five compounds and four labels, sweeps tiny
water changes around former boundaries, and executes short races in both
engines with unlimited or finite replacement sets. It includes an explicit
used-soft start in damp conditions and an automatic opening in mild damp
conditions. It holds rainfall and water at equilibrium, removes incidents and
lap variation, and uses expected service time. Apart from descriptive condition
metadata, each label produces identical executed results and final race random
state for the same case, engine and inventory. Finite histories account for all
six completed laps and conserve physical wear.

The transition diagnostic independently executes bounded alternative schedules
to check the chosen pit policy against actual race times. Both examples print
JSON and make no network requests or default file writes. These controlled
checks establish internal consistency; they do not validate real-race pace or
strategy frequencies.
