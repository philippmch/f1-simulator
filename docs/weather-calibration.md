# Weather and interruption calibration

Updated 2026-09-07. These checks constrain event frequency and weather behavior;
they do not estimate a forecast for a particular venue or validate winner odds.

## Observed reference

The diagnostic queries [OpenF1 race sessions](https://api.openf1.org/v1/sessions?year=2026&session_name=Race),
race-control messages and weather records for completed current-season races.
The cutoff is midnight UTC on the day the check runs, excluding races still in
progress. It excludes cancelled sessions and requires a chequered-flag record.
Only weather observations between scheduled start and the final chequered flag
are included. Binary rainfall readings indicate observed rain, not intensity or
standing-water depth. All provider rows stay in memory.

Before 2026-09-06, the sample contained 12 races: 11 with no rainfall observed
(one with a red flag), and one with rainfall observed (also with a red flag).
The flagged races were Monaco and the Netherlands. Current race-control records
can have `flag=null` and `message="RED FLAG - RACE SUSPENDED"`, so checking only
the flag field incorrectly reports zero interruptions.

This is a small, mixed-cause sample. The official
[Dutch race report video](https://www.formula1.com/en/video/2026-dutch-grand-prix-race-red-flagged-as-verstappen-crashes-out.1874321001211871979)
attributes its flag to a crash. A single rainy race cannot establish a
weather-caused interruption probability. The observed ratios are sanity checks,
not precise fitted estimates.

Filtering the weather window matters: Miami has rainfall records after the
chequered flag, which this diagnostic correctly excludes.

## Model choices

Aggregated `RaceEventStatistics` records `num_simulations`, including races with
no events. Its `safety_car_rate` and `red_flag_rate` properties are percentages
of races with at least one deployment; several flags in one race count once.
Directly constructed statistics must supply the denominator to compute these
properties; an empty sample returns zero. Statistics JSON includes `event_rates`,
whose occurrence rates are fractions from zero to one, and whose `avg_*` values
are deployment or incident counts per simulated race.

Incident totals include contact during overtaking as well as random collisions,
spins, punctures and mechanical failures. One contact between two cars counts
as one incident. The event ledger records both drivers and their separately
sampled losses in `applied_time_losses`; those losses have already been charged
by battle resolution and are not applied again. The contact appears before any
race-control deployment it helps trigger on the same lap. Normal race runs
record the actual lap; direct internal battle calls without a lap use zero to
indicate unknown timing. Successful passes and clean failed attempts add no
incident record.

- Random racing incidents use the existing 22-car grid as the reference
  exposure. After circuit, consistency and weather modifiers, its per-lap
  probability `h22` is bounded between 0.0005 and 0.08. For `N` active cars,
  `hN = 1 - (1 - h22) ** (N / 22)`. Thus retirements reduce combined incident
  risk instead of concentrating the full-grid risk on the remaining drivers.
  A single survivor can still spin, suffer a puncture or crash; zero active
  cars receive no incident draw. The 0.08 defensive ceiling also applies to
  oversized synthetic fields, but the lower floor is not reapplied after
  exposure scaling. The 22-car reference and these bounds are model priors.
- Random-incident severity and driver-selection weights are unchanged.
  Punctures remain part of this incident channel; there is no separately
  calibrated tyre-wear failure hazard. Mechanical failures remain per-car
  events, on-track overtakes retain their own contact model, and the independent
  background safety-intervention channel is unchanged. Neutralized laps still
  suppress random racing incidents. At most one such incident is emitted per
  lap, so this is an aggregate exposure model, not independent incident draws
  for every car.
- While rain falls, surface wetness moves 20% of the distance toward normalized
  rainfall intensity each lap. This represents rainfall balanced by drainage:
  sustained intensity 0.35 tends toward wetness 0.35, not a flooded track.
  An already flooded surface drains even during light rain. Without rain,
  wetness falls by 0.03 per lap. The response rate and dimensionless intensity
  mapping are explicit assumptions, not measured hydrology.
- Minor overtake contacts retain their time losses and safety-car influence.
  Their count no longer represents multiple serious crashes triggering red flags.
- The background interruption prior is 10% over a full green-flag race,
  converted to per-lap hazard as `1 - (1 - 0.10) ** (1 / total_laps)`.
  Neutralized laps skip sampling; overall rates can therefore be lower.
  This channel represents major incidents or obstructions not individually
  resolved by the contact model.
- A continuous severe-weather episode receives one 50% interruption decision,
  including when that decision declines a flag. An episode qualifies at wetness
  and intensity both at least 0.8, or wetness at least 0.95. A new episode requires
  intensity below 0.65 and wetness below 0.8. The 50% prior and thresholds are
  modeling choices; the sparse observed sample does not fit them.
- Manual flags and independent background incidents remain possible. The race
  engine abstracts suspension duration and does not simulate elapsed waiting
  minutes, race abandonment, or a weather-conditioned restart forecast. One
  unchanged storm is not repeatedly sampled as a new weather interruption.

These choices preserve the distinction between damp/intermediate and heavily
waterlogged conditions described by
[Pirelli](https://www.pirelli.com/global/en-ww/race/racingspot/formula-1/when-it-s-time-to-change-from-slick-to-wet-tyres-in-formula-1-52943/).
An actual weather suspension can include a substantial wait for visibility to
improve, as in the official
[2025 Belgian race account](https://www.formula1.com/en/latest/article/piastri-wins-wet-dry-belgian-gp-after-late-pressure-from-title-rival-and.7QmPcUP90MvR5iX0w3j91).
Neither source supplies numerical parameters for this normalized model.

## Reproducible comparison

Recorded with the weather-calibration implementation at commit `d5aac75`.
Later simulation changes may alter these fixed-seed observations; rerun the
diagnostic against the current checkout for current values.

Synthetic 22-driver field, 50 laps, safety-car prior 0.3, identical car models,
100 simulations per scenario, seed 42. Fixed scenarios disable condition
transitions but still update surface water. The evolving case uses probability
0.1 per lap. These fixtures are not replays of observed races.

| Scenario | Old races with red flags | New races with red flags | Old flags/race | New flags/race |
|---|---:|---:|---:|---:|
| Fixed dry | 84% | 11% | 1.48 | 0.11 |
| Fixed light rain | 100% | 13% | 14.23 | 0.13 |
| Fixed heavy rain | 100% | 53% | 15.57 | 0.58 |
| Evolving dry start | 93% | 12% | 3.39 | 0.12 |

A separate seed-17 run gave 9%, 13%, 48%, and 14% respectively. Deterministic
tests cover surface-water equilibrium, storm resets and forced flags;
statistical tests check the prior across different lap counts and minor-contact
volumes. Numerical priors can be revisited as more current-season evidence arrives.

```powershell
python examples/check_weather_calibration.py --simulations 100 --seed 42
python examples/check_weather_calibration.py --simulations 100 --seed 17
python examples/check_weather_calibration.py --observed --simulations 100
pytest -q tests/test_weather_calibration.py
```

Only `--observed` makes network requests. The diagnostic prints summaries and
does not add runtime feed dependencies, replay data, or a persistent cache.
