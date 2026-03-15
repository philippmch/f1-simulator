# F1 Monte Carlo Simulator

Monte Carlo simulation for Formula 1 races using historical data from FastF1.

## Features

- **Historical Data Integration**: Uses FastF1 to load real qualifying grids, lap times, and driver performance
- **Weighted Driver Stats**: Combines track-specific performance, recent form, and qualifying data
- **Track-Specific Behavior**:
  - High-difficulty tracks (Monaco, Singapore): Minimal overtaking, grid position is king
  - Normal tracks (Bahrain, Monza): Realistic overtaking and pit strategy
- **Realistic Race Simulation**:
  - Pit stop strategy with position-aware timing
  - DRS effectiveness scaled by track difficulty
  - DNFs from mechanical failures and incidents
  - Safety car deployments

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

## Usage

```bash
# Simulate a race (uses historical qualifying grid)
python examples/simulate_race.py --race Monaco --simulations 100

# Different races
python examples/simulate_race.py --race Bahrain --simulations 500
python examples/simulate_race.py --race Silverstone

# Export results (CSV + JSON + interactive HTML report)
python examples/simulate_race.py --race Monza --export --output-dir results/
# includes run history + browser index at results/index.html

# Reproducible run with explicit seed + worker cap
python examples/simulate_race.py --race Bahrain --simulations 1000 --seed 7 --max-workers 4

# Compare weather scenarios in one run
python examples/simulate_race.py --race Silverstone --simulations 300 --scenarios dry,light_rain,heavy_rain --export
# CLI now also prints event calibration diagnostics (expected vs observed SC race rate)

# Run minimal dashboard UI/API (after `pip install -e '.[web]'`)
python -m f1sim.web.dashboard
# open http://127.0.0.1:8080 in browser and run scenarios
# API also exposes /api/runs for recent exported run history
# dashboard shows scenario cards (incl. runtime/sims-sec), top-N win chart/matrix controls, trend metric+scale controls, sortable/filterable/exportable matrix, result JSON download, quick links, quick example presets, rerun, run filtering, and persistent UI prefs
```

When exporting, `statistics.json` includes run metadata (`seed`, `parallel`, `max_workers`), team championship projection, top-3/top-10 finish probabilities, and per-driver position percentiles (P10/P50/P90) for richer race analysis.

## Example Output

```
WIN PROBABILITIES:
--------------------------------------------------
VER                   66.5% #################################
PER                   24.0% ############
SAI                    8.0% ####
...
```

## Project Structure

```
src/f1sim/
├── models/          # Driver, Car, Track, Tire, Weather models
├── simulation/      # Race, qualifying, lap time, overtaking logic
├── analysis/        # Monte Carlo runner
├── data/            # FastF1 data loader with weighted stats
└── output/          # Console and file exporters
```

## Current Status

Working simulation with:
- 2025 season data (default), also supports 2024
- Historical qualifying grids as starting positions
- Track-specific overtake difficulty (Monaco 0.95, Bahrain 0.35, etc.)
- Realistic pit stop windows (1-2 stops per race)
- Driver skill affecting lap times (~0.1s/lap difference between teammates)
- Realism-aware event model: lap-progression reliability, component-level failure risk, heat stress, weather/consistency incident scaling, plus mechanical-failure calibration metrics, tuning suggestions, and numeric reliability adjustment recommendations
- Calibrated safety controls: SC/VSC probabilities scale with track risk, incidents, weather, and race phase
- Tire crossover realism: dynamic slick/inter/wet mismatch thresholds for changing conditions
- Team strategy archetypes with multi-plan pit strategies, stint-compound optimization, dynamic switching, tunable strategy thresholds, and configurable strategy profiles
- Strategy-aware pit logic: free-stop detection under SC/VSC, undercut/overcut bias, and late-race soft-tire sprinting

## Requirements

- Python 3.11+
- FastF1 >= 3.3.0
- NumPy, Pandas, Pydantic
