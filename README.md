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

# Export results
python examples/simulate_race.py --race Monza --export --output-dir results/
```

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

## Requirements

- Python 3.11+
- FastF1 >= 3.3.0
- NumPy, Pandas, Pydantic
