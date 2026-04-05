# F1 Monte Carlo Simulator

Monte Carlo simulation for Formula 1 races using historical data from FastF1.

## Features

- **Historical Data Integration**: Uses FastF1 to load real qualifying grids, lap times, and driver performance
- **Weighted Driver Stats**: Combines track-specific performance, recent form, and qualifying data
- **Real-Time Ratings API**: Backend serves per-race driver skill and car pace ratings derived from actual F1 data
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

# Install core dependencies
pip install -e .

# Install web server dependencies (required for real data ratings)
pip install -e '.[web]'
```

## Quick Start

The recommended way to use the simulator is with the **web frontend + backend server**, which gives you real FastF1-derived ratings for every race:

```bash
# 1. Start the backend server
python -m f1sim.web.server
# Server starts at http://127.0.0.1:8080

# 2. Open the frontend
open frontend/index.html
# Or serve it: python3 -m http.server 3000 -d frontend
```

When you run a simulation in the frontend, it automatically fetches real driver/car ratings from the backend (derived from FastF1 historical data). If the backend is unavailable, it falls back to hardcoded defaults.

The backend API dashboard is also available at http://127.0.0.1:8080 for running scenario comparisons.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ratings?year=2025&race=Bahrain` | GET | Real driver skill + car pace ratings from FastF1 |
| `/api/run` | POST | Run full Monte Carlo simulation on the backend |
| `/api/runs` | GET | Recent exported run history |
| `/api/health` | GET | Health check |

## CLI Usage

```bash
# Simulate a race (uses historical qualifying grid)
python examples/simulate_race.py --race Monaco --simulations 100

# Different races
python examples/simulate_race.py --race Bahrain --simulations 500
python examples/simulate_race.py --race Silverstone

# Export results (CSV + JSON + interactive HTML report)
python examples/simulate_race.py --race Monza --export --output-dir results/

# Reproducible run with explicit seed + worker cap
python examples/simulate_race.py --race Bahrain --simulations 1000 --seed 7 --max-workers 4

# Compare weather scenarios in one run
python examples/simulate_race.py --race Silverstone --simulations 300 --scenarios dry,light_rain,heavy_rain --export
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

## Web Frontend

A standalone browser-based UI is included in `frontend/index.html`.

Features:
- Season and circuit selection with configurable simulation count
- **Real-data ratings** from FastF1 when the backend server is running (green badge indicator)
- Fallback to hardcoded ratings when offline (gray badge)
- Race results timing tower (positions, gaps, tire strategy, pit stops, DNFs)
- Qualifying results with Q1/Q2/Q3 elimination rounds
- Monte Carlo statistics: win probabilities, podium rates, average points, DNF rates, and position distribution heatmap

## Project Structure

```
src/f1sim/
├── models/          # Driver, Car, Track, Tire, Weather models
├── simulation/      # Race, qualifying, lap time, overtaking logic
├── analysis/        # Monte Carlo runner
├── data/            # FastF1 data loader with weighted stats
├── output/          # Console and file exporters
└── web/             # FastAPI server and API endpoints
frontend/
└── index.html       # Browser-based simulation UI
```

## Current Status

Working simulation with:
- 2025 season data (default), also supports 2023-2024
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
- FastAPI + Uvicorn (for web server: `pip install -e '.[web]'`)
