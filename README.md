# F1 Championship Win Probability Calculator

A Python script that calculates the probability of each Formula 1 driver winning the championship based on current standings, recent performance, and reliability metrics.

## Features

- Fetches real-time F1 data from the OpenF1 API
- Calculates current championship standings
- Analyzes driver performance with emphasis on recent races
- Accounts for DNFs and disqualifications
- Uses Monte Carlo simulation to predict championship outcomes
- Identifies drivers in mathematical contention
- Displays probability percentages with visual bars

## How It Works

### Data Collection
- Fetches race sessions and results from OpenF1 API
- Calculates current championship points
- Analyzes historical performance for all drivers

### Performance Analysis
- **Recent Performance Weight**: 70% weight on last 5 races, 30% on full season
- **DNF/DSQ Tracking**: Calculates reliability rate based on past finishes
- **Position Variance**: Uses standard deviation to model consistency

### Monte Carlo Simulation
- Runs 10,000 simulations of remaining races
- Each simulation:
  - Applies DNF probability for each driver
  - Predicts finishing position based on weighted performance
  - Adds randomness using performance variance
  - Calculates points and updates standings
- Counts championship wins across all simulations
- Converts to percentages

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Show current standings and stats
python f1_championship_odds.py

# Calculate probabilities with remaining races
python f1_championship_odds.py --remaining-races 2 --remaining-sprints 1
```

### Command-Line Arguments

```
usage: f1_championship_odds.py [-h] [--year YEAR] [--simulations SIMULATIONS]
                                [--remaining-races REMAINING_RACES]
                                [--remaining-sprints REMAINING_SPRINTS]
                                [--clear-cache]

Calculate F1 championship win probabilities

optional arguments:
  -h, --help            show this help message and exit
  --year YEAR           F1 season year (default: 2025)
  --simulations SIMULATIONS
                        Number of Monte Carlo simulations (default: 10000)
  --remaining-races REMAINING_RACES
                        Manually specify number of remaining GP races (useful
                        when API schedule is incomplete)
  --remaining-sprints REMAINING_SPRINTS
                        Manually specify number of remaining sprint races
  --clear-cache         Clear cached API data before running
```

### Examples

```bash
# Show current 2025 season standings
python f1_championship_odds.py

# Calculate probabilities with 2 GP races and 1 sprint remaining
python f1_championship_odds.py --remaining-races 2 --remaining-sprints 1

# Use more simulations for higher accuracy (slower)
python f1_championship_odds.py --remaining-races 2 --simulations 50000

# Analyze previous season
python f1_championship_odds.py --year 2024

# Clear cache and fetch fresh data
python f1_championship_odds.py --clear-cache --remaining-races 2 --remaining-sprints 1
```

### How It Works

The script will:
1. Fetch current season data from OpenF1 API (or use cached data)
2. Calculate championship standings including sprint race points
3. Display current standings for drivers in mathematical contention
4. Run Monte Carlo simulations of remaining races (if any specified)
5. Display championship win probabilities with statistics

## Example Output

```
F1 Championship Win Probability Calculator
==================================================

Fetching 2025 season data from OpenF1 API...
Calculating current championship standings...
Processing 22 completed GP races (including sprint points)...

Season Progress: 22/24 GP races completed
Remaining GP races: 2
Remaining sprint races: 1
  (Note: Remaining races manually specified)

Analyzing driver performance and reliability...

Drivers in mathematical contention: 3

Current Championship Standings (Contenders):
--------------------------------------------------
 1. Lando Norris              390.0 points
 2. Max Verstappen            366.0 points (-24.0)
 3. Oscar Piastri             366.0 points (-24.0)

Running 10,000 simulations for 2 GP races and 1 sprint...

==================================================
CHAMPIONSHIP WIN PROBABILITIES
==================================================
 1. Lando Norris               67.41% █████████████████████████████████
    Current points: 390.0
    Recent avg position: 5.4
    DNF rate: 13.6%

 2. Max Verstappen             27.94% █████████████
    Current points: 366.0
    Recent avg position: 2.0
    DNF rate: 4.5%

 3. Oscar Piastri               4.65% ██
    Current points: 366.0
    Recent avg position: 7.8
    DNF rate: 9.1%

==================================================
SIMULATION DETAILS
==================================================
Simulations run: 10,000
Remaining GP races: 2
Remaining sprint races: 1
Maximum points available: 58
```

## How the Simulation Works

### Points Systems
- **GP Races**: Standard F1 points (25-18-15-12-10-8-6-4-2-1 for top 10)
- **Sprint Races**: Sprint points (8-7-6-5-4-3-2-1 for top 8)

### Performance Modeling
The simulation uses historical data to predict future performance:

- **Recent vs Overall**: 70% weight on last 5 races, 30% on full season
- **Reliability**: DNF/DSQ/DNS rate calculated from all races
- **Consistency**: Standard deviation of finishing positions
- **Sprint Adjustment**: 50% lower DNF rate for sprints (shorter races)

### Monte Carlo Method
Each simulation:
1. Starts with current championship points
2. Randomly orders remaining GP races and sprints
3. For each session:
   - Predicts each driver's position using weighted average ± variance
   - Applies DNF probability
   - Awards points based on session type (GP or sprint)
4. Determines championship winner
5. Repeats 10,000 times to calculate win probability

## Data Source

This script uses the [OpenF1 API](https://openf1.org/) which provides:
- Historical and real-time F1 data
- No authentication required for historical data
- JSON format responses
- Comprehensive race results and driver information

## Requirements

- Python 3.7+
- `requests` library

## Important Notes

### API Schedule Limitations
The OpenF1 API may not have all future races in the schedule yet. Use the `--remaining-races` and `--remaining-sprints` arguments to manually specify remaining sessions when:
- The API hasn't updated with upcoming race weekends
- You want to simulate "what if" scenarios

### Caching
- API responses are cached in `.cache/` directory to minimize API calls
- Use `--clear-cache` to fetch fresh data
- Cache persists between runs for faster subsequent executions

### Limitations
- Predictions are based on historical performance and probability models
- Cannot account for unexpected events (team changes, car upgrades, penalties, etc.)
- Accuracy depends on amount of historical data available
- Assumes consistent performance patterns from recent races
- Real-time data requires paid OpenF1 account (script uses freely available historical data)

## License

This is a free tool for F1 fans and data enthusiasts.
