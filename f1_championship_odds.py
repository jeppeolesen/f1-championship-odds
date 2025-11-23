#!/usr/bin/env python3
"""
F1 Championship Win Probability Calculator

This script calculates the probability of each driver winning the championship based on current standings, recent performance, and reliability metrics.
"""

import requests
import json
import time
import os
import statistics
import random
from datetime import datetime
from collections import defaultdict
from pathlib import Path


class F1ChampionshipPredictor:
    """Predicts F1 championship win probabilities using Monte Carlo simulation."""

    BASE_URL = "https://api.openf1.org/v1"
    CURRENT_YEAR = 2025

    # Points system for GPs
    POINTS_MAP = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }

    # Points system for sprints
    SPRINT_POINTS_MAP = {
        1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1
    }

    def __init__(self, simulations=10000, cache_dir=".cache", override_remaining_races=None, override_remaining_sprints=None):
        """
        Initialize the predictor.

        Args:
            simulations: Number of Monte Carlo simulations to run
            cache_dir: Directory to store cached API responses
            override_remaining_races: Manually specify remaining GPs
            override_remaining_sprints: Manually specify remaining sprints
        """
        
        self.simulations = simulations
        self.drivers_data = {}
        self.current_standings = {}
        self.remaining_races = 0
        self.override_remaining_races = override_remaining_races
        self.override_remaining_sprints = override_remaining_sprints
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum seconds between requests, to avoid OpenF1 rate limits

    def _rate_limit(self):
        """Ensure minimum time between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()

    def _get_cache_path(self, cache_key):
        """Get the cache file path for a given key."""
        return self.cache_dir / f"{cache_key}.json"

    def _read_cache(self, cache_key):
        """Read data from cache if it exists."""
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        return None

    def _write_cache(self, cache_key, data):
        """Write data to cache."""
        cache_path = self._get_cache_path(cache_key)
        
        with open(cache_path, 'w') as f:
            json.dump(data, f)

    def _fetch_with_cache(self, url, cache_key):
        """Fetch data from URL with caching and rate limiting."""
        
        # Check cache first
        cached_data = self._read_cache(cache_key)
        
        if cached_data is not None:
            return cached_data

        # Rate limit before making request
        self._rate_limit()

        # Fetch from API
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Cache the result
        self._write_cache(cache_key, data)
        
        return data

    def fetch_sessions(self, year=None):
        """Fetch all sessions for the specified year."""
        year = year or self.CURRENT_YEAR
        url = f"{self.BASE_URL}/sessions?year={year}"
        cache_key = f"sessions_{year}"
        
        return self._fetch_with_cache(url, cache_key)

    def fetch_session_results(self, session_key):
        """Fetch results for a specific session."""
        url = f"{self.BASE_URL}/session_result?session_key={session_key}"
        cache_key = f"session_result_{session_key}"
        
        return self._fetch_with_cache(url, cache_key)

    def fetch_drivers(self, session_key):
        """Fetch driver information for a session."""
        url = f"{self.BASE_URL}/drivers?session_key={session_key}"
        cache_key = f"drivers_{session_key}"
        
        return self._fetch_with_cache(url, cache_key)

    def get_race_sessions(self):
        """Get all race sessions (excluding practice, qualifying, and sprints)."""
        sessions = self.fetch_sessions()
        
        # Filter for Race sessions but exclude Sprint races
        race_sessions = [s for s in sessions
                        if s['session_type'] == 'Race'
                        and s.get('session_name', '').lower() != 'sprint']
        
        return sorted(race_sessions, key=lambda x: x['date_start'])

    def calculate_current_standings(self, race_sessions):
        """Calculate current championship standings from completed races and sprints."""
        standings = defaultdict(int)
        driver_names = {}

        # Get all sessions (including sprints) for points calculation
        all_sessions = self.fetch_sessions()
        all_race_sessions = [s for s in all_sessions if s['session_type'] == 'Race']

        # Count only completed GP races (not sprints) for progress tracking
        completed_count = 0
        
        for session in race_sessions:
            date_start = datetime.fromisoformat(session['date_start'].replace('Z', '+00:00'))
            if date_start <= datetime.now(date_start.tzinfo):
                completed_count += 1

        print(f"Processing {completed_count} completed GPs (including sprint points)...")
        race_num = 0

        # Process all race sessions (GPs and sprints) for points
        for session in sorted(all_race_sessions, key=lambda x: x['date_start']):
            # Skip if race hasn't happened yet
            date_start = datetime.fromisoformat(session['date_start'].replace('Z', '+00:00'))
            
            if date_start > datetime.now(date_start.tzinfo):
                continue

            # Only increment counter and show progress for actual races
            is_sprint = session.get('session_name', '').lower() == 'sprint'
            
            if not is_sprint:
                race_num += 1
                race_name = session.get('location', session.get('country_name', 'Unknown'))
                
                print(f"  Fetching race {race_num}/{completed_count}: {race_name}...", end='\r')

            results = self.fetch_session_results(session['session_key'])

            for result in results:
                driver_number = result.get('driver_number')
                points = result.get('points', 0)

                if driver_number is not None:
                    # Get driver name
                    if driver_number not in driver_names:
                        drivers = self.fetch_drivers(session['session_key'])
                        
                        for d in drivers:
                            if d['driver_number'] == driver_number:
                                driver_names[driver_number] = f"{d['first_name']} {d['last_name']}"
                                break

                    # Add points (from both GPs and sprints)
                    standings[driver_number] += points

        print(f"  Completed processing {completed_count} GP races.{' ' * 30}")
        
        return standings, driver_names

    def analyze_recent_performance(self, race_sessions, num_recent_races=5):
        """
        Analyze recent performance to estimate average finishing position and reliability.

        Returns:
            dict: Driver statistics including avg position, DNF rate, points per race
        """
        
        driver_stats = defaultdict(lambda: {
            'positions': [],
            'points': [],
            'dnf_count': 0,
            'race_count': 0,
            'recent_positions': [],
            'recent_points': []
        })

        completed_races = []
        
        for session in race_sessions:
            date_start = datetime.fromisoformat(session['date_start'].replace('Z', '+00:00'))
        
            if date_start <= datetime.now(date_start.tzinfo):
                completed_races.append(session)

        # Analyze all races
        for session in completed_races:
            results = self.fetch_session_results(session['session_key'])

            for result in results:
                driver_number = result.get('driver_number')
                position = result.get('position')
                is_dnf = result.get('dnf', False) or result.get('dsq', False) or result.get('dns', False)

                if driver_number:
                    stats = driver_stats[driver_number]
                    stats['race_count'] += 1

                    if is_dnf:
                        # DNF, DSQ, or DNS
                        stats['dnf_count'] += 1
                        stats['positions'].append(20)  # Penalty for DNF
                        stats['points'].append(0)
                    elif position:
                        stats['positions'].append(position)
                        # Use points from API if available, otherwise calculate
                        points = result.get('points', self.POINTS_MAP.get(position, 0))
                        stats['points'].append(points)
                    else:
                        # Fallback
                        stats['positions'].append(20)
                        stats['points'].append(0)

        # Analyze recent races (more heavily weighted)
        recent_races = completed_races[-num_recent_races:]
        
        for session in recent_races:
            results = self.fetch_session_results(session['session_key'])

            for result in results:
                driver_number = result.get('driver_number')
                position = result.get('position')
                is_dnf = result.get('dnf', False) or result.get('dsq', False) or result.get('dns', False)

                if driver_number:
                    stats = driver_stats[driver_number]

                    if is_dnf:
                        stats['recent_positions'].append(20)
                        stats['recent_points'].append(0)
                    elif position:
                        stats['recent_positions'].append(position)
                        points = result.get('points', self.POINTS_MAP.get(position, 0))
                        stats['recent_points'].append(points)
                    else:
                        stats['recent_positions'].append(20)
                        stats['recent_points'].append(0)

        # Calculate metrics
        analyzed_stats = {}
        
        for driver_number, stats in driver_stats.items():
            if stats['race_count'] > 0:
                analyzed_stats[driver_number] = {
                    'avg_position': statistics.mean(stats['positions']) if stats['positions'] else 15,
                    'recent_avg_position': statistics.mean(stats['recent_positions']) if stats['recent_positions'] else 15,
                    'avg_points': statistics.mean(stats['points']) if stats['points'] else 0,
                    'recent_avg_points': statistics.mean(stats['recent_points']) if stats['recent_points'] else 0,
                    'dnf_rate': stats['dnf_count'] / stats['race_count'],
                    'position_std': statistics.stdev(stats['positions']) if len(stats['positions']) > 1 else 5,
                    'race_count': stats['race_count']
                }

        return analyzed_stats

    def simulate_race(self, driver_stats, current_points, is_sprint=False):
        """
        Simulate a single race outcome for all drivers.

        Uses recent performance weighted 70%, overall performance 30%.
        Accounts for DNF probability and position variance.

        Args:
            driver_stats: Performance statistics for drivers
            current_points: Current championship points
            is_sprint: If True, use sprint points system (top 8 only)
        """
        race_results = {}
        points_map = self.SPRINT_POINTS_MAP if is_sprint else self.POINTS_MAP

        for driver_number in current_points.keys():
            if driver_number not in driver_stats:
                continue

            stats = driver_stats[driver_number]

            # Check for DNF (slightly lower rate for sprints due to shorter race)
            dnf_rate = stats['dnf_rate'] * (0.5 if is_sprint else 1.0)
        
            if random.random() < dnf_rate:
                race_results[driver_number] = 0
                continue

            # Weighted average: 70% recent, 30% overall
            expected_position = (0.7 * stats['recent_avg_position'] + 0.3 * stats['avg_position'])

            # Add randomness based on standard deviation
            position = max(1, min(20, int(random.gauss(expected_position, stats['position_std']))))

            # Convert position to points
            points = points_map.get(position, 0)
            race_results[driver_number] = points

        return race_results

    def run_monte_carlo_simulation(self, current_standings, driver_stats, remaining_races, remaining_sprints=0):
        """
        Run Monte Carlo simulation to estimate championship win probabilities.

        Args:
            current_standings: Current championship points
            driver_stats: Performance statistics for each driver
            remaining_races: Number of GP races left in season
            remaining_sprints: Number of sprint races left in season

        Returns:
            dict: Win probability for each driver
        """
        win_counts = defaultdict(int)

        total_sessions = remaining_races + remaining_sprints
        sessions_desc = f"{remaining_races} GP race{'s' if remaining_races != 1 else ''}"
        
        if remaining_sprints > 0:
            sessions_desc += f" and {remaining_sprints} sprint{'s' if remaining_sprints != 1 else ''}"

        print(f"\nRunning {self.simulations:,} simulations for {sessions_desc}...")

        for sim in range(self.simulations):
            if (sim + 1) % 1000 == 0:
                print(f"  Completed {sim + 1:,} simulations...")

            # Start with current points
            sim_standings = current_standings.copy()

            # Simulate remaining sprints and races
            # Note: We're not bothering with the exact correct order, so we interleave them randomly
            sessions = ['sprint'] * remaining_sprints + ['race'] * remaining_races
            random.shuffle(sessions)

            for session_type in sessions:
                is_sprint = (session_type == 'sprint')
                race_results = self.simulate_race(driver_stats, sim_standings, is_sprint=is_sprint)
                
                for driver, points in race_results.items():
                    sim_standings[driver] += points

            # Find winner
            if sim_standings:
                winner = max(sim_standings, key=sim_standings.get)
                win_counts[winner] += 1

        # Calculate probabilities
        probabilities = {}
        
        for driver, wins in win_counts.items():
            probabilities[driver] = (wins / self.simulations) * 100

        return probabilities

    def get_drivers_in_contention(self, current_standings, remaining_races, threshold_gap=None):
        """
        Identify drivers still in mathematical contention.

        Args:
            current_standings: Current points
            remaining_races: Races remaining
            threshold_gap: Maximum points gap to consider (default: max points available)
        """
        if not current_standings:
            return {}

        max_points_available = remaining_races * 25
        leader_points = max(current_standings.values())

        if threshold_gap is None:
            threshold_gap = max_points_available

        in_contention = {}
        
        for driver, points in current_standings.items():
            gap = leader_points - points
            if gap <= threshold_gap:
                in_contention[driver] = points

        return in_contention

    def predict_championship(self):
        """Main method to predict championship probabilities."""
        print("F1 Championship Win Probability Calculator")
        print("=" * 50)
        print(f"\nFetching {self.CURRENT_YEAR} season data from OpenF1 API...")

        # Get all race sessions
        race_sessions = self.get_race_sessions()
        total_races = len(race_sessions)

        # Calculate current standings
        print("Calculating current championship standings...")
        current_standings, driver_names = self.calculate_current_standings(race_sessions)

        # Count completed races
        completed_races = 0
        
        for session in race_sessions:
            date_start = datetime.fromisoformat(session['date_start'].replace('Z', '+00:00'))
            if date_start <= datetime.now(date_start.tzinfo):
                completed_races += 1

        # Use override if provided, otherwise calculate from API data
        if self.override_remaining_races is not None:
            remaining_races = self.override_remaining_races
            total_races = completed_races + remaining_races
        else:
            remaining_races = total_races - completed_races

        # Handle remaining sprints
        remaining_sprints = self.override_remaining_sprints or 0

        print(f"\nSeason Progress: {completed_races}/{total_races} GP races completed")
        print(f"Remaining GP races: {remaining_races}")
        
        if remaining_sprints > 0:
            print(f"Remaining sprint races: {remaining_sprints}")
        if self.override_remaining_races is not None or self.override_remaining_sprints is not None:
            print(f"  (Note: Remaining races manually specified)")

        if remaining_races == 0 and remaining_sprints == 0:
            print("\nSeason complete! Final standings:")
            sorted_standings = sorted(current_standings.items(), key=lambda x: x[1], reverse=True)
        
            for i, (driver, points) in enumerate(sorted_standings[:10], 1):
                name = driver_names.get(driver, f"Driver #{driver}")
                print(f"{i:2d}. {name:25s} - {points:.1f} points")
        
            return

        # Analyze performance
        print("\nAnalyzing driver performance and reliability...")
        driver_stats = self.analyze_recent_performance(race_sessions)

        # Get drivers in contention
        in_contention = self.get_drivers_in_contention(current_standings, remaining_races)

        print(f"\nDrivers in mathematical contention: {len(in_contention)}")

        # Display current standings for contenders
        print("\nCurrent Championship Standings (Contenders):")
        print("-" * 50)
        sorted_contenders = sorted(in_contention.items(), key=lambda x: x[1], reverse=True)
        
        for i, (driver, points) in enumerate(sorted_contenders, 1):
            name = driver_names.get(driver, f"Driver #{driver}")
            gap = max(in_contention.values()) - points
            gap_str = f"(-{gap})" if gap > 0 else ""
        
            print(f"{i:2d}. {name:25s} {points:.1f} points {gap_str}")

        # Run simulation
        probabilities = self.run_monte_carlo_simulation(
            in_contention, driver_stats, remaining_races, remaining_sprints
        )

        # Display results
        print("\n" + "=" * 50)
        print("CHAMPIONSHIP WIN PROBABILITIES")
        print("=" * 50)

        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        for i, (driver, prob) in enumerate(sorted_probs, 1):
            name = driver_names.get(driver, f"Driver #{driver}")
            points = in_contention[driver]

            # Create probability bar
            bar_length = int(prob / 2)  # Scale to 50 chars max
            bar = "█" * bar_length

            print(f"{i:2d}. {name:25s} {prob:6.2f}% {bar}")
            print(f"    Current points: {points:.1f}")

            if driver in driver_stats:
                stats = driver_stats[driver]
                print(f"    Recent avg position: {stats['recent_avg_position']:.1f}")
                print(f"    DNF rate: {stats['dnf_rate']*100:.1f}%")
            print()

        # Summary statistics
        print("=" * 50)
        print("SIMULATION DETAILS")
        print("=" * 50)
        print(f"Simulations run: {self.simulations:,}")
        print(f"Remaining GP races: {remaining_races}")
 
        if remaining_sprints > 0:
            print(f"Remaining sprint races: {remaining_sprints}")
        max_points = (remaining_races * 25) + (remaining_sprints * 8)
 
        print(f"Maximum points available: {max_points}")


def main():
    """Run the championship predictor."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Calculate F1 championship win probabilities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Show current standings
            python f1_championship_odds.py

            # Calculate probabilities with 2 remaining races
            python f1_championship_odds.py --remaining-races 2

            # Use different year
            python f1_championship_odds.py --year 2024
        """
    )
    parser.add_argument(
        '--year',
        type=int,
        default=2025,
        help='F1 season year (default: 2025)'
    )
    parser.add_argument(
        '--simulations',
        type=int,
        default=10000,
        help='Number of Monte Carlo simulations (default: 10000)'
    )
    parser.add_argument(
        '--remaining-races',
        type=int,
        default=None,
        help='Manually specify number of remaining GPs'
    )
    parser.add_argument(
        '--remaining-sprints',
        type=int,
        default=None,
        help='Manually specify number of remaining sprints'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear cached API data before running'
    )

    args = parser.parse_args()

    if args.clear_cache:
        import shutil
        cache_dir = Path('.cache')
        
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("Cache cleared.\n")

    try:
        predictor = F1ChampionshipPredictor(
            simulations=args.simulations,
            override_remaining_races=args.remaining_races,
            override_remaining_sprints=args.remaining_sprints
        )
        predictor.CURRENT_YEAR = args.year
        predictor.predict_championship()
    except requests.exceptions.RequestException as e:
        print(f"\nError fetching data from API: {e}")
        print("Please check your internet connection and try again.")
        print("\nIf you're seeing rate limit errors, the data may have been cached anyway.")
        print("Run the script again to use cached data.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
