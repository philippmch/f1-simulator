"""Historical data fetching and processing using FastF1."""

from pathlib import Path
from typing import Any

import fastf1
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from f1sim.models import Car, Driver, Sector, Track, TireCompound


class DriverStats(BaseModel):
    """Statistical summary of driver performance from historical data."""

    driver_id: str
    driver_name: str
    team_id: str
    team_name: str
    avg_lap_time: float
    lap_time_std: float
    avg_sector1: float
    avg_sector2: float
    avg_sector3: float
    pit_stop_avg: float
    pit_stop_std: float
    dnf_rate: float
    sample_size: int


class TrackStats(BaseModel):
    """Statistical summary of track characteristics from historical data."""

    track_id: str
    track_name: str
    country: str
    total_laps: int
    fastest_lap: float
    avg_lap_time: float
    sector1_avg: float
    sector2_avg: float
    sector3_avg: float
    pit_lane_time: float
    safety_car_rate: float
    drs_zones: int


class HistoricalDataLoader:
    """Loads and processes F1 historical data using FastF1."""

    def __init__(self, cache_dir: str | Path = "data/cache"):
        """Initialize the data loader.

        Args:
            cache_dir: Directory for FastF1 cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))

        self._sessions: dict[str, Any] = {}
        self._driver_stats: dict[str, DriverStats] = {}
        self._track_stats: dict[str, TrackStats] = {}

    def load_race(self, year: int, race: str | int) -> Any:
        """Load a race session.

        Args:
            year: Season year
            race: Race name or round number

        Returns:
            FastF1 Session object
        """
        cache_key = f"{year}_{race}_R"
        if cache_key not in self._sessions:
            session = fastf1.get_session(year, race, "R")
            session.load()
            self._sessions[cache_key] = session
        return self._sessions[cache_key]

    def load_qualifying(self, year: int, race: str | int) -> Any:
        """Load a qualifying session.

        Args:
            year: Season year
            race: Race name or round number

        Returns:
            FastF1 Session object
        """
        cache_key = f"{year}_{race}_Q"
        if cache_key not in self._sessions:
            session = fastf1.get_session(year, race, "Q")
            session.load()
            self._sessions[cache_key] = session
        return self._sessions[cache_key]

    def get_historical_grid(self, year: int, race: str | int) -> list[str]:
        """Get the historical qualifying grid (starting positions).

        Args:
            year: Season year
            race: Race name or round number

        Returns:
            List of driver abbreviations in grid order (P1 first)
        """
        quali_session = self.load_qualifying(year, race)

        # Get the qualifying results
        results = quali_session.results
        if results is None or results.empty:
            return []

        # Sort by position and return driver abbreviations
        sorted_results = results.sort_values("Position")
        return sorted_results["Abbreviation"].tolist()

    def get_driver_lap_stats(
        self,
        year: int,
        races: list[str | int] | None = None,
    ) -> dict[str, DriverStats]:
        """Calculate driver statistics from historical races.

        Args:
            year: Season year
            races: List of races to include (None = all races)

        Returns:
            Dictionary of driver_id -> DriverStats
        """
        if races is None:
            schedule = fastf1.get_event_schedule(year)
            races = list(range(1, len(schedule) + 1))

        # Store per-race normalized deltas (percentage slower than fastest)
        driver_race_deltas: dict[str, dict[str | int, float]] = {}  # driver -> {race: delta}
        driver_race_stds: dict[str, dict[str | int, float]] = {}
        driver_teams: dict[str, str] = {}
        driver_names: dict[str, str] = {}
        pit_data: list[dict] = []
        dnf_data: dict[str, dict] = {}

        for race in races:
            try:
                session = self.load_race(year, race)
                laps = session.laps.pick_quicklaps()

                if laps.empty:
                    continue

                # Calculate fastest lap of the session for normalization
                all_lap_times = laps["LapTime"].dt.total_seconds().dropna()
                if len(all_lap_times) == 0:
                    continue
                session_fastest = all_lap_times.min()

                # Process each driver's laps relative to session fastest
                for driver in laps["Driver"].unique():
                    driver_laps = laps[laps["Driver"] == driver]
                    lap_times = driver_laps["LapTime"].dt.total_seconds().dropna()

                    # Require minimum 5 laps per race for reliability
                    if len(lap_times) < 5:
                        continue

                    # Use median for robustness, calculate as % delta from fastest
                    median_time = lap_times.median()
                    delta_pct = (median_time - session_fastest) / session_fastest

                    if driver not in driver_race_deltas:
                        driver_race_deltas[driver] = {}
                        driver_race_stds[driver] = {}

                    driver_race_deltas[driver][race] = delta_pct
                    driver_race_stds[driver][race] = lap_times.std() / median_time

                    # Store team/name info
                    team = driver_laps.iloc[-1].get("Team", "Unknown")
                    driver_teams[driver] = team if isinstance(team, str) else "Unknown"
                    if "FullName" in driver_laps.columns:
                        driver_names[driver] = driver_laps.iloc[-1].get("FullName", driver)
                    else:
                        driver_names[driver] = driver

                # Extract pit stop data
                for _, lap in session.laps.iterrows():
                    if lap["PitInTime"] is not pd.NaT and lap["PitOutTime"] is not pd.NaT:
                        pit_duration = (lap["PitOutTime"] - lap["PitInTime"]).total_seconds()
                        stationary_time = max(1.5, pit_duration - 18)
                        pit_data.append({
                            "Driver": lap["Driver"],
                            "PitTime": stationary_time,
                        })

                # Track DNFs
                results = session.results
                for _, result in results.iterrows():
                    driver = result["Abbreviation"]
                    if driver not in dnf_data:
                        dnf_data[driver] = {"races": 0, "dnfs": 0}
                    dnf_data[driver]["races"] += 1
                    if result["Status"] not in ["Finished", "+1 Lap", "+2 Laps", "+3 Laps"]:
                        dnf_data[driver]["dnfs"] += 1

            except Exception as e:
                print(f"Warning: Could not load race {race}: {e}")
                continue

        if not driver_race_deltas:
            return {}

        pit_df = pd.DataFrame(pit_data) if pit_data else pd.DataFrame(columns=["Driver", "PitTime"])

        # Get reference lap time for absolute times
        reference_lap_time = 90.0
        try:
            session = self.load_race(year, races[-1])  # Use target race for reference
            quick_laps = session.laps.pick_quicklaps()["LapTime"].dt.total_seconds().dropna()
            if len(quick_laps) > 0:
                reference_lap_time = quick_laps.min()
        except Exception:
            pass

        stats: dict[str, DriverStats] = {}

        for driver, race_deltas in driver_race_deltas.items():
            # Require at least 2 races for stats
            if len(race_deltas) < 2:
                continue

            # Simple average of all available races
            avg_delta = np.median(list(race_deltas.values()))
            avg_std = np.median(list(driver_race_stds.get(driver, {0: 0.01}).values()))

            # Convert back to absolute time using reference
            avg_lap_time = reference_lap_time * (1 + avg_delta)
            lap_time_std = reference_lap_time * avg_std

            team = driver_teams.get(driver, "Unknown")

            # Pit stop stats
            driver_pits = pit_df[pit_df["Driver"] == driver]["PitTime"]
            pit_avg = driver_pits.mean() if len(driver_pits) > 0 else 2.5
            pit_std = driver_pits.std() if len(driver_pits) > 1 else 0.3

            # DNF rate
            dnf_info = dnf_data.get(driver, {"races": 1, "dnfs": 0})
            dnf_rate = dnf_info["dnfs"] / max(1, dnf_info["races"])

            stats[driver] = DriverStats(
                driver_id=driver,
                driver_name=driver_names.get(driver, driver),
                team_id=team.lower().replace(" ", "_") if isinstance(team, str) else "unknown",
                team_name=team,
                avg_lap_time=float(avg_lap_time),
                lap_time_std=float(lap_time_std),
                avg_sector1=0,
                avg_sector2=0,
                avg_sector3=0,
                pit_stop_avg=float(pit_avg) if not pd.isna(pit_avg) else 2.5,
                pit_stop_std=float(pit_std) if not pd.isna(pit_std) else 0.3,
                dnf_rate=float(dnf_rate),
                sample_size=len(race_deltas),
            )

        self._driver_stats = stats
        return stats

    def get_weighted_driver_stats(
        self,
        year: int,
        target_race: str | int,
        form_races: int = 3,
        track_weight: float = 0.5,
        form_weight: float = 0.3,
        quali_weight: float = 0.2,
    ) -> dict[str, DriverStats]:
        """Calculate driver stats with weighted form + track + qualifying performance.

        Args:
            year: Season year
            target_race: The race to simulate
            form_races: Number of previous races for form (default: 3)
            track_weight: Weight for track-specific race performance (default: 0.5)
            form_weight: Weight for recent form (default: 0.3)
            quali_weight: Weight for qualifying performance (default: 0.2)

        Returns:
            Dictionary of driver_id -> DriverStats
        """
        # Get the race calendar to find previous races
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule["EventFormat"] != "testing"]

        # Find target race index (flexible matching)
        target_idx = None
        target_race_lower = str(target_race).lower()
        for idx, row in schedule.iterrows():
            event_name = str(row["EventName"]).lower()
            round_num = row["RoundNumber"]
            # Match by round number, exact name, or partial name match
            if (round_num == target_race or
                event_name == target_race_lower or
                target_race_lower in event_name or
                event_name.startswith(target_race_lower)):
                target_idx = idx
                break

        if target_idx is None:
            raise ValueError(f"Race '{target_race}' not found in {year} calendar")

        # Get race names/numbers
        race_list = schedule["RoundNumber"].tolist()
        target_pos = race_list.index(schedule.loc[target_idx, "RoundNumber"])

        # Determine which races to use
        # Form races: up to `form_races` races before target
        form_start = max(0, target_pos - form_races)
        form_race_ids = race_list[form_start:target_pos]

        # Target race
        target_race_id = race_list[target_pos]

        print(f"  Form races: {form_race_ids}")
        print(f"  Target race: {target_race_id}")
        print(f"  Weights: track={track_weight:.0%}, form={form_weight:.0%}, quali={quali_weight:.0%}")

        # Load data for all relevant races
        all_races = form_race_ids + [target_race_id]
        driver_race_deltas: dict[str, dict] = {}
        driver_race_stds: dict[str, dict] = {}
        driver_quali_deltas: dict[str, float] = {}  # Qualifying performance at target track
        driver_teams: dict[str, str] = {}
        driver_names: dict[str, str] = {}
        pit_data: list[dict] = []
        dnf_data: dict[str, dict] = {}

        # Load qualifying data for target race
        try:
            quali_session = self.load_qualifying(year, target_race_id)
            quali_results = quali_session.results
            if not quali_results.empty and "Q3" in quali_results.columns:
                # Get best qualifying time for each driver
                pole_time = None
                driver_quali_times: dict[str, float] = {}

                for _, result in quali_results.iterrows():
                    driver = result["Abbreviation"]
                    # Try Q3, then Q2, then Q1
                    best_time = None
                    for q in ["Q3", "Q2", "Q1"]:
                        if q in result and pd.notna(result[q]):
                            time_val = result[q]
                            if hasattr(time_val, "total_seconds"):
                                best_time = time_val.total_seconds()
                            elif isinstance(time_val, (int, float)):
                                best_time = float(time_val)
                            break

                    if best_time and best_time > 0:
                        driver_quali_times[driver] = best_time
                        if pole_time is None or best_time < pole_time:
                            pole_time = best_time

                # Calculate deltas from pole
                if pole_time:
                    for driver, q_time in driver_quali_times.items():
                        driver_quali_deltas[driver] = (q_time - pole_time) / pole_time

                print(f"  Qualifying data loaded: {len(driver_quali_deltas)} drivers")
        except Exception as e:
            print(f"  Warning: Could not load qualifying data: {e}")

        for race in all_races:
            try:
                session = self.load_race(year, race)
                laps = session.laps.pick_quicklaps()

                if laps.empty:
                    continue

                all_lap_times = laps["LapTime"].dt.total_seconds().dropna()
                if len(all_lap_times) == 0:
                    continue
                session_fastest = all_lap_times.min()

                for driver in laps["Driver"].unique():
                    driver_laps = laps[laps["Driver"] == driver]
                    lap_times = driver_laps["LapTime"].dt.total_seconds().dropna()

                    if len(lap_times) < 5:
                        continue

                    median_time = lap_times.median()
                    delta_pct = (median_time - session_fastest) / session_fastest

                    if driver not in driver_race_deltas:
                        driver_race_deltas[driver] = {}
                        driver_race_stds[driver] = {}

                    driver_race_deltas[driver][race] = delta_pct
                    driver_race_stds[driver][race] = lap_times.std() / median_time

                    team = driver_laps.iloc[-1].get("Team", "Unknown")
                    driver_teams[driver] = team if isinstance(team, str) else "Unknown"
                    if "FullName" in driver_laps.columns:
                        driver_names[driver] = driver_laps.iloc[-1].get("FullName", driver)
                    else:
                        driver_names[driver] = driver

                # Pit stops
                for _, lap in session.laps.iterrows():
                    if lap["PitInTime"] is not pd.NaT and lap["PitOutTime"] is not pd.NaT:
                        pit_duration = (lap["PitOutTime"] - lap["PitInTime"]).total_seconds()
                        stationary_time = max(1.5, pit_duration - 18)
                        pit_data.append({"Driver": lap["Driver"], "PitTime": stationary_time})

                # DNFs
                results = session.results
                for _, result in results.iterrows():
                    driver = result["Abbreviation"]
                    if driver not in dnf_data:
                        dnf_data[driver] = {"races": 0, "dnfs": 0}
                    dnf_data[driver]["races"] += 1
                    if result["Status"] not in ["Finished", "+1 Lap", "+2 Laps", "+3 Laps"]:
                        dnf_data[driver]["dnfs"] += 1

            except Exception as e:
                print(f"Warning: Could not load race {race}: {e}")
                continue

        if not driver_race_deltas:
            return {}

        pit_df = pd.DataFrame(pit_data) if pit_data else pd.DataFrame(columns=["Driver", "PitTime"])

        # Reference lap time from target race
        reference_lap_time = 90.0
        try:
            session = self.load_race(year, target_race_id)
            quick_laps = session.laps.pick_quicklaps()["LapTime"].dt.total_seconds().dropna()
            if len(quick_laps) > 0:
                reference_lap_time = quick_laps.min()
        except Exception:
            pass

        stats: dict[str, DriverStats] = {}

        for driver, race_deltas in driver_race_deltas.items():
            # Need data from target race OR at least 2 form races OR qualifying
            has_target = target_race_id in race_deltas
            has_quali = driver in driver_quali_deltas
            form_deltas = [race_deltas[r] for r in form_race_ids if r in race_deltas]

            if not has_target and not has_quali and len(form_deltas) < 2:
                continue

            # Calculate weighted average delta with qualifying
            components = []
            weights = []

            if has_target:
                components.append(race_deltas[target_race_id])
                weights.append(track_weight)

            if form_deltas:
                components.append(np.median(form_deltas))
                weights.append(form_weight)

            if has_quali:
                components.append(driver_quali_deltas[driver])
                weights.append(quali_weight)

            # Normalize weights and calculate weighted average
            if weights:
                total_weight = sum(weights)
                avg_delta = sum(c * w for c, w in zip(components, weights)) / total_weight
            else:
                avg_delta = 0.02  # Default ~2% off pace

            # Std dev (simple average)
            all_stds = list(driver_race_stds.get(driver, {0: 0.01}).values())
            avg_std = np.median(all_stds)

            avg_lap_time = reference_lap_time * (1 + avg_delta)
            lap_time_std = reference_lap_time * avg_std

            team = driver_teams.get(driver, "Unknown")

            driver_pits = pit_df[pit_df["Driver"] == driver]["PitTime"]
            pit_avg = driver_pits.mean() if len(driver_pits) > 0 else 2.5
            pit_std = driver_pits.std() if len(driver_pits) > 1 else 0.3

            dnf_info = dnf_data.get(driver, {"races": 1, "dnfs": 0})
            dnf_rate = dnf_info["dnfs"] / max(1, dnf_info["races"])

            stats[driver] = DriverStats(
                driver_id=driver,
                driver_name=driver_names.get(driver, driver),
                team_id=team.lower().replace(" ", "_") if isinstance(team, str) else "unknown",
                team_name=team,
                avg_lap_time=float(avg_lap_time),
                lap_time_std=float(lap_time_std),
                avg_sector1=0,
                avg_sector2=0,
                avg_sector3=0,
                pit_stop_avg=float(pit_avg) if not pd.isna(pit_avg) else 2.5,
                pit_stop_std=float(pit_std) if not pd.isna(pit_std) else 0.3,
                dnf_rate=float(dnf_rate),
                sample_size=len(race_deltas),
            )

        self._driver_stats = stats
        return stats

    def get_track_stats(self, year: int, race: str | int) -> TrackStats:
        """Get track statistics from a specific race.

        Args:
            year: Season year
            race: Race name or round number

        Returns:
            TrackStats for the track
        """
        session = self.load_race(year, race)
        event = session.event

        laps = session.laps.pick_quicklaps()
        if laps.empty:
            laps = session.laps[session.laps["LapTime"].notna()]

        lap_times = laps["LapTime"].dt.total_seconds()
        s1_times = laps["Sector1Time"].dt.total_seconds().dropna()
        s2_times = laps["Sector2Time"].dt.total_seconds().dropna()
        s3_times = laps["Sector3Time"].dt.total_seconds().dropna()

        # Count safety car laps (approximation based on slow laps)
        all_laps = session.laps["LapTime"].dt.total_seconds().dropna()
        avg_lap = lap_times.mean()
        slow_laps = (all_laps > avg_lap * 1.3).sum()
        sc_rate = min(1.0, slow_laps / len(all_laps) * 5) if len(all_laps) > 0 else 0.3

        # Get circuit info
        circuit_info = session.get_circuit_info() if hasattr(session, "get_circuit_info") else None
        drs_zones = 0
        if circuit_info is not None and hasattr(circuit_info, "marshal_sectors"):
            # Estimate DRS zones from circuit data
            drs_zones = max(1, len(circuit_info.marshal_sectors) // 10)
        else:
            drs_zones = 2  # Default assumption

        stats = TrackStats(
            track_id=event["EventName"].lower().replace(" ", "_") if "EventName" in event else str(race),
            track_name=event.get("EventName", str(race)),
            country=event.get("Country", "Unknown"),
            total_laps=int(session.total_laps) if hasattr(session, "total_laps") else 60,
            fastest_lap=float(lap_times.min()) if len(lap_times) > 0 else 90.0,
            avg_lap_time=float(avg_lap) if not pd.isna(avg_lap) else 90.0,
            sector1_avg=float(s1_times.mean()) if len(s1_times) > 0 else 30.0,
            sector2_avg=float(s2_times.mean()) if len(s2_times) > 0 else 30.0,
            sector3_avg=float(s3_times.mean()) if len(s3_times) > 0 else 30.0,
            pit_lane_time=20.0,  # Standard assumption
            safety_car_rate=float(sc_rate),
            drs_zones=drs_zones,
        )

        self._track_stats[stats.track_id] = stats
        return stats

    def create_drivers_from_stats(
        self,
        stats: dict[str, DriverStats] | None = None,
    ) -> list[Driver]:
        """Create Driver models from historical statistics.

        Args:
            stats: Driver statistics (uses cached if None)

        Returns:
            List of Driver models
        """
        if stats is None:
            stats = self._driver_stats

        if not stats:
            raise ValueError("No driver stats available. Run get_driver_lap_stats first.")

        # Normalize stats relative to field
        # In F1, the spread from fastest to slowest is typically ~3% of lap time
        avg_times = [s.avg_lap_time for s in stats.values()]
        fastest_avg = min(avg_times)

        # Use expected range of ~3% for skill mapping (maps to 0.85-1.0 skill range)
        expected_range = fastest_avg * 0.04  # 4% covers full field with margin

        std_devs = [s.lap_time_std for s in stats.values()]
        min_std = min(std_devs)
        max_std = max(std_devs)
        std_range = max_std - min_std if max_std > min_std else 0.1

        drivers = []
        for driver_stats in stats.values():
            # Skill: faster average = higher skill
            # Map to 0.75-1.0 range for meaningful simulation differences
            pace_delta = driver_stats.avg_lap_time - fastest_avg
            pace_pct = pace_delta / expected_range  # 0 = fastest, 1 = 4% slower
            skill = 1.0 - (pace_pct * 0.25)  # Maps to 0.75-1.0 range

            # Consistency: lower std dev = higher consistency
            std_delta = driver_stats.lap_time_std - min_std
            consistency = 1.0 - (std_delta / std_range) * 0.15 if std_range > 0 else 0.9

            drivers.append(Driver(
                id=driver_stats.driver_id,
                name=driver_stats.driver_name,
                team_id=driver_stats.team_id,
                skill_rating=max(0.75, min(1.0, skill)),
                consistency=max(0.85, min(1.0, consistency)),
                wet_skill_modifier=np.random.uniform(0.95, 1.05),
                overtaking_skill=max(0.75, min(1.0, skill)),
                tire_management=consistency,
            ))

        return drivers

    def create_cars_from_stats(
        self,
        stats: dict[str, DriverStats] | None = None,
    ) -> dict[str, Car]:
        """Create Car models from historical statistics.

        Args:
            stats: Driver statistics (uses cached if None)

        Returns:
            Dictionary of team_id -> Car model
        """
        if stats is None:
            stats = self._driver_stats

        if not stats:
            raise ValueError("No driver stats available. Run get_driver_lap_stats first.")

        # Group by team and calculate team averages
        team_data: dict[str, list[DriverStats]] = {}
        for driver_stats in stats.values():
            team_id = driver_stats.team_id
            if team_id not in team_data:
                team_data[team_id] = []
            team_data[team_id].append(driver_stats)

        # Calculate team pace
        team_paces: dict[str, float] = {}
        for team_id, drivers in team_data.items():
            # Use best driver's pace as team pace
            team_paces[team_id] = min(d.avg_lap_time for d in drivers)

        fastest_team = min(team_paces.values())
        # Use expected range of ~3% for pace mapping (realistic F1 car spread)
        expected_range = fastest_team * 0.04  # 4% covers full field

        cars = {}
        for team_id, drivers in team_data.items():
            pace_delta = team_paces[team_id] - fastest_team
            pace_pct = pace_delta / expected_range  # 0 = fastest, 1 = 4% slower
            # Map to 0.75-1.0 range for meaningful simulation differences
            base_pace = 1.0 - (pace_pct * 0.25)

            # Average pit stop stats from team drivers
            pit_avg = np.mean([d.pit_stop_avg for d in drivers])
            pit_std = np.mean([d.pit_stop_std for d in drivers])
            dnf_rate = np.mean([d.dnf_rate for d in drivers])

            cars[team_id] = Car(
                team_id=team_id,
                team_name=drivers[0].team_name,
                base_pace=max(0.75, min(1.0, base_pace)),
                downforce_level=0.85,
                straight_line_speed=0.85,
                reliability=max(0.90, 1.0 - dnf_rate * 2),
                tire_degradation_factor=1.0,
                wet_performance=0.85,
                pit_stop_avg=float(pit_avg),
                pit_stop_std=float(pit_std),
            )

        return cars

    # Track-specific characteristics (overtake difficulty: 0=easy, 1=impossible)
    TRACK_OVERTAKE_DIFFICULTY = {
        "monaco": 0.95,           # Street circuit, nearly impossible to pass
        "singapore": 0.85,        # Street circuit, very hard
        "hungary": 0.80,          # Tight and twisty
        "barcelona": 0.75,        # Dirty air issues
        "melbourne": 0.70,        # Street circuit but with DRS
        "imola": 0.65,            # Traditional, limited opportunities
        "zandvoort": 0.70,        # Narrow, hard to pass
        "miami": 0.50,            # Modern circuit with DRS zones
        "jeddah": 0.45,           # Fast street circuit, some opportunities
        "bahrain": 0.35,          # Multiple overtaking zones
        "spa": 0.30,              # Kemmel straight great for passing
        "monza": 0.25,            # Low downforce, long straights
        "baku": 0.35,             # Long straight despite street circuit
        "china": 0.40,            # Good passing opportunities
        "canada": 0.40,           # Long straights
        "austria": 0.40,          # Short lap but good DRS zones
        "silverstone": 0.50,      # High speed, some opportunities
        "cota": 0.45,             # Long straight to turn 1
        "mexico": 0.40,           # Long straight, thin air helps
        "brazil": 0.40,           # Good overtaking track
        "qatar": 0.55,            # Limited opportunities
        "las_vegas": 0.40,        # Long straights
        "abu_dhabi": 0.45,        # Modern, decent DRS
        "japan": 0.60,            # Figure 8, tricky
        "saudi_arabia": 0.45,     # Same as jeddah
    }

    def create_track_from_stats(
        self,
        stats: TrackStats | None = None,
        track_id: str | None = None,
    ) -> Track:
        """Create Track model from historical statistics.

        Args:
            stats: Track statistics (uses cached if None)
            track_id: Track ID to look up from cache

        Returns:
            Track model
        """
        if stats is None:
            if track_id and track_id in self._track_stats:
                stats = self._track_stats[track_id]
            else:
                raise ValueError("No track stats provided and none cached")

        # Look up track-specific overtake difficulty
        track_key = stats.track_id.lower().replace(" ", "_").replace("grand_prix", "").strip("_")
        # Try to match against known tracks
        overtake_diff = 0.5  # Default
        for known_track, difficulty in self.TRACK_OVERTAKE_DIFFICULTY.items():
            if known_track in track_key or track_key in known_track:
                overtake_diff = difficulty
                break

        return Track(
            id=stats.track_id,
            name=stats.track_name,
            country=stats.country,
            total_laps=stats.total_laps,
            base_lap_time=stats.avg_lap_time,
            pit_lane_delta=stats.pit_lane_time,
            sectors=[
                Sector(number=1, base_time=stats.sector1_avg),
                Sector(number=2, base_time=stats.sector2_avg),
                Sector(number=3, base_time=stats.sector3_avg),
            ],
            overtake_difficulty=overtake_diff,
            tire_stress=0.5,
            safety_car_probability=stats.safety_car_rate,
        )
