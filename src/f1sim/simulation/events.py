"""Race events: safety car, failures, incidents."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from f1sim.models import Car, Driver, Track, Weather

# Broad model priors, not precise estimates from the small observed race sample.
# Background interruptions cover unmodelled major crashes/track blockages.
BACKGROUND_RED_FLAG_RACE_PROBABILITY = 0.10
SEVERE_WEATHER_RED_FLAG_EPISODE_PROBABILITY = 0.50
# The existing incident prior describes a full current-era grid, not one car.
INCIDENT_REFERENCE_FIELD_SIZE = 22


class EventType(str, Enum):
    """Types of race events."""

    SAFETY_CAR = "safety_car"
    VIRTUAL_SAFETY_CAR = "virtual_safety_car"
    RED_FLAG = "red_flag"
    MECHANICAL_FAILURE = "mechanical_failure"
    COLLISION = "collision"
    SPIN = "spin"
    PUNCTURE = "puncture"


@dataclass
class RaceEvent:
    """Represents a race event."""

    event_type: EventType
    lap: int
    drivers_involved: list[str] = field(default_factory=list)
    duration_laps: int = 0
    description: str = ""
    # Consequences are carried with the event so the race engine can apply
    # them to its per-driver state after event detection.  Existing callers
    # remain compatible because both fields are optional.
    time_loss_seconds: float = 0.0
    forces_pit_stop: bool = False
    # Reporting only: these asymmetric losses were already applied by the
    # battle resolver and must not be charged again as pending consequences.
    applied_time_losses: dict[str, float] = field(default_factory=dict)


class EventManager:
    """Manages race events and their effects."""

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize the event manager.

        Args:
            rng: Random number generator
        """
        self.rng = rng if rng is not None else np.random.default_rng()
        self.events: list[RaceEvent] = []
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.vsc_active = False
        self.vsc_laps_remaining = 0
        self.sc_just_ended = False  # Flag for restart lap
        self.sc_restart_lap = False  # True on the lap after SC ends
        self.sc_restart_lap_number: int | None = None
        self.current_lap: int | None = None
        self._lap_overtake_mode_snapshot: bool | None = None
        self._lap_started_neutralized = False
        # Red flag state
        self.red_flag_active = False
        self.red_flag_just_ended = False  # Flag for restart lap after red flag
        self.red_flag_restart_lap = False  # True on the lap after red flag ends
        self.red_flag_restart_lap_number: int | None = None
        self._severe_weather_episode_decided = False
        # Manual trigger configuration
        self.forced_red_flag_laps: set[int] = set()  # Laps to force red flags
        self.forced_safety_car_laps: set[int] = set()  # Laps to force safety cars

        # Event counters for calibration during a race
        self.safety_car_deployments = 0
        self.vsc_deployments = 0
        self.red_flag_deployments = 0

    def reset(self) -> None:
        """Reset event state for new race."""
        self.events = []
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.vsc_active = False
        self.vsc_laps_remaining = 0
        self.sc_just_ended = False
        self.sc_restart_lap = False
        self.sc_restart_lap_number = None
        self.current_lap = None
        self._lap_overtake_mode_snapshot = None
        self._lap_started_neutralized = False
        self.red_flag_active = False
        self.red_flag_just_ended = False
        self.red_flag_restart_lap = False
        self.red_flag_restart_lap_number: int | None = None
        self._severe_weather_episode_decided = False

        self.safety_car_deployments = 0
        self.vsc_deployments = 0
        self.red_flag_deployments = 0
        # Note: forced laps are NOT reset - they persist across races

    def set_forced_red_flag(self, laps: list[int] | int) -> None:
        """Configure laps where red flags will be forced.

        Args:
            laps: Single lap number or list of lap numbers to force red flags
        """
        if isinstance(laps, int):
            self.forced_red_flag_laps.add(laps)
        else:
            self.forced_red_flag_laps.update(laps)

    def set_forced_safety_car(self, laps: list[int] | int) -> None:
        """Configure laps where safety cars will be forced.

        Args:
            laps: Single lap number or list of lap numbers to force safety cars
        """
        if isinstance(laps, int):
            self.forced_safety_car_laps.add(laps)
        else:
            self.forced_safety_car_laps.update(laps)

    def clear_forced_events(self) -> None:
        """Clear all forced event configurations."""
        self.forced_red_flag_laps.clear()
        self.forced_safety_car_laps.clear()

    def process_lap(
        self,
        lap: int,
        drivers: list[Driver],
        cars: dict[str, Car],
        track: Track,
        weather: Weather,
        incidents_this_lap: int = 0,
    ) -> list[RaceEvent]:
        """Process events for a lap.

        Args:
            lap: Current lap number
            drivers: Field to sample, or an empty list for race-control-only updates
            cars: Dictionary of cars by team_id
            track: Current track
            weather: Current weather
            incidents_this_lap: Number of overtake incidents

        Returns:
            List of events that occurred
        """
        self.current_lap = lap
        self._update_weather_episode(weather)
        # A countdown expiring during this call does not make the already
        # completed lap a green-flag lap.  Keep this snapshot separate from
        # the live state, which is updated below for the following lap.
        self._lap_started_neutralized = self.safety_car_active or self.vsc_active
        # Capture control state before any SC/VSC/red-flag transition for the
        # lap.  Consumers querying this lap after process_lap gets the same
        # answer as consumers that queried it before the transition.
        self._lap_overtake_mode_snapshot = None
        self._lap_overtake_mode_snapshot = self.is_overtake_mode_allowed(
            lap,
            weather,
        )
        lap_events: list[RaceEvent] = []

        # Track restart lap (lap after SC ended)
        self.sc_restart_lap = self.sc_just_ended
        self.sc_just_ended = False

        # Track red flag restart lap
        self.red_flag_restart_lap = self.red_flag_just_ended
        self.red_flag_just_ended = False

        # If red flag is active, race is suspended - no events processed
        # Red flag ending is handled externally by RaceSimulator
        if self.red_flag_active:
            return lap_events

        # Update active safety car/VSC
        if self.safety_car_active:
            self.safety_car_laps_remaining -= 1
            if self.safety_car_laps_remaining <= 0:
                self.safety_car_active = False
                self.sc_just_ended = True  # Next lap is restart
                self.sc_restart_lap_number = lap + 1

        if self.vsc_active:
            self.vsc_laps_remaining -= 1
            if self.vsc_laps_remaining <= 0:
                self.vsc_active = False

        # Check for mechanical failures
        for driver in drivers:
            if driver.dnf:
                continue

            car = cars.get(driver.team_id)
            if car is None:
                continue

            failure = self._check_mechanical_failure(driver, car, track, lap, weather)
            if failure:
                lap_events.append(failure)

        # Check for random incidents (spins, etc.)
        if (
            not self._lap_started_neutralized
            and not self.safety_car_active
            and not self.vsc_active
            and any(not driver.dnf for driver in drivers)
        ):
            random_incident = self._check_random_incident(drivers, track, weather, lap)
            if random_incident:
                lap_events.append(random_incident)
                incidents_this_lap += 1

        # A supplied field losing its last survivor ends the race. Keep the
        # retirement incidents, but do not deploy fresh control for an empty
        # track. An empty input is the chronological engine's intentional
        # control-only call; that caller owns its individual retirement checks.
        if drivers and not any(not driver.dnf for driver in drivers):
            self.events.extend(lap_events)
            return lap_events

        # Check for forced red flag
        if (
            lap in self.forced_red_flag_laps
            and not self.red_flag_active
        ):
            red_flag_event = self.deploy_red_flag(lap, "Manual trigger")
            lap_events.append(red_flag_event)
            self.events.extend(lap_events)
            return lap_events

        # Neutralization suppresses fresh incident/SC hazards, but a newly
        # severe storm can still require suspension. Include countdown expiry:
        # the lap that just ran was neutralized, so it must not redraw an SC.
        if self._lap_started_neutralized:
            red_flag_event = self._check_severe_weather_red_flag(lap, weather)
            if red_flag_event:
                lap_events.append(red_flag_event)

        # Check for forced safety car
        if (
            lap in self.forced_safety_car_laps
            and not self._lap_started_neutralized
            and not self.safety_car_active
            and not self.vsc_active
            and not self.red_flag_active
        ):
            self.safety_car_active = True
            self.safety_car_laps_remaining = self.rng.integers(3, 7)
            self.safety_car_deployments += 1
            sc_event = RaceEvent(
                event_type=EventType.SAFETY_CAR,
                lap=lap,
                duration_laps=self.safety_car_laps_remaining,
                description="Safety car deployed (manual trigger)",
            )
            lap_events.append(sc_event)

        # Convert the track's race-level event probability into a lap hazard.
        # Calling this once per lap also allows an unmodelled marshal/debris
        # event to produce a safety intervention; incident context still
        # increases the hazard when one is already known this lap.
        if (
            not self._lap_started_neutralized
            and not self.safety_car_active
            and not self.vsc_active
            and not self.red_flag_active
        ):
            sc_event = self._deploy_safety_measure(lap, incidents_this_lap, track, weather)
            if sc_event:
                lap_events.append(sc_event)

        self.events.extend(lap_events)
        return lap_events

    def _mechanical_failure_probability(
        self,
        car: Car,
        track: Track,
        weather: Weather | None,
        lap: int,
    ) -> float:
        """Estimate per-lap mechanical failure probability.

        Scales with:
        - car reliability baseline
        - race progression (late-race attrition)
        - tire/track stress proxy
        - high track temperatures
        """
        # Combine team-level reliability with weakest component pressure.
        components = car.component_reliability_map()
        avg_component_rel = float(np.mean(list(components.values())))
        weakest_component_rel = min(components.values())
        effective_reliability = (
            car.reliability * 0.55
            + avg_component_rel * 0.3
            + weakest_component_rel * 0.15
        )

        if effective_reliability >= 1.0:
            return 0.0
        if effective_reliability <= 0.0:
            return 1.0

        # Cars are more likely to fail late in races.
        distance = max(track.total_laps, 1)
        race_progress = max(lap, 1) / distance
        progression_modifier = 0.85 + 0.35 * race_progress
        # Normalize the scheduled progression weights analytically. With
        # neutral stress/temperature, multiplying all lap survivals recovers
        # the combined reliability exactly, including low reliability values.
        progression_total = 0.85 * distance + 0.35 * (distance + 1) / 2

        # Tire-stress tracks tend to be harder on components.
        stress_modifier = 0.9 + 0.25 * track.tire_stress

        # Heat stress from very hot tracks (worse with weak cooling reliability).
        temp_modifier = 1.0
        if weather is not None and weather.track_temperature >= 45.0:
            cooling_penalty = 1.0 + (1.0 - car.cooling_reliability) * 0.8
            temp_modifier += min((weather.track_temperature - 45.0) * 0.01, 0.25) * cooling_penalty

        hazard_share = progression_modifier / progression_total
        return float(-np.expm1(np.log(effective_reliability) * hazard_share
                              * stress_modifier * temp_modifier))

    def _check_mechanical_failure(
        self,
        driver: Driver,
        car: Car,
        track: Track,
        lap: int,
        weather: Weather | None = None,
    ) -> RaceEvent | None:
        """Check if a car has mechanical failure.

        Returns event if failure occurred.
        """
        failure_prob = self._mechanical_failure_probability(car, track, weather, lap)

        if self.rng.random() < failure_prob:
            components = car.component_reliability_map()
            component_keys = list(components.keys())
            # Lower reliability component => higher failure chance.
            raw_weights = np.array([max(1e-6, 1.0 - components[k]) for k in component_keys])
            probs = raw_weights / raw_weights.sum()
            failing_component = str(self.rng.choice(component_keys, p=probs))

            failure_labels = {
                "engine": "engine failure",
                "gearbox": "gearbox failure",
                "brakes": "brake failure",
                "electrical": "electrical failure",
                "cooling": "cooling failure",
            }
            failure = failure_labels[failing_component]

            driver.dnf = True
            driver.dnf_reason = failure

            return RaceEvent(
                event_type=EventType.MECHANICAL_FAILURE,
                lap=lap,
                drivers_involved=[driver.id],
                description=f"{driver.name} retired with {failure}",
            )

        return None

    def _incident_probability(
        self,
        drivers: list[Driver],
        track: Track,
        weather: Weather,
    ) -> float:
        """Estimate the remaining field's per-lap random-incident probability."""
        active_drivers = [d for d in drivers if not d.dnf]
        if not active_drivers:
            return 0.0

        # Derive lap-level incident risk from the same race-level safety-car
        # signal used for deployment.  Incidents are still a little more
        # frequent than full SC deployments, hence the calibrated 0.6 factor.
        base_prob = self._race_probability_to_lap_hazard(
            track.safety_car_probability,
            track.total_laps,
        ) / 0.6

        # Hard-to-pass tracks create compression and mistakes.
        base_prob *= 0.85 + track.overtake_difficulty * 0.7

        # Inconsistency in the field increases incidents.
        avg_consistency = float(np.mean([d.consistency for d in active_drivers]))
        consistency_modifier = 1.0 + (1.0 - avg_consistency) * 0.8
        base_prob *= consistency_modifier

        # Weather significantly increases risk.
        if weather.is_wet():
            base_prob *= 1.8
        if weather.requires_wet_tires():
            base_prob *= 1.6

        # Preserve the calibrated full-grid prior, then scale its survival
        # probability by active-car exposure. Retirements reduce field risk
        # rather than concentrating the whole prior on the last few cars.
        # A lone car can still spin, puncture a tyre, or hit a barrier.
        full_field_probability = float(np.clip(base_prob, 0.0005, 0.08))
        exposure = len(active_drivers) / INCIDENT_REFERENCE_FIELD_SIZE
        return min(0.08, float(-np.expm1(np.log1p(-full_field_probability) * exposure)))

    @staticmethod
    def _race_probability_to_lap_hazard(
        race_probability: float,
        total_laps: int,
    ) -> float:
        """Convert a probability of at least one event in a race to a hazard.

        For an independent, constant per-lap hazard ``h``,
        ``1 - (1 - h) ** total_laps`` is the probability of seeing an event
        during the race.  Using the inverse keeps low-risk circuits low risk
        instead of imposing an arbitrary per-lap floor.
        """
        race_probability = float(np.clip(race_probability, 0.0, 1.0))
        laps = max(int(total_laps), 1)
        if race_probability <= 0.0:
            return 0.0
        if race_probability >= 1.0:
            return 1.0
        return float(-np.expm1(np.log1p(-race_probability) / laps))

    def _incident_subset_probability(
        self,
        drivers: list[Driver],
        exposure_drivers: list[Driver],
        track: Track,
        weather: Weather,
    ) -> float:
        """Allocate the active field's hazard to a selected active subset.

        Retired entries are ignored. Active IDs must be unique in each list,
        and every selected ID must exist in the exposure field. Field ratings
        define the risk weights; this calculation neither samples nor mutates.
        """
        selected = [driver for driver in drivers if not driver.dnf]
        field = [driver for driver in exposure_drivers if not driver.dnf]
        selected_ids = {driver.id for driver in selected}
        field_ids = {driver.id for driver in field}
        if len(selected_ids) != len(selected) or len(field_ids) != len(field):
            raise ValueError("Incident exposure requires unique active driver IDs")
        if not selected_ids <= field_ids:
            raise ValueError("Incident drivers must belong to the active exposure field")
        if not selected:
            return 0.0
        probability = self._incident_probability(field, track, weather)
        if selected_ids == field_ids:
            return probability
        weights = self._incident_driver_weights(field, weather)
        share = float(sum(weight for driver, weight in zip(field, weights)
                          if driver.id in selected_ids))
        return float(-np.expm1(np.log1p(-probability) * share))

    def _check_random_incident(
        self,
        drivers: list[Driver],
        track: Track,
        weather: Weather,
        lap: int,
        *,
        exposure_drivers: list[Driver] | None = None,
    ) -> RaceEvent | None:
        """Check incidents, optionally allocating a full field's subset hazard."""
        active_drivers = [d for d in drivers if not d.dnf]
        incident_prob = (
            self._incident_subset_probability(drivers, exposure_drivers, track, weather)
            if exposure_drivers is not None else None
        )
        if not active_drivers:
            return None

        if incident_prob is None:
            incident_prob = self._incident_probability(active_drivers, track, weather)

        if self.rng.random() < incident_prob:
            # Drivers with lower consistency are more exposed to spins and
            # contact.  In wet conditions, lower wet skill adds a second risk
            # channel.  The weights are intentionally modest so a single
            # rating cannot make a driver deterministic.
            weights = self._incident_driver_weights(active_drivers, weather)
            driver = self.rng.choice(active_drivers, p=weights)

            # Determine incident severity
            severity_roll = self.rng.random()

            if severity_roll < 0.3:
                # Spin, continues
                return RaceEvent(
                    event_type=EventType.SPIN,
                    lap=lap,
                    drivers_involved=[driver.id],
                    description=f"{driver.name} spun but continues",
                    time_loss_seconds=float(self.rng.uniform(2.0, 6.0)),
                )
            elif severity_roll < 0.6:
                # Puncture
                return RaceEvent(
                    event_type=EventType.PUNCTURE,
                    lap=lap,
                    drivers_involved=[driver.id],
                    description=f"{driver.name} suffered a puncture",
                    # The driver loses time limping to the pits, then must
                    # take a complete stop on the following lap.
                    time_loss_seconds=float(self.rng.uniform(6.0, 14.0)),
                    forces_pit_stop=True,
                )
            else:
                # Crash, DNF
                driver.dnf = True
                driver.dnf_reason = "crash"
                return RaceEvent(
                    event_type=EventType.COLLISION,
                    lap=lap,
                    drivers_involved=[driver.id],
                    description=f"{driver.name} crashed and retired",
                    time_loss_seconds=float(self.rng.uniform(8.0, 18.0)),
                )

        return None

    @staticmethod
    def _incident_driver_weights(
        drivers: list[Driver],
        weather: Weather,
    ) -> np.ndarray:
        """Return deterministic, normalized incident-victim risk weights."""
        if not drivers:
            return np.asarray([], dtype=float)

        weights: list[float] = []
        for driver in drivers:
            # Keep a non-zero baseline: even a very consistent driver can be
            # caught by debris or another car.
            consistency_risk = 0.45 + 1.55 * (1.0 - float(np.clip(driver.consistency, 0.0, 1.0)))
            if weather.is_wet() or weather.rain_intensity > 0.0:
                wet_skill_risk = float(
                    np.clip(1.5 - driver.wet_skill_modifier, 0.0, 1.0)
                )
                consistency_risk *= 1.0 + 0.9 * wet_skill_risk
            weights.append(consistency_risk)

        normalized = np.asarray(weights, dtype=float)
        total = float(normalized.sum())
        if total <= 0.0:
            return np.full(len(drivers), 1.0 / len(drivers), dtype=float)
        return normalized / total

    def _calibrated_safety_probs(
        self,
        track: Track,
        weather: Weather | None,
        incidents: int,
        lap: int,
    ) -> tuple[float, float]:
        """Calibrate SC/VSC probabilities from track+weather+incident context."""
        # ``Track.safety_car_probability`` is specifically the chance of at
        # least one full safety car across a race.  VSC is modelled as a
        # separate, lower-risk process; it is not subtracted from the SC
        # probability.  Both are converted with the inverse cumulative hazard
        # equation, with no arbitrary per-lap floor.
        sc_pressure = self._race_probability_to_lap_hazard(
            track.safety_car_probability,
            track.total_laps,
        )
        vsc_race_probability = float(np.clip(track.safety_car_probability * 0.5, 0.0, 1.0))
        vsc_pressure = self._race_probability_to_lap_hazard(
            vsc_race_probability,
            track.total_laps,
        )

        race_progress = float(np.clip(lap / max(track.total_laps, 1), 0.0, 1.0))
        # Centred at one over a dry, incident-free race so the full-SC
        # probability remains statistically calibrated across all laps.
        progress_modifier = 0.8 + 0.4 * race_progress

        weather_modifier = 1.0
        if weather is not None:
            if weather.is_wet():
                weather_modifier *= 1.4
            if weather.requires_wet_tires():
                weather_modifier *= 1.35

        incident_modifier = 1.0 + incidents * 0.5

        context_modifier = progress_modifier * weather_modifier * incident_modifier
        sc_prob = float(np.clip(sc_pressure * context_modifier, 0.0, 0.95))
        vsc_prob = float(np.clip(vsc_pressure * context_modifier, 0.0, 0.95))

        return sc_prob, vsc_prob

    def _deploy_safety_measure(
        self,
        lap: int,
        incidents: int,
        track: Track,
        weather: Weather | None = None,
    ) -> RaceEvent | None:
        """Deploy safety car, VSC, or red flag based on incidents and conditions."""
        # Check for red flag conditions first
        # Red flags are rare but occur for major incidents or dangerous weather
        red_flag_event = self._check_red_flag_conditions(
            lap, incidents, weather, total_laps=track.total_laps
        )
        if red_flag_event:
            return red_flag_event

        sc_prob, vsc_prob = self._calibrated_safety_probs(track, weather, incidents, lap)
        roll = self.rng.random()

        if roll < sc_prob:
            # Full safety car
            self.safety_car_active = True
            self.safety_car_laps_remaining = self.rng.integers(3, 7)
            self.safety_car_deployments += 1

            return RaceEvent(
                event_type=EventType.SAFETY_CAR,
                lap=lap,
                duration_laps=self.safety_car_laps_remaining,
                description="Safety car deployed",
            )

        if roll < sc_prob + (1.0 - sc_prob) * vsc_prob:
            # Virtual safety car
            self.vsc_active = True
            self.vsc_laps_remaining = self.rng.integers(2, 4)
            self.vsc_deployments += 1

            return RaceEvent(
                event_type=EventType.VIRTUAL_SAFETY_CAR,
                lap=lap,
                duration_laps=self.vsc_laps_remaining,
                description="Virtual safety car deployed",
            )

        return None

    def _check_red_flag_conditions(
        self,
        lap: int,
        incidents: int,
        weather: Weather | None = None,
        *,
        total_laps: int,
    ) -> RaceEvent | None:
        """Sample background interruptions and one decision per severe storm.

        ``incidents`` includes minor overtake contact, not major wrecks, so it
        does not increase the red-flag prior. Separate major incidents remain
        possible through the background hazard even within a decided storm.
        Suspension duration is abstracted by the race engine; a weather flag
        does not imply that the next simulated lap has physically dried out.
        """
        self._update_weather_episode(weather)
        red_flag_event = self._check_severe_weather_red_flag(lap, weather)
        if red_flag_event:
            return red_flag_event

        probability = self._race_probability_to_lap_hazard(
            BACKGROUND_RED_FLAG_RACE_PROBABILITY, total_laps
        )
        if self.rng.random() < probability:
            return self.deploy_red_flag(lap, "Major incident or track obstruction")

        return None

    def _check_severe_weather_red_flag(
        self, lap: int, weather: Weather | None,
    ) -> RaceEvent | None:
        """Decide once per severe episode, including laps under SC or VSC."""
        if weather is not None:
            severe = weather.track_wetness >= 0.95 or (
                weather.track_wetness >= 0.8 and weather.rain_intensity >= 0.8
            )
            if severe and not self._severe_weather_episode_decided:
                self._severe_weather_episode_decided = True
                if self.rng.random() < SEVERE_WEATHER_RED_FLAG_EPISODE_PROBABILITY:
                    return self.deploy_red_flag(lap, "Severe weather")

        return None

    def _update_weather_episode(self, weather: Weather | None) -> None:
        """Require a clear improvement before a new storm can be sampled."""
        if (
            weather is not None
            and weather.rain_intensity < 0.65
            and weather.track_wetness < 0.8
        ):
            self._severe_weather_episode_decided = False

    def deploy_red_flag(self, lap: int, reason: str = "Incident") -> RaceEvent:
        """Deploy a red flag, stopping the race.

        Args:
            lap: Current lap number
            reason: Description of why red flag was deployed

        Returns:
            RaceEvent for the red flag
        """
        self.red_flag_active = True
        self.red_flag_deployments += 1
        # Clear any active SC/VSC
        self.safety_car_active = False
        self.safety_car_laps_remaining = 0
        self.vsc_active = False
        self.vsc_laps_remaining = 0
        self.sc_just_ended = False
        self.sc_restart_lap = False
        self.sc_restart_lap_number = None
        self.red_flag_just_ended = False
        self.red_flag_restart_lap = False
        self.red_flag_restart_lap_number = None

        return RaceEvent(
            event_type=EventType.RED_FLAG,
            lap=lap,
            duration_laps=0,  # Duration determined by race director
            description=f"Red flag: {reason}",
        )

    def end_red_flag(self) -> None:
        """End the red flag period and prepare for restart."""
        self.red_flag_active = False
        self.red_flag_just_ended = True
        if self.current_lap is not None:
            self.red_flag_restart_lap_number = self.current_lap + 1

    def is_red_flag_active(self) -> bool:
        """Check if red flag is currently active."""
        return self.red_flag_active

    def is_red_flag_restart_lap(self) -> bool:
        """Check if this is the restart lap after a red flag."""
        return self.red_flag_restart_lap

    def get_lap_time_modifier(self) -> float:
        """Get lap time modifier based on current safety status.

        Returns:
            Multiplier for lap times (>1 = slower)
        """
        if self.red_flag_active:
            return 0.0  # Race is stopped during red flag
        elif self.safety_car_active:
            return 1.4  # 40% slower under SC
        elif self.vsc_active:
            return 1.2  # 20% slower under VSC
        return 1.0

    def is_pit_window_open(self) -> bool:
        """Check if it's a good time to pit (under SC/VSC/red flag)."""
        return self.safety_car_active or self.vsc_active or self.red_flag_active

    def is_restart_lap(self, lap: int | None = None) -> bool:
        """Check if this is a restart lap after SC or red flag."""
        target_lap = lap if lap is not None else self.current_lap
        if target_lap is not None:
            if target_lap in {
                self.sc_restart_lap_number,
                self.red_flag_restart_lap_number,
            }:
                return True
            return (
                self.sc_restart_lap_number is None
                and (self.sc_just_ended or self.sc_restart_lap)
            ) or (
                self.red_flag_restart_lap_number is None
                and (self.red_flag_just_ended or self.red_flag_restart_lap)
            )
        return (
            self.sc_restart_lap
            or self.red_flag_restart_lap
            or self.sc_just_ended
            or self.red_flag_just_ended
        )

    def is_active_aero_allowed(self) -> bool:
        """Return whether green-flag straight mode is currently available.

        Active Aero is a car mode rather than a following aid: every car may
        use it on the track's configured sections whenever the race is green.
        There is consequently no detection-gap check here.
        """
        return not (
            self.safety_car_active
            or self.vsc_active
            or self.red_flag_active
        )

    def is_overtake_mode_allowed(
        self,
        lap: int | None = None,
        weather: Weather | None = None,
    ) -> bool:
        """Return whether Overtake Mode may be deployed this lap.

        Overtake Mode is a distinct, detection-gap-gated deployment.  The
        first useful detection opportunity is approximated as lap two, and
        deployment remains unavailable through safety-car/VSC/red-flag
        running and the first lap after a restart.  Wet or otherwise low-grip
        conditions also disable the mode.
        """
        effective_lap = lap if lap is not None else self.current_lap
        if (
            effective_lap is not None
            and self._lap_overtake_mode_snapshot is not None
            and effective_lap == self.current_lap
        ):
            return self._lap_overtake_mode_snapshot
        if self.safety_car_active or self.vsc_active or self.red_flag_active:
            return False
        if effective_lap is not None:
            if effective_lap in {
                self.sc_restart_lap_number,
                self.red_flag_restart_lap_number,
            }:
                return False
            # Preserve compatibility for callers that set the legacy boolean
            # restart flags directly without a lap-number timeline.
            if (
                self.sc_restart_lap_number is None
                and (self.sc_just_ended or self.sc_restart_lap)
            ) or (
                self.red_flag_restart_lap_number is None
                and (self.red_flag_just_ended or self.red_flag_restart_lap)
            ):
                return False
        elif (
            self.sc_just_ended
            or self.sc_restart_lap
            or self.red_flag_just_ended
            or self.red_flag_restart_lap
        ):
            return False
        if effective_lap is not None and effective_lap < 2:
            return False
        if weather is not None and (
            weather.is_wet()
            or weather.track_wetness > 0.2
            or weather.rain_intensity > 0.25
        ):
            return False
        return True

    def bunch_field(self, driver_states: list) -> None:
        """Bunch up the field behind safety car.

        Sets all gaps between cars to ~1 second, simulating
        the field catching the safety car and forming a queue.

        Args:
            driver_states: List of DriverRaceState objects
        """
        # Sort by position
        racing = [s for s in driver_states if s.status.value == "racing"]
        racing.sort(key=lambda s: s.position)

        if len(racing) < 2:
            return

        # Leader stays unchanged; each following car is set to ~0.8-1.2s
        # behind.  RaceSimulator classifies positions by elapsed time before
        # calling this method, so a same-lap incident's position loss survives
        # the gap reset even though ordinary clean gaps are closed here.
        for i, state in enumerate(racing[1:], 1):
            gap_to_ahead = self.rng.uniform(0.8, 1.2)
            state.total_time = racing[i - 1].total_time + gap_to_ahead
