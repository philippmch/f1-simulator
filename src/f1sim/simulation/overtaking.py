"""Overtaking probability model."""

import numpy as np

from f1sim.models import Car, Driver, Track


class OvertakingModel:
    """Models overtaking attempts and success probability."""

    def __init__(self, rng: np.random.Generator | None = None):
        """Initialize the overtaking model.

        Args:
            rng: Random number generator
        """
        self.rng = rng if rng is not None else np.random.default_rng()

    def attempt_overtake(
        self,
        attacker: Driver,
        attacker_car: Car,
        defender: Driver,
        defender_car: Car,
        track: Track,
        gap: float,
        has_drs: bool = False,
        is_wet: bool = False,
        restart_boost: bool = False,
    ) -> tuple[bool, bool]:
        """Attempt an overtake maneuver.

        Args:
            attacker: Driver attempting to pass
            attacker_car: Attacker's car
            defender: Driver being passed
            defender_car: Defender's car
            track: Current track
            gap: Time gap between cars in seconds
            has_drs: Whether attacker has DRS
            is_wet: Whether track is wet
            restart_boost: Whether this is a SC restart lap (increased aggression)

        Returns:
            Tuple of (overtake_successful, incident_occurred)
        """
        # Check if overtake is possible (wider window on restarts)
        max_gap = 2.0 if restart_boost else 1.5
        if gap > max_gap:
            return False, False  # Too far behind to attempt

        # Calculate success probability
        prob = self._calculate_probability(
            attacker, attacker_car,
            defender, defender_car,
            track, gap, has_drs, is_wet,
            restart_boost=restart_boost,
        )

        # Attempt the overtake
        roll = self.rng.random()

        if roll < prob:
            # Successful overtake
            return True, False
        elif roll < prob + self._incident_probability(gap, track.overtake_difficulty):
            # Incident during attempt
            return False, True
        else:
            # Failed attempt, no incident
            return False, False

    def _calculate_probability(
        self,
        attacker: Driver,
        attacker_car: Car,
        defender: Driver,
        defender_car: Car,
        track: Track,
        gap: float,
        has_drs: bool,
        is_wet: bool,
        restart_boost: bool = False,
    ) -> float:
        """Calculate overtake success probability.

        Returns probability between 0 and 1.
        """
        # Base probability from track difficulty
        # On restarts, track difficulty matters less (everyone bunched, cold tires)
        effective_difficulty = track.overtake_difficulty * 0.6 if restart_boost else track.overtake_difficulty
        base_prob = 0.5 * (1.0 - effective_difficulty)

        # Pace advantage factor
        pace_delta = attacker_car.base_pace - defender_car.base_pace
        pace_factor = self._sigmoid(pace_delta * 3, offset=0)  # ~0-1 based on pace diff

        # Gap factor (closer = higher chance)
        gap_factor = max(0, 1.0 - gap / 1.5)

        # DRS bonus - scaled by track overtake opportunity (less effective at tight tracks)
        if has_drs and not is_wet:
            # At Monaco (difficulty 0.95), DRS bonus is nearly zero
            # At Monza (difficulty 0.25), DRS bonus is nearly full
            drs_bonus = 0.2 * (1.0 - track.overtake_difficulty)
        else:
            drs_bonus = 0.0

        # Driver skill difference
        skill_diff = attacker.overtaking_skill - defender.overtaking_skill * 0.5
        skill_factor = 0.5 + skill_diff * 0.3

        # Weather modifier (harder to overtake in wet)
        wet_modifier = 0.7 if is_wet else 1.0

        # Straight line speed advantage helps overtaking
        speed_factor = 1.0 + (attacker_car.straight_line_speed - defender_car.straight_line_speed) * 0.2

        # Combine factors
        probability = base_prob * pace_factor * gap_factor * skill_factor * speed_factor * wet_modifier + drs_bonus

        # SC restart bonus - drivers are more aggressive, tires cold, field bunched
        if restart_boost:
            probability *= 1.5  # 50% more likely to succeed on restart

        return min(0.9, max(0.0, probability))

    def _incident_probability(self, gap: float, track_difficulty: float) -> float:
        """Calculate probability of incident during overtake attempt.

        Args:
            gap: Time gap between cars
            track_difficulty: Track overtake difficulty

        Returns:
            Probability of incident (0-1)
        """
        # Closer battles = more incident risk
        gap_risk = max(0, (1.0 - gap) * 0.05)

        # Harder tracks = more incident risk
        track_risk = track_difficulty * 0.03

        return min(0.1, gap_risk + track_risk)

    @staticmethod
    def _sigmoid(x: float, offset: float = 0.5) -> float:
        """Sigmoid function for smooth probability transitions."""
        return 1.0 / (1.0 + np.exp(-x)) - offset + 0.5

    def should_attempt_overtake(
        self,
        attacker: Driver,
        defender: Driver,
        gap: float,
        remaining_laps: int,
        position_of_attacker: int,
    ) -> bool:
        """Decide if driver should attempt overtake based on race situation.

        Args:
            attacker: Driver considering attempt
            defender: Driver ahead
            gap: Current gap in seconds
            remaining_laps: Laps remaining in race
            position_of_attacker: Current position of attacker

        Returns:
            Whether to attempt overtake
        """
        if gap > 1.5:
            return False

        # More aggressive if fighting for points
        points_positions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        in_points_battle = position_of_attacker in points_positions

        # More aggressive with fewer laps remaining
        urgency = min(1.0, (20 - remaining_laps) / 20) if remaining_laps < 20 else 0.0

        # Driver aggression based on overtaking skill
        aggression = attacker.overtaking_skill

        threshold = 0.3 + urgency * 0.2 + aggression * 0.2
        if in_points_battle:
            threshold -= 0.1

        # Attempt if gap is small enough relative to threshold
        return gap < 1.5 * (1.0 - threshold + 0.5)
