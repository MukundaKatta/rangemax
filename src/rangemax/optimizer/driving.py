"""Driving style analyzer scoring efficiency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rangemax.models import DrivingProfile


@dataclass
class DrivingEvent:
    """A single driving event (acceleration, braking, cruising)."""

    timestamp_s: float
    speed_kmh: float
    acceleration_ms2: float
    is_braking: bool = False
    is_regen: bool = False


@dataclass
class DrivingScore:
    """Comprehensive driving efficiency score."""

    overall_score: float  # 0-100
    acceleration_score: float  # 0-100
    braking_score: float  # 0-100
    speed_consistency_score: float  # 0-100
    regen_score: float  # 0-100
    grade: str  # A+, A, B, C, D, F
    tips: list[str]
    profile: DrivingProfile


class DrivingStyleAnalyzer:
    """Analyze driving patterns and score efficiency.

    Evaluates:
    - Acceleration patterns (smooth vs aggressive)
    - Braking patterns (regen vs friction)
    - Speed consistency (steady vs variable)
    - Overall energy efficiency
    """

    # Thresholds for scoring
    GENTLE_ACCEL_MS2 = 1.5  # m/s^2
    MODERATE_ACCEL_MS2 = 2.5
    HARD_ACCEL_MS2 = 3.5
    GENTLE_BRAKE_MS2 = 1.5
    HARD_BRAKE_MS2 = 3.0
    IDEAL_SPEED_VARIABILITY = 0.08  # coefficient of variation

    def analyze(self, events: list[DrivingEvent]) -> DrivingScore:
        """Analyze driving events and produce a score."""
        if not events:
            return self._empty_score()

        accel_events = [e for e in events if e.acceleration_ms2 > 0.1 and not e.is_braking]
        brake_events = [e for e in events if e.is_braking]
        regen_events = [e for e in events if e.is_regen]
        speeds = np.array([e.speed_kmh for e in events])

        # Acceleration score
        accel_score = self._score_acceleration(accel_events)

        # Braking score
        braking_score = self._score_braking(brake_events)

        # Speed consistency
        speed_score = self._score_speed_consistency(speeds)

        # Regen usage
        regen_score = self._score_regen(brake_events, regen_events)

        # Overall weighted score
        overall = (
            accel_score * 0.30
            + braking_score * 0.25
            + speed_score * 0.25
            + regen_score * 0.20
        )

        grade = self._grade(overall)
        tips = self._generate_tips(accel_score, braking_score, speed_score, regen_score)

        # Build driving profile
        accels = [e.acceleration_ms2 for e in accel_events] if accel_events else [1.5]
        brakes = [abs(e.acceleration_ms2) for e in brake_events] if brake_events else [2.0]
        avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 50.0
        speed_var = float(np.std(speeds) / max(avg_speed, 1)) if len(speeds) > 1 else 0.15

        total_distance_km = self._estimate_distance(events)
        hard_brakes = sum(1 for e in brake_events if abs(e.acceleration_ms2) > self.HARD_BRAKE_MS2)
        hard_accels = sum(1 for e in accel_events if e.acceleration_ms2 > self.HARD_ACCEL_MS2)

        profile = DrivingProfile(
            avg_acceleration_ms2=round(float(np.mean(accels)), 2),
            avg_braking_decel_ms2=round(float(np.mean(brakes)), 2),
            avg_speed_kmh=round(avg_speed, 1),
            speed_variability=round(speed_var, 3),
            hard_braking_events_per_km=round(hard_brakes / max(total_distance_km, 0.1), 3),
            hard_accel_events_per_km=round(hard_accels / max(total_distance_km, 0.1), 3),
            regen_usage_pct=round(
                len(regen_events) / max(len(brake_events), 1) * 100, 1
            ),
            efficiency_score=round(overall / 100.0, 3),
        )

        return DrivingScore(
            overall_score=round(overall, 1),
            acceleration_score=round(accel_score, 1),
            braking_score=round(braking_score, 1),
            speed_consistency_score=round(speed_score, 1),
            regen_score=round(regen_score, 1),
            grade=grade,
            tips=tips,
            profile=profile,
        )

    def _score_acceleration(self, events: list[DrivingEvent]) -> float:
        """Score acceleration smoothness (100 = very gentle, 0 = very aggressive)."""
        if not events:
            return 75.0
        accels = np.array([e.acceleration_ms2 for e in events])
        avg_accel = float(np.mean(accels))

        if avg_accel <= self.GENTLE_ACCEL_MS2:
            base = 90.0
        elif avg_accel <= self.MODERATE_ACCEL_MS2:
            base = 70.0 + 20.0 * (self.MODERATE_ACCEL_MS2 - avg_accel) / (
                self.MODERATE_ACCEL_MS2 - self.GENTLE_ACCEL_MS2
            )
        else:
            base = max(20.0, 70.0 - 30.0 * (avg_accel - self.MODERATE_ACCEL_MS2))

        # Penalty for hard acceleration events
        hard_pct = np.sum(accels > self.HARD_ACCEL_MS2) / len(accels)
        penalty = hard_pct * 20.0

        return max(0, min(100, base - penalty))

    def _score_braking(self, events: list[DrivingEvent]) -> float:
        """Score braking smoothness."""
        if not events:
            return 75.0
        decels = np.array([abs(e.acceleration_ms2) for e in events])
        avg_decel = float(np.mean(decels))

        if avg_decel <= self.GENTLE_BRAKE_MS2:
            base = 90.0
        elif avg_decel <= self.HARD_BRAKE_MS2:
            base = 60.0 + 30.0 * (self.HARD_BRAKE_MS2 - avg_decel) / (
                self.HARD_BRAKE_MS2 - self.GENTLE_BRAKE_MS2
            )
        else:
            base = max(15.0, 60.0 - 20.0 * (avg_decel - self.HARD_BRAKE_MS2))

        hard_pct = np.sum(decels > self.HARD_BRAKE_MS2) / len(decels)
        penalty = hard_pct * 25.0

        return max(0, min(100, base - penalty))

    def _score_speed_consistency(self, speeds: np.ndarray) -> float:
        """Score speed consistency (less variation = more efficient)."""
        if len(speeds) < 2:
            return 75.0

        mean_speed = float(np.mean(speeds))
        if mean_speed < 1:
            return 50.0

        cv = float(np.std(speeds) / mean_speed)

        if cv <= self.IDEAL_SPEED_VARIABILITY:
            return 95.0
        elif cv <= 0.15:
            return 80.0 + 15.0 * (0.15 - cv) / (0.15 - self.IDEAL_SPEED_VARIABILITY)
        elif cv <= 0.30:
            return 50.0 + 30.0 * (0.30 - cv) / 0.15
        else:
            return max(10.0, 50.0 - 80.0 * (cv - 0.30))

    def _score_regen(
        self,
        brake_events: list[DrivingEvent],
        regen_events: list[DrivingEvent],
    ) -> float:
        """Score regenerative braking usage."""
        if not brake_events:
            return 75.0

        regen_ratio = len(regen_events) / len(brake_events)

        if regen_ratio >= 0.9:
            return 95.0
        elif regen_ratio >= 0.7:
            return 75.0 + 20.0 * (regen_ratio - 0.7) / 0.2
        elif regen_ratio >= 0.4:
            return 50.0 + 25.0 * (regen_ratio - 0.4) / 0.3
        else:
            return max(10.0, regen_ratio / 0.4 * 50.0)

    def _grade(self, score: float) -> str:
        if score >= 95:
            return "A+"
        elif score >= 85:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 45:
            return "D"
        return "F"

    def _generate_tips(
        self, accel: float, brake: float, speed: float, regen: float
    ) -> list[str]:
        tips = []
        if accel < 70:
            tips.append("Accelerate more gently. Aim for smooth, gradual acceleration.")
        if brake < 70:
            tips.append("Brake earlier and more gradually to maximize regenerative braking.")
        if speed < 70:
            tips.append("Maintain a more consistent speed. Use cruise control when possible.")
        if regen < 70:
            tips.append("Use one-pedal driving mode to maximize energy recovery through regen.")
        if not tips:
            tips.append("Excellent driving style! Keep it up for maximum range.")
        return tips

    def _estimate_distance(self, events: list[DrivingEvent]) -> float:
        """Estimate total distance from events in km."""
        if len(events) < 2:
            return 1.0
        total_m = 0.0
        for i in range(1, len(events)):
            dt = events[i].timestamp_s - events[i - 1].timestamp_s
            avg_speed_ms = (events[i].speed_kmh + events[i - 1].speed_kmh) / 2.0 / 3.6
            total_m += avg_speed_ms * dt
        return max(total_m / 1000.0, 0.1)

    def _empty_score(self) -> DrivingScore:
        return DrivingScore(
            overall_score=0.0,
            acceleration_score=0.0,
            braking_score=0.0,
            speed_consistency_score=0.0,
            regen_score=0.0,
            grade="F",
            tips=["No driving data available."],
            profile=DrivingProfile(),
        )
