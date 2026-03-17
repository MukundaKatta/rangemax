"""Battery health tracker monitoring capacity degradation over time."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class HealthReading:
    """A single battery health measurement."""

    date: date
    cycle_count: int
    soh_pct: float
    capacity_kwh: float
    temperature_avg_celsius: float = 25.0


@dataclass
class HealthReport:
    """Battery health analysis report."""

    current_soh_pct: float
    current_capacity_kwh: float
    original_capacity_kwh: float
    capacity_loss_pct: float
    cycle_count: int
    projected_80pct_date: date | None
    projected_80pct_cycles: int | None
    degradation_rate_pct_per_year: float
    degradation_rate_pct_per_cycle: float
    health_grade: str  # A, B, C, D, F
    recommendation: str


class BatteryHealthTracker:
    """Monitor battery capacity degradation over time.

    Models degradation using:
    - Calendar aging: sqrt(time) degradation model
    - Cycle aging: linear + sqrt(cycles) model
    - Temperature stress: Arrhenius acceleration factor
    """

    # Typical NMC degradation parameters
    CALENDAR_AGING_FACTOR = 0.02  # 2% per sqrt(year) at 25C
    CYCLE_AGING_FACTOR = 0.00005  # per cycle
    TEMP_REF_CELSIUS = 25.0
    TEMP_ACTIVATION_ENERGY = 0.8  # Arrhenius factor (simplified)

    def __init__(self, original_capacity_kwh: float) -> None:
        self.original_capacity_kwh = original_capacity_kwh
        self._readings: list[HealthReading] = []

    def add_reading(self, reading: HealthReading) -> None:
        """Add a health measurement reading."""
        self._readings.append(reading)
        self._readings.sort(key=lambda r: r.date)

    @property
    def readings(self) -> list[HealthReading]:
        return list(self._readings)

    @property
    def latest_reading(self) -> HealthReading | None:
        return self._readings[-1] if self._readings else None

    def estimate_soh(
        self,
        age_years: float,
        cycle_count: int,
        avg_temperature_celsius: float = 25.0,
    ) -> float:
        """Estimate state of health from age and usage.

        Uses a combined calendar + cycle aging model.
        """
        # Calendar aging: capacity_loss = a * sqrt(t) * temp_factor
        temp_factor = self._temperature_factor(avg_temperature_celsius)
        calendar_loss = self.CALENDAR_AGING_FACTOR * np.sqrt(age_years) * temp_factor

        # Cycle aging: capacity_loss = b * cycles
        cycle_loss = self.CYCLE_AGING_FACTOR * cycle_count

        total_loss = calendar_loss + cycle_loss
        soh = max(0.0, 1.0 - total_loss)
        return round(soh * 100, 2)

    def project_degradation(
        self,
        years_ahead: int = 5,
        annual_cycles: int = 300,
        avg_temperature_celsius: float = 25.0,
    ) -> list[tuple[float, float]]:
        """Project future degradation.

        Returns list of (years_from_now, projected_soh_pct).
        """
        if not self._readings:
            current_age = 0.0
            current_cycles = 0
        else:
            latest = self._readings[-1]
            first = self._readings[0]
            current_age = (latest.date - first.date).days / 365.25
            current_cycles = latest.cycle_count

        projections = []
        for yr in range(years_ahead + 1):
            age = current_age + yr
            cycles = current_cycles + annual_cycles * yr
            soh = self.estimate_soh(age, cycles, avg_temperature_celsius)
            projections.append((float(yr), soh))

        return projections

    def generate_report(self) -> HealthReport:
        """Generate a comprehensive health report."""
        if not self._readings:
            return HealthReport(
                current_soh_pct=100.0,
                current_capacity_kwh=self.original_capacity_kwh,
                original_capacity_kwh=self.original_capacity_kwh,
                capacity_loss_pct=0.0,
                cycle_count=0,
                projected_80pct_date=None,
                projected_80pct_cycles=None,
                degradation_rate_pct_per_year=0.0,
                degradation_rate_pct_per_cycle=0.0,
                health_grade="A",
                recommendation="No data yet. Continue monitoring.",
            )

        latest = self._readings[-1]
        first = self._readings[0]
        current_soh = latest.soh_pct
        capacity_loss = 100.0 - current_soh
        age_years = max((latest.date - first.date).days / 365.25, 0.01)

        # Degradation rates
        deg_per_year = capacity_loss / age_years if age_years > 0 else 0.0
        deg_per_cycle = (
            capacity_loss / latest.cycle_count
            if latest.cycle_count > 0
            else 0.0
        )

        # Project when SoH hits 80%
        projected_80_date = None
        projected_80_cycles = None
        if current_soh > 80.0 and deg_per_year > 0:
            years_to_80 = (current_soh - 80.0) / deg_per_year
            projected_80_date = latest.date + timedelta(days=int(years_to_80 * 365.25))
        if current_soh > 80.0 and deg_per_cycle > 0:
            projected_80_cycles = latest.cycle_count + int(
                (current_soh - 80.0) / deg_per_cycle
            )

        # Health grade
        if current_soh >= 95:
            grade = "A"
        elif current_soh >= 90:
            grade = "B"
        elif current_soh >= 85:
            grade = "C"
        elif current_soh >= 80:
            grade = "D"
        else:
            grade = "F"

        # Recommendation
        if current_soh >= 90:
            rec = "Battery is in good condition. Continue normal usage."
        elif current_soh >= 85:
            rec = "Minor degradation detected. Avoid frequent fast charging and extreme temperatures."
        elif current_soh >= 80:
            rec = "Significant degradation. Consider reducing fast charge usage. Plan for battery replacement within 1-2 years."
        else:
            rec = "Battery below recommended threshold. Replacement recommended."

        return HealthReport(
            current_soh_pct=round(current_soh, 1),
            current_capacity_kwh=round(latest.capacity_kwh, 2),
            original_capacity_kwh=self.original_capacity_kwh,
            capacity_loss_pct=round(capacity_loss, 1),
            cycle_count=latest.cycle_count,
            projected_80pct_date=projected_80_date,
            projected_80pct_cycles=projected_80_cycles,
            degradation_rate_pct_per_year=round(deg_per_year, 2),
            degradation_rate_pct_per_cycle=round(deg_per_cycle, 4),
            health_grade=grade,
            recommendation=rec,
        )

    def _temperature_factor(self, temperature_celsius: float) -> float:
        """Arrhenius-like temperature acceleration factor.

        Higher temperatures accelerate degradation.
        """
        delta_t = temperature_celsius - self.TEMP_REF_CELSIUS
        return float(np.exp(self.TEMP_ACTIVATION_ENERGY * delta_t / self.TEMP_REF_CELSIUS))

    def fit_degradation_curve(self) -> tuple[float, float] | None:
        """Fit a degradation curve to recorded readings.

        Returns (a, b) for model: SoH = 100 - a*sqrt(cycles) - b*cycles
        Returns None if insufficient data.
        """
        if len(self._readings) < 3:
            return None

        cycles = np.array([r.cycle_count for r in self._readings], dtype=float)
        soh = np.array([r.soh_pct for r in self._readings], dtype=float)

        def model(x, a, b):
            return 100.0 - a * np.sqrt(x) - b * x

        try:
            popt, _ = curve_fit(model, cycles, soh, p0=[0.5, 0.001], maxfev=5000)
            return (float(popt[0]), float(popt[1]))
        except RuntimeError:
            return None
