"""Range estimator predicting remaining range from current conditions."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from rangemax.battery.model import BatteryModel
from rangemax.models import (
    BatteryState,
    DrivingProfile,
    EVehicle,
    RangeEstimate,
    Route,
    WeatherCondition,
)


# Physical constants
AIR_DENSITY_KG_M3 = 1.225  # at sea level, 15C
GRAVITY_MS2 = 9.81


class RangeEstimator:
    """Estimate remaining EV range from SoC, conditions, and driving style.

    Uses a physics-based energy consumption model considering:
    - Aerodynamic drag
    - Rolling resistance
    - Elevation changes (grade resistance)
    - Regenerative braking energy recovery
    - HVAC and auxiliary loads
    - Battery temperature effects
    - Driving style efficiency
    """

    def __init__(self, vehicle: EVehicle, battery_model: Optional[BatteryModel] = None) -> None:
        self.vehicle = vehicle
        self.battery = battery_model or BatteryModel(vehicle)

    def energy_per_km(
        self,
        speed_kmh: float,
        grade_pct: float = 0.0,
        driving_profile: Optional[DrivingProfile] = None,
        weather: Optional[WeatherCondition] = None,
        climate_power_w: float = 0.0,
    ) -> float:
        """Compute energy consumption in kWh/km at steady state.

        Args:
            speed_kmh: Vehicle speed in km/h.
            grade_pct: Road grade in percent (positive = uphill).
            driving_profile: Driving style profile.
            weather: Weather conditions.
            climate_power_w: HVAC power draw in watts.

        Returns:
            Energy consumption in kWh/km (can be negative for steep downhill with regen).
        """
        v = vehicle = self.vehicle
        speed_ms = max(speed_kmh / 3.6, 0.1)  # m/s, min to avoid div by zero
        grade = grade_pct / 100.0
        mass = v.curb_weight_kg + 80.0  # add driver weight

        # --- Aerodynamic drag ---
        # F_drag = 0.5 * rho * Cd * A * v^2
        air_density = AIR_DENSITY_KG_M3
        effective_speed = speed_ms
        if weather and weather.wind_speed_kmh > 0:
            # Simple headwind/tailwind model
            wind_ms = weather.wind_speed_kmh / 3.6
            # Assume heading = 0, wind_direction is relative
            wind_component = wind_ms * np.cos(np.radians(weather.wind_direction_deg))
            effective_speed = speed_ms + wind_component  # headwind increases drag

        f_drag = 0.5 * air_density * v.drag_coefficient * v.frontal_area_m2 * effective_speed**2
        # Drag always opposes motion
        if effective_speed < 0:
            f_drag = -f_drag

        # --- Rolling resistance ---
        # F_rr = Crr * m * g * cos(theta)
        cos_theta = np.cos(np.arctan(grade))
        f_rolling = v.rolling_resistance_coeff * mass * GRAVITY_MS2 * cos_theta

        # --- Grade resistance ---
        # F_grade = m * g * sin(theta)
        sin_theta = np.sin(np.arctan(grade))
        f_grade = mass * GRAVITY_MS2 * sin_theta

        # --- Total tractive force ---
        f_total = f_drag + f_rolling + f_grade

        # --- Driving style adjustment ---
        style_factor = 1.0
        if driving_profile:
            # Aggressive driving increases consumption
            style_factor = 1.0 / max(driving_profile.efficiency_score, 0.3)

        # --- Power at wheels ---
        p_wheels = f_total * speed_ms * style_factor  # watts

        if p_wheels >= 0:
            # Driving: motor efficiency loss
            p_battery = p_wheels / v.drivetrain_efficiency
        else:
            # Regen: recovering energy
            regen_power = abs(p_wheels)
            # Limit regen by max regen deceleration
            max_regen_force = mass * v.max_regen_decel_ms2
            max_regen_power = max_regen_force * speed_ms
            regen_power = min(regen_power, max_regen_power)

            # Apply regen and drivetrain efficiency
            regen_usage = 0.7
            if driving_profile:
                regen_usage = driving_profile.regen_usage_pct / 100.0
            p_battery = -(regen_power * v.regen_efficiency * v.drivetrain_efficiency * regen_usage)

        # --- Auxiliary loads ---
        p_aux = v.aux_power_w + climate_power_w

        # --- Total battery power ---
        p_total = p_battery + p_aux  # watts

        # Energy per km = power / speed
        energy_kwh_per_km = p_total / (speed_ms * 1000.0)  # kWh/km
        # speed_ms * 1000 converts W/(m/s) to kWh/km via:
        # W / (m/s) = J/m = Ws/m; /1000 = Wh/m... actually:
        # p_total (W) / speed_ms (m/s) = J/m = Ws/m
        # Ws/m * 1000m/km / 3600 s/h / 1000 W/kW = kWh/km
        energy_kwh_per_km = (p_total / speed_ms) * (1.0 / 3_600_000.0) * 1000.0

        return energy_kwh_per_km

    def estimate_range(
        self,
        battery_state: BatteryState,
        driving_profile: Optional[DrivingProfile] = None,
        weather: Optional[WeatherCondition] = None,
        climate_power_w: float = 0.0,
        avg_speed_kmh: float = 50.0,
    ) -> RangeEstimate:
        """Estimate remaining range given current conditions.

        Provides best-case, expected, and worst-case estimates.
        """
        remaining_energy = battery_state.energy_remaining_kwh

        # Expected consumption
        consumption = self.energy_per_km(
            speed_kmh=avg_speed_kmh,
            driving_profile=driving_profile,
            weather=weather,
            climate_power_w=climate_power_w,
        )

        # Best case: eco driving, no climate, no elevation
        eco_profile = DrivingProfile(efficiency_score=0.95, regen_usage_pct=95)
        best_consumption = self.energy_per_km(
            speed_kmh=avg_speed_kmh * 0.85,
            driving_profile=eco_profile,
        )

        # Worst case: aggressive driving, full climate, headwind
        worst_profile = DrivingProfile(efficiency_score=0.5, regen_usage_pct=40)
        worst_weather = WeatherCondition(
            wind_speed_kmh=30,
            wind_direction_deg=0,
            ambient_temp_celsius=-5 if not weather else weather.ambient_temp_celsius,
        )
        worst_consumption = self.energy_per_km(
            speed_kmh=avg_speed_kmh * 1.2,
            driving_profile=worst_profile,
            weather=worst_weather,
            climate_power_w=self.vehicle.hvac_max_power_w,
        )

        # Compute ranges (avoid negative/zero consumption)
        consumption = max(consumption, 0.01)
        best_consumption = max(best_consumption, 0.01)
        worst_consumption = max(worst_consumption, 0.01)

        est_range = remaining_energy / consumption
        best_range = remaining_energy / best_consumption
        worst_range = remaining_energy / worst_consumption

        # Factor breakdown
        base = self.energy_per_km(avg_speed_kmh)
        factors = {
            "aero_drag": round(base * 0.45, 4),  # approximate breakdown
            "rolling_resistance": round(base * 0.25, 4),
            "auxiliary_loads": round(self.vehicle.aux_power_w / (avg_speed_kmh / 3.6) / 3_600_000 * 1000, 4),
            "climate": round(climate_power_w / (avg_speed_kmh / 3.6) / 3_600_000 * 1000, 4) if climate_power_w else 0.0,
        }

        eff_score = 1.0
        if driving_profile:
            eff_score = driving_profile.efficiency_score

        climate_impact = climate_power_w * (est_range / avg_speed_kmh) / 1000.0 if avg_speed_kmh > 0 else 0.0

        return RangeEstimate(
            estimated_range_km=round(max(0, est_range), 1),
            best_case_km=round(max(0, best_range), 1),
            worst_case_km=round(max(0, worst_range), 1),
            energy_consumption_kwh_per_km=round(consumption, 4),
            remaining_energy_kwh=round(remaining_energy, 2),
            factors=factors,
            driving_efficiency_score=round(eff_score, 2),
            climate_impact_kwh=round(climate_impact, 2),
            timestamp=datetime.now(),
        )

    def estimate_route_energy(
        self,
        route: Route,
        battery_state: BatteryState,
        driving_profile: Optional[DrivingProfile] = None,
        weather: Optional[WeatherCondition] = None,
        climate_power_w: float = 0.0,
    ) -> dict[str, float]:
        """Estimate total energy consumption for a specific route.

        Returns:
            Dict with total_energy_kwh, can_complete (bool), soc_at_end, etc.
        """
        total_energy = 0.0
        remaining = battery_state.energy_remaining_kwh
        segment_energies = []

        for seg in route.segments:
            speed = seg.expected_speed_kmh / max(seg.traffic_factor, 0.1)
            grade = seg.grade * 100.0  # convert to pct
            distance_km = seg.distance_m / 1000.0

            consumption = self.energy_per_km(
                speed_kmh=speed,
                grade_pct=grade,
                driving_profile=driving_profile,
                weather=weather,
                climate_power_w=climate_power_w,
            )
            energy = consumption * distance_km
            total_energy += energy
            remaining -= energy
            segment_energies.append(energy)

        can_complete = remaining > 0
        soc_at_end = max(0, remaining / battery_state.capacity_kwh * 100) if battery_state.capacity_kwh > 0 else 0

        return {
            "total_energy_kwh": round(total_energy, 3),
            "remaining_energy_kwh": round(max(0, remaining), 3),
            "can_complete": float(can_complete),
            "soc_at_end_pct": round(soc_at_end, 1),
            "avg_consumption_kwh_per_km": round(
                total_energy / route.total_distance_km, 4
            )
            if route.total_distance_km > 0
            else 0.0,
            "segment_count": len(route.segments),
        }
