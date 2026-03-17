"""Climate impact estimator for battery drain from HVAC systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from rangemax.models import ClimateMode, EVehicle, WeatherCondition


@dataclass
class ClimateImpact:
    """Result of climate impact estimation."""

    hvac_power_w: float
    range_reduction_pct: float
    energy_per_hour_kwh: float
    cabin_target_celsius: float
    mode: ClimateMode
    details: dict[str, float]


class ClimateImpactEstimator:
    """Estimate battery drain from heating, cooling, and defrost systems.

    Models:
    - Cabin heat loss/gain based on temperature differential
    - Heat pump COP (coefficient of performance) for cooling/heating
    - Resistive heating for extreme cold (below heat pump threshold)
    - Defrost mode power draw
    - Seat heater / steering wheel heater alternatives
    """

    # Cabin thermal parameters
    CABIN_THERMAL_MASS_KJ_PER_C = 8.0  # kJ per degree C
    CABIN_HEAT_TRANSFER_COEFF_W_PER_C = 40.0  # W per degree C difference
    CABIN_TARGET_COOLING = 22.0  # Celsius
    CABIN_TARGET_HEATING = 22.0

    # Heat pump performance
    HEAT_PUMP_COP_COOLING = 3.0  # COP for cooling (AC)
    HEAT_PUMP_COP_HEATING = 2.5  # COP for heating (mild weather)
    HEAT_PUMP_MIN_TEMP_C = -10.0  # Below this, resistive heating is used
    RESISTIVE_HEATING_EFFICIENCY = 0.95

    # Additional loads
    DEFROST_POWER_W = 2500.0
    SEAT_HEATER_POWER_W = 75.0  # per seat
    STEERING_HEATER_POWER_W = 50.0

    def __init__(self, vehicle: EVehicle) -> None:
        self.vehicle = vehicle

    def estimate_hvac_power(
        self,
        weather: WeatherCondition,
        mode: ClimateMode = ClimateMode.AUTO,
        cabin_temp_celsius: Optional[float] = None,
        num_seat_heaters: int = 0,
        steering_heater: bool = False,
    ) -> ClimateImpact:
        """Estimate HVAC power draw for given conditions.

        Args:
            weather: Current weather conditions.
            mode: HVAC mode.
            cabin_temp_celsius: Current cabin temperature (if None, assumes ambient).
            num_seat_heaters: Number of active seat heaters.
            steering_heater: Whether steering wheel heater is on.

        Returns:
            ClimateImpact with power draw and range impact.
        """
        ambient = weather.ambient_temp_celsius
        if cabin_temp_celsius is None:
            cabin_temp_celsius = ambient

        # Determine effective mode if AUTO
        if mode == ClimateMode.AUTO:
            if ambient < 18.0:
                mode = ClimateMode.HEATING
            elif ambient > 26.0:
                mode = ClimateMode.COOLING
            else:
                mode = ClimateMode.OFF

        # Calculate thermal load
        target_temp = self.CABIN_TARGET_HEATING if mode == ClimateMode.HEATING else self.CABIN_TARGET_COOLING
        delta_t = abs(ambient - target_temp)

        # Steady-state heat transfer: Q = U * delta_T
        thermal_load_w = self.CABIN_HEAT_TRANSFER_COEFF_W_PER_C * delta_t

        # Wind increases heat transfer
        wind_factor = 1.0 + 0.01 * weather.wind_speed_kmh
        thermal_load_w *= wind_factor

        # Solar load for cooling (simplified)
        if mode == ClimateMode.COOLING and weather.humidity_pct > 60:
            thermal_load_w *= 1.15  # humidity makes cooling harder

        # Compute electrical power based on mode
        hvac_power = 0.0
        details: dict[str, float] = {}

        if mode == ClimateMode.HEATING:
            if ambient >= self.HEAT_PUMP_MIN_TEMP_C:
                # Heat pump mode
                cop = self.HEAT_PUMP_COP_HEATING
                # COP degrades as temperature drops
                cop *= max(0.5, 1.0 - 0.02 * max(0, 10.0 - ambient))
                hvac_power = thermal_load_w / cop
                details["heat_pump_cop"] = round(cop, 2)
                details["mode"] = 1  # heat pump
            else:
                # Resistive heating (PTC heater)
                hvac_power = thermal_load_w / self.RESISTIVE_HEATING_EFFICIENCY
                details["mode"] = 2  # resistive
                details["resistive_efficiency"] = self.RESISTIVE_HEATING_EFFICIENCY

        elif mode == ClimateMode.COOLING:
            cop = self.HEAT_PUMP_COP_COOLING
            hvac_power = thermal_load_w / cop
            details["heat_pump_cop"] = round(cop, 2)
            details["mode"] = 3  # cooling

        elif mode == ClimateMode.DEFROST:
            hvac_power = self.DEFROST_POWER_W
            details["mode"] = 4  # defrost

        # Clamp to vehicle's max HVAC power
        hvac_power = min(hvac_power, self.vehicle.hvac_max_power_w)

        # Add seat/steering heaters
        seat_power = num_seat_heaters * self.SEAT_HEATER_POWER_W
        steering_power = self.STEERING_HEATER_POWER_W if steering_heater else 0.0
        total_power = hvac_power + seat_power + steering_power

        details["hvac_compressor_w"] = round(hvac_power, 0)
        details["seat_heaters_w"] = round(seat_power, 0)
        details["steering_heater_w"] = round(steering_power, 0)
        details["thermal_load_w"] = round(thermal_load_w, 0)
        details["ambient_temp_c"] = ambient
        details["delta_t"] = round(delta_t, 1)

        # Range reduction estimate
        # Baseline consumption at ~50km/h, no climate
        baseline_wh_per_km = (
            self.vehicle.usable_capacity_kwh * 1000.0 / self.vehicle.wltp_range_km
        )
        climate_wh_per_km = total_power / (50.0 / 3.6) * (1.0 / 3600.0) * 1000.0
        # Simpler: at 50km/h, power_w / (50/3.6) gives J/m, * 1000/3.6e6...
        # Actually: total_power (W) at speed 50km/h for time to travel 1km = 1/50 h = 72s
        climate_wh_per_km = total_power * (1.0 / 50.0)  # Wh/km (since 1km at 50km/h = 0.02h, P*t = P/50)
        range_reduction = (
            climate_wh_per_km / (baseline_wh_per_km + climate_wh_per_km) * 100.0
            if baseline_wh_per_km > 0
            else 0.0
        )

        energy_per_hour = total_power / 1000.0  # kWh

        return ClimateImpact(
            hvac_power_w=round(total_power, 0),
            range_reduction_pct=round(range_reduction, 1),
            energy_per_hour_kwh=round(energy_per_hour, 2),
            cabin_target_celsius=target_temp,
            mode=mode,
            details=details,
        )

    def recommend_strategy(
        self, weather: WeatherCondition
    ) -> list[str]:
        """Recommend strategies to minimize climate-related range loss."""
        tips: list[str] = []
        ambient = weather.ambient_temp_celsius

        if ambient < 5:
            tips.append("Pre-condition the cabin while plugged in to save battery.")
            tips.append("Use seat and steering wheel heaters instead of cabin heating when possible.")
            if ambient < -10:
                tips.append("Expect 30-40% range reduction in extreme cold. Plan charging stops accordingly.")
            else:
                tips.append("Expect 15-25% range reduction in cold weather.")

        elif ambient < 15:
            tips.append("Use ECO climate mode to reduce heating power.")
            tips.append("Seat heaters are more efficient than cabin heating for short trips.")

        elif ambient > 35:
            tips.append("Pre-cool the cabin while plugged in.")
            tips.append("Park in shade when possible to reduce cooling load.")
            tips.append("Expect 10-20% range reduction from air conditioning.")
            if weather.humidity_pct > 70:
                tips.append("High humidity increases AC load. Use recirculation mode.")

        elif ambient > 28:
            tips.append("Use ventilation or low AC setting to save energy.")

        else:
            tips.append("Mild weather - minimal climate impact on range.")

        if weather.precipitation:
            tips.append("Rain increases rolling resistance slightly. Wipers and lights add minor load.")

        return tips
