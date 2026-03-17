"""Route optimizer computing energy-optimal routes."""

from __future__ import annotations

from typing import Optional

import numpy as np

from rangemax.battery.estimator import RangeEstimator
from rangemax.battery.model import BatteryModel
from rangemax.models import (
    BatteryState,
    DrivingProfile,
    EVehicle,
    Route,
    RouteSegment,
    WeatherCondition,
)


class RouteOptimizer:
    """Compute energy-optimal routes considering elevation, speed, and traffic.

    Optimizes speed per segment to minimize total energy consumption
    while respecting speed limits and time constraints.
    """

    def __init__(
        self,
        vehicle: EVehicle,
        battery_model: Optional[BatteryModel] = None,
    ) -> None:
        self.vehicle = vehicle
        self.battery = battery_model or BatteryModel(vehicle)
        self.estimator = RangeEstimator(vehicle, self.battery)

    def optimize_speed_profile(
        self,
        route: Route,
        battery_state: BatteryState,
        driving_profile: Optional[DrivingProfile] = None,
        weather: Optional[WeatherCondition] = None,
        climate_power_w: float = 0.0,
        max_time_hours: Optional[float] = None,
    ) -> list[dict[str, float]]:
        """Optimize speed for each segment to minimize energy consumption.

        Uses a simple gradient-free optimization: for each segment, find the
        speed that minimizes energy/km within constraints.

        Args:
            route: The route to optimize.
            battery_state: Current battery state.
            driving_profile: Driver profile.
            weather: Weather conditions.
            climate_power_w: HVAC power draw.
            max_time_hours: Optional time constraint.

        Returns:
            List of dicts per segment with optimal_speed_kmh, energy_kwh,
            time_minutes.
        """
        results = []
        remaining_energy = battery_state.energy_remaining_kwh

        for seg in route.segments:
            grade_pct = seg.grade * 100.0
            distance_km = seg.distance_m / 1000.0

            # Speed range: 20 km/h to speed limit (adjusted for traffic)
            max_speed = seg.speed_limit_kmh / max(seg.traffic_factor, 0.5)
            min_speed = min(20.0, max_speed)

            # Search for optimal speed
            best_speed = max_speed
            best_energy = float("inf")
            test_speeds = np.linspace(min_speed, max_speed, 20)

            for speed in test_speeds:
                e_per_km = self.estimator.energy_per_km(
                    speed_kmh=speed,
                    grade_pct=grade_pct,
                    driving_profile=driving_profile,
                    weather=weather,
                    climate_power_w=climate_power_w,
                )
                total_e = e_per_km * distance_km
                # Add time penalty for aux/climate loads
                time_h = distance_km / max(speed, 1)
                time_energy = (self.vehicle.aux_power_w + climate_power_w) * time_h / 1000.0
                effective_energy = total_e  # time cost already in energy_per_km

                if effective_energy < best_energy:
                    best_energy = effective_energy
                    best_speed = speed

            time_min = (distance_km / max(best_speed, 1)) * 60.0
            remaining_energy -= best_energy

            results.append(
                {
                    "segment_id": seg.segment_id,
                    "optimal_speed_kmh": round(float(best_speed), 1),
                    "energy_kwh": round(float(best_energy), 4),
                    "time_minutes": round(float(time_min), 1),
                    "distance_km": round(distance_km, 2),
                    "grade_pct": round(grade_pct, 1),
                    "remaining_energy_kwh": round(float(remaining_energy), 2),
                }
            )

        # If time constraint exists, adjust speeds upward where needed
        if max_time_hours is not None:
            total_time_h = sum(r["time_minutes"] for r in results) / 60.0
            if total_time_h > max_time_hours:
                speed_factor = total_time_h / max_time_hours
                for r, seg in zip(results, route.segments):
                    new_speed = min(
                        r["optimal_speed_kmh"] * speed_factor,
                        seg.speed_limit_kmh,
                    )
                    r["optimal_speed_kmh"] = round(new_speed, 1)
                    r["time_minutes"] = round(
                        (r["distance_km"] / max(new_speed, 1)) * 60.0, 1
                    )

        return results

    def compare_routes(
        self,
        routes: list[Route],
        battery_state: BatteryState,
        driving_profile: Optional[DrivingProfile] = None,
        weather: Optional[WeatherCondition] = None,
        climate_power_w: float = 0.0,
    ) -> list[dict[str, float]]:
        """Compare multiple routes by energy consumption.

        Returns list of route analyses sorted by energy efficiency.
        """
        analyses = []

        for route in routes:
            result = self.estimator.estimate_route_energy(
                route=route,
                battery_state=battery_state,
                driving_profile=driving_profile,
                weather=weather,
                climate_power_w=climate_power_w,
            )
            analyses.append(
                {
                    "route_id": route.route_id,
                    "route_name": route.name,
                    "distance_km": round(route.total_distance_km, 1),
                    "elevation_gain_m": round(route.total_elevation_gain_m, 0),
                    "total_energy_kwh": result["total_energy_kwh"],
                    "avg_consumption_kwh_per_km": result["avg_consumption_kwh_per_km"],
                    "soc_at_end_pct": result["soc_at_end_pct"],
                    "can_complete": bool(result["can_complete"]),
                }
            )

        analyses.sort(key=lambda a: a["total_energy_kwh"])
        return analyses

    def suggest_eco_speed(
        self,
        grade_pct: float = 0.0,
        weather: Optional[WeatherCondition] = None,
    ) -> float:
        """Suggest the most energy-efficient speed for given conditions.

        Returns optimal speed in km/h.
        """
        speeds = np.linspace(20, 130, 50)
        consumptions = []

        for speed in speeds:
            e = self.estimator.energy_per_km(
                speed_kmh=speed,
                grade_pct=grade_pct,
                weather=weather,
            )
            consumptions.append(e)

        best_idx = int(np.argmin(consumptions))
        return round(float(speeds[best_idx]), 0)
