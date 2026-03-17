"""EV simulation for generating test scenarios."""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Optional

import numpy as np

from rangemax.battery.health import BatteryHealthTracker, HealthReading
from rangemax.models import (
    DrivingProfile,
    EVehicle,
    Route,
    RouteSegment,
    WeatherCondition,
)
from rangemax.optimizer.driving import DrivingEvent


class EVSimulator:
    """Generate realistic EV data for testing and demonstration."""

    # Reference EVs with real-world parameters
    REFERENCE_VEHICLES = [
        EVehicle(
            vehicle_id="tesla-model3-lr",
            make="Tesla",
            model="Model 3 Long Range",
            year=2024,
            battery_capacity_kwh=82.0,
            usable_capacity_kwh=75.0,
            curb_weight_kg=1830,
            frontal_area_m2=2.22,
            drag_coefficient=0.23,
            rolling_resistance_coeff=0.009,
            drivetrain_efficiency=0.92,
            regen_efficiency=0.70,
            max_regen_decel_ms2=2.0,
            aux_power_w=250,
            hvac_max_power_w=5000,
            max_speed_kmh=233,
            wltp_range_km=629,
        ),
        EVehicle(
            vehicle_id="hyundai-ioniq5-lr",
            make="Hyundai",
            model="IONIQ 5 Long Range",
            year=2024,
            battery_capacity_kwh=77.4,
            usable_capacity_kwh=74.0,
            curb_weight_kg=2020,
            frontal_area_m2=2.47,
            drag_coefficient=0.288,
            rolling_resistance_coeff=0.010,
            drivetrain_efficiency=0.90,
            regen_efficiency=0.65,
            max_regen_decel_ms2=1.5,
            aux_power_w=300,
            hvac_max_power_w=5500,
            max_speed_kmh=185,
            wltp_range_km=507,
        ),
        EVehicle(
            vehicle_id="bmw-ix-xdrive50",
            make="BMW",
            model="iX xDrive50",
            year=2024,
            battery_capacity_kwh=111.5,
            usable_capacity_kwh=105.2,
            curb_weight_kg=2510,
            frontal_area_m2=2.61,
            drag_coefficient=0.25,
            rolling_resistance_coeff=0.010,
            drivetrain_efficiency=0.89,
            regen_efficiency=0.63,
            max_regen_decel_ms2=1.5,
            aux_power_w=350,
            hvac_max_power_w=6000,
            max_speed_kmh=200,
            wltp_range_km=630,
        ),
        EVehicle(
            vehicle_id="chevy-equinox-ev",
            make="Chevrolet",
            model="Equinox EV 3LT",
            year=2024,
            battery_capacity_kwh=85.0,
            usable_capacity_kwh=79.0,
            curb_weight_kg=2168,
            frontal_area_m2=2.55,
            drag_coefficient=0.28,
            rolling_resistance_coeff=0.010,
            drivetrain_efficiency=0.90,
            regen_efficiency=0.64,
            max_regen_decel_ms2=1.5,
            aux_power_w=280,
            hvac_max_power_w=5000,
            max_speed_kmh=190,
            wltp_range_km=515,
        ),
    ]

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

    def get_vehicle(self, index: int = 0) -> EVehicle:
        """Get a reference vehicle by index."""
        return self.REFERENCE_VEHICLES[index % len(self.REFERENCE_VEHICLES)]

    def get_all_vehicles(self) -> list[EVehicle]:
        """Get all reference vehicles."""
        return list(self.REFERENCE_VEHICLES)

    def generate_route(
        self,
        name: str = "City Commute",
        num_segments: int = 10,
        route_type: str = "city",
    ) -> Route:
        """Generate a realistic route.

        Args:
            name: Route name.
            num_segments: Number of segments.
            route_type: One of 'city', 'highway', 'mountain', 'mixed'.
        """
        segments = []
        elevation = self.rng.uniform(50, 200)

        for i in range(num_segments):
            if route_type == "city":
                distance = self.rng.uniform(200, 2000)
                speed_limit = self.rng.choice([30, 40, 50, 60])
                traffic = self.rng.uniform(1.0, 2.0)
                elev_change = self.rng.normal(0, 3)
            elif route_type == "highway":
                distance = self.rng.uniform(2000, 10000)
                speed_limit = self.rng.choice([100, 110, 120, 130])
                traffic = self.rng.uniform(1.0, 1.5)
                elev_change = self.rng.normal(0, 5)
            elif route_type == "mountain":
                distance = self.rng.uniform(500, 3000)
                speed_limit = self.rng.choice([40, 50, 60, 80])
                traffic = self.rng.uniform(1.0, 1.3)
                elev_change = self.rng.normal(15, 20)
            else:  # mixed
                distance = self.rng.uniform(500, 5000)
                speed_limit = float(self.rng.choice([40, 60, 80, 100, 120]))
                traffic = self.rng.uniform(1.0, 1.8)
                elev_change = self.rng.normal(0, 10)

            new_elevation = elevation + elev_change
            expected_speed = speed_limit / traffic

            segments.append(
                RouteSegment(
                    segment_id=i,
                    distance_m=round(float(distance), 0),
                    elevation_start_m=round(float(elevation), 1),
                    elevation_end_m=round(float(new_elevation), 1),
                    speed_limit_kmh=float(speed_limit),
                    expected_speed_kmh=round(float(expected_speed), 1),
                    traffic_factor=round(float(traffic), 2),
                )
            )
            elevation = new_elevation

        return Route(
            route_id=f"route-{route_type}-{num_segments}",
            name=name,
            segments=segments,
        )

    def generate_driving_events(
        self,
        duration_s: float = 1800,
        style: str = "normal",
    ) -> list[DrivingEvent]:
        """Generate a sequence of driving events.

        Args:
            duration_s: Total duration in seconds.
            style: One of 'eco', 'normal', 'aggressive'.
        """
        events = []
        t = 0.0
        speed = 0.0
        dt = 1.0  # 1 second intervals

        while t < duration_s:
            # Random driving behavior
            r = self.rng.random()

            if style == "eco":
                target_speed = self.rng.uniform(30, 60)
                max_accel = 1.2
                max_brake = 1.5
            elif style == "aggressive":
                target_speed = self.rng.uniform(50, 100)
                max_accel = 4.0
                max_brake = 5.0
            else:
                target_speed = self.rng.uniform(35, 80)
                max_accel = 2.5
                max_brake = 3.0

            if r < 0.3 and speed < target_speed:
                # Accelerate
                accel = self.rng.uniform(0.5, max_accel)
                speed = min(speed + accel * dt * 3.6, target_speed)
                is_braking = False
            elif r < 0.5 and speed > 10:
                # Brake
                decel = self.rng.uniform(0.5, max_brake)
                speed = max(0, speed - decel * dt * 3.6)
                accel = -decel
                is_braking = True
            else:
                # Coast
                accel = self.rng.normal(0, 0.2)
                speed = max(0, speed + accel * dt * 3.6)
                is_braking = False

            is_regen = is_braking and abs(accel) < 2.0

            events.append(
                DrivingEvent(
                    timestamp_s=t,
                    speed_kmh=round(speed, 1),
                    acceleration_ms2=round(abs(accel) if not is_braking else -abs(accel), 2),
                    is_braking=is_braking,
                    is_regen=is_regen,
                )
            )
            t += dt

        return events

    def generate_weather_scenarios(self) -> list[tuple[str, WeatherCondition]]:
        """Generate typical weather scenarios for range testing."""
        return [
            (
                "Mild Spring",
                WeatherCondition(
                    ambient_temp_celsius=18.0,
                    wind_speed_kmh=10.0,
                    wind_direction_deg=90.0,
                ),
            ),
            (
                "Hot Summer",
                WeatherCondition(
                    ambient_temp_celsius=38.0,
                    wind_speed_kmh=5.0,
                    wind_direction_deg=180.0,
                    humidity_pct=75.0,
                ),
            ),
            (
                "Cold Winter",
                WeatherCondition(
                    ambient_temp_celsius=-10.0,
                    wind_speed_kmh=25.0,
                    wind_direction_deg=0.0,
                    humidity_pct=40.0,
                ),
            ),
            (
                "Rainy Autumn",
                WeatherCondition(
                    ambient_temp_celsius=10.0,
                    wind_speed_kmh=30.0,
                    wind_direction_deg=45.0,
                    precipitation=True,
                    humidity_pct=90.0,
                ),
            ),
            (
                "Ideal",
                WeatherCondition(
                    ambient_temp_celsius=22.0,
                    wind_speed_kmh=0.0,
                    wind_direction_deg=0.0,
                    humidity_pct=50.0,
                ),
            ),
        ]

    def generate_health_history(
        self,
        original_capacity_kwh: float,
        age_years: float = 3.0,
        annual_cycles: int = 300,
        avg_temp: float = 25.0,
    ) -> BatteryHealthTracker:
        """Generate a battery health history."""
        tracker = BatteryHealthTracker(original_capacity_kwh)
        start_date = date.today() - timedelta(days=int(age_years * 365.25))
        total_cycles = int(annual_cycles * age_years)
        num_readings = min(int(age_years * 4), 20)  # quarterly readings

        for i in range(num_readings + 1):
            frac = i / num_readings
            reading_date = start_date + timedelta(days=int(frac * age_years * 365.25))
            cycles = int(frac * total_cycles)

            soh = tracker.estimate_soh(
                frac * age_years, cycles, avg_temp
            )
            capacity = original_capacity_kwh * soh / 100.0

            tracker.add_reading(
                HealthReading(
                    date=reading_date,
                    cycle_count=cycles,
                    soh_pct=soh,
                    capacity_kwh=round(capacity, 2),
                    temperature_avg_celsius=avg_temp,
                )
            )

        return tracker
