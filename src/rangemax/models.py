"""Pydantic models for RangeMax."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DriveMode(str, Enum):
    ECO = "eco"
    NORMAL = "normal"
    SPORT = "sport"


class ClimateMode(str, Enum):
    OFF = "off"
    HEATING = "heating"
    COOLING = "cooling"
    DEFROST = "defrost"
    AUTO = "auto"


class EVehicle(BaseModel):
    """An electric vehicle with physical parameters."""

    vehicle_id: str
    make: str
    model: str
    year: int
    battery_capacity_kwh: float = Field(gt=0, description="Total battery capacity in kWh")
    usable_capacity_kwh: float = Field(gt=0, description="Usable battery capacity in kWh")
    curb_weight_kg: float = Field(gt=0, description="Vehicle curb weight in kg")
    frontal_area_m2: float = Field(default=2.3, gt=0, description="Frontal area in m^2")
    drag_coefficient: float = Field(default=0.28, gt=0, description="Aerodynamic drag coefficient Cd")
    rolling_resistance_coeff: float = Field(default=0.01, gt=0, description="Rolling resistance coefficient Crr")
    drivetrain_efficiency: float = Field(default=0.90, gt=0, le=1.0, description="Motor+inverter+gearbox efficiency")
    regen_efficiency: float = Field(default=0.65, gt=0, le=1.0, description="Regenerative braking efficiency")
    max_regen_decel_ms2: float = Field(default=1.5, gt=0, description="Max regen braking deceleration m/s^2")
    aux_power_w: float = Field(default=300.0, ge=0, description="Baseline auxiliary power draw in watts")
    hvac_max_power_w: float = Field(default=5000.0, ge=0, description="Max HVAC power in watts")
    max_speed_kmh: float = Field(default=200.0, gt=0)
    wltp_range_km: float = Field(gt=0, description="WLTP rated range in km")


class RouteSegment(BaseModel):
    """A segment of a route."""

    segment_id: int
    distance_m: float = Field(ge=0)
    elevation_start_m: float = 0.0
    elevation_end_m: float = 0.0
    speed_limit_kmh: float = Field(default=60.0, gt=0)
    expected_speed_kmh: float = Field(default=50.0, gt=0)
    traffic_factor: float = Field(default=1.0, ge=0.1, le=3.0, description="1.0=free flow, >1=congestion")
    road_grade_pct: Optional[float] = None  # computed from elevation if None

    @property
    def elevation_change_m(self) -> float:
        return self.elevation_end_m - self.elevation_start_m

    @property
    def grade(self) -> float:
        """Road grade as a fraction (rise/run)."""
        if self.road_grade_pct is not None:
            return self.road_grade_pct / 100.0
        if self.distance_m > 0:
            return self.elevation_change_m / self.distance_m
        return 0.0


class Route(BaseModel):
    """A driving route composed of segments."""

    route_id: str
    name: str
    segments: list[RouteSegment] = Field(default_factory=list)

    @property
    def total_distance_km(self) -> float:
        return sum(s.distance_m for s in self.segments) / 1000.0

    @property
    def total_elevation_gain_m(self) -> float:
        return sum(
            s.elevation_change_m for s in self.segments if s.elevation_change_m > 0
        )

    @property
    def total_elevation_loss_m(self) -> float:
        return abs(
            sum(
                s.elevation_change_m
                for s in self.segments
                if s.elevation_change_m < 0
            )
        )


class BatteryState(BaseModel):
    """Current state of the battery."""

    soc_pct: float = Field(ge=0, le=100, description="State of charge in percent")
    temperature_celsius: float = Field(default=25.0, description="Battery temperature")
    voltage_v: float = Field(default=400.0, gt=0)
    current_a: float = Field(default=0.0)
    capacity_kwh: float = Field(gt=0, description="Current usable capacity (accounts for degradation)")
    state_of_health_pct: float = Field(default=100.0, ge=0, le=100, description="Battery health percentage")
    cycle_count: int = Field(default=0, ge=0)
    energy_remaining_kwh: float = Field(ge=0)

    @property
    def energy_used_kwh(self) -> float:
        return self.capacity_kwh - self.energy_remaining_kwh


class RangeEstimate(BaseModel):
    """A range estimate with confidence bounds."""

    estimated_range_km: float = Field(ge=0)
    best_case_km: float = Field(ge=0)
    worst_case_km: float = Field(ge=0)
    energy_consumption_kwh_per_km: float = Field(ge=0)
    remaining_energy_kwh: float = Field(ge=0)
    factors: dict[str, float] = Field(
        default_factory=dict,
        description="Contribution of each factor to energy consumption",
    )
    driving_efficiency_score: float = Field(default=1.0, ge=0)
    climate_impact_kwh: float = Field(default=0.0, ge=0)
    elevation_impact_kwh: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=datetime.now)


class DrivingProfile(BaseModel):
    """A driver's driving style profile."""

    avg_acceleration_ms2: float = Field(default=1.5)
    avg_braking_decel_ms2: float = Field(default=2.0)
    avg_speed_kmh: float = Field(default=50.0)
    speed_variability: float = Field(default=0.15, description="Coefficient of variation of speed")
    hard_braking_events_per_km: float = Field(default=0.05)
    hard_accel_events_per_km: float = Field(default=0.05)
    regen_usage_pct: float = Field(default=70.0, ge=0, le=100)
    efficiency_score: float = Field(default=0.7, ge=0, le=1.0)


class WeatherCondition(BaseModel):
    """Weather conditions affecting range."""

    ambient_temp_celsius: float = Field(default=22.0)
    wind_speed_kmh: float = Field(default=0.0, ge=0)
    wind_direction_deg: float = Field(default=0.0, ge=0, lt=360)
    precipitation: bool = False
    humidity_pct: float = Field(default=50.0, ge=0, le=100)
