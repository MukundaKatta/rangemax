"""Tests for RangeMax."""

from datetime import date

import numpy as np
import pytest

from rangemax.battery.estimator import RangeEstimator
from rangemax.battery.health import BatteryHealthTracker, HealthReading
from rangemax.battery.model import BatteryModel
from rangemax.models import (
    BatteryState,
    ClimateMode,
    DrivingProfile,
    EVehicle,
    Route,
    RouteSegment,
    WeatherCondition,
)
from rangemax.optimizer.climate import ClimateImpactEstimator
from rangemax.optimizer.driving import DrivingEvent, DrivingStyleAnalyzer
from rangemax.optimizer.route import RouteOptimizer
from rangemax.simulator import EVSimulator


# --- Fixtures ---


@pytest.fixture
def ev():
    return EVehicle(
        vehicle_id="test-ev",
        make="Test",
        model="EV",
        year=2024,
        battery_capacity_kwh=75.0,
        usable_capacity_kwh=70.0,
        curb_weight_kg=1800,
        frontal_area_m2=2.3,
        drag_coefficient=0.28,
        rolling_resistance_coeff=0.01,
        drivetrain_efficiency=0.90,
        regen_efficiency=0.65,
        max_regen_decel_ms2=1.5,
        aux_power_w=300,
        hvac_max_power_w=5000,
        max_speed_kmh=200,
        wltp_range_km=500,
    )


@pytest.fixture
def battery_model(ev):
    return BatteryModel(ev)


@pytest.fixture
def battery_state(battery_model):
    return battery_model.get_state(soc_pct=80.0, temperature_celsius=25.0)


@pytest.fixture
def sample_route():
    segments = [
        RouteSegment(segment_id=0, distance_m=2000, elevation_start_m=100, elevation_end_m=110, speed_limit_kmh=50, expected_speed_kmh=45),
        RouteSegment(segment_id=1, distance_m=5000, elevation_start_m=110, elevation_end_m=100, speed_limit_kmh=100, expected_speed_kmh=90),
        RouteSegment(segment_id=2, distance_m=3000, elevation_start_m=100, elevation_end_m=130, speed_limit_kmh=60, expected_speed_kmh=50),
    ]
    return Route(route_id="test-route", name="Test Route", segments=segments)


@pytest.fixture
def simulator():
    return EVSimulator(seed=42)


# --- Battery Model Tests ---


class TestBatteryModel:
    def test_ocv_increases_with_soc(self, battery_model):
        v_low = battery_model.open_circuit_voltage(0.1)
        v_mid = battery_model.open_circuit_voltage(0.5)
        v_high = battery_model.open_circuit_voltage(0.9)
        assert v_low < v_mid < v_high

    def test_internal_resistance_increases_cold(self, battery_model):
        r_warm = battery_model.internal_resistance(0.5, 25.0)
        r_cold = battery_model.internal_resistance(0.5, 0.0)
        r_very_cold = battery_model.internal_resistance(0.5, -15.0)
        assert r_warm < r_cold < r_very_cold

    def test_internal_resistance_increases_low_soc(self, battery_model):
        r_normal = battery_model.internal_resistance(0.5, 25.0)
        r_low = battery_model.internal_resistance(0.1, 25.0)
        assert r_low > r_normal

    def test_temperature_capacity_factor(self, battery_model):
        f_normal = battery_model.temperature_capacity_factor(25.0)
        f_cold = battery_model.temperature_capacity_factor(0.0)
        f_very_cold = battery_model.temperature_capacity_factor(-20.0)
        assert f_normal == 1.0
        assert f_cold < f_normal
        assert f_very_cold < f_cold

    def test_usable_capacity_degrades(self, battery_model):
        full = battery_model.usable_capacity_kwh()
        battery_model.state_of_health = 0.85
        degraded = battery_model.usable_capacity_kwh()
        assert degraded < full
        assert abs(degraded - 0.85 * 70.0) < 0.01

    def test_get_state(self, battery_model):
        state = battery_model.get_state(80.0, 25.0)
        assert state.soc_pct == 80.0
        assert state.energy_remaining_kwh > 0
        assert state.voltage_v > 300

    def test_energy_for_power_discharge(self, battery_model):
        energy = battery_model.compute_energy_for_power(10000, 3600, 0.5, 25.0)
        assert energy > 0  # positive = discharge
        # At 10kW for 1 hour, ~10kWh plus losses
        assert 9.5 < energy < 12.0

    def test_energy_for_power_regen(self, battery_model):
        energy = battery_model.compute_energy_for_power(-5000, 60, 0.5, 25.0)
        assert energy < 0  # negative = charging


# --- Range Estimator Tests ---


class TestRangeEstimator:
    def test_energy_per_km_positive(self, ev, battery_model):
        est = RangeEstimator(ev, battery_model)
        e = est.energy_per_km(50.0)
        assert e > 0

    def test_energy_increases_with_speed(self, ev, battery_model):
        est = RangeEstimator(ev, battery_model)
        e_slow = est.energy_per_km(50.0)
        e_fast = est.energy_per_km(130.0)
        assert e_fast > e_slow

    def test_uphill_uses_more_energy(self, ev, battery_model):
        est = RangeEstimator(ev, battery_model)
        e_flat = est.energy_per_km(60.0, grade_pct=0.0)
        e_uphill = est.energy_per_km(60.0, grade_pct=5.0)
        assert e_uphill > e_flat

    def test_downhill_uses_less_energy(self, ev, battery_model):
        est = RangeEstimator(ev, battery_model)
        e_flat = est.energy_per_km(60.0, grade_pct=0.0)
        e_downhill = est.energy_per_km(60.0, grade_pct=-5.0)
        assert e_downhill < e_flat

    def test_estimate_range(self, ev, battery_model, battery_state):
        est = RangeEstimator(ev, battery_model)
        result = est.estimate_range(battery_state, avg_speed_kmh=50.0)
        assert result.estimated_range_km > 0
        assert result.best_case_km >= result.estimated_range_km
        assert result.worst_case_km <= result.estimated_range_km

    def test_climate_reduces_range(self, ev, battery_model, battery_state):
        est = RangeEstimator(ev, battery_model)
        range_no_climate = est.estimate_range(battery_state, avg_speed_kmh=50.0)
        range_climate = est.estimate_range(
            battery_state, avg_speed_kmh=50.0, climate_power_w=3000
        )
        assert range_climate.estimated_range_km < range_no_climate.estimated_range_km

    def test_route_energy(self, ev, battery_model, battery_state, sample_route):
        est = RangeEstimator(ev, battery_model)
        result = est.estimate_route_energy(sample_route, battery_state)
        assert result["total_energy_kwh"] > 0
        assert "can_complete" in result
        assert "soc_at_end_pct" in result

    def test_headwind_increases_consumption(self, ev, battery_model):
        est = RangeEstimator(ev, battery_model)
        no_wind = est.energy_per_km(80.0)
        headwind = est.energy_per_km(
            80.0,
            weather=WeatherCondition(wind_speed_kmh=30, wind_direction_deg=0),
        )
        assert headwind > no_wind


# --- Route Optimizer Tests ---


class TestRouteOptimizer:
    def test_optimize_speed_profile(self, ev, battery_model, battery_state, sample_route):
        optimizer = RouteOptimizer(ev, battery_model)
        profile = optimizer.optimize_speed_profile(sample_route, battery_state)
        assert len(profile) == len(sample_route.segments)
        for p in profile:
            assert p["optimal_speed_kmh"] > 0
            assert p["energy_kwh"] is not None

    def test_eco_speed_reasonable(self, ev, battery_model):
        optimizer = RouteOptimizer(ev, battery_model)
        speed = optimizer.suggest_eco_speed()
        assert 20 <= speed <= 100  # eco speed should be moderate

    def test_compare_routes(self, ev, battery_model, battery_state, simulator):
        optimizer = RouteOptimizer(ev, battery_model)
        routes = [
            simulator.generate_route("City", 5, "city"),
            simulator.generate_route("Highway", 5, "highway"),
        ]
        comparison = optimizer.compare_routes(routes, battery_state)
        assert len(comparison) == 2
        # Should be sorted by energy
        assert comparison[0]["total_energy_kwh"] <= comparison[1]["total_energy_kwh"]


# --- Driving Style Analyzer Tests ---


class TestDrivingStyleAnalyzer:
    def test_eco_driving_scores_high(self, simulator):
        analyzer = DrivingStyleAnalyzer()
        events = simulator.generate_driving_events(600, "eco")
        score = analyzer.analyze(events)
        assert score.overall_score > 50

    def test_aggressive_driving_scores_low(self, simulator):
        analyzer = DrivingStyleAnalyzer()
        eco_events = simulator.generate_driving_events(600, "eco")
        agg_events = simulator.generate_driving_events(600, "aggressive")
        eco_score = analyzer.analyze(eco_events)
        agg_score = analyzer.analyze(agg_events)
        assert eco_score.overall_score > agg_score.overall_score

    def test_empty_events(self):
        analyzer = DrivingStyleAnalyzer()
        score = analyzer.analyze([])
        assert score.grade == "F"

    def test_produces_tips(self, simulator):
        analyzer = DrivingStyleAnalyzer()
        events = simulator.generate_driving_events(600, "normal")
        score = analyzer.analyze(events)
        assert len(score.tips) > 0

    def test_produces_profile(self, simulator):
        analyzer = DrivingStyleAnalyzer()
        events = simulator.generate_driving_events(600, "normal")
        score = analyzer.analyze(events)
        assert score.profile.avg_speed_kmh > 0
        assert 0 <= score.profile.efficiency_score <= 1.0


# --- Climate Impact Tests ---


class TestClimateImpactEstimator:
    def test_heating_power(self, ev):
        est = ClimateImpactEstimator(ev)
        weather = WeatherCondition(ambient_temp_celsius=-5.0)
        impact = est.estimate_hvac_power(weather, mode=ClimateMode.HEATING)
        assert impact.hvac_power_w > 0
        assert impact.range_reduction_pct > 0

    def test_cooling_power(self, ev):
        est = ClimateImpactEstimator(ev)
        weather = WeatherCondition(ambient_temp_celsius=38.0)
        impact = est.estimate_hvac_power(weather, mode=ClimateMode.COOLING)
        assert impact.hvac_power_w > 0

    def test_no_power_mild_weather(self, ev):
        est = ClimateImpactEstimator(ev)
        weather = WeatherCondition(ambient_temp_celsius=22.0)
        impact = est.estimate_hvac_power(weather, mode=ClimateMode.AUTO)
        assert impact.hvac_power_w == 0  # AUTO should be OFF at 22C

    def test_cold_uses_more_than_mild(self, ev):
        est = ClimateImpactEstimator(ev)
        cold = est.estimate_hvac_power(
            WeatherCondition(ambient_temp_celsius=-10.0), mode=ClimateMode.HEATING
        )
        mild = est.estimate_hvac_power(
            WeatherCondition(ambient_temp_celsius=10.0), mode=ClimateMode.HEATING
        )
        assert cold.hvac_power_w > mild.hvac_power_w

    def test_recommend_strategy(self, ev):
        est = ClimateImpactEstimator(ev)
        tips = est.recommend_strategy(WeatherCondition(ambient_temp_celsius=-15.0))
        assert len(tips) > 0


# --- Battery Health Tests ---


class TestBatteryHealth:
    def test_estimate_soh_new(self):
        tracker = BatteryHealthTracker(70.0)
        soh = tracker.estimate_soh(0, 0)
        assert soh == 100.0

    def test_soh_decreases_with_age(self):
        tracker = BatteryHealthTracker(70.0)
        soh_new = tracker.estimate_soh(0, 0)
        soh_3yr = tracker.estimate_soh(3, 900)
        soh_6yr = tracker.estimate_soh(6, 1800)
        assert soh_new > soh_3yr > soh_6yr

    def test_high_temp_degrades_faster(self):
        tracker = BatteryHealthTracker(70.0)
        soh_normal = tracker.estimate_soh(3, 900, 25.0)
        soh_hot = tracker.estimate_soh(3, 900, 40.0)
        assert soh_hot < soh_normal

    def test_generate_report_empty(self):
        tracker = BatteryHealthTracker(70.0)
        report = tracker.generate_report()
        assert report.current_soh_pct == 100.0
        assert report.health_grade == "A"

    def test_generate_report_with_data(self):
        tracker = BatteryHealthTracker(70.0)
        tracker.add_reading(HealthReading(
            date=date(2022, 1, 1), cycle_count=0, soh_pct=100.0, capacity_kwh=70.0
        ))
        tracker.add_reading(HealthReading(
            date=date(2024, 1, 1), cycle_count=600, soh_pct=95.0, capacity_kwh=66.5
        ))
        report = tracker.generate_report()
        assert report.current_soh_pct == 95.0
        assert report.capacity_loss_pct == 5.0
        assert report.degradation_rate_pct_per_year > 0

    def test_project_degradation(self):
        tracker = BatteryHealthTracker(70.0)
        tracker.add_reading(HealthReading(
            date=date(2022, 1, 1), cycle_count=0, soh_pct=100.0, capacity_kwh=70.0
        ))
        projections = tracker.project_degradation(years_ahead=5)
        assert len(projections) == 6  # 0 through 5
        # SoH should decrease over time
        sohs = [p[1] for p in projections]
        assert sohs == sorted(sohs, reverse=True)

    def test_health_grades(self):
        tracker = BatteryHealthTracker(70.0)
        tracker.add_reading(HealthReading(
            date=date(2020, 1, 1), cycle_count=0, soh_pct=100.0, capacity_kwh=70.0
        ))
        tracker.add_reading(HealthReading(
            date=date(2025, 1, 1), cycle_count=1500, soh_pct=82.0, capacity_kwh=57.4
        ))
        report = tracker.generate_report()
        assert report.health_grade == "D"


# --- Simulator Tests ---


class TestSimulator:
    def test_generate_route(self, simulator):
        route = simulator.generate_route("Test", 10, "city")
        assert len(route.segments) == 10
        assert route.total_distance_km > 0

    def test_generate_driving_events(self, simulator):
        events = simulator.generate_driving_events(300, "normal")
        assert len(events) > 0
        assert events[0].timestamp_s == 0.0

    def test_weather_scenarios(self, simulator):
        scenarios = simulator.generate_weather_scenarios()
        assert len(scenarios) >= 4

    def test_reference_vehicles(self, simulator):
        vehicles = simulator.get_all_vehicles()
        assert len(vehicles) >= 4
        for v in vehicles:
            assert v.battery_capacity_kwh > 0
            assert v.wltp_range_km > 0

    def test_health_history(self, simulator):
        tracker = simulator.generate_health_history(70.0, age_years=3.0)
        assert len(tracker.readings) > 0
        report = tracker.generate_report()
        assert report.current_soh_pct < 100.0
