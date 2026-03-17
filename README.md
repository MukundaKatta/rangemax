# RangeMax

EV Range Optimizer -- predict and maximize electric vehicle range under real-world conditions.

## Features

- **Range Estimation**: Physics-based energy consumption model with aerodynamic drag, rolling resistance, grade resistance, and regenerative braking.
- **Route Optimization**: Compute energy-optimal speed profiles considering elevation, traffic, and weather.
- **Driving Style Analysis**: Score driving efficiency across acceleration, braking, speed consistency, and regen usage.
- **Climate Impact**: Estimate battery drain from heating, cooling, and defrost systems with heat pump and resistive heater modeling.
- **Battery Model**: State-of-charge tracking with temperature-dependent capacity, internal resistance, and Coulombic losses.
- **Battery Health**: Monitor capacity degradation using calendar + cycle aging models with Arrhenius temperature acceleration.
- **Reference Vehicles**: Includes real-world parameters for Tesla Model 3 LR, Hyundai IONIQ 5, BMW iX, and Chevrolet Equinox EV.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Full demo with range estimate, route comparison, driving score, and battery health
rangemax demo --vehicle 0 --soc 80 --temp 22

# Estimate remaining range
rangemax range-est --vehicle 0 --soc 60 --speed 80 --temp 5

# Optimize route speed profile
rangemax optimize --vehicle 1 --route-type highway --soc 90

# List reference vehicles
rangemax vehicles

# Battery health analysis
rangemax health --vehicle 0 --years 4 --cycles 350
```

## Project Structure

```
src/rangemax/
  cli.py              # Click CLI
  models.py            # Pydantic models (EVehicle, Route, BatteryState, RangeEstimate)
  simulator.py         # EV data simulator with reference vehicles
  report.py            # Rich console reporting
  optimizer/
    route.py           # RouteOptimizer (energy-optimal speed profiles)
    driving.py         # DrivingStyleAnalyzer (efficiency scoring)
    climate.py         # ClimateImpactEstimator (HVAC battery drain)
  battery/
    model.py           # BatteryModel (OCV, internal resistance, temperature effects)
    estimator.py       # RangeEstimator (physics-based range prediction)
    health.py          # BatteryHealthTracker (degradation monitoring)
```

## EV Physics

The energy consumption model accounts for:

- **Aerodynamic drag**: F = 0.5 * rho * Cd * A * v^2 (including wind effects)
- **Rolling resistance**: F = Crr * m * g * cos(theta)
- **Grade resistance**: F = m * g * sin(theta)
- **Regenerative braking**: Energy recovery limited by max regen deceleration and regen efficiency
- **Battery losses**: Internal resistance modeled with SoC and temperature dependence
- **HVAC loads**: Heat pump COP modeling with fallback to resistive heating below -10C

## Dependencies

- numpy, scipy -- numerical computation and curve fitting
- pydantic -- data validation and models
- click -- CLI framework
- rich -- terminal reporting

## Author

Mukunda Katta
