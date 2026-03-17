"""CLI interface for RangeMax."""

from __future__ import annotations

import click
from rich.console import Console

from rangemax.battery.estimator import RangeEstimator
from rangemax.battery.model import BatteryModel
from rangemax.models import DrivingProfile, WeatherCondition
from rangemax.optimizer.climate import ClimateImpactEstimator
from rangemax.optimizer.driving import DrivingStyleAnalyzer
from rangemax.optimizer.route import RouteOptimizer
from rangemax.report import (
    print_battery_state,
    print_driving_score,
    print_health_report,
    print_range_estimate,
    print_route_comparison,
    print_vehicle_info,
)
from rangemax.simulator import EVSimulator

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="rangemax")
def cli() -> None:
    """RangeMax - EV Range Optimizer.

    Predict and maximize electric vehicle range under real-world conditions.
    """


@cli.command()
@click.option("--vehicle", "-v", default=0, help="Vehicle index (0-3).")
@click.option("--soc", default=80.0, help="Starting state of charge (%).")
@click.option("--temp", default=22.0, help="Ambient temperature (C).")
@click.option("--seed", default=42, help="Random seed.")
def demo(vehicle: int, soc: float, temp: float, seed: int) -> None:
    """Run a full demonstration."""
    console.print("[bold cyan]RangeMax Demo[/]")
    console.print("=" * 60)

    sim = EVSimulator(seed=seed)
    ev = sim.get_vehicle(vehicle)
    print_vehicle_info(ev, console)

    # Battery model and state
    battery = BatteryModel(ev)
    state = battery.get_state(soc_pct=soc, temperature_celsius=temp)
    print_battery_state(state, console)

    # Range estimate
    estimator = RangeEstimator(ev, battery)
    weather = WeatherCondition(ambient_temp_celsius=temp)
    profile = DrivingProfile(efficiency_score=0.75)

    # Climate impact
    climate_est = ClimateImpactEstimator(ev)
    climate = climate_est.estimate_hvac_power(weather)
    console.print(f"\n[bold]HVAC Power:[/] {climate.hvac_power_w:.0f}W  |  Range impact: -{climate.range_reduction_pct:.1f}%")

    estimate = estimator.estimate_range(
        battery_state=state,
        driving_profile=profile,
        weather=weather,
        climate_power_w=climate.hvac_power_w,
        avg_speed_kmh=50.0,
    )
    print_range_estimate(estimate, f"{ev.make} {ev.model}", console)

    # Route comparison
    console.print("\n[bold]Route Comparison[/]")
    routes = [
        sim.generate_route("City Commute", 10, "city"),
        sim.generate_route("Highway Trip", 8, "highway"),
        sim.generate_route("Mountain Drive", 12, "mountain"),
    ]
    optimizer = RouteOptimizer(ev, battery)
    comparison = optimizer.compare_routes(routes, state, profile, weather, climate.hvac_power_w)
    print_route_comparison(comparison, console)

    # Driving analysis
    console.print("\n[bold]Driving Style Analysis[/]")
    analyzer = DrivingStyleAnalyzer()
    events = sim.generate_driving_events(1800, "normal")
    score = analyzer.analyze(events)
    print_driving_score(score, console)

    # Battery health
    console.print("\n[bold]Battery Health[/]")
    tracker = sim.generate_health_history(ev.usable_capacity_kwh, age_years=3.0)
    report = tracker.generate_report()
    print_health_report(report, console)


@cli.command()
@click.option("--vehicle", "-v", default=0, help="Vehicle index (0-3).")
@click.option("--soc", default=80.0, help="State of charge (%).")
@click.option("--speed", default=50.0, help="Average speed (km/h).")
@click.option("--temp", default=22.0, help="Ambient temperature (C).")
def range_est(vehicle: int, soc: float, speed: float, temp: float) -> None:
    """Estimate remaining range."""
    sim = EVSimulator()
    ev = sim.get_vehicle(vehicle)
    battery = BatteryModel(ev)
    state = battery.get_state(soc_pct=soc, temperature_celsius=temp)

    estimator = RangeEstimator(ev, battery)
    weather = WeatherCondition(ambient_temp_celsius=temp)

    climate_est = ClimateImpactEstimator(ev)
    climate = climate_est.estimate_hvac_power(weather)

    estimate = estimator.estimate_range(
        battery_state=state,
        weather=weather,
        climate_power_w=climate.hvac_power_w,
        avg_speed_kmh=speed,
    )
    print_vehicle_info(ev, console)
    print_range_estimate(estimate, f"{ev.make} {ev.model}", console)


@cli.command()
@click.option("--vehicle", "-v", default=0, help="Vehicle index (0-3).")
@click.option("--route-type", "-r", default="mixed", help="Route type: city/highway/mountain/mixed.")
@click.option("--soc", default=80.0, help="Starting SoC (%).")
@click.option("--seed", default=42, help="Random seed.")
def optimize(vehicle: int, route_type: str, soc: float, seed: int) -> None:
    """Optimize route speed profile for minimum energy."""
    sim = EVSimulator(seed=seed)
    ev = sim.get_vehicle(vehicle)
    battery = BatteryModel(ev)
    state = battery.get_state(soc_pct=soc)

    route = sim.generate_route(f"{route_type.title()} Route", 10, route_type)
    optimizer = RouteOptimizer(ev, battery)
    profile = optimizer.optimize_speed_profile(route, state)

    from rich.table import Table
    table = Table(title=f"Optimized Speed Profile - {route.name}", show_lines=True)
    table.add_column("Seg", justify="center")
    table.add_column("Distance", justify="right")
    table.add_column("Grade", justify="right")
    table.add_column("Optimal Speed", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Remaining", justify="right")

    for p in profile:
        table.add_row(
            str(p["segment_id"]),
            f"{p['distance_km']:.2f} km",
            f"{p['grade_pct']:.1f}%",
            f"{p['optimal_speed_kmh']:.0f} km/h",
            f"{p['energy_kwh']:.3f} kWh",
            f"{p['time_minutes']:.1f} min",
            f"{p['remaining_energy_kwh']:.1f} kWh",
        )

    console.print(table)
    total_energy = sum(p["energy_kwh"] for p in profile)
    total_time = sum(p["time_minutes"] for p in profile)
    console.print(f"\n[bold]Total energy:[/] {total_energy:.2f} kWh  |  [bold]Total time:[/] {total_time:.0f} min")


@cli.command()
@click.option("--vehicle", "-v", default=0, help="Vehicle index (0-3).")
def vehicles(vehicle: int) -> None:
    """List available reference vehicles."""
    sim = EVSimulator()
    for i, ev in enumerate(sim.get_all_vehicles()):
        marker = " <--" if i == vehicle else ""
        print_vehicle_info(ev, console)
        if marker:
            console.print(f"  [bold cyan]Selected{marker}[/]")
        console.print()


@cli.command()
@click.option("--vehicle", "-v", default=0, help="Vehicle index (0-3).")
@click.option("--years", default=3.0, help="Vehicle age in years.")
@click.option("--cycles", default=300, help="Annual charge cycles.")
def health(vehicle: int, years: float, cycles: int) -> None:
    """Show battery health analysis."""
    sim = EVSimulator()
    ev = sim.get_vehicle(vehicle)
    tracker = sim.generate_health_history(ev.usable_capacity_kwh, age_years=years, annual_cycles=cycles)
    report = tracker.generate_report()
    print_health_report(report, console)

    # Projections
    projections = tracker.project_degradation(years_ahead=5, annual_cycles=cycles)
    from rich.table import Table
    table = Table(title="Degradation Projection", show_lines=False)
    table.add_column("Year", justify="right")
    table.add_column("Projected SoH", justify="right")
    for yr, soh in projections:
        color = "green" if soh > 90 else ("yellow" if soh > 80 else "red")
        table.add_row(f"+{yr:.0f}", f"[{color}]{soh:.1f}%[/]")
    console.print(table)


if __name__ == "__main__":
    cli()
