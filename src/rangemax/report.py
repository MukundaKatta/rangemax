"""Rich console reporting for RangeMax."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rangemax.battery.health import HealthReport
from rangemax.models import BatteryState, EVehicle, RangeEstimate
from rangemax.optimizer.driving import DrivingScore


def print_vehicle_info(vehicle: EVehicle, console: Console | None = None) -> None:
    """Print vehicle specifications."""
    console = console or Console()

    table = Table(title=f"{vehicle.year} {vehicle.make} {vehicle.model}", show_lines=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Battery Capacity", f"{vehicle.battery_capacity_kwh} kWh")
    table.add_row("Usable Capacity", f"{vehicle.usable_capacity_kwh} kWh")
    table.add_row("WLTP Range", f"{vehicle.wltp_range_km} km")
    table.add_row("Curb Weight", f"{vehicle.curb_weight_kg} kg")
    table.add_row("Drag Coefficient", f"{vehicle.drag_coefficient}")
    table.add_row("Frontal Area", f"{vehicle.frontal_area_m2} m^2")
    table.add_row("Rolling Resistance", f"{vehicle.rolling_resistance_coeff}")
    table.add_row("Drivetrain Efficiency", f"{vehicle.drivetrain_efficiency:.0%}")
    table.add_row("Regen Efficiency", f"{vehicle.regen_efficiency:.0%}")

    console.print(table)


def print_range_estimate(
    estimate: RangeEstimate,
    vehicle_name: str = "",
    console: Console | None = None,
) -> None:
    """Print range estimate."""
    console = console or Console()

    title = "Range Estimate"
    if vehicle_name:
        title += f" - {vehicle_name}"

    range_color = "green" if estimate.estimated_range_km > 200 else ("yellow" if estimate.estimated_range_km > 100 else "red")

    panel_text = (
        f"[bold {range_color}]{estimate.estimated_range_km:.0f} km[/] estimated range\n"
        f"Best case: {estimate.best_case_km:.0f} km  |  Worst case: {estimate.worst_case_km:.0f} km\n"
        f"Consumption: {estimate.energy_consumption_kwh_per_km * 1000:.0f} Wh/km  |  "
        f"Remaining: {estimate.remaining_energy_kwh:.1f} kWh\n"
        f"Driving efficiency: {estimate.driving_efficiency_score:.0%}"
    )

    if estimate.climate_impact_kwh > 0:
        panel_text += f"  |  Climate impact: {estimate.climate_impact_kwh:.1f} kWh"

    console.print(Panel(panel_text, title=title))

    if estimate.factors:
        table = Table(title="Energy Breakdown", show_lines=False)
        table.add_column("Factor", style="cyan")
        table.add_column("kWh/km", justify="right")
        for factor, value in estimate.factors.items():
            table.add_row(factor.replace("_", " ").title(), f"{value:.4f}")
        console.print(table)


def print_battery_state(
    state: BatteryState, console: Console | None = None
) -> None:
    """Print battery state."""
    console = console or Console()

    soc_color = "green" if state.soc_pct > 50 else ("yellow" if state.soc_pct > 20 else "red")

    table = Table(title="Battery State", show_lines=True)
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("State of Charge", f"[{soc_color}]{state.soc_pct:.1f}%[/]")
    table.add_row("Energy Remaining", f"{state.energy_remaining_kwh:.1f} kWh")
    table.add_row("Capacity", f"{state.capacity_kwh:.1f} kWh")
    table.add_row("Voltage", f"{state.voltage_v:.1f} V")
    table.add_row("Temperature", f"{state.temperature_celsius:.1f} C")
    table.add_row("State of Health", f"{state.state_of_health_pct:.1f}%")
    table.add_row("Cycle Count", str(state.cycle_count))

    console.print(table)


def print_driving_score(
    score: DrivingScore, console: Console | None = None
) -> None:
    """Print driving efficiency score."""
    console = console or Console()

    grade_colors = {"A+": "bold green", "A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "bold red"}
    color = grade_colors.get(score.grade, "white")

    console.print(
        Panel(
            f"[{color}]Grade: {score.grade}  |  Score: {score.overall_score:.0f}/100[/]",
            title="Driving Efficiency",
        )
    )

    table = Table(show_lines=False)
    table.add_column("Category", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Bar")

    for name, val in [
        ("Acceleration", score.acceleration_score),
        ("Braking", score.braking_score),
        ("Speed Consistency", score.speed_consistency_score),
        ("Regen Usage", score.regen_score),
    ]:
        bar_len = int(val / 5)
        bar = "[green]" + "#" * bar_len + "[/]" + "-" * (20 - bar_len)
        table.add_row(name, f"{val:.0f}", bar)

    console.print(table)

    if score.tips:
        console.print("\n[bold]Tips:[/]")
        for tip in score.tips:
            console.print(f"  - {tip}")


def print_health_report(
    report: HealthReport, console: Console | None = None
) -> None:
    """Print battery health report."""
    console = console or Console()

    grade_colors = {"A": "green", "B": "cyan", "C": "yellow", "D": "red", "F": "bold red"}
    color = grade_colors.get(report.health_grade, "white")

    table = Table(title="Battery Health Report", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Health Grade", f"[{color}]{report.health_grade}[/]")
    table.add_row("State of Health", f"{report.current_soh_pct:.1f}%")
    table.add_row("Current Capacity", f"{report.current_capacity_kwh:.1f} kWh")
    table.add_row("Original Capacity", f"{report.original_capacity_kwh:.1f} kWh")
    table.add_row("Capacity Loss", f"{report.capacity_loss_pct:.1f}%")
    table.add_row("Cycle Count", str(report.cycle_count))
    table.add_row("Degradation/Year", f"{report.degradation_rate_pct_per_year:.2f}%")
    table.add_row("Degradation/Cycle", f"{report.degradation_rate_pct_per_cycle:.4f}%")
    if report.projected_80pct_date:
        table.add_row("80% SoH Date", str(report.projected_80pct_date))
    if report.projected_80pct_cycles:
        table.add_row("80% SoH Cycles", str(report.projected_80pct_cycles))

    console.print(table)
    console.print(f"\n[bold]Recommendation:[/] {report.recommendation}")


def print_route_comparison(
    analyses: list[dict[str, float]],
    console: Console | None = None,
) -> None:
    """Print route comparison table."""
    console = console or Console()

    table = Table(title="Route Comparison", show_lines=True)
    table.add_column("Route", style="cyan")
    table.add_column("Distance", justify="right")
    table.add_column("Elev. Gain", justify="right")
    table.add_column("Energy", justify="right")
    table.add_column("Wh/km", justify="right")
    table.add_column("SoC at End", justify="right")
    table.add_column("Feasible", justify="center")

    for a in analyses:
        feasible = "[green]YES[/]" if a["can_complete"] else "[red]NO[/]"
        table.add_row(
            str(a.get("route_name", a["route_id"])),
            f"{a['distance_km']:.1f} km",
            f"{a['elevation_gain_m']:.0f} m",
            f"{a['total_energy_kwh']:.2f} kWh",
            f"{a['avg_consumption_kwh_per_km'] * 1000:.0f}",
            f"{a['soc_at_end_pct']:.1f}%",
            feasible,
        )

    console.print(table)
