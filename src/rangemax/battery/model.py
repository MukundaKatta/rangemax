"""Battery model with state-of-charge, degradation, and temperature effects."""

from __future__ import annotations

import numpy as np

from rangemax.models import BatteryState, EVehicle


class BatteryModel:
    """Physics-based battery model for EVs.

    Models:
    - Open-circuit voltage as function of SoC (polynomial fit for NMC cells)
    - Internal resistance varying with temperature and SoC
    - Temperature effects on capacity and efficiency
    - Coulombic efficiency losses
    """

    # NMC cell OCV polynomial coefficients (SoC 0-1 -> voltage per cell ~3.0-4.2V)
    # Approximation for a typical NMC 622 cell
    OCV_COEFFS = np.array([-0.5, 2.5, -3.8, 3.1, 0.2, 3.0])
    # Nominal pack voltage at 50% SoC
    NOMINAL_PACK_VOLTAGE = 400.0
    # Number of series cells (approx for 400V pack with NMC)
    NUM_SERIES_CELLS = 108

    # Internal resistance at 25C, 50% SoC (ohms per cell)
    BASE_CELL_RESISTANCE = 0.002
    # Coulombic efficiency
    COULOMBIC_EFFICIENCY = 0.995

    def __init__(self, vehicle: EVehicle) -> None:
        self.vehicle = vehicle
        self._soh = 1.0  # State of health (fraction)

    @property
    def state_of_health(self) -> float:
        return self._soh

    @state_of_health.setter
    def state_of_health(self, value: float) -> None:
        self._soh = max(0.0, min(1.0, value))

    def usable_capacity_kwh(self) -> float:
        """Current usable capacity accounting for degradation."""
        return self.vehicle.usable_capacity_kwh * self._soh

    def open_circuit_voltage(self, soc: float) -> float:
        """Compute pack open-circuit voltage from SoC.

        Args:
            soc: State of charge as fraction (0-1).

        Returns:
            Pack voltage in volts.
        """
        soc = np.clip(soc, 0.0, 1.0)
        # Cell voltage from polynomial
        cell_v = float(np.polyval(self.OCV_COEFFS, soc))
        cell_v = np.clip(cell_v, 2.8, 4.2)
        return cell_v * self.NUM_SERIES_CELLS

    def internal_resistance(
        self, soc: float, temperature_celsius: float
    ) -> float:
        """Compute pack internal resistance.

        Resistance increases at low SoC, low temperature, and with degradation.

        Args:
            soc: State of charge as fraction (0-1).
            temperature_celsius: Battery temperature.

        Returns:
            Pack resistance in ohms.
        """
        # SoC effect: resistance increases below 20% and above 90%
        soc_factor = 1.0
        if soc < 0.2:
            soc_factor = 1.0 + 2.0 * (0.2 - soc)
        elif soc > 0.9:
            soc_factor = 1.0 + 0.5 * (soc - 0.9)

        # Temperature effect: Arrhenius-like
        # Doubles roughly every 15C below 25C
        temp_factor = 1.0
        if temperature_celsius < 25.0:
            temp_factor = 2.0 ** ((25.0 - temperature_celsius) / 15.0)
        elif temperature_celsius > 40.0:
            temp_factor = 0.9  # slightly lower at high temp

        # Degradation increases resistance
        degradation_factor = 1.0 + 0.5 * (1.0 - self._soh)

        cell_r = (
            self.BASE_CELL_RESISTANCE
            * soc_factor
            * temp_factor
            * degradation_factor
        )
        return cell_r * self.NUM_SERIES_CELLS

    def temperature_capacity_factor(self, temperature_celsius: float) -> float:
        """Fraction of nominal capacity available at given temperature.

        Capacity drops significantly below 0C and slightly above 45C.
        """
        if temperature_celsius >= 20.0 and temperature_celsius <= 35.0:
            return 1.0
        if temperature_celsius < 20.0:
            # Linear drop: ~1% per degree below 20C, accelerating below 0C
            if temperature_celsius >= 0.0:
                return max(0.8, 1.0 - 0.01 * (20.0 - temperature_celsius))
            else:
                return max(0.5, 0.8 - 0.015 * abs(temperature_celsius))
        # High temperature: slight capacity increase but bad for health
        return max(0.95, 1.0 - 0.005 * (temperature_celsius - 35.0))

    def compute_energy_for_power(
        self,
        power_w: float,
        duration_s: float,
        soc: float,
        temperature_celsius: float,
    ) -> float:
        """Compute energy drawn from battery for a given power demand.

        Accounts for internal resistance losses.

        Args:
            power_w: Power demand in watts (positive = discharge, negative = charge/regen).
            duration_s: Duration in seconds.
            soc: Current SoC as fraction.
            temperature_celsius: Battery temperature.

        Returns:
            Energy in kWh drawn from battery (positive = discharge).
        """
        r_pack = self.internal_resistance(soc, temperature_celsius)
        v_oc = self.open_circuit_voltage(soc)

        if power_w >= 0:
            # Discharging: battery provides power + resistive losses
            # P = V*I - I^2*R => solve for I
            # V*I = P + I^2*R => R*I^2 - V*I + P = 0
            discriminant = v_oc**2 - 4 * r_pack * power_w
            if discriminant < 0:
                # Cannot deliver this power
                current = v_oc / (2 * r_pack)
            else:
                current = (v_oc - np.sqrt(discriminant)) / (2 * r_pack)
            total_power = power_w + current**2 * r_pack
        else:
            # Charging (regen): battery receives power minus losses
            regen_power = abs(power_w)
            discriminant = v_oc**2 + 4 * r_pack * regen_power
            current = (-v_oc + np.sqrt(discriminant)) / (2 * r_pack)
            total_power = -(regen_power - current**2 * r_pack)
            total_power *= self.COULOMBIC_EFFICIENCY

        energy_kwh = total_power * duration_s / 3_600_000.0
        return energy_kwh

    def get_state(
        self,
        soc_pct: float,
        temperature_celsius: float = 25.0,
        cycle_count: int = 0,
    ) -> BatteryState:
        """Get current battery state."""
        soc_frac = soc_pct / 100.0
        capacity = self.usable_capacity_kwh()
        temp_factor = self.temperature_capacity_factor(temperature_celsius)
        effective_capacity = capacity * temp_factor
        energy_remaining = effective_capacity * soc_frac

        return BatteryState(
            soc_pct=soc_pct,
            temperature_celsius=temperature_celsius,
            voltage_v=round(self.open_circuit_voltage(soc_frac), 1),
            current_a=0.0,
            capacity_kwh=round(effective_capacity, 2),
            state_of_health_pct=round(self._soh * 100, 1),
            cycle_count=cycle_count,
            energy_remaining_kwh=round(energy_remaining, 2),
        )
