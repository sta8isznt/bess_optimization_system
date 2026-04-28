"""Battery simulator - physics-based synthetic data generation."""

import numpy as np
from typing import Tuple


class BatterySimulator:
    """ECM-based battery simulator."""

    def __init__(self, capacity: float, voltage_nominal: float, resistance: float, **kwargs):
        """Initialize simulator.
        
        Args:
            capacity: Battery capacity in Ah
            voltage_nominal: Nominal voltage in V
            resistance: Internal resistance in Ohms
        """
        self.capacity = capacity
        self.voltage_nominal = voltage_nominal
        self.resistance = resistance
        self.soc = kwargs.get("initial_soc", 0.5)
        self.temperature = kwargs.get("temperature", 25)
        self.noise_std = kwargs.get("noise_std", 0.01)

    def step(self, current: float, dt: float = 1.0) -> Tuple[float, float, float]:
        """Advance simulator by one step.
        
        Args:
            current: Current in Amps (positive = discharge)
            dt: Time step in seconds
            
        Returns:
            soc: State of charge (0-1)
            voltage: Terminal voltage in V
            power: Power in W
        """
        # Update SoC (coulomb counting)
        dq = current * dt / 3600  # Coulombs to Ah
        self.soc -= dq / self.capacity

        # Clamp SoC
        self.soc = np.clip(self.soc, 0.0, 1.0)

        # Simple voltage model
        ocv = self.voltage_nominal * self.soc
        voltage = ocv - self.resistance * current
        voltage = np.clip(voltage, 0.0, self.voltage_nominal * 1.2)

        # Power
        power = voltage * current

        # Add noise
        soc_noise = self.soc + np.random.normal(0, self.noise_std)
        voltage_noise = voltage + np.random.normal(0, self.noise_std)

        return float(np.clip(soc_noise, 0, 1)), float(voltage_noise), float(power)

    def reset(self, soc: float = 0.5):
        """Reset simulator state."""
        self.soc = soc
