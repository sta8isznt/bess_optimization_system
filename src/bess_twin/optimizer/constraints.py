"""Operational constraints for optimization."""

import torch
from typing import Dict


class OperationalConstraints:
    """Enforce operational constraints."""

    def __init__(self, config: Dict):
        self.config = config
        self.max_discharge = config.get("max_discharge_rate", 0.5)
        self.max_charge = config.get("max_charge_rate", 0.5)
        self.min_soc = config.get("min_soc", 0.1)
        self.max_soc = config.get("max_soc", 0.9)
        self.power_ramp = config.get("power_ramp_limit", 10.0)

    def check_feasible(self, state: torch.Tensor, action: torch.Tensor) -> bool:
        """Check if state-action is feasible."""
        # SoC bounds
        if state.shape[-1] > 0:
            soc = state[..., 0]
            if (soc < self.min_soc).any() or (soc > self.max_soc).any():
                return False

        # Power bounds
        if (torch.abs(action) > self.max_discharge).any():
            return False

        return True

    def project(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Project action to feasible set."""
        action = torch.clamp(action, -self.max_discharge, self.max_charge)
        return action
