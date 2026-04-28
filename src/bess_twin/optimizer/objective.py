"""Objective function for optimization."""

import torch
import torch.nn as nn
from typing import Dict


class OptimizationObjective(nn.Module):
    """Objective to optimize."""

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.cost_weights = config.get("cost_weights", {})

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute cost/reward.
        
        Args:
            state: Current state [batch, features]
            action: Action [batch, actions]
            
        Returns:
            cost: Cost to minimize
        """
        # Simple cost = weighted combination of objectives
        cost = torch.zeros(state.shape[0], device=state.device)

        # Energy cost
        energy_weight = self.cost_weights.get("energy", 0.5)
        energy_cost = (action ** 2).sum(dim=1)
        cost = cost + energy_weight * energy_cost

        # Degradation penalty (SoC-based)
        deg_weight = self.cost_weights.get("degradation", 0.3)
        if state.shape[-1] > 0:
            soc = state[..., 0]
            degradation = torch.abs(soc - 0.5) * 0.1
            cost = cost + deg_weight * degradation.mean()

        return cost
