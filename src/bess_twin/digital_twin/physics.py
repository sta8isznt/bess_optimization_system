"""Physics-informed loss functions for TimeGAN."""

import torch
import torch.nn as nn


class PhysicsLoss(nn.Module):
    """Physics-constrained losses for battery dynamics."""

    def __init__(self, 
                 soc_weight: float = 1.0,
                 voltage_weight: float = 0.5,
                 smooth_weight: float = 0.1,
                 min_voltage: float = 40.0,
                 max_voltage: float = 55.0):
        super().__init__()
        self.soc_weight = soc_weight
        self.voltage_weight = voltage_weight
        self.smooth_weight = smooth_weight
        self.min_voltage = min_voltage
        self.max_voltage = max_voltage

    def forward(self, synthetic: torch.Tensor, real: torch.Tensor = None) -> torch.Tensor:
        """Compute physics loss.
        
        Args:
            synthetic: Generated sequences [batch, seq_len, features]
            real: Real sequences (optional)
            
        Returns:
            loss: Scalar loss value
        """
        loss = torch.tensor(0.0, device=synthetic.device)

        # SoC consistency (column 0)
        if synthetic.shape[-1] > 0:
            soc = synthetic[..., 0]
            soc_penalty = torch.nn.functional.relu(soc - 1.0) + torch.nn.functional.relu(-soc)
            loss = loss + self.soc_weight * soc_penalty.mean()

        # Voltage bounds (column 1)
        if synthetic.shape[-1] > 1:
            voltage = synthetic[..., 1]
            v_lower = torch.nn.functional.relu(self.min_voltage - voltage)
            v_upper = torch.nn.functional.relu(voltage - self.max_voltage)
            loss = loss + self.voltage_weight * (v_lower.mean() + v_upper.mean())

        # Smoothness regularization
        if synthetic.shape[1] > 1:
            diff = torch.diff(synthetic, dim=1)
            smooth_penalty = (diff ** 2).mean()
            loss = loss + self.smooth_weight * smooth_penalty

        return loss


def compute_physics_aware_loss(real: torch.Tensor, 
                               fake: torch.Tensor,
                               physics_loss_fn: PhysicsLoss) -> torch.Tensor:
    """Combined physics + adversarial loss."""
    return physics_loss_fn(fake) + physics_loss_fn(real)
