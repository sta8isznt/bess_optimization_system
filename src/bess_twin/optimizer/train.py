"""Optimizer training loop."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class OptimizerNetwork(nn.Module):
    """Neural network for optimization policy."""

    def __init__(self, input_dim: int, output_dim: int, config: Dict):
        super().__init__()
        hidden_layers = config.get("hidden_layers", [64, 64])
        dropout = config.get("dropout", 0.1)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class OptimizerTrainer:
    """Trainer for optimizer."""

    def __init__(self, objective, constraints, config: Dict, device: str = "cpu"):
        """Initialize optimizer trainer.
        
        Args:
            objective: Objective function
            constraints: Constraints
            config: Config dict
            device: torch device
        """
        self.objective = objective
        self.constraints = constraints
        self.config = config
        self.device = device

        # Create network
        input_dim = 4  # SoC, V, I, T
        output_dim = 1  # Power command

        self.network = OptimizerNetwork(input_dim, output_dim, config).to(device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.get("learning_rate", 0.001)
        )

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.network.train()
        total_loss = 0.0

        for batch_states in dataloader:
            states = batch_states.to(self.device)

            # Forward
            self.optimizer.zero_grad()
            actions = self.network(states)

            # Project to feasible
            actions = self.constraints.project(states, actions)

            # Compute cost
            cost = self.objective(states, actions)
            loss = cost.mean()

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Optimizer Epoch {epoch} - Loss: {avg_loss:.6f}")

        return {"loss": avg_loss}
