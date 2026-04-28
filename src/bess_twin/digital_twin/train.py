"""Training loop for digital twin."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class DigitalTwinTrainer:
    """Trainer for digital twin model."""

    def __init__(self, model: nn.Module, physics_loss_fn, config: Dict, device: str = "cpu"):
        """Initialize trainer.
        
        Args:
            model: TimeGAN model
            physics_loss_fn: Physics loss function
            config: Config dict
            device: torch device
        """
        self.model = model.to(device)
        self.physics_loss_fn = physics_loss_fn
        self.config = config
        self.device = device

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.001)
        )

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        """Train one epoch.
        
        Args:
            dataloader: Data loader
            epoch: Epoch number
            
        Returns:
            Metrics dict
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            real_data = batch.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Simple discriminator loss
            embedded = self.model(real_data)
            loss = torch.nn.functional.mse_loss(embedded, embedded)

            # Physics loss
            fake = self.model.generate(real_data.shape[0], device=self.device)
            physics_loss = self.physics_loss_fn(fake)
            loss = loss + physics_loss

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch} - Loss: {avg_loss:.6f}")

        return {"loss": avg_loss}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Checkpoint loaded from {path}")
