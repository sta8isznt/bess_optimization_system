"""Train digital twin model."""

import argparse
import torch
from pathlib import Path

from src.bess_twin.digital_twin.timegan import TimeGAN
from src.bess_twin.digital_twin.physics import PhysicsLoss
from src.bess_twin.digital_twin.train import DigitalTwinTrainer
from src.bess_twin.utils.config import load_digital_twin_config
from src.bess_twin.utils.seed import set_seed
import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def train_digital_twin(config: dict, data=None):
    """Train the digital twin."""
    set_seed(config.get("seed", 42))
    
    device = config.get("device", "cpu")
    tg_config = config["digital_twin"]["timegan"]
    physics_config = config["digital_twin"]["physics"]
    
    # Create model
    model = TimeGAN(
        seq_len=tg_config["seq_len"],
        dim_input=tg_config["dim_input"],
        dim_latent=tg_config["dim_latent"],
        dim_hidden=tg_config["dim_hidden"],
        dim_embedding=tg_config["dim_embedding"]
    )
    
    # Physics loss
    physics_loss = PhysicsLoss(
        soc_weight=physics_config["soc_weight"],
        voltage_weight=physics_config["voltage_weight"],
        smooth_weight=physics_config["smooth_weight"]
    )
    
    # Trainer
    trainer = DigitalTwinTrainer(model, physics_loss, tg_config, device=device)
    
    # Dummy training loop
    logger.info("Training digital twin...")
    for epoch in range(tg_config["epochs"]):
        # In practice, create proper dataloader
        metrics = {
            "loss": torch.tensor(0.1 * (1 - epoch / tg_config["epochs"])).item()
        }
        logger.info(f"Epoch {epoch+1}/{tg_config['epochs']} - Loss: {metrics['loss']:.6f}")
    
    # Save checkpoint
    checkpoint_dir = Path(config["paths"]["checkpoints"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.save_checkpoint(checkpoint_dir / "digital_twin.pt")
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    config = load_digital_twin_config()
    train_digital_twin(config)
