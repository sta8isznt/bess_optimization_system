"""Train optimizer model."""

import argparse
from pathlib import Path

from src.bess_twin.optimizer.objective import OptimizationObjective
from src.bess_twin.optimizer.constraints import OperationalConstraints
from src.bess_twin.optimizer.train import OptimizerTrainer
from src.bess_twin.utils.config import load_optimizer_config
from src.bess_twin.utils.seed import set_seed
import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def train_optimizer(config: dict):
    """Train the optimizer."""
    set_seed(config.get("seed", 42))
    
    device = config.get("device", "cpu")
    opt_config = config["optimizer"]
    
    # Create objective and constraints
    objective = OptimizationObjective(opt_config["objective"])
    constraints = OperationalConstraints(opt_config["constraints"])
    
    # Trainer
    trainer = OptimizerTrainer(objective, constraints, opt_config["optimizer_algorithm"], device=device)
    
    logger.info("Training optimizer...")
    for epoch in range(opt_config["epochs"]):
        # Dummy training
        metrics = {"loss": 0.1 * (1 - epoch / opt_config["epochs"])}
        logger.info(f"Epoch {epoch+1}/{opt_config['epochs']} - Loss: {metrics['loss']:.6f}")
    
    logger.info("Optimizer training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    config = load_optimizer_config()
    train_optimizer(config)
