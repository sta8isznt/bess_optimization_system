"""Generate synthetic battery data."""

import argparse
import numpy as np
import torch
from pathlib import Path

from src.bess_twin.digital_twin.simulator import BatterySimulator
from src.bess_twin.utils.config import load_digital_twin_config
from src.bess_twin.utils.seed import set_seed
import logging

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def generate_synthetic_data(config: dict, num_trajectories: int = 100):
    """Generate synthetic trajectories."""
    sim_config = config["digital_twin"]["simulator"]
    
    set_seed(config.get("seed", 42))
    
    simulator = BatterySimulator(
        capacity=sim_config["capacity"],
        voltage_nominal=sim_config["nominal_voltage"],
        resistance=sim_config["internal_resistance"],
        initial_soc=sim_config["initial_soc"],
        noise_std=sim_config["noise_std"]
    )
    
    trajectories = []
    
    for traj_idx in range(num_trajectories):
        simulator.reset(soc=np.random.uniform(0.2, 0.8))
        
        trajectory = []
        trajectory_len = sim_config["trajectory_length"]
        
        for _ in range(trajectory_len):
            # Random current profile
            current = np.random.uniform(-10, 10)  # Amps
            
            soc, voltage, power = simulator.step(current)
            
            trajectory.append([soc, voltage, current, simulator.temperature])
        
        trajectories.append(np.array(trajectory))
        
        if (traj_idx + 1) % 10 == 0:
            logger.info(f"Generated {traj_idx + 1}/{num_trajectories} trajectories")
    
    # Save
    output_dir = Path(config["paths"]["data_synthetic"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = np.array(trajectories)
    np.save(output_dir / "synthetic_trajectories.npy", data)
    logger.info(f"Saved {num_trajectories} trajectories to {output_dir}")
    
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trajectories", type=int, default=100)
    args = parser.parse_args()
    
    config = load_digital_twin_config()
    generate_synthetic_data(config, num_trajectories=args.num_trajectories)
