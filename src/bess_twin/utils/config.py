"""Config loading utilities."""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_base_config() -> Dict[str, Any]:
    """Load base configuration."""
    return load_yaml("configs/base.yaml")


def load_digital_twin_config() -> Dict[str, Any]:
    """Load digital twin config."""
    config = load_base_config()
    config.update(load_yaml("configs/digital_twin.yaml"))
    return config


def load_optimizer_config() -> Dict[str, Any]:
    """Load optimizer config."""
    config = load_base_config()
    config.update(load_yaml("configs/optimizer.yaml"))
    return config


def load_dashboard_config() -> Dict[str, Any]:
    """Load dashboard config."""
    return load_yaml("configs/dashboard.yaml")
