"""
Configuration module for Cities: Skylines 2 agent.

This module provides centralized configuration management for the entire project.
"""

from .hardware_config import HardwareConfig
from typing import Dict, Any, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a file.
    
    Args:
        config_path (Optional[str]): Path to the configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.warning("Falling back to default configuration")
    
    # Load default configuration
    default_config_path = os.path.join(CONFIG_DIR, "default_config.json")
    if os.path.exists(default_config_path):
        try:
            with open(default_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load default config: {e}")
    
    # If all else fails, return empty config
    logger.warning("No configuration found, using empty configuration")
    return {}

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """Save configuration to a file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        config_path (str): Path to save the configuration file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Failed to save config to {config_path}: {e}")
        return False 