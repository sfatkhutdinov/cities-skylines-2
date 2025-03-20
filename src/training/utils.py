"""
Utility functions for training.

This module provides various utility functions to support the training process.
"""

import torch
import os
import logging
import argparse
from typing import Dict, Any, Optional
import time
from pathlib import Path
import sys

# Fix path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from src.config.hardware_config import HardwareConfig
from src.config.config_loader import ConfigLoader
from src.agent.core import PPOAgent
from src.environment.core import Environment

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for training.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train reinforcement learning agent")
    
    # Config file options
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--save_config", type=str, help="Save current configuration to file")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, help="Number of episodes to train for")
    parser.add_argument("--max_steps", type=int, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, help="Batch size for updates")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, help="GAE lambda parameter")
    parser.add_argument("--clip_param", type=float, help="PPO clipping parameter")
    
    # Checkpointing and resumption
    parser.add_argument("--checkpoint_dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, help="Episodes between checkpoints")
    parser.add_argument("--autosave_interval", type=int, help="Minutes between auto-saves (0 to disable)")
    parser.add_argument("--backup_checkpoints", type=int, help="Number of backup checkpoints to keep")
    parser.add_argument("--max_checkpoints", type=int, help="Max regular checkpoints to keep")
    parser.add_argument("--max_disk_usage_gb", type=float, help="Maximum disk usage in GB for logs and checkpoints")
    parser.add_argument("--fresh_start", action="store_true", help="Force starting training from scratch, ignoring existing checkpoints")
    parser.add_argument("--use_best", action="store_true", help="Resume training from best checkpoint instead of latest")
    
    # Environment settings
    parser.add_argument("--mock_env", action="store_true", help="Use mock environment for testing")
    parser.add_argument("--disable_menu_detection", action="store_true", help="Disable menu detection")
    
    # Monitoring and logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, help="Weights & Biases project name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--log_dir", type=str, help="Directory for logs")
    
    # Hardware acceleration
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (FP16) if available")
    parser.add_argument("--cpu_threads", type=int, help="Number of CPU threads to use (0 for all)")
    
    return parser.parse_args()

def setup_config(args) -> ConfigLoader:
    """Set up configuration from args and config file.
    
    Args:
        args: Parsed arguments
        
    Returns:
        ConfigLoader: Configuration loader
    """
    # Create config loader from file or defaults
    config_loader = ConfigLoader(args.config)
    
    # Convert args to dictionary and merge with config
    args_dict = args_to_dict(args)
    config_loader.merge_with_args(args_dict)
    
    # Validate config
    errors = config_loader.validate_config()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        logger.warning("Using default values for invalid configuration entries")
    
    # Save config if requested
    if args.save_config:
        if config_loader.save(args.save_config):
            logger.info(f"Configuration saved to {args.save_config}")
        else:
            logger.error(f"Failed to save configuration to {args.save_config}")
    
    return config_loader

def setup_hardware_config(args, config_loader: Optional[ConfigLoader] = None):
    """Set up hardware configuration based on arguments and config.
    
    Args:
        args: Parsed arguments
        config_loader: Optional configuration loader
        
    Returns:
        Hardware configuration
    """
    if config_loader is not None:
        # Use configuration from config loader
        config = config_loader.get_hardware_config()
        
        # Override with CLI arguments if specified
        args_dict = args_to_dict(args)
        if args.force_cpu:
            config.force_cpu = True
        if args.fp16:
            config.use_fp16 = True
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.learning_rate:
            config.learning_rate = args.learning_rate
    else:
        # Fallback to old method if no config loader
        config = HardwareConfig()
        
        # Configure hardware settings
        if args.force_cpu:
            config.force_cpu = True
        
        if args.fp16:
            config.use_fp16 = True
            
        if args.batch_size:
            config.batch_size = args.batch_size
            
        if args.learning_rate:
            config.learning_rate = args.learning_rate
    
    # Log hardware configuration
    logger.info(f"Hardware configuration: device={config.get_device()}, "
               f"batch_size={config.batch_size}, fp16={config.use_fp16}")
    
    return config

def setup_environment(config, args, config_loader: Optional[ConfigLoader] = None):
    """Set up the environment based on arguments.
    
    Args:
        config: Hardware configuration
        args: Parsed arguments
        config_loader: Optional configuration loader
        
    Returns:
        Environment instance
    """
    from src.environment.core import Environment
    from src.environment.mock_environment import MockEnvironment
    from src.environment.core.error_recovery import ErrorRecovery
    
    env_config = {}
    if config_loader:
        env_config = config_loader.get_section('environment')
    
    # Create a wrapper class for HardwareConfig that adds a 'get' method
    class ConfigWrapper:
        def __init__(self, hardware_config, env_config):
            self.hardware_config = hardware_config
            self.env_config = env_config
            
            # Create a dictionary mapping section names to their contents
            self.sections = {
                'hardware': hardware_config.to_dict(),
                'capture': env_config.get('capture', {}),
                'metrics': env_config.get('metrics', {}),
                'input': env_config.get('input', {}),
                'detection': env_config.get('detection', {})
            }
        
        def get(self, section, default=None):
            """Get a configuration section by name.
            
            Args:
                section: Section name
                default: Default value if section is not found
                
            Returns:
                Section contents or default value
            """
            return self.sections.get(section, default)
            
        # Forward all other hardware config methods
        def __getattr__(self, name):
            return getattr(self.hardware_config, name)
    
    # Use mock environment if requested
    if args.mock_env:
        logger.info("Using mock environment for testing")
        mock_env_params = {
            'frame_height': env_config.get('frame_height', 240),
            'frame_width': env_config.get('frame_width', 320),
            'max_steps': args.max_steps if args.max_steps else env_config.get('max_steps', 1000),
            'crash_probability': env_config.get('crash_probability', 0.005),
            'freeze_probability': env_config.get('freeze_probability', 0.01),
            'menu_probability': env_config.get('menu_probability', 0.02)
        }
        wrapped_config = ConfigWrapper(config, env_config)
        return MockEnvironment(config=wrapped_config, **mock_env_params)
    
    # Create real environment
    logger.info("Setting up real game environment")
    
    # Create a wrapped config that has both HardwareConfig methods and a 'get' method
    wrapped_config = ConfigWrapper(config, env_config)
    
    env = Environment(
        config=wrapped_config,
        disable_menu_detection=args.disable_menu_detection,
        **env_config
    )
    
    return env

def setup_agent(config, env, args, config_loader: Optional[ConfigLoader] = None):
    """Set up agent using hardware configuration and environment.
    
    Args:
        config: Hardware configuration
        env: Environment instance
        args: Parsed arguments
        config_loader: Optional configuration loader
        
    Returns:
        Agent instance
    """
    model_config = {}
    if config_loader:
        model_config = config_loader.get_section('model')
    
    # Extract only parameters accepted by PPOAgent
    ppo_params = {
        'state_dim': env.observation_space.shape,
        'action_dim': env.action_space.n,
        'config': config,
        'use_amp': model_config.get('use_amp', False)
    }
    
    # Create agent
    agent = PPOAgent(**ppo_params)
    
    agent.to(config.get_device())
    logger.info(f"Agent initialized on {config.get_device()}")
    
    # Log agent configuration
    logger.info(f"Agent configuration: {str(agent)}")
    
    return agent

def args_to_dict(args):
    """Convert arguments to dictionary.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Dictionary of arguments
    """
    # Convert argparse Namespace to dictionary
    return {k: v for k, v in vars(args).items() if v is not None}