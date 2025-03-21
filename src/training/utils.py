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
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to train for")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--learning_rate", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--early_stop_reward", type=float, default=None, help="Early stopping reward threshold")
    
    # Checkpointing and resumption
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=10, help="Episodes between checkpoints")
    parser.add_argument("--load_checkpoint", type=str, help="Path to checkpoint to load")
    
    # Environment settings
    parser.add_argument("--mock_env", action="store_true", help="Use mock environment instead of real game")
    parser.add_argument("--skip_game_check", action="store_true", help="Skip checking if game is running")
    parser.add_argument("--disable_menu_detection", action="store_true", help="Disable menu detection")
    parser.add_argument("--window_title", type=str, help="Game window title to connect to")
    parser.add_argument("--game_path", type=str, help="Path to game executable")
    
    # Hardware settings
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use (cpu or cuda)")
    parser.add_argument("--resolution", type=str, help="Observation resolution, format: WIDTHxHEIGHT")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    
    # Visualization and monitoring
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    parser.add_argument("--visualize_performance", action="store_true", help="Visualize performance metrics")
    parser.add_argument("--monitor_performance", action="store_true", help="Monitor hardware performance")
    
    args = parser.parse_args()
    return args

def setup_config(args):
    """Set up configuration from command-line arguments and optionally a config file.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary containing configuration
    """
    logger = logging.getLogger(__name__)
    config = {}
    
    # Start with defaults
    config = {
        "training": {
            "num_episodes": 100,
            "max_steps": 1000,
            "batch_size": 64,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_param": 0.2,
            "early_stop_reward": None,
        },
        "environment": {
            "mock_env": False,
            "disable_menu_detection": False,
            "skip_game_check": False,
            "window_title": "Cities: Skylines II",
        },
        "hardware": {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "resolution": "84x84",
            "mixed_precision": False,
        },
        "checkpointing": {
            "checkpoint_dir": "checkpoints",
            "checkpoint_interval": 10,
        }
    }
    
    # Override with command-line arguments if provided
    if args.num_episodes is not None:
        config["training"]["num_episodes"] = args.num_episodes
    if args.max_steps is not None:
        config["training"]["max_steps"] = args.max_steps
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.early_stop_reward is not None:
        config["training"]["early_stop_reward"] = args.early_stop_reward
        
    if args.mock_env:
        config["environment"]["mock_env"] = True
    if args.disable_menu_detection:
        config["environment"]["disable_menu_detection"] = True
    if args.skip_game_check:
        config["environment"]["skip_game_check"] = True
    if args.window_title:
        config["environment"]["window_title"] = args.window_title
    if args.game_path:
        config["environment"]["game_path"] = args.game_path
        
    if args.device:
        config["hardware"]["device"] = args.device
    if args.resolution:
        config["hardware"]["resolution"] = args.resolution
    if args.mixed_precision:
        config["hardware"]["mixed_precision"] = True
        
    if args.checkpoint_dir:
        config["checkpointing"]["checkpoint_dir"] = args.checkpoint_dir
    if args.checkpoint_interval:
        config["checkpointing"]["checkpoint_interval"] = args.checkpoint_interval
    
    logger.info(f"Configuration: {config}")
    return config

def setup_hardware_config(args):
    """Set up hardware configuration from command-line arguments and configuration.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        HardwareConfig: Hardware configuration
    """
    from src.config.hardware_config import HardwareConfig
    
    logger = logging.getLogger(__name__)
    
    # Parse resolution if provided
    resolution = None
    if hasattr(args, 'resolution') and args.resolution:
        try:
            width, height = map(int, args.resolution.split('x'))
            resolution = (width, height)
        except (ValueError, AttributeError):
            logger.warning(f"Invalid resolution format: {args.resolution}. Using default.")
    
    # Determine device to use
    device = None
    if hasattr(args, 'device') and args.device:
        device = args.device
    elif hasattr(args, 'force_cpu') and args.force_cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create hardware configuration
    mixed_precision = hasattr(args, 'mixed_precision') and args.mixed_precision
    batch_size = args.batch_size if hasattr(args, 'batch_size') else 64
    
    hardware_config = HardwareConfig(
        device=device,
        resolution=resolution,  # If None, will use default from HardwareConfig
        batch_size=batch_size,
        use_fp16=mixed_precision
    )
    
    # Log hardware configuration
    logger.info(f"Hardware configuration:")
    logger.info(f"  Device: {hardware_config.get_device()}")
    logger.info(f"  Resolution: {hardware_config.resolution}")
    logger.info(f"  Batch size: {hardware_config.batch_size}")
    logger.info(f"  Mixed precision: {hardware_config.use_fp16}")
    
    return hardware_config

def setup_environment(args, hardware_config, env_config=None, override_game_path=None, override_window_title=None):
    """Set up environment using hardware configuration and arguments.
    
    Args:
        args: Command-line arguments
        hardware_config: Hardware configuration
        env_config: Optional environment configuration
        override_game_path: Optional path to game executable
        override_window_title: Optional window title
        
    Returns:
        Environment instance
    """
    from src.environment.core.environment import Environment
    from src.environment.mock_environment import MockEnvironment
    import time
    
    logger = logging.getLogger(__name__)
    
    # Create configuration wrapper for easy access
    class ConfigWrapper:
        def __init__(self, hardware_config, env_config):
            self.hardware_config = hardware_config
            self.env_config = env_config or {}
        
        def get_device(self):
            return self.hardware_config.get_device()
            
        def get_observation_shape(self):
            return self.hardware_config.resolution
            
        def get_dtype(self):
            return self.hardware_config.get_dtype()
            
        def get(self, key, default=None):
            return self.env_config.get(key, default)
    
    wrapped_config = ConfigWrapper(hardware_config, env_config or {})
    
    # Get environment-specific parameters
    game_path = override_game_path
    window_title = override_window_title
    
    # If mock environment is requested, use it
    if hasattr(args, 'mock_env') and args.mock_env:
        logger.info("Using mock environment")
        env = MockEnvironment(
            config=wrapped_config,
            observation_shape=hardware_config.get_observation_shape(),
            action_size=137  # Default action space size, should match real env
        )
    else:
        logger.info("Setting up real game environment")
        # Get additional settings
        env_config = env_config or {}
        
        # Prepare environment parameters
        env_kwargs = {
            'config': wrapped_config,
            'disable_menu_detection': args.disable_menu_detection if hasattr(args, 'disable_menu_detection') else False,
        }
        
        # Add these parameters only if not already in env_config
        if game_path and 'game_path' not in env_config:
            env_kwargs['game_path'] = game_path
            
        if window_title and 'window_title' not in env_config:
            env_kwargs['window_title'] = window_title
            
        # Pass skip_game_check if specified
        if hasattr(args, 'skip_game_check') and args.skip_game_check:
            env_kwargs['skip_game_check'] = args.skip_game_check
        
        # Filter env_config to avoid duplicate parameters
        filtered_env_config = {k: v for k, v in env_config.items() 
                              if k not in ('game_path', 'window_title', 'disable_menu_detection')}
        
        env = Environment(
            **env_kwargs,
            **filtered_env_config
        )
        
        # Add additional initialization for improved window focus
        logger.info("Waiting for environment to initialize...")
        time.sleep(2)  # Wait for environment initialization
        
        # Force window focus with multiple attempts
        logger.info("Ensuring initial window focus for environment setup")
        focus_attempts = 3
        focus_success = False
        for attempt in range(focus_attempts):
            logger.info(f"Focus attempt {attempt+1}/{focus_attempts}")
            try:
                if env._ensure_window_focused():
                    focus_success = True
                    logger.info("Window focus successful")
                    break
            except Exception as e:
                logger.warning(f"Error during focus attempt {attempt+1}: {e}")
            time.sleep(1)
            
        if not focus_success:
            logger.warning("Could not get reliable window focus during initialization")
            
        # Set minimum action delay for more reliable action execution
        env.min_action_delay = 0.25  # Increased from the default 0.1
        logger.info(f"Set minimum action delay to {env.min_action_delay}s for environment setup")
        
        # Ensure initial observation is available
        logger.info("Getting initial observation")
        try:
            initial_obs = env.reset()
            logger.info(f"Initial observation obtained successfully: shape={initial_obs.shape}")
        except Exception as e:
            logger.warning(f"Error getting initial observation: {e}")
    
    return env

def setup_agent(args, hardware_config, observation_space, action_space):
    """Set up agent using hardware configuration and environment spaces.
    
    Args:
        args: Command-line arguments
        hardware_config: Hardware configuration
        observation_space: Environment observation space
        action_space: Environment action space
        
    Returns:
        Agent instance
    """
    from src.agent.core.ppo_agent import PPOAgent
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating PPO agent with state_dim={observation_space.shape}, action_dim={action_space.n}")
    
    # Extract only parameters accepted by PPOAgent
    ppo_params = {
        'state_dim': observation_space.shape,
        'action_dim': action_space.n,
        'config': hardware_config,
        'use_amp': args.mixed_precision if hasattr(args, 'mixed_precision') else False
    }
    
    # Create agent
    agent = PPOAgent(**ppo_params)
    
    agent.to(hardware_config.get_device())
    logger.info(f"Agent initialized on {hardware_config.get_device()}")
    
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