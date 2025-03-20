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
    parser.add_argument("--game_path", type=str, help="Path to the game executable")
    parser.add_argument("--window_title", type=str, help="Window title to search for")
    parser.add_argument("--skip_game_check", action="store_true", help="Skip game process verification (use when game is already running)")
    
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

def setup_environment(
    config: HardwareConfig,
    args: argparse.Namespace,
    config_loader: ConfigLoader
) -> Environment:
    """Set up the environment.
    
    Args:
        config: Hardware configuration
        args: Command-line arguments
        config_loader: Configuration loader
        
    Returns:
        Environment: Environment instance
    """
    # Get environment configuration
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
        from src.environment.mock_environment import MockEnvironment
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
        env = MockEnvironment(config=wrapped_config, **mock_env_params)
    else:
        # Set up real environment
        logger.info("Setting up real game environment")
        
        # Create a wrapped config that has both HardwareConfig methods and a 'get' method
        wrapped_config = ConfigWrapper(config, env_config)
        
        # Get game path from args or config
        game_path = args.game_path if hasattr(args, 'game_path') else None
        if not game_path:
            game_path = env_config.get('game_path')
        
        # Get window title from args or config
        window_title = args.window_title if hasattr(args, 'window_title') else None
        if not window_title:
            window_title = env_config.get('window_title', "Cities: Skylines II")
        
        logger.info(f"Game path: {game_path or 'Not specified'}")
        logger.info(f"Window title: {window_title or 'Using default'}")
        
        # Remove arguments that might conflict with env_config
        # to avoid multiple values error
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
    time.sleep(2)  # Wait for environment initialization
    
    # Force window focus with multiple attempts
    logger.info("Ensuring initial window focus for training")
    focus_success = False
    for attempt in range(3):
        logger.info(f"Focus attempt {attempt+1}/3")
        if env._ensure_window_focused():
            focus_success = True
            logger.info("Window focus successful")
            break
        time.sleep(1)
        
    if not focus_success:
        logger.warning("Could not get reliable window focus during initialization")
        
    # Set minimum action delay for more reliable action execution
    env.min_action_delay = 0.3  # Increased from the default 0.1
    logger.info(f"Set minimum action delay to {env.min_action_delay}s")
    
    # Ensure initial observation is available
    logger.info("Getting initial observation")
    try:
        env.get_observation()
        logger.info("Initial observation obtained successfully")
    except Exception as e:
        logger.warning(f"Error getting initial observation: {e}")
    
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