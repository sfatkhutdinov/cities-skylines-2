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

def parse_args(parser=None):
    """Parse command line arguments for training.
    
    Args:
        parser: Optional ArgumentParser instance
    
    Returns:
        Parsed arguments
    """
    if parser is None:
        parser = argparse.ArgumentParser(description="Train reinforcement learning agent")
    
    def add_argument_if_not_exists(parser, *args, **kwargs):
        """Add argument only if it doesn't already exist."""
        dest = kwargs.get('dest')
        if not dest:
            # Extract dest from the argument name
            for arg in args:
                if arg.startswith('--'):
                    dest = arg[2:].replace('-', '_')
                    break
        
        # Check if argument already exists
        if dest and any(action.dest == dest for action in parser._actions):
            return
        
        # Add the argument
        parser.add_argument(*args, **kwargs)
    
    # Config file options
    add_argument_if_not_exists(parser, "--config", type=str, help="Path to configuration file")
    add_argument_if_not_exists(parser, "--save_config", type=str, help="Save current configuration to file")
    
    # Training parameters
    add_argument_if_not_exists(parser, "--num_episodes", type=int, default=1000, help="Number of episodes to train for")
    add_argument_if_not_exists(parser, "--max_steps", type=int, default=2000, help="Maximum steps per episode")
    add_argument_if_not_exists(parser, "--batch_size", type=int, default=128, help="Batch size for updates")
    add_argument_if_not_exists(parser, "--learning_rate", type=float, default=0.0001, help="Learning rate")
    add_argument_if_not_exists(parser, "--gamma", type=float, default=0.995, help="Discount factor")
    add_argument_if_not_exists(parser, "--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    add_argument_if_not_exists(parser, "--clip_param", type=float, default=0.2, help="PPO clipping parameter")
    add_argument_if_not_exists(parser, "--early_stop_reward", type=float, default=None, help="Early stopping reward threshold")
    
    # Checkpointing and resumption
    add_argument_if_not_exists(parser, "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    add_argument_if_not_exists(parser, "--checkpoint_interval", type=int, default=5, help="Episodes between checkpoints")
    add_argument_if_not_exists(parser, "--load_checkpoint", type=str, help="Path to checkpoint to load")
    
    # Environment options
    add_argument_if_not_exists(parser, "--mock_env", action="store_true", help="Use mock environment")
    add_argument_if_not_exists(parser, "--game_path", type=str, help="Path to Cities Skylines 2 executable")
    add_argument_if_not_exists(parser, "--pixel_size", type=int, default=128, help="Size of observation pixels")
    add_argument_if_not_exists(parser, "--disable_menu_detection", action="store_true", help="Disable menu detection")
    add_argument_if_not_exists(parser, "--skip_game_check", action="store_true", default=True, help="Skip game presence check")
    add_argument_if_not_exists(parser, "--window_title", type=str, default="Cities: Skylines II", help="Game window title")
    
    # Rendering options
    add_argument_if_not_exists(parser, "--render", action="store_true", help="Render environment")
    add_argument_if_not_exists(parser, "--verbose", action="store_true", help="Print verbose output")
    
    # Hardware options
    add_argument_if_not_exists(parser, "--cpu", action="store_true", help="Force CPU usage")
    add_argument_if_not_exists(parser, "--gpu", type=int, default=0, help="GPU device to use")
    add_argument_if_not_exists(parser, "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use (cuda/cpu)")
    add_argument_if_not_exists(parser, "--resolution", type=str, default="84x84", help="Observation resolution")
    add_argument_if_not_exists(parser, "--mixed_precision", action="store_true", default=True, help="Use mixed precision training")
    add_argument_if_not_exists(parser, "--monitor_hardware", action="store_true", default=True, help="Monitor hardware usage")
    
    # Agent options
    add_argument_if_not_exists(parser, "--action_repeat", type=int, default=1, help="Number of times to repeat actions")
    add_argument_if_not_exists(parser, "--action_delay", type=float, default=0.25, help="Delay between actions")
    
    # Memory options (MANN is enabled by default)
    add_argument_if_not_exists(parser, "--disable_memory", action="store_true", help="Disable memory-augmented agent (uses standard PPO)")
    add_argument_if_not_exists(parser, "--memory_size", type=int, default=2000, help="Size of episodic memory")
    add_argument_if_not_exists(parser, "--memory_use_prob", type=float, default=0.9, help="Probability of using memory during inference")
    add_argument_if_not_exists(parser, "--memory_warmup", type=int, default=10, help="Episodes before memory is used")
    add_argument_if_not_exists(parser, "--memory_curriculum", action="store_true", default=True, help="Use curriculum learning for memory usage")
    
    return parser.parse_args()

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
            # Expect resolution like 'widthxheight', e.g., '800x600'
            # HardwareConfig expects (height, width)
            width, height = map(int, args.resolution.split('x'))
            resolution = (height, width) # Correct order for HardwareConfig
        except (ValueError, AttributeError):
            logger.warning(f"Invalid resolution format: {args.resolution}. Using default.")
    
    # Determine device to use
    device = 'auto' # Default to auto-detection within HardwareConfig
    if hasattr(args, 'device') and args.device:
        device = args.device
    elif hasattr(args, 'cpu') and args.cpu: # Check for --cpu flag
        device = 'cpu'
        
    # Extract arguments relevant to HardwareConfig constructor
    # Use getattr with defaults to handle missing args gracefully
    batch_size = getattr(args, 'batch_size', 64) # Default from HardwareConfig used if arg missing
    learning_rate = getattr(args, 'learning_rate', 1e-4) # Default from HardwareConfig used if arg missing
    use_fp16 = getattr(args, 'mixed_precision', False) # Default from HardwareConfig used if arg missing
    force_cpu = getattr(args, 'cpu', False) # Check for --cpu flag
    
    hardware_config = HardwareConfig(
        device=device,
        resolution=resolution,  # Pass parsed tuple or None
        batch_size=batch_size,
        learning_rate=learning_rate,
        use_fp16=use_fp16,
        force_cpu=force_cpu 
    )
    
    # Add memory configuration using args
    hardware_config.memory = {
        'enabled': not (hasattr(args, 'disable_memory') and args.disable_memory),
        'memory_size': getattr(args, 'memory_size', 2000), # Use getattr
        'memory_use_probability': getattr(args, 'memory_use_prob', 0.9), # Use getattr
        'key_size': 128, # Default value
        'value_size': 256, # Default value
        'retrieval_threshold': 0.5, # Default value
        'warmup_episodes': getattr(args, 'memory_warmup', 10), # Use getattr
        'use_curriculum': getattr(args, 'memory_curriculum', True), # Use getattr
        # Example curriculum phases (can be adjusted or made configurable)
        'curriculum_phases': {
            'observation': 10,
            'retrieval': 30,
            'integration': 50,
            'refinement': 100
        }
    }
    
    # Add hierarchical configuration using args
    use_hierarchical = not (hasattr(args, 'disable_hierarchical') and args.disable_hierarchical)
    hardware_config.hierarchical = {
        'enabled': use_hierarchical,
        # Add other hierarchical defaults or get from args if defined
        'feature_dim': 512,
        'latent_dim': 256,
        'prediction_horizon': 5,
        'adaptive_memory_use': True,
        'adaptive_memory_threshold': 0.7,
        # Example schedules/phases (can be adjusted or made configurable)
        'training_schedules': {
            'visual_network': 10,      
            'world_model': 5,          
            'error_detection': 20      
        },
        'batch_sizes': {
            'visual_network': 32,
            'world_model': 64,
            'error_detection': 32
        },
        'progressive_training': True,
        'progressive_phases': {
            'visual_network': 50,      
            'world_model': 100,        
            'error_detection': 150     
        }
    }
    
    # Log hardware configuration
    logger.info(f"Hardware configuration initialized - Device: {hardware_config.get_device()}, Resolution: {hardware_config.resolution}, Batch size: {hardware_config.batch_size}, FP16: {hardware_config.use_fp16}")
    
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

def args_to_dict(args):
    """Convert arguments to dictionary.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Dictionary of arguments
    """
    # Convert argparse Namespace to dictionary
    return {k: v for k, v in vars(args).items() if v is not None}