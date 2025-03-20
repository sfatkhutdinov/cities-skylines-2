"""
Utility functions for training.

This module provides various utility functions to support the training process.
"""

import torch
import os
import logging
import argparse
from typing import Dict, Any
import time
from pathlib import Path

from ..config.hardware_config import HardwareConfig
from ..agent.ppo_agent import PPOAgent
from ..environment.game_env import CitiesEnvironment

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments for training.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train reinforcement learning agent")
    
    # Training parameters
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of episodes to train for")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip_param", type=float, default=0.2, help="PPO clipping parameter")
    
    # Checkpointing and resumption
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--checkpoint_freq", type=int, default=100, help="Episodes between checkpoints")
    parser.add_argument("--autosave_interval", type=int, default=15, help="Minutes between auto-saves")
    parser.add_argument("--backup_checkpoints", type=int, default=5, help="Number of backup checkpoints to keep")
    parser.add_argument("--max_checkpoints", type=int, default=10, help="Max regular checkpoints to keep")
    parser.add_argument("--max_disk_usage_gb", type=float, default=5.0, help="Maximum disk usage in GB for logs and checkpoints")
    parser.add_argument("--fresh_start", action="store_true", help="Force starting training from scratch, ignoring existing checkpoints")
    parser.add_argument("--use_best", action="store_true", help="Resume training from best checkpoint instead of latest")
    
    # Environment settings
    parser.add_argument("--mock_env", action="store_true", help="Use mock environment for testing")
    parser.add_argument("--disable_menu_detection", action="store_true", help="Disable menu detection")
    
    # Monitoring and logging
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="cities-skylines-rl", help="Weights & Biases project name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--render", action="store_true", help="Render environment during training")
    
    # Hardware acceleration
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (FP16) if available")
    
    return parser.parse_args()

def setup_hardware_config(args):
    """Set up hardware configuration based on arguments.
    
    Args:
        args: Parsed arguments
        
    Returns:
        Hardware configuration
    """
    config = HardwareConfig()
    
    # Configure hardware settings
    if args.force_cpu:
        config.device = "cpu"
    
    if args.fp16:
        config.use_fp16 = True
    
    # Validate configuration
    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU")
        config.device = "cpu"
    
    # Log configuration
    logger.info(f"Using device: {config.device}")
    if config.device == "cuda":
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    if config.use_fp16:
        logger.info("Using mixed precision (FP16)")
    
    return config

def setup_environment(config, args):
    """Set up the environment based on configuration and arguments.
    
    Args:
        config: Hardware configuration
        args: Parsed arguments
        
    Returns:
        Environment instance
    """
    # Find menu reference path
    menu_reference_path = None
    if os.path.exists("menu_reference.png"):
        menu_reference_path = os.path.abspath("menu_reference.png")
        logger.info(f"Using menu reference: {menu_reference_path}")
    
    # Create environment
    env = CitiesEnvironment(
        config=config,
        mock_mode=args.mock_env,
        menu_screenshot_path=menu_reference_path,
        disable_menu_detection=args.disable_menu_detection,
        max_steps=args.max_steps
    )
    
    return env

def setup_agent(config, env, args):
    """Set up the agent based on configuration and arguments.
    
    Args:
        config: Hardware configuration
        env: Environment instance
        args: Parsed arguments
        
    Returns:
        Agent instance
    """
    # Get environment properties
    if hasattr(env, 'observation_shape'):
        observation_shape = env.observation_shape
    else:
        test_state = env.reset()
        observation_shape = test_state.shape
    
    if hasattr(env, 'num_actions'):
        num_actions = env.num_actions
    else:
        num_actions = env.action_space.n if hasattr(env, 'action_space') else 10
    
    # Create agent
    agent = PPOAgent(
        state_dim=observation_shape,
        action_dim=num_actions,
        lr=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        batch_size=args.batch_size,
        device=config.get_device(),
        dtype=config.get_dtype()
    )
    
    return agent

def args_to_dict(args):
    """Convert argparse Namespace to dictionary.
    
    Args:
        args: Argparse Namespace
        
    Returns:
        Dictionary of arguments
    """
    return {k: v for k, v in vars(args).items()} 