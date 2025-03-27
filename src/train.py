"""
Main training script for the Cities: Skylines 2 reinforcement learning agent.

This script handles the command line interface and training process.
"""

import os
import time
import logging
import sys
import signal
import argparse
import traceback
from datetime import datetime
import atexit
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter

# Import directly from utils.py
from src.training.utils import parse_args, setup_config, setup_hardware_config, setup_environment
from src.training.trainer import Trainer, HierarchicalTrainer, MemoryTrainer
from src.agent.core.ppo_agent import PPOAgent
from src.agent.memory_agent import MemoryAugmentedAgent
from src.agent.hierarchical_agent import HierarchicalAgent
from src.model.optimized_network import OptimizedNetwork
from src.memory.memory_augmented_network import MemoryAugmentedNetwork
from src.model.visual_understanding_network import VisualUnderstandingNetwork
from src.model.world_model import WorldModel
from src.model.error_detection_network import ErrorDetectionNetwork
from src.utils import (
    get_logs_dir, 
    get_output_dir, 
    get_checkpoints_dir,
    get_path
)
from src.config.hardware_config import HardwareConfig
from src.environment.core.environment import GameEnvironment
from src.environment.mock.mock_environment import MockEnvironment

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Global variables for cleanup
_trainer: Optional[Trainer] = None
_environment: Optional[GameEnvironment] = None
_exit_requested = False

# Default game path for Steam installation
DEFAULT_GAME_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Cities Skylines II\Cities2.exe"

def setup_file_logging() -> str:
    """Set up file logging with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return log_file

def handle_exit_signal(signum, frame):
    """Handle exit signals gracefully."""
    global _exit_requested
    _exit_requested = True
    logger.info("Received exit signal, cleaning up...")

def cleanup():
    """Clean up resources before exit."""
    global _trainer, _environment
    try:
        if _trainer is not None:
            _trainer.save_checkpoint()
        if _environment is not None:
            _environment.close()
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def focus_game_window(env, max_attempts=5):
    """Focus the game window before starting training.
    
    Args:
        env: Environment instance
        max_attempts: Maximum number of focus attempts
    """
    if hasattr(env, 'error_recovery') and hasattr(env.error_recovery, 'focus_game_window'):
        logger.critical("===== INITIAL WINDOW FOCUS =====")
        logger.critical("Attempting to find and focus the game window...")
        
        success = False
        
        # First try direct mouse controller approach
        if hasattr(env, 'input_simulator') and hasattr(env.input_simulator, 'mouse_controller'):
            logger.critical("Trying mouse controller find_game_window method...")
            window_found = env.input_simulator.mouse_controller.find_game_window("Cities: Skylines II")
            if window_found:
                logger.critical("Game window found and focused successfully through mouse controller!")
                success = True
        
        # If that didn't work, try the error recovery approach
        if not success:
            for attempt in range(max_attempts):
                logger.critical(f"Focus attempt {attempt+1}/{max_attempts}...")
                
                focus_success = env.error_recovery.focus_game_window()
                
                if focus_success:
                    logger.critical("Game window found and focused successfully!")
                    success = True
                    break
                else:
                    logger.critical("Failed to focus game window, retrying...")
                    time.sleep(1.0)  # Wait a bit before retrying
        
        if not success:
            logger.critical("WARNING: Failed to focus game window after multiple attempts")
            logger.critical("Training may not work correctly if the game window is not focused")
            logger.critical("Please ensure the game is running and manually focus the window if possible")
        
        # Force a delay after focusing to ensure stability
        logger.critical("Waiting for window focus to stabilize...")
        time.sleep(2.0)
        
        logger.critical("===== WINDOW FOCUS COMPLETE =====")
    else:
        logger.warning("Environment doesn't support window focusing, input may not work correctly")

def extend_parser_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Extend argument parser with additional arguments."""
    # Training parameters
    parser.add_argument('--num-episodes', type=int, help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, help='Maximum steps per episode')
    parser.add_argument('--early-stop-reward', type=float, help='Reward threshold for early stopping')
    parser.add_argument('--update-frequency', type=int, help='Frequency of policy updates')
    
    # Agent type selection
    parser.add_argument('--agent-type', choices=['ppo', 'memory', 'hierarchical'], default='ppo',
                       help='Type of agent to use')
    parser.add_argument('--disable-memory', action='store_true', help='Disable memory augmentation')
    parser.add_argument('--disable-hierarchical', action='store_true', help='Disable hierarchical learning')
    
    # Memory parameters
    parser.add_argument('--memory-size', type=int, help='Size of memory buffer')
    parser.add_argument('--memory-use-prob', type=float, help='Probability of using memory')
    
    # Hierarchical parameters
    parser.add_argument('--feature-dim', type=int, help='Feature dimension for hierarchical agent')
    parser.add_argument('--latent-dim', type=int, help='Latent dimension for hierarchical agent')
    
    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, help='Directory for saving checkpoints')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    # Logging
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--tensorboard-dir', type=str, help='Directory for TensorBoard logs')
    
    return parser

def setup_agent(args: argparse.Namespace, config: HardwareConfig, 
                observation_space: Any, action_space: Any) -> PPOAgent:
    """Set up the appropriate agent based on arguments."""
    agent_type = args.agent_type
    
    if agent_type == 'hierarchical' and not args.disable_hierarchical:
        logger.info("Creating hierarchical agent")
        return HierarchicalAgent(
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )
    elif agent_type == 'memory' and not args.disable_memory:
        logger.info("Creating memory-augmented agent")
        return MemoryAugmentedAgent(
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )
    else:
        logger.info("Creating standard PPO agent")
        return PPOAgent(
            observation_space=observation_space,
            action_space=action_space,
            config=config
        )

def main():
    """Main training function."""
    global _trainer, _environment, _exit_requested

    try:
        # Set up signal handlers and cleanup
        atexit.register(cleanup)
        signal.signal(signal.SIGINT, handle_exit_signal)
        signal.signal(signal.SIGTERM, handle_exit_signal)
        log_file = setup_file_logging()

        # Parse command line arguments
        parser = argparse.ArgumentParser()
        parser = extend_parser_args(parser)
        args = parse_args(parser)

        # Create base configuration
        config = HardwareConfig()

        # Update configuration from command line arguments
        if args.num_episodes is not None:
            config.num_episodes = args.num_episodes
        if args.max_steps is not None:
            config.max_steps = args.max_steps
        if args.early_stop_reward is not None:
            config.early_stop_reward = args.early_stop_reward
        if args.update_frequency is not None:
            config.update_frequency = args.update_frequency
        if args.checkpoint_dir is not None:
            config.checkpoint_dir = args.checkpoint_dir
        if args.tensorboard_dir is not None:
            config.tensorboard_dir = args.tensorboard_dir
        if args.use_wandb:
            config.use_wandb = True

        # Update memory settings if provided
        if args.memory_size is not None:
            config.memory['memory_size'] = args.memory_size
        if args.memory_use_prob is not None:
            config.memory['memory_use_probability'] = args.memory_use_prob
        if args.disable_memory:
            config.memory['enabled'] = False

        # Update hierarchical settings if provided
        if args.feature_dim is not None:
            config.hierarchical['feature_dim'] = args.feature_dim
        if args.latent_dim is not None:
            config.hierarchical['latent_dim'] = args.latent_dim
        if args.disable_hierarchical:
            config.hierarchical['enabled'] = False

        logger.info(f"Final Configuration: {config.to_dict()}")

        # Set up environment
        env_type = "mock" if args.mock_env else "real game"
        logger.info(f"Setting up {env_type} environment")
        _environment = setup_environment(args, config)

        # Set up agent
        agent = setup_agent(args, config, _environment.observation_space, _environment.action_space)

        # Create appropriate trainer
        trainer_args = {
            'agent': agent,
            'env': _environment,
            'config': config,
            'device': config.get_device(),
            'checkpoint_dir': config.checkpoint_dir,
            'tensorboard_dir': config.tensorboard_dir
        }

        if isinstance(agent, HierarchicalAgent):
            logger.info("Creating hierarchical trainer")
            _trainer = HierarchicalTrainer(**trainer_args)
        elif isinstance(agent, MemoryAugmentedAgent):
            logger.info("Creating memory-augmented trainer")
            _trainer = MemoryTrainer(**trainer_args)
        else:
            logger.info("Creating standard trainer")
            _trainer = Trainer(**trainer_args)

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            _trainer.load_checkpoint(args.resume)

        # Train the agent
        logger.info(f"Starting training for {config.num_episodes} episodes")
        _trainer.train()
        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    finally:
        cleanup()
        logger.info("Training process complete")

if __name__ == "__main__":
    main() 