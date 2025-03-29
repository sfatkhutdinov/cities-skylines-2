"""
Main training script for the Cities: Skylines 2 reinforcement learning agent.

This script handles the command line interface and training process.
"""

import os
import sys
if sys.platform == 'win32' and sys.version_info >= (3, 8):
    # Explicitly add Python's DLL directory to the search path
    # This attempts to fix DLL load errors for packages like pywin32 on newer Python
    python_dll_path = os.path.join(sys.base_prefix, 'DLLs')
    if os.path.isdir(python_dll_path):
        try:
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(python_dll_path)
        except (FileNotFoundError, OSError):
            pass
    # Also try adding the main Python installation directory
    if os.path.isdir(sys.base_prefix):
         try:
            if hasattr(os, 'add_dll_directory'):
                os.add_dll_directory(sys.base_prefix)
         except (FileNotFoundError, OSError):
            pass

import time
import logging
import signal
import argparse
import traceback
from datetime import datetime
import atexit
from pathlib import Path

# Import directly from utils.py
from src.training.utils import parse_args, setup_config, setup_hardware_config, setup_environment
from src.training.trainer import Trainer
from src.training.memory_trainer import MemoryTrainer
from src.training.hierarchical_trainer import HierarchicalTrainer
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
_trainer = None
_environment = None
_exit_requested = False

# Default game path for Steam installation
DEFAULT_GAME_PATH = r"C:\Program Files (x86)\Steam\steamapps\common\Cities Skylines II\Cities2.exe"

def setup_file_logging():
    """Configure logging to write to a timestamped file in the logs directory."""
    logs_dir = get_logs_dir()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"training_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    return log_file

def handle_exit_signal(signum, frame):
    """Handle exit signals by requesting a clean exit."""
    global _exit_requested
    logger.info(f"Received signal {signum}. Requesting clean exit...")
    _exit_requested = True
    
    # If trainer exists, request exit
    if _trainer is not None:
        _trainer.request_exit()

def cleanup():
    """Clean up resources on exit."""
    global _environment
    try:
        if _environment is not None:
            logger.info("Closing environment during cleanup...")
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
    logger.info(f"Creating agent with state_dim={observation_space.shape}, action_dim={action_space.n}")
    
    # Get device
    device = hardware_config.get_device()
    
    # Determine which agent type to use - hierarchical is now default
    use_memory = not (hasattr(args, 'disable_memory') and args.disable_memory)
    use_hierarchical = not (hasattr(args, 'disable_hierarchical') and args.disable_hierarchical)
    memory_size = args.memory_size if hasattr(args, 'memory_size') else 2000  # Increased default memory size
    
    if use_hierarchical:
        logger.info("Creating hierarchical agent with specialized neural networks (default)")
        
        # Get hierarchical config from hardware config
        if not hasattr(hardware_config, 'hierarchical'):
            hardware_config.hierarchical = {
                'enabled': True,
                'feature_dim': 512,
                'latent_dim': 256,
                'prediction_horizon': 5,
                'adaptive_memory_use': True,
                'adaptive_memory_threshold': 0.7
            }
        
        # Determine which components to use
        use_visual_network = not (hasattr(args, 'no_visual') and args.no_visual)
        use_world_model = not (hasattr(args, 'no_world') and args.no_world)
        use_error_detection = not (hasattr(args, 'no_error') and args.no_error)
        
        # Create memory-augmented network as the base policy network
        policy_network = MemoryAugmentedNetwork(
            input_shape=observation_space.shape,
            num_actions=action_space.n,
            memory_size=memory_size,
            device=device,
            use_lstm=True,  # Enable LSTM by default
            lstm_hidden_size=384,  # Larger hidden size
            use_attention=True,  # Enable attention by default
            attention_heads=8  # More attention heads for better learning
        )
        
        # Create hierarchical agent
        agent = HierarchicalAgent(
            policy_network=policy_network,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            memory_size=memory_size,
            memory_use_prob=hardware_config.memory.get('memory_use_probability', 0.8),
            use_visual_network=use_visual_network,
            use_world_model=use_world_model,
            use_error_detection=use_error_detection,
            feature_dim=hardware_config.hierarchical.get('feature_dim', 512),
            latent_dim=hardware_config.hierarchical.get('latent_dim', 256),
            prediction_horizon=hardware_config.hierarchical.get('prediction_horizon', 5),
            adaptive_memory_use=hardware_config.hierarchical.get('adaptive_memory_use', True),
            adaptive_memory_threshold=hardware_config.hierarchical.get('adaptive_memory_threshold', 0.7)
        )
        
        logger.info(f"Created hierarchical agent with components: "
                  f"Visual={use_visual_network}, World={use_world_model}, "
                  f"Error={use_error_detection}")
    
    elif use_memory:
        logger.info("Creating memory-augmented network and agent")
        
        # Create memory-augmented network
        policy_network = MemoryAugmentedNetwork(
            input_shape=observation_space.shape,
            num_actions=action_space.n,
            memory_size=memory_size,
            device=device,
            use_lstm=True,  # Enable LSTM by default
            lstm_hidden_size=384,  # Larger hidden size
            use_attention=True,  # Enable attention by default
            attention_heads=8  # More attention heads for better learning
        )
        
        # Create memory-augmented agent
        agent = MemoryAugmentedAgent(
            policy_network=policy_network,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            memory_size=memory_size,
            memory_use_prob=hardware_config.memory.get('memory_use_probability', 0.8)
        )
    else:
        logger.info("Creating standard PPO agent (memory disabled via flag)")
        
        # Create the PPO agent with the state_dim, action_dim pattern since we're
        # not using a memory-augmented network
        agent = PPOAgent(
            state_dim=observation_space.shape,
            action_dim=action_space.n,
            config=hardware_config,
            use_amp=hardware_config.use_fp16
        )
    
    agent.to(device)
    logger.info(f"Agent initialized on {device}")
    
    return agent

def extend_parser_args(parser):
    """Extend the argument parser with hierarchical agent options.
    
    Args:
        parser: Argument parser
        
    Returns:
        Extended argument parser
    """
    # Add hierarchical agent arguments - now using disable flags instead
    parser.add_argument('--disable_hierarchical', action='store_true',
                       help='Disable hierarchical agent structure (use simpler agent instead)')
    parser.add_argument('--no_visual', action='store_true',
                       help='Disable visual understanding network in hierarchical agent')
    parser.add_argument('--no_world', action='store_true',
                       help='Disable world model in hierarchical agent')
    parser.add_argument('--no_error', action='store_true',
                       help='Disable error detection network in hierarchical agent')
    
    # Only add --memory_size if not already there
    if not any(action.dest == 'memory_size' for action in parser._actions):
        parser.add_argument('--memory_size', type=int, default=2000,
                          help='Size of episodic memory (default: 2000)')
    
    # Only add --mixed_precision if not already there
    if not any(action.dest == 'mixed_precision' for action in parser._actions):
        parser.add_argument('--mixed_precision', action='store_true', default=True,
                          help='Enable mixed precision training (default: enabled)')
    
    return parser

def main():
    """Main training function."""
    global _trainer, _environment, _exit_requested
    
    try:
        # Register the cleanup function
        atexit.register(cleanup)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, handle_exit_signal)
        signal.signal(signal.SIGTERM, handle_exit_signal)
        
        # Setup file logging
        log_file = setup_file_logging()
        
        # Parse command line arguments - use a single unified approach
        parser = argparse.ArgumentParser()
        # Add our custom arguments
        parser = extend_parser_args(parser)
        # Allow parse_args to add the standard arguments
        args = parse_args(parser)
        
        # Add memory-specific arguments if not present
        if not hasattr(args, 'disable_memory'):
            args.disable_memory = False
        if not hasattr(args, 'memory_size'):
            args.memory_size = 2000  # Increased default memory size
        if not hasattr(args, 'disable_hierarchical'):
            args.disable_hierarchical = False
        
        # Setup hardware configuration
        hardware_config = setup_hardware_config(args)
        
        # Setup the environment
        env_type = "mock" if args.mock_env else "real game"
        logger.info(f"Setting up {env_type} environment")
        
        # Handle game path - check command line args, then use default path if it exists
        game_path = args.game_path if hasattr(args, 'game_path') and args.game_path else None
        
        # If no path provided via args, try the default Steam path
        if not game_path and not args.mock_env:
            if os.path.exists(DEFAULT_GAME_PATH):
                game_path = DEFAULT_GAME_PATH
                logger.info(f"Using default Steam installation path: {game_path}")
            else:
                logger.warning("Default game path not found. Game auto-restart will be disabled.")
                logger.warning("Use --game_path to specify the path to the Cities: Skylines 2 executable")
        
        _environment = setup_environment(args, hardware_config)
        
        # If game path is available, set it on the error recovery system
        if game_path and not args.mock_env and hasattr(_environment, 'error_recovery'):
            logger.info(f"Setting game path: {game_path}")
            _environment.error_recovery.game_path = game_path
        
        # Ensure window focus before proceeding
        if not args.mock_env:
            # Call our updated focus_game_window function
            focus_game_window(_environment)
            
            # Set minimum action delay for better reliability
            logger.info("Setting minimum action delay to 0.3s")
            _environment.min_action_delay = 0.3
        
        # Setup the agent
        agent = setup_agent(args, hardware_config, _environment.observation_space, _environment.action_space)
        
        # Gather trainer-specific parameters
        num_episodes = args.num_episodes
        max_steps = args.max_steps
        checkpoint_dir = args.checkpoint_dir
        tensorboard_dir = 'logs' # Keep explicit tensorboard_dir
        checkpoint_freq = getattr(args, 'checkpoint_interval', 5)
        early_stop_reward = getattr(args, 'early_stop_reward', None)
        use_wandb = False # Keep explicit wandb setting
        
        # Add trainer-specific parameters to the hardware_config object
        # The Trainer classes expect these to be attributes of the config object passed
        hardware_config.num_episodes = num_episodes
        hardware_config.max_steps = max_steps
        hardware_config.checkpoint_freq = checkpoint_freq
        hardware_config.early_stop_reward = early_stop_reward
        hardware_config.use_wandb = use_wandb
        # Note: hardware_config already has learning_rate and mixed_precision (use_fp16)
        
        # Create appropriate trainer based on agent type
        # Pass the unified hardware_config object and specific dirs
        if isinstance(agent, HierarchicalAgent):
            logger.info("Creating hierarchical trainer")
            _trainer = HierarchicalTrainer(
                agent=agent,
                env=_environment,
                config=hardware_config, # Pass the unified config
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir
            )
        elif isinstance(agent, MemoryAugmentedAgent):
            logger.info("Creating memory-augmented trainer")
            _trainer = MemoryTrainer(
                agent=agent,
                env=_environment,
                config=hardware_config, # Pass the unified config
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir
            )
        else:
            logger.info("Creating standard trainer")
            _trainer = Trainer(
                agent=agent,
                env=_environment,
                config=hardware_config, # Pass the unified config
                checkpoint_dir=checkpoint_dir,
                tensorboard_dir=tensorboard_dir
            )
        
        # Resume from checkpoint if specified
        if hasattr(args, 'resume') and args.resume:
            logger.info(f"Resuming training from checkpoint: {args.resume}")
            _trainer.load_checkpoint(args.resume)
        
        # Override number of episodes if specified
        if hasattr(args, 'num_episodes') and args.num_episodes:
            _trainer.max_episodes = args.num_episodes
        
        # Train the agent
        logger.info(f"Starting training for {_trainer.max_episodes} episodes")
        _trainer.train()
        logger.info("Training completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up
        try:
            if _environment is not None:
                logger.info("Closing environment...")
                _environment.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")
        
        logger.info("Training process complete")

if __name__ == "__main__":
    main() 