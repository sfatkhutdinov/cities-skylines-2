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

# Import directly from utils.py
from src.training.utils import parse_args, setup_config, setup_hardware_config, setup_environment, setup_agent
from src.training.trainer import Trainer

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

def setup_file_logging(log_dir="logs"):
    """Configure logging to write to a timestamped file in the specified directory."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
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
        
        # Parse command line arguments
        args = parse_args()
        
        # Setup configuration
        config = setup_config(args)
        
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
        
        # Create config dict for trainer
        trainer_config = {
            'num_episodes': args.num_episodes,
            'max_steps': args.max_steps,
            'checkpoint_dir': args.checkpoint_dir,
            'tensorboard_dir': 'logs',  # Add explicit tensorboard_dir parameter
            'checkpoint_freq': args.checkpoint_interval if hasattr(args, 'checkpoint_interval') else 10,
            'learning_rate': args.learning_rate if hasattr(args, 'learning_rate') else 0.0003,
            'early_stop_reward': args.early_stop_reward if hasattr(args, 'early_stop_reward') else None,
            'use_wandb': False,  # Disable wandb for now
            'mixed_precision': args.mixed_precision if hasattr(args, 'mixed_precision') else False,
        }
        
        # Update hardware config with trainer config values
        hardware_config.num_episodes = trainer_config['num_episodes']
        hardware_config.max_steps = trainer_config['max_steps']
        hardware_config.learning_rate = trainer_config['learning_rate']
        hardware_config.early_stop_reward = trainer_config['early_stop_reward']
        hardware_config.mixed_precision = trainer_config['mixed_precision']
        
        # Add optimizer related attributes 
        hardware_config.optimizer = 'adam'  # Default optimizer
        hardware_config.weight_decay = 0.0  # Default weight decay
        hardware_config.clip_param = 0.2    # PPO clip parameter
        
        # Add additional required attributes
        hardware_config.max_checkpoints = 5  # Maximum number of checkpoints to keep
        hardware_config.value_loss_coef = 0.5  # Value loss coefficient
        hardware_config.entropy_coef = 0.01  # Entropy coefficient
        hardware_config.max_grad_norm = 0.5  # Max gradient norm
        hardware_config.visualizer_update_interval = 10  # Visualizer update interval
        hardware_config.monitor_hardware = False  # Hardware monitoring
        hardware_config.min_fps = 10  # Minimum FPS
        hardware_config.max_memory_usage = 0.9  # Maximum memory usage (90%)
        hardware_config.safeguard_cooldown = 60  # Safeguard cooldown
        
        # Create trainer
        _trainer = Trainer(
            agent=agent,
            env=_environment,
            config=hardware_config,
            checkpoint_dir=trainer_config['checkpoint_dir'],
            tensorboard_dir=trainer_config['tensorboard_dir']
        )
        
        # Train the agent
        logger.info("Starting training")
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