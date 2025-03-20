"""
Main training script for the Cities: Skylines 2 reinforcement learning agent.

This script handles the command line interface and training process.
"""

import logging
import sys
import time
import os

# Add project root to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.debug("Starting script...")

def setup_file_logging(log_dir="logs"):
    """Configure logging to write to timestamped log files.
    
    Args:
        log_dir (str): Directory for log files
        
    Returns:
        str: Path to the log file
    """
    try:
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate log filename with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Add the file handler to the root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
        return log_file
    except Exception as e:
        logger.error(f"Failed to set up file logging: {e}")
        print(f"ERROR: Failed to set up file logging: {e}")
        return None

def main():
    """Main entry point for training."""
    try:
        # Import modules
        logger.debug("Importing training modules...")
        from src.training.utils import parse_args, setup_config, setup_hardware_config, setup_environment, setup_agent
        from src.training.trainer import Trainer
        from src.training.signal_handlers import setup_signal_handlers, create_autosave_thread
        from src.utils.hardware_monitor import HardwareMonitor
        from src.utils.performance_safeguards import PerformanceSafeguards
        
        # Parse command-line arguments
        args = parse_args()
        
        # Set up configuration
        logger.info("Setting up configuration...")
        config_loader = setup_config(args)
        
        # Get logging configuration
        logging_config = config_loader.get_section('logging')
        log_dir = logging_config.get('log_dir', 'logs')
        
        # Initialize file logging with configured directory
        log_file_path = setup_file_logging(log_dir=log_dir)
        
        # Set up hardware configuration
        logger.info("Setting up hardware configuration...")
        hardware_config = setup_hardware_config(args, config_loader)
        
        # Set up performance monitoring
        hardware_monitor = HardwareMonitor(hardware_config)
        hardware_monitor.start_monitoring()
        
        # Set up performance safeguards
        perf_config = config_loader.get_section('performance')
        performance_safeguards = PerformanceSafeguards(hardware_config)
        
        # Set up environment
        logger.info("Setting up environment...")
        env_type = "mock" if args.mock_env else "real game"
        logger.info(f"Using {env_type} environment")
        env = setup_environment(hardware_config, args, config_loader)
        
        # Set up agent
        logger.info("Setting up agent...")
        agent = setup_agent(hardware_config, env, args, config_loader)
        
        # Get training configuration
        training_config = config_loader.get_section('training')
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(
            agent=agent,
            env=env,
            config=hardware_config,
            config_dict=training_config,
            hardware_monitor=hardware_monitor,
            performance_safeguards=performance_safeguards
        )
        
        # Set up signal handlers
        def save_checkpoint():
            trainer.save_progress()
            
        def cleanup():
            trainer.cleanup()
            hardware_monitor.stop_monitoring()
            
        setup_signal_handlers(cleanup_function=cleanup, save_function=save_checkpoint)
        
        # Create autosave thread if requested
        autosave_interval = training_config.get('autosave_interval', 15)
        if autosave_interval > 0:
            autosave_thread = create_autosave_thread(
                save_checkpoint,
                interval_minutes=autosave_interval
            )
            autosave_thread.start()
            logger.info(f"Started autosave thread with {autosave_interval} minute interval")
        
        # Start training
        logger.info("Starting training...")
        trainer.train(render=args.render if args.render else training_config.get('render', False))
        
        # Clean up
        trainer.cleanup()
        hardware_monitor.stop_monitoring()
        logger.info("Training completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 