"""
Main training script for the Cities: Skylines 2 reinforcement learning agent.

This script handles the command line interface and training process.
"""

import logging
import sys
import time
import os

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

def setup_file_logging():
    """Configure logging to write to timestamped log files."""
    try:
        # Create logs directory if it doesn't exist
        log_dir = "logs"
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
    # Initialize file logging
    log_file_path = setup_file_logging()
    
    try:
        # Import modules
        logger.debug("Importing training modules...")
        from training.utils import parse_args, setup_hardware_config, setup_environment, setup_agent, args_to_dict
        from training.trainer import Trainer
        from training.signal_handlers import setup_signal_handlers, create_autosave_thread
        
        # Parse command-line arguments
        args = parse_args()
        
        # Set up hardware configuration
        logger.info("Setting up hardware configuration...")
        config = setup_hardware_config(args)
        
        # Set up environment
        logger.info("Setting up environment...")
        env = setup_environment(config, args)
        
        # Set up agent
        logger.info("Setting up agent...")
        agent = setup_agent(config, env, args)
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = Trainer(agent, env, config, args_to_dict(args))
        
        # Set up signal handlers
        def save_checkpoint():
            trainer.save_progress()
            
        def cleanup():
            trainer.cleanup()
            
        setup_signal_handlers(cleanup_function=cleanup, save_function=save_checkpoint)
        
        # Create autosave thread if requested
        if args.autosave_interval > 0:
            autosave_thread = create_autosave_thread(
                save_checkpoint,
                interval_minutes=args.autosave_interval
            )
            autosave_thread.start()
            logger.info(f"Started autosave thread with {args.autosave_interval} minute interval")
        
        # Start training
        logger.info("Starting training...")
        trainer.train(render=args.render)
        
        # Clean up
        trainer.cleanup()
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