#!/usr/bin/env python
"""
Script to run training with the mock environment.

This script provides a simplified interface for running training with the mock environment,
which is useful for development and testing.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train import main as train_main
from src.utils import get_output_dir, get_path, ensure_dir_exists

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run training with mock environment")
    
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to train")
    parser.add_argument("--steps", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--no-save", action="store_true", help="Don't save checkpoints")
    parser.add_argument("--output_dir", type=str, default="output/mock_training", help="Output directory")
    
    return parser.parse_args()

def setup_logging():
    """Set up logging configuration."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    return logger

def main():
    """Main function to run training with mock environment."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting mock environment training")
    
    # Create output directory using path utilities
    output_path = get_path(args.output_dir)
    ensure_dir_exists(output_path)
    
    # Set up configuration
    config_dict = {
        "hardware": {
            "use_gpu": args.gpu,
            "use_cpu": args.cpu,
            "use_mixed_precision": args.mixed_precision,
            "batch_size": args.batch_size
        },
        "environment": {
            "mock": True,
            "render": args.render
        },
        "training": {
            "num_episodes": args.episodes,
            "max_steps": args.steps,
            "learning_rate": args.learning_rate,
            "checkpoint_dir": os.path.join(args.output_dir, "checkpoints") if not args.no_save else None,
            "early_stop_reward": None  # Don't use early stopping for mock training
        }
    }
    
    # Create a namespace object for setup functions
    class Args:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    
    arg_obj = Args(
        mock_env=True, 
        max_steps=args.steps,
        batch_size=args.batch_size,
        force_cpu=args.cpu,
        fp16=args.mixed_precision
    )
    
    # Set up hardware configuration
    hardware_config = setup_hardware_config(arg_obj)
    
    # Set up environment
    logger.info("Setting up mock environment")
    env = setup_environment(
        hardware_config, 
        arg_obj, 
        config_dict={
            "environment": config_dict["environment"]
        }
    )
    
    # Set up agent
    logger.info("Setting up agent")
    agent = setup_agent(hardware_config, env, arg_obj)
    
    # Set up hardware monitor
    hardware_monitor = HardwareMonitor(hardware_config)
    hardware_monitor.start_monitoring()
    
    # Set up performance safeguards
    performance_safeguards = PerformanceSafeguards(hardware_config)
    
    # Create trainer
    logger.info("Creating trainer")
    trainer = Trainer(
        agent=agent,
        env=env,
        config=hardware_config,
        config_dict=config_dict["training"],
        hardware_monitor=hardware_monitor,
        performance_safeguards=performance_safeguards
    )
    
    # Run training
    logger.info(
        f"Starting training with {args.episodes} episodes, {args.steps} steps per episode, "
        f"batch size {args.batch_size}, render={args.render}"
    )
    
    try:
        # Start training
        trainer.train(render=args.render)
        
        # Clean up
        trainer.cleanup()
        hardware_monitor.stop_monitoring()
        
        logger.info(f"Training completed successfully - results saved to {args.output_dir}")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.cleanup()
        hardware_monitor.stop_monitoring()
        return 1
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 