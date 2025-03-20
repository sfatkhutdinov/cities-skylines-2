#!/usr/bin/env python
"""
Convenience script for running training with the mock environment.

This script provides a simplified interface for running training using
the mock environment for testing and development.
"""

import os
import sys
import argparse
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import trainer components
from src.training.utils import setup_config, setup_hardware_config, setup_environment, setup_agent
from src.training.trainer import Trainer
from src.utils.hardware_monitor import HardwareMonitor
from src.utils.performance_safeguards import PerformanceSafeguards

def parse_args():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description="Run training with mock environment")
    
    # Training parameters
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to train")
    parser.add_argument("--steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for updates")
    
    # Output and visualization
    parser.add_argument("--output_dir", type=str, default="output/mock_training", help="Output directory")
    parser.add_argument("--render", action="store_true", help="Render visualization during training")
    parser.add_argument("--no_save", action="store_true", help="Don't save checkpoints")
    
    # Environment configuration
    parser.add_argument("--crash_prob", type=float, default=0.005, help="Probability of simulated crash")
    parser.add_argument("--freeze_prob", type=float, default=0.01, help="Probability of simulated freeze")
    parser.add_argument("--menu_prob", type=float, default=0.02, help="Probability of simulated menu")
    
    # Hardware acceleration
    parser.add_argument("--gpu", action="store_true", help="Force GPU usage")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision")
    
    return parser.parse_args()

def setup_logging():
    """Set up logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def main():
    """Main function to run training with mock environment."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging()
    logger.info("Starting mock environment training")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up configuration
    config_dict = {
        "hardware": {
            "use_gpu": args.gpu,
            "use_cpu": args.cpu,
            "use_mixed_precision": args.mixed_precision,
            "batch_size": args.batch_size
        },
        "environment": {
            "max_steps": args.steps,
            "mock_settings": {
                "crash_probability": args.crash_prob,
                "freeze_probability": args.freeze_prob,
                "menu_probability": args.menu_prob
            }
        },
        "training": {
            "num_episodes": args.episodes,
            "max_steps": args.steps,
            "checkpoint_dir": os.path.join(args.output_dir, "checkpoints") if not args.no_save else None,
            "checkpoint_freq": max(1, args.episodes // 10),  # Save 10 checkpoints
            "render": args.render
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