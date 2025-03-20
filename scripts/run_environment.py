#!/usr/bin/env python3
"""
Script to run the Cities Skylines 2 environment without training.

This is useful for debugging the environment and visualizing its behavior.
"""

import argparse
import logging
import time
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.core import Environment
from src.config.hardware_config import HardwareConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Cities Skylines 2 environment without training")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--mock", action="store_true", help="Use mock environment for testing")
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps to run")
    parser.add_argument("--random_actions", action="store_true", help="Take random actions")
    return parser.parse_args()


def main():
    """Run the environment."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("environment_runner")
    
    logger.info("Initializing environment...")
    
    # Create the environment
    config = HardwareConfig()
    env = Environment(config=config, mock_mode=args.mock)
    
    # Reset the environment
    logger.info("Resetting environment...")
    obs = env.reset()
    logger.info(f"Observation shape: {obs.shape}")
    
    # Run for the specified number of steps
    logger.info(f"Running for {args.steps} steps...")
    for step in range(args.steps):
        # Take action
        if args.random_actions:
            action = env.action_space.sample()
        else:
            # Just take "no-op" action (usually index 0)
            action = 0
        
        # Step the environment
        obs, reward, done, info = env.step(action)
        
        # Log progress
        if step % 10 == 0:
            logger.info(f"Step {step}/{args.steps}, Reward: {reward:.4f}")
        
        # Break if done
        if done:
            logger.info(f"Environment is done after {step} steps")
            break
        
        # Sleep to make it visually followable
        if args.render:
            time.sleep(0.1)
    
    logger.info("Done!")
    env.close()


if __name__ == "__main__":
    main() 