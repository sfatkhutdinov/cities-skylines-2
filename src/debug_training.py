#!/usr/bin/env python
"""
Debug training script for Cities Skylines 2 agent.
This script provides enhanced logging to diagnose training issues.
"""

import os
import time
import sys
import logging
import argparse
import torch
from pathlib import Path

# Add project root to path to fix import issues
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"debug_training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("debug_training")

# Import custom modules
from src.environment.core.environment import Environment
from src.agent.core.ppo_agent import PPOAgent
from src.config.hardware_config import HardwareConfig

class ConfigWrapper:
    """Wrapper for dictionary config to provide HardwareConfig-like interface."""
    
    def __init__(self, config):
        """Initialize wrapper with config dictionary."""
        self.config = config
        self.sections = config
        
    def get(self, section, default=None):
        """Get a section from the config."""
        return self.config.get(section, default)
    
    def get_device(self):
        """Get the device for training."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_dtype(self):
        """Get the data type for training."""
        return torch.float32
    
    def to_dict(self):
        """Convert config to dictionary."""
        return self.config

def setup_environment():
    """Set up the environment with fixes for input handling."""
    logger.info("Setting up environment with input handling fixes")
    
    # Create config dictionary
    env_config = {
        'capture': {
            'window_title': "Cities: Skylines II",
            'capture_method': 'windows',
            'capture_fps': 30
        },
        'input': {
            'key_delay': 0.2,  # Increased from original 0.1
            'mouse_delay': 0.2
        }
    }
    
    # Wrap config
    wrapped_config = ConfigWrapper(env_config)
    
    # Create environment
    env = Environment(
        config=wrapped_config,
        skip_game_check=True,  # Skip automatic game checking
        window_title="Cities: Skylines II",  # Explicit window title
        disable_menu_detection=True  # Disable menu detection for simplicity
    )
    
    # Ensure proper initialization
    time.sleep(2)
    
    # Force window focus with multiple attempts
    logger.info("Forcing window focus")
    for i in range(3):
        logger.info(f"Focus attempt {i+1}/3")
        focus_success = env._ensure_window_focused()
        if focus_success:
            logger.info("Focus successful!")
            break
        time.sleep(1)
    
    # Set minimum action delay
    env.min_action_delay = 0.3  # Increased delay between actions
    logger.info(f"Set minimum action delay to {env.min_action_delay}s")
    
    return env

def setup_agent(env):
    """Set up the agent."""
    logger.info("Setting up agent")
    
    # Create hardware config
    hardware_config = HardwareConfig()
    
    # Create agent
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    logger.info(f"Creating PPO agent with state_dim={state_dim}, action_dim={action_dim}")
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=hardware_config
    )
    
    return agent

def debug_train_loop(env, agent, num_episodes=3, max_steps=100):
    """Training loop with detailed debugging."""
    logger.info(f"Starting debug training: {num_episodes} episodes, {max_steps} steps per episode")
    
    # Track statistics
    all_rewards = []
    all_steps = []
    
    for episode in range(num_episodes):
        logger.info(f"==== EPISODE {episode+1}/{num_episodes} ====")
        
        # Reset environment
        logger.info("Resetting environment")
        observation = env.reset()
        logger.info(f"Observation shape: {observation.shape}, type: {type(observation)}")
        
        # Ensure window focus at episode start
        logger.info("Re-focusing window before episode start")
        env._ensure_window_focused()
        time.sleep(1)
        
        # Episode variables
        done = False
        step = 0
        episode_reward = 0
        action_successes = 0
        action_failures = 0
        
        while not done and step < max_steps:
            # Log current step
            logger.info(f"--- Step {step+1}/{max_steps} ---")
            
            # Re-ensure focus periodically
            if step % 10 == 0 and step > 0:
                logger.info("Periodic window re-focus")
                env._ensure_window_focused()
                time.sleep(0.5)
            
            # Select action
            if not isinstance(observation, torch.Tensor):
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            else:
                obs_tensor = observation.unsqueeze(0)
                
            with torch.no_grad():
                logger.debug(f"Selecting action for observation tensor of shape {obs_tensor.shape}")
                action, action_log_prob, value = agent.select_action(obs_tensor)
                action_value = action.item() if hasattr(action, 'item') else action
                logger.info(f"Selected action: {action_value}")
            
            # Execute step with timing
            start_time = time.time()
            logger.info(f"Executing action {action_value}")
            
            try:
                next_observation, reward, done, info = env.step(action_value)
                step_time = time.time() - start_time
                logger.info(f"Step completed in {step_time:.3f}s")
                logger.info(f"Reward: {reward}, Done: {done}")
                
                # Log success/failure
                if 'success' in info and info['success']:
                    action_successes += 1
                    logger.info("Action execution SUCCESS")
                else:
                    action_failures += 1
                    logger.warning("Action execution FAILED")
                    
                # Log info
                logger.debug(f"Step info: {info}")
                
            except Exception as e:
                logger.error(f"Error during step: {e}")
                # Continue with default values
                next_observation = observation
                reward = -1
                done = True
                info = {"error": str(e)}
            
            # Update counters
            episode_reward += reward
            step += 1
            
            # Update observation
            observation = next_observation
            
            # Small delay
            time.sleep(0.1)
        
        # Log episode results
        logger.info(f"Episode {episode+1} finished: steps={step}, reward={episode_reward}")
        logger.info(f"Action successes: {action_successes}, failures: {action_failures}, " 
                   f"success rate: {action_successes/(action_successes+action_failures)*100:.1f}%")
        
        all_rewards.append(episode_reward)
        all_steps.append(step)
        
        # Episode wait between episodes
        time.sleep(2)
    
    # Overall statistics
    logger.info("==== TRAINING SUMMARY ====")
    logger.info(f"Episodes completed: {num_episodes}")
    logger.info(f"Average reward: {sum(all_rewards)/len(all_rewards):.2f}")
    logger.info(f"Average steps: {sum(all_steps)/len(all_steps):.2f}")
    
    return {
        "rewards": all_rewards,
        "steps": all_steps
    }

def main():
    """Main entry point."""
    logger.info("Starting debug training script")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Debug training for Cities Skylines 2")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=100, help="Maximum steps per episode")
    args = parser.parse_args()
    
    try:
        # Set up environment
        env = setup_environment()
        
        # Set up agent
        agent = setup_agent(env)
        
        # Run training loop
        debug_train_loop(env, agent, num_episodes=args.episodes, max_steps=args.steps)
        
        # Clean up
        env.close()
        logger.info("Training completed successfully")
        return 0
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 