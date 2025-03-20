"""
Custom training script for Cities Skylines 2 with debugging features.
This version focuses on ensuring proper input handling.
"""

import os
import time
import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.environment.core.environment import Environment
from src.agent.core.ppo_agent import PPOAgent
from src.config.hardware_config import HardwareConfig
from src.custom_visualizer import Visualizer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigWrapper:
    """Wrapper for dictionary config to provide HardwareConfig-like interface."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize wrapper with config dictionary.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.sections = config
        
    def get(self, section: str, default: Any = None) -> Any:
        """Get a section from the config.
        
        Args:
            section: Section name
            default: Default value if section doesn't exist
            
        Returns:
            Section value or default
        """
        return self.config.get(section, default)
    
    def get_device(self) -> torch.device:
        """Get the device for training.
        
        Returns:
            torch.device: CPU device
        """
        return torch.device("cpu")
    
    def get_dtype(self) -> torch.dtype:
        """Get the data type for training.
        
        Returns:
            torch.dtype: Float32 data type
        """
        return torch.float32

def setup_environment():
    """Set up the environment with fixes for input handling."""
    logger.info("Setting up environment with input handling fixes")
    
    # Create config dictionary that works with Environment
    env_config = {
        'capture': {
            'window_title': "Cities: Skylines II",
            'capture_method': 'windows',
            'capture_fps': 30
        },
        'input': {
            'key_delay': 0.1,
            'mouse_delay': 0.1
        }
    }
    
    # Wrap config to provide required methods
    wrapped_config = ConfigWrapper(env_config)
    
    env = Environment(
        config=wrapped_config,
        skip_game_check=True,  # Skip automatic game checking
        window_title="Cities: Skylines II"  # Explicit window title
    )
    
    # Ensure proper initialization
    time.sleep(2)
    
    # Force window focus
    logger.info("Forcing window focus")
    for i in range(3):
        logger.info(f"Focus attempt {i+1}")
        env._ensure_window_focused()
        time.sleep(1)
    
    return env

def setup_agent(env):
    """Set up the PPO agent."""
    logger.info("Setting up agent")
    
    # Create hardware config
    hardware_config = HardwareConfig()
    
    # Create agent
    state_dim = env.observation_space.shape
    action_dim = env.action_space.n
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        config=hardware_config
    )
    
    return agent

def custom_train_loop(env, agent, num_episodes=5, max_steps=100):
    """Custom training loop with debug output."""
    logger.info(f"Starting custom training loop: {num_episodes} episodes, {max_steps} steps per episode")
    
    # Initialize visualizer
    log_dir = Path("logs/custom_run")
    log_dir.mkdir(parents=True, exist_ok=True)
    visualizer = Visualizer(log_dir=log_dir)
    
    # Initialize metrics
    all_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        logger.info(f"Starting episode {episode+1}/{num_episodes}")
        
        # Reset environment
        observation = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        # Ensure window focus at the start of each episode
        env._ensure_window_focused()
        time.sleep(1)
        
        # Storage for episode data
        states = []
        actions = []
        action_probs = []
        values = []
        rewards = []
        dones = []
        
        while not done and step_count < max_steps:
            # Log current step
            logger.info(f"Episode {episode+1}, Step {step_count+1}")
            
            # Ensure window is still focused periodically
            if step_count % 10 == 0:
                env._ensure_window_focused()
                time.sleep(0.5)
            
            # Convert observation to tensor if needed
            if not isinstance(observation, torch.Tensor):
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(agent.device)
            else:
                obs_tensor = observation.unsqueeze(0)
            
            # Select action
            with torch.no_grad():
                action, action_log_prob, action_info = agent.select_action(obs_tensor)
                value = action_info['value']
                logger.info(f"Selected action: {action.item()}")
            
            # Take step with explicit debugging
            try:
                logger.info(f"Executing action {action.item()}")
                next_observation, reward, done, info = env.step(action.item())
                logger.info(f"Step result: reward={reward}, done={done}, info={info}")
            except Exception as e:
                logger.error(f"Error executing step: {e}")
                # Try to continue with default observation
                next_observation = observation
                reward = -1
                done = True
                info = {"error": str(e)}
            
            # Add experience directly to agent memory
            try:
                # Convert next observation to tensor if needed
                if not isinstance(next_observation, torch.Tensor):
                    next_obs_tensor = torch.FloatTensor(next_observation).unsqueeze(0).to(agent.device)
                else:
                    next_obs_tensor = next_observation.unsqueeze(0)
                    
                # Add to memory
                agent.memory.add(
                    state=obs_tensor, 
                    action=action, 
                    reward=reward, 
                    next_state=next_obs_tensor, 
                    done=done, 
                    log_prob=action_log_prob, 
                    value=value
                )
                logger.debug("Added experience to memory")
            except Exception as e:
                logger.error(f"Error adding to memory: {e}")
            
            # Update metrics
            episode_reward += reward
            step_count += 1
            
            # Update observation
            observation = next_observation
            
            # Add small delay to ensure proper rendering and state updates
            time.sleep(0.1)
        
        # Log episode results
        logger.info(f"Episode {episode+1} finished: steps={step_count}, reward={episode_reward}")
        all_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # Only try to update if we have enough steps
        if step_count > 0:
            logger.info(f"Processing episode data with {step_count} steps")
            
            # Compute returns and advantages
            try:
                # Update policy if memory has enough data
                logger.info("Updating policy")
                loss_info = agent.update()
                logger.info(f"Policy update: {loss_info}")
            except Exception as e:
                logger.error(f"Error updating policy: {e}")
        
        # Visualize progress
        visualizer.log_episode_metrics(
            episode=episode,
            reward=episode_reward,
            length=step_count
        )
        
        # Generate visualizations periodically
        if (episode + 1) % 1 == 0 or episode == num_episodes - 1:
            visualizer.generate_visualizations(
                rewards=all_rewards,
                episode_lengths=episode_lengths
            )
    
    # Final visualizations
    visualizer.generate_dashboard(
        rewards=all_rewards,
        episode_lengths=episode_lengths
    )
    
    logger.info("Training complete!")
    return all_rewards

def main():
    """Main execution function."""
    logger.info("Starting custom training run")
    
    # Setup environment and agent
    env = setup_environment()
    agent = setup_agent(env)
    
    try:
        # Run training loop
        rewards = custom_train_loop(env, agent)
        logger.info(f"Final rewards: {rewards}")
    except Exception as e:
        logger.error(f"Error during training: {e}")
    finally:
        # Clean up
        env.close()
        logger.info("Environment closed")

if __name__ == "__main__":
    main() 