"""
PPO agent module for Cities: Skylines 2.

This module implements the main PPO agent class, integrating the policy,
value function, memory, and updater components.
"""

import torch
import logging
import os
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union
from collections import deque

from src.model.optimized_network import OptimizedNetwork
from src.config.hardware_config import HardwareConfig
from src.agent.core.policy import Policy
from src.agent.core.value import ValueFunction
from src.agent.core.memory import Memory
from src.agent.core.updater import PPOUpdater

logger = logging.getLogger(__name__)

class PPOAgent:
    """Proximal Policy Optimization agent for Cities: Skylines 2."""
    
    def __init__(self, state_dim: tuple, action_dim: int, config: Optional[HardwareConfig] = None):
        """Initialize the PPO agent.
        
        Args:
            state_dim: Dimensions of the state space (C, H, W)
            action_dim: Number of possible actions
            config: Hardware configuration
        """
        # Basic setup
        self.config = config or HardwareConfig()
        self.device = self.config.get_device()
        self.dtype = self.config.get_dtype()
        
        # State and action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create neural network
        self.network = OptimizedNetwork(state_dim, action_dim, device=self.device)
        
        # Set up agent components
        self.policy = Policy(self.network, action_dim, self.device)
        self.value_function = ValueFunction(self.network, self.device)
        self.memory = Memory(self.device)
        self.updater = PPOUpdater(self.network, self.device)
        
        # Agent parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Apply parameters from config if available
        if config:
            if hasattr(config, 'gamma'):
                self.gamma = config.gamma
                self.value_function.set_params(gamma=config.gamma)
            if hasattr(config, 'gae_lambda'):
                self.gae_lambda = config.gae_lambda
                self.value_function.set_params(gae_lambda=config.gae_lambda)
        
        # Track training state
        self.training = True
        self.last_state = None
        self.steps_taken = 0
        self.episodes_completed = 0
        
        logger.info(f"Initialized PPO agent with state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
    
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Select an action based on the current state.
        
        Args:
            state: Current observation
            deterministic: Whether to select action deterministically
            
        Returns:
            Tuple containing:
                - Selected action
                - Info dictionary with additional data
        """
        # Convert state to tensor if needed
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=self.dtype, device=self.device)
        
        # Store last state for continued episodes
        self.last_state = state
        
        # Select action using policy
        action, log_prob, info = self.policy.select_action(state, deterministic)
        
        # Extract value from info
        value = info['value']
        
        # Track steps
        self.steps_taken += 1
        
        # Return action and info
        return action, {'log_prob': log_prob, 'value': value, 'entropy': info['entropy']}
    
    def store_experience(self, state: torch.Tensor, 
                        action: torch.Tensor, 
                        reward: float, 
                        next_state: Optional[torch.Tensor], 
                        done: bool, 
                        info: Dict[str, Any]) -> None:
        """Store an experience in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            info: Additional information
        """
        if not self.training:
            return
            
        # Extract log_prob and value from info
        log_prob = info.get('log_prob', None)
        value = info.get('value', None)
        action_probs = info.get('action_probs', None)
        
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done, log_prob, value, action_probs)
        
        # Update episode counter if episode ended
        if done:
            self.episodes_completed += 1
    
    def update(self) -> Dict[str, float]:
        """Update policy and value function.
        
        Returns:
            Dict with update statistics
        """
        if not self.training:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
            
        # Check if enough experiences are stored
        if self.memory.size() < self.updater.batch_size:
            logger.warning(f"Not enough experiences for update: {self.memory.size()} < {self.updater.batch_size}")
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0}
            
        # Compute returns and advantages if not already computed
        self.memory.compute_returns_and_advantages(self.gamma, self.gae_lambda)
        
        # Update policy and value function
        update_stats = self.updater.update(self.memory)
        
        # Clear memory after update
        self.memory.clear()
        
        # Update menu penalties
        self.policy.decay_penalties()
        
        return update_stats
    
    def set_training(self, training: bool) -> None:
        """Set agent to training or evaluation mode.
        
        Args:
            training: Whether to enable training mode
        """
        self.training = training
        if training:
            self.network.train()
        else:
            self.network.eval()
    
    def register_menu_action(self, action_idx: int, penalty: float = 0.5) -> None:
        """Register an action as a menu action to be penalized.
        
        Args:
            action_idx: Index of action to register
            penalty: Penalty factor (0 to 1)
        """
        self.policy.register_menu_action(action_idx, penalty)
    
    def update_from_reward(self, action_idx: int, reward: float) -> None:
        """Update action penalties based on reward.
        
        Args:
            action_idx: Index of action to update
            reward: Reward received after taking action
        """
        self.policy.update_penalty(action_idx, reward)
    
    def save(self, path: str) -> None:
        """Save agent state to file.
        
        Args:
            path: Path to save directory
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Create state dict
        state_dict = {
            'network': self.network.state_dict(),
            'policy': self.policy.state_dict(),
            'value_function': self.value_function.state_dict(),
            'updater': self.updater.state_dict(),
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'steps_taken': self.steps_taken,
            'episodes_completed': self.episodes_completed
        }
        
        # Save state dict
        model_path = os.path.join(path, 'agent.pth')
        torch.save(state_dict, model_path)
        
        logger.info(f"Saved agent state to {model_path}")
        
        # Save optimizer separately (it can be large)
        optimizer_path = os.path.join(path, 'optimizer.pth')
        torch.save({'optimizer': self.updater.optimizer.state_dict()}, optimizer_path)
        
        # Save episode statistics
        stats_path = os.path.join(path, 'stats.pth')
        torch.save({
            'episode_rewards': list(self.memory.episode_rewards),
            'episode_lengths': list(self.memory.episode_lengths)
        }, stats_path)
        
        logger.info(f"Saved agent state to {path}")
    
    def load(self, path: str) -> None:
        """Load agent state from file.
        
        Args:
            path: Path to save directory
        """
        # Check if path exists
        model_path = os.path.join(path, 'agent.pth')
        if not os.path.exists(model_path):
            logger.warning(f"Agent state file not found at {model_path}")
            return
            
        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)
        
        # Check if state dimensions match
        if 'state_dim' in state_dict and state_dict['state_dim'] != self.state_dim:
            logger.warning(f"State dimensions don't match: {state_dict['state_dim']} vs {self.state_dim}")
        
        if 'action_dim' in state_dict and state_dict['action_dim'] != self.action_dim:
            logger.warning(f"Action dimensions don't match: {state_dict['action_dim']} vs {self.action_dim}")
            return
        
        # Load network state
        if 'network' in state_dict:
            try:
                self.network.load_state_dict(state_dict['network'])
            except Exception as e:
                logger.error(f"Error loading network state: {e}")
        
        # Load component states
        if 'policy' in state_dict:
            self.policy.load_state_dict(state_dict['policy'])
        
        if 'value_function' in state_dict:
            self.value_function.load_state_dict(state_dict['value_function'])
        
        if 'updater' in state_dict:
            self.updater.load_state_dict(state_dict['updater'])
        
        # Load optimizer separately if exists
        optimizer_path = os.path.join(path, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            try:
                optimizer_dict = torch.load(optimizer_path, map_location=self.device)
                if 'optimizer' in optimizer_dict:
                    self.updater.optimizer.load_state_dict(optimizer_dict['optimizer'])
            except Exception as e:
                logger.error(f"Error loading optimizer state: {e}")
        
        # Load other parameters
        self.gamma = state_dict.get('gamma', self.gamma)
        self.gae_lambda = state_dict.get('gae_lambda', self.gae_lambda)
        self.steps_taken = state_dict.get('steps_taken', 0)
        self.episodes_completed = state_dict.get('episodes_completed', 0)
        
        # Update value function parameters
        self.value_function.set_params(gamma=self.gamma, gae_lambda=self.gae_lambda)
        
        # Load episode statistics if exists
        stats_path = os.path.join(path, 'stats.pth')
        if os.path.exists(stats_path):
            try:
                stats_dict = torch.load(stats_path, map_location=self.device)
                if 'episode_rewards' in stats_dict:
                    self.memory.episode_rewards = deque(stats_dict['episode_rewards'], maxlen=100)
                if 'episode_lengths' in stats_dict:
                    self.memory.episode_lengths = deque(stats_dict['episode_lengths'], maxlen=100)
            except Exception as e:
                logger.error(f"Error loading statistics: {e}")
        
        logger.info(f"Loaded agent state from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dict with agent statistics
        """
        return {
            'steps_taken': self.steps_taken,
            'episodes_completed': self.episodes_completed,
            'learning_rate': self.updater.get_learning_rate(),
            'memory_size': self.memory.size(),
            **self.memory.get_episode_stats()
        } 