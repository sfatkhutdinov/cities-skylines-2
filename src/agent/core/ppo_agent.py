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
import time
import random

from src.model.optimized_network import OptimizedNetwork
from src.config.hardware_config import HardwareConfig
from src.agent.core.policy import Policy
from src.agent.core.value import ValueFunction
from src.agent.core.memory import Memory
from src.agent.core.updater import PPOUpdater

logger = logging.getLogger(__name__)

class PPOAgent:
    """Proximal Policy Optimization agent for Cities: Skylines 2."""
    
    def __init__(
        self, 
        state_dim: tuple, 
        action_dim: int, 
        config: Optional[HardwareConfig] = None,
        use_amp: bool = False
    ):
        """Initialize the PPO agent.
        
        Args:
            state_dim: Dimensions of the state space (C, H, W)
            action_dim: Number of possible actions
            config: Hardware configuration
            use_amp: Whether to use automatic mixed precision (FP16)
        """
        # Basic setup
        self.config = config or HardwareConfig()
        self.device = self.config.get_device()
        self.dtype = self.config.get_dtype()
        
        # Mixed precision setup
        self.use_amp = use_amp and torch.cuda.is_available()
        if self.use_amp:
            logger.info("Using automatic mixed precision (FP16)")
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
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
        self.update_frequency = 32  # Update policy every 32 steps
        
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
        
        # Add menu action tracking to help avoid menu transitions
        self.menu_actions = {}  # Maps action index to count of menu occurrences
        self.menu_action_memory = 50  # Remember the last 50 menu occurrences
        self.menu_penalty_factor = 0.5  # Penalty factor for actions that cause menus
        
        # Initialize LSTM hidden state
        self.hidden_state = None
        self.use_lstm = hasattr(self.policy, 'use_lstm') and self.policy.use_lstm
        if self.use_lstm:
            logger.info("LSTM is enabled in the policy network")
        
        logger.info(f"Initialized PPO agent with state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
    
    def select_action(self, state, deterministic=False):
        """Select an action based on the current state.
        
        Args:
            state: Current state
            deterministic: Whether to select the best action deterministically
            
        Returns:
            dict: Containing action, log_prob, value, and other info
        """
        logger.critical(f"Selecting action for state with shape {state.shape if hasattr(state, 'shape') else 'unknown'}")
        
        # Ensure state is a tensor
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, device=self.device).float()
        
        try:
            with torch.no_grad():
                # Forward pass through policy network
                # Pass and update hidden_state for LSTM memory
                if self.use_lstm:
                    logger.critical("Using LSTM in action selection")
                    action_probs, value, next_hidden = self.policy(state, self.hidden_state)
                    # Update hidden state for next time
                    self.hidden_state = next_hidden
                else:
                    action_probs, value, _ = self.policy(state)
                
                # Create categorical distribution
                action_distribution = torch.distributions.Categorical(action_probs)
                
                # Choose action
                if deterministic:
                    action = torch.argmax(action_probs, dim=1)
                else:
                    action = action_distribution.sample()
                
                # Get log probability
                log_prob = action_distribution.log_prob(action)
                
                # Create info dictionary
                info = {
                    'action_probs': action_probs,
                    'action_distribution': action_distribution,
                    'log_prob': log_prob,
                    'value': value
                }
                
                # Update steps counter
                self.steps_taken += 1
                
                # Apply menu action penalty to reduce likelihood of selecting
                # actions that have frequently led to menus
                if self.training and hasattr(self, 'get_menu_action_penalty'):
                    action_probs = info['action_probs'].clone()
                    
                    # Apply penalties to actions known to cause menus
                    for action_idx in range(self.action_dim):
                        penalty = self.get_menu_action_penalty(action_idx)
                        if penalty > 0:
                            action_probs[0, action_idx] *= (1.0 - penalty)
                            
                    # Renormalize probabilities
                    action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)
                    
                    # Create new distribution
                    penalized_distribution = torch.distributions.Categorical(action_probs)
                    
                    # Sample from penalized distribution
                    if deterministic:
                        action = torch.argmax(action_probs, dim=1)
                    else:
                        action = penalized_distribution.sample()
                        
                    # Get log probability from original distribution for learning
                    log_prob = info['log_prob']
                
                # Return action, log_prob, and value
                logger.critical(f"Returning action: {action.cpu().numpy().item()}, with value: {value}")
                return {
                    'action': action,
                    'log_prob': log_prob,
                    'value': value,
                    'action_probs': action_probs,
                }
                
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Return a random action as fallback
            random_action = torch.randint(0, self.action_dim, (1,), device=self.device)
            return {
                'action': random_action,
                'log_prob': torch.tensor(0.0, device=self.device),
                'value': torch.tensor(0.0, device=self.device),
                'action_probs': torch.ones(1, self.action_dim, device=self.device) / self.action_dim,
            }
    
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
    
    def update(self, experiences: List) -> Dict[str, float]:
        """Update policy and value function.
        
        Args:
            experiences: List of experience tuples (state, action, reward, next_state, done, log_prob, value)
            
        Returns:
            Dict with update statistics
        """
        if not self.training:
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
            
        # Process experiences
        for exp in experiences:
            state, action, reward, next_state, done, log_prob, value = exp
            
            # Store experience
            info = {'log_prob': log_prob, 'value': value}
            self.store_experience(state, action, reward, next_state, done, info)
            
        # Check if enough experiences are stored
        if self.memory.size() < self.updater.batch_size:
            logger.warning(f"Not enough experiences for update: {self.memory.size()} < {self.updater.batch_size}")
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
            
        # Compute returns and advantages
        self.memory.compute_returns(self.gamma, self.gae_lambda)
        
        # If using mixed precision, update with gradient scaling
        if self.use_amp:
            # Update with mixed precision
            with torch.cuda.amp.autocast():
                # Run standard update through updater but with mixed precision
                update_stats = self._update_with_amp()
        else:
            # Standard update
            update_stats = self.updater.update(self.memory)
            
        # Clear memory after update
        self.memory.clear()
        
        return update_stats
    
    def _update_with_amp(self) -> Dict[str, float]:
        """Perform update with automatic mixed precision.
        
        Returns:
            Dict with update statistics
        """
        # Initialize stats
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        # Custom update loop similar to updater.update but with AMP
        for epoch in range(self.updater.ppo_epochs):
            # Get batch iterator
            batch_iter = self.memory.get_batch_iterator(self.updater.batch_size, shuffle=True)
            
            # Process each batch
            for batch in batch_iter:
                # Extract batch data
                states, actions, old_log_probs, returns, advantages, old_values = batch
                
                # Forward pass with autocast
                with torch.cuda.amp.autocast():
                    # Get action distributions and values
                    action_dists, values = self.network(states)
                    
                    # Calculate new log probs
                    new_log_probs = action_dists.log_prob(actions)
                    
                    # Calculate policy loss (clipped PPO objective)
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.updater.clip_param, 1.0 + self.updater.clip_param) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate value loss (clipped)
                    value_pred_clipped = old_values + torch.clamp(values - old_values, -self.updater.clip_param, self.updater.clip_param)
                    value_loss_unclipped = (values - returns) ** 2
                    value_loss_clipped = (value_pred_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                    
                    # Calculate entropy bonus
                    entropy = action_dists.entropy().mean()
                    
                    # Calculate total loss
                    loss = policy_loss + self.updater.value_coef * value_loss - self.updater.entropy_coef * entropy
                
                # Zero gradients
                self.updater.optimizer.zero_grad()
                
                # Backward pass with scaler
                self.scaler.scale(loss).backward()
                
                # Clip gradients using scaler
                self.scaler.unscale_(self.updater.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.updater.max_grad_norm)
                
                # Step optimizer with scaler
                self.scaler.step(self.updater.optimizer)
                
                # Update scaler
                self.scaler.update()
                
                # Update stats
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                num_updates += 1
        
        # Calculate average stats
        if num_updates > 0:
            avg_policy_loss = total_policy_loss / num_updates
            avg_value_loss = total_value_loss / num_updates
            avg_entropy = total_entropy / num_updates
        else:
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            avg_entropy = 0.0
        
        return {
            'actor_loss': avg_policy_loss,
            'critic_loss': avg_value_loss,
            'entropy': avg_entropy
        }
            
    def set_training(self, training: bool) -> None:
        """Set agent to training or evaluation mode.
        
        Args:
            training: Whether agent should be in training mode
        """
        self.training = training
        
        # Set network mode
        if training:
            self.network.train()
        else:
            self.network.eval()
            
        logger.debug(f"Set agent to {'training' if training else 'evaluation'} mode")
    
    def train(self) -> None:
        """Set agent to training mode."""
        self.set_training(True)
    
    def eval(self) -> None:
        """Set agent to evaluation mode."""
        self.set_training(False)
    
    def to(self, device: Union[str, torch.device]) -> 'PPOAgent':
        """Move agent to specified device.
        
        Args:
            device: Device to move agent to
            
        Returns:
            Self for chaining
        """
        self.device = torch.device(device)
        
        # Move network to device
        self.network.to(self.device)
        
        # Update components with new device
        self.policy.device = self.device
        self.value_function.device = self.device
        self.memory.device = self.device
        self.updater.device = self.device
        
        return self
    
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
            Dict[str, Any]: Dictionary of statistics
        """
        return {
            'steps_taken': self.steps_taken,
            'episodes_completed': self.episodes_completed,
            'memory_size': len(self.memory) if hasattr(self.memory, '__len__') else 0,
            'device': str(self.device)
        }
    
    def parameters(self):
        """Get network parameters for optimization.
        
        Returns:
            Parameters of the neural network
        """
        return self.network.parameters()
        
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for checkpointing.
        
        Returns:
            Dict[str, Any]: State dictionary
        """
        return {
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
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary for checkpointing.
        
        Args:
            state_dict: State dictionary containing agent state
        """
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
        
        # Load other parameters
        self.gamma = state_dict.get('gamma', self.gamma)
        self.gae_lambda = state_dict.get('gae_lambda', self.gae_lambda)
        self.steps_taken = state_dict.get('steps_taken', 0)
        self.episodes_completed = state_dict.get('episodes_completed', 0)
        
        # Update value function parameters
        self.value_function.set_params(gamma=self.gamma, gae_lambda=self.gae_lambda)
        
        logger.info("Agent state loaded successfully")
    
    def update_menu_action_tracking(self, action):
        """Track which actions tend to cause menu transitions.
        
        Args:
            action: The action that led to a menu
        """
        if isinstance(action, torch.Tensor):
            action = action.item()
            
        # Initialize counter if this is the first time seeing this action
        if action not in self.menu_actions:
            self.menu_actions[action] = deque(maxlen=self.menu_action_memory)
            
        # Add timestamp to track recency
        self.menu_actions[action].append(time.time())
        
        logger.info(f"Updated menu action tracking for action {action}: {len(self.menu_actions[action])} occurrences")
    
    def get_menu_action_penalty(self, action):
        """Get penalty for actions that frequently cause menus.
        
        Args:
            action: The action to check
            
        Returns:
            float: Penalty factor (0 to menu_penalty_factor)
        """
        if isinstance(action, torch.Tensor):
            action = action.item()
            
        if action not in self.menu_actions or len(self.menu_actions[action]) == 0:
            return 0.0
            
        # Calculate decay based on time since last occurrence
        now = time.time()
        recent_count = sum(1 for t in self.menu_actions[action] if now - t < 300)  # Count occurrences in last 5 minutes
        
        # Normalize by memory size
        recent_ratio = recent_count / self.menu_action_memory
        
        # Return penalty scaled by factor
        return recent_ratio * self.menu_penalty_factor 

    def reset(self):
        """Reset agent state between episodes."""
        logger.critical("Resetting agent state")
        # Reset LSTM hidden state between episodes
        if self.use_lstm:
            logger.critical("Resetting LSTM hidden state")
            self.hidden_state = None
        
        # Other resets if needed
        self.episodes_completed += 1
        
        # Reset menu action tracking
        self.menu_actions = {}
        self.menu_actions = deque(maxlen=self.menu_action_memory)
        
        logger.info("Agent state reset successfully") 