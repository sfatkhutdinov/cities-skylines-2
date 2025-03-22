"""
PPO agent module for Cities: Skylines 2.

This module implements the main PPO agent class, integrating the policy,
value function, memory, and updater components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import os
import numpy as np
import random
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import deque
import math

from src.config.hardware_config import HardwareConfig
from src.agent.core.network import OptimizedNetwork
from src.agent.core.policy import Policy
from src.agent.core.value_function import ValueFunction
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
            state_dim: Dimensions of the state space
            action_dim: Dimensions of the action space
            config: Hardware configuration with device preferences
            use_amp: Whether to use automatic mixed precision
        """
        # Hardware setup
        if config is None:
            config = HardwareConfig()
        
        self.device = torch.device(config.device if hasattr(config, "device") else "cpu")
        self.use_amp = use_amp and 'cuda' in str(self.device)
        
        if self.use_amp:
            logger.info("Using automatic mixed precision (AMP)")
        
        # Print logging info on initialization
        device_name = torch.cuda.get_device_name(self.device) if 'cuda' in str(self.device) else "CPU"
        logger.critical(f"Initializing PPO agent on {self.device} ({device_name})")
        
        # Create neural network with proper parameters based on state shape
        if isinstance(state_dim, tuple):
            if len(state_dim) == 3:  # Visual input (channels, height, width)
                input_channels = state_dim[0]
                frame_size = (state_dim[1], state_dim[2])
                is_visual_input = True
            elif len(state_dim) == 1:  # Vector input (size,)
                input_channels = state_dim[0]
                frame_size = (1, 1)  # Dummy size for vector input
                is_visual_input = False
            else:
                raise ValueError(f"Unsupported state shape: {state_dim}")
        elif isinstance(state_dim, int):  # Vector input as int
            input_channels = state_dim
            frame_size = (1, 1)
            is_visual_input = False
        else:
            raise ValueError(f"Unsupported state_dim type: {type(state_dim)}")
            
        logger.info(f"Creating OptimizedNetwork with input_channels={input_channels}, "
                   f"frame_size={frame_size}, is_visual_input={is_visual_input}")
        
        self.network = OptimizedNetwork(
            input_channels=input_channels,
            num_actions=action_dim,
            frame_size=frame_size,
            feature_size=512,
            hidden_size=256,
            use_lstm=True,
            frames_to_stack=4 if is_visual_input else 1,
            device=self.device,
            use_attention=True,
            is_visual_input=is_visual_input
        )
        
        # Set up agent components
        self.policy = Policy(self.network, action_dim, self.device)
        self.value_function = ValueFunction(self.network, self.device)
        self.memory = Memory(self.device)
        self.updater = PPOUpdater(self.network, self.device)
        
        # Agent parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.update_frequency = 32
        
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
        
        # Track if we're currently in a menu - initially assume we're not
        self.in_menu = False
        
        # Enhanced menu action tracking with severe penalties
        self.menu_actions = {}  # Maps action index to count of menu occurrences
        self.menu_action_memory = 50  # Remember the last 50 menu occurrences
        self.menu_penalty_factor = 5.0  # Increased penalty factor for actions that cause menus
        
        # Track specific menu-causing actions
        self.menu_entry_causes = {
            'escape_key': [],  # Track actions that involved ESC key and caused menu entry
            'gear_icon': [],   # Track actions that involved potential gear icon clicks
            'unknown': []      # Track other actions that somehow caused menu entry
        }
        self.menu_exit_rewards = []  # Track actions that successfully exited menus
        
        # Maximum number of entries to remember per category
        self.max_menu_cause_entries = 25
        
        # Initialize LSTM hidden state
        self.hidden_state = None
        self.use_lstm = hasattr(self.policy, 'use_lstm') and self.policy.use_lstm
        if self.use_lstm:
            logger.info("LSTM is enabled in the policy network")
        
        # Action smoothing parameters
        self.action_history = []
        self.action_memory_size = 5  # Remember last 5 actions for smoothing
        self.smooth_factor = 0.7  # Weight for current action vs. historical actions
        self.continuous_actions = set([0, 1, 2, 3, 4, 5, 6, 7])  # Indices of movement/camera actions
        self.last_continuous_action = None
        self.action_momentum = 0.8  # Momentum factor for continuous actions
        logger.info(f"Action smoothing enabled with smooth_factor={self.smooth_factor}")
        
        logger.info(f"Initialized PPO agent with state_dim={state_dim}, action_dim={action_dim}, device={self.device}")
    
    def select_action(self, state, deterministic=False, info=None):
        """Select an action based on current state.
        
        Args:
            state: Current state observation
            deterministic: Whether to select the highest probability action
            info: Additional information from environment for action selection
            
        Returns:
            Selected action
        """
        try:
            with torch.no_grad():
                # Forward pass through policy network
                action_probs, value, next_hidden = self.policy(state, self.hidden_state)
                
                # Update hidden state for next time
                self.hidden_state = next_hidden
                
                # Process environment info if provided
                if info is not None:
                    self.process_step_info(info)
                    
                    # Track if we're currently in a menu
                    self.in_menu = info.get('in_menu', False)
                
                # Apply menu action penalties
                action_probs = self.adjust_action_probs(action_probs)
                
                # Apply action smoothing from parent class
                action_probs = self._apply_action_smoothing(action_probs)
                
                # Create categorical distribution
                action_distribution = torch.distributions.Categorical(action_probs)
                
                # Choose action
                if deterministic:
                    action = torch.argmax(action_probs, dim=1)
                else:
                    action = action_distribution.sample()
                
                # Store action in history for future smoothing
                action_item = action.cpu().item()
                self._update_action_history(action_item)
                
                # Store current value and log probability for update
                self.last_state = state
                self.last_action = action_item
                self.last_value = value
                self.last_log_prob = action_distribution.log_prob(action)
                
                return action_item
        except Exception as e:
            logger.error(f"Error selecting action: {e}")
            # Fallback to random action
            return random.randint(0, self.action_dim - 1)
    
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
    
    def process_step_info(self, info):
        """Process additional information from environment step.
        
        Args:
            info: Information dict from environment step
            
        Returns:
            Modified action probabilities with penalties applied
        """
        try:
            # Check if menu-related info is available
            if 'menu_entered' in info and info['menu_entered']:
                # Extract menu entry cause if available
                cause = 'unknown'
                # Check for specific known causes
                if 'is_escape_action' in info and info['is_escape_action']:
                    cause = 'escape_key'
                elif 'is_potential_gear_click' in info and info['is_potential_gear_click']:
                    cause = 'gear_icon'
                
                # Get the action that caused menu entry
                action_idx = self.last_action if hasattr(self, 'last_action') else -1
                
                # Track this action in appropriate category
                if cause in self.menu_entry_causes:
                    self.menu_entry_causes[cause].append(action_idx)
                    # Keep only the most recent entries
                    if len(self.menu_entry_causes[cause]) > self.max_menu_cause_entries:
                        self.menu_entry_causes[cause].pop(0)
                    
                    logger.warning(f"Tracked menu-causing action {action_idx} as '{cause}'")
                
                # Track in overall menu actions dict
                if action_idx >= 0:
                    if action_idx in self.menu_actions:
                        self.menu_actions[action_idx] += 1
                    else:
                        self.menu_actions[action_idx] = 1
            
            # Track successful menu exits
            if 'menu_exited' in info and info['menu_exited']:
                if 'is_escape_action' in info and info['is_escape_action']:
                    # Get the action that successfully exited the menu
                    action_idx = self.last_action if hasattr(self, 'last_action') else -1
                    if action_idx >= 0:
                        self.menu_exit_rewards.append(action_idx)
                        # Keep only the most recent entries
                        if len(self.menu_exit_rewards) > self.max_menu_cause_entries:
                            self.menu_exit_rewards.pop(0)
                        
                        logger.info(f"Tracked menu exit action {action_idx}")
            
            # Return success
            return True
        except Exception as e:
            logger.error(f"Error processing step info: {e}")
            return False
    
    def adjust_action_probs(self, action_probs):
        """Adjust action probabilities to penalize menu-causing actions.
        
        Args:
            action_probs: Original action probabilities
            
        Returns:
            Modified action probabilities with penalties applied
        """
        try:
            # Clone the probabilities tensor to avoid modifying the original
            if isinstance(action_probs, torch.Tensor):
                adjusted_probs = action_probs.clone()
            else:
                return action_probs
            
            # Apply penalties for menu-causing actions
            with torch.no_grad():
                for cause, actions in self.menu_entry_causes.items():
                    for action_idx in actions:
                        if action_idx >= 0 and action_idx < action_probs.shape[1]:
                            # Apply severe penalties for escape key and gear icon clicks
                            penalty = self.menu_penalty_factor
                            if cause in ['escape_key', 'gear_icon']:
                                # More severe penalty for these specific actions
                                penalty = self.menu_penalty_factor * 2.0
                            
                            # Reduce probability of this action
                            adjusted_probs[0, action_idx] *= max(0.01, 1.0 - penalty / 10.0)
                
                # Apply bonuses for successful menu exit actions
                for action_idx in self.menu_exit_rewards:
                    if action_idx >= 0 and action_idx < action_probs.shape[1]:
                        # Small bonus for actions that successfully exited menus
                        # But only apply this if we're actually in a menu
                        if self.in_menu:
                            adjusted_probs[0, action_idx] *= 1.2  # 20% bonus
                
                # Renormalize the probabilities
                if adjusted_probs.sum() > 0:
                    adjusted_probs = adjusted_probs / adjusted_probs.sum(dim=1, keepdim=True)
            
            return adjusted_probs
        except Exception as e:
            logger.error(f"Error adjusting action probabilities: {e}")
            return action_probs
    
    def reset(self):
        """Reset agent state for a new episode."""
        logger.critical("Resetting agent state for new episode")
        
        # Reset memory
        self.reset_memory()
        
        # Reset LSTM/RNN hidden state if present
        if self.use_lstm:
            logger.critical("Resetting LSTM hidden state")
            self.hidden_state = None
        
        # Reset action history
        self.action_history = []
        
        # Reset episode counter and reward
        self.episode_reward = 0.0
        self.steps_since_reset = 0
        
        # Reset last actions and states
        self.last_action = None
        self.last_state = None
        self.last_value = None
        self.last_log_prob = None
        
        # We don't reset menu_actions tracking between episodes
        # This helps the agent learn which actions cause menus across episodes
        
        # Increment episode counter
        self.episodes_completed += 1
        
        logger.info("Agent state reset for new episode")
    
    def reset_memory(self):
        """Reset memory buffer."""
        if hasattr(self, 'memory') and self.memory is not None:
            logger.critical("Resetting experience memory")
            self.memory.clear()
    
    def _apply_action_smoothing(self, action_probs):
        """Apply smoothing to action probabilities to reduce jerky movements.
        
        Args:
            action_probs: Current raw action probabilities
            
        Returns:
            Smoothed action probabilities
        """
        smoothed_probs = action_probs.clone()
        
        # Apply momentum for continuous actions if we have a history
        if self.last_continuous_action is not None:
            # Increase probability of the last continuous action
            for action_idx in self.continuous_actions:
                if action_idx == self.last_continuous_action:
                    # Boost probability of continuing the same continuous action
                    # but only for movement/camera actions
                    current_prob = smoothed_probs[0, action_idx].item()
                    boosted_prob = current_prob + (1 - current_prob) * self.action_momentum
                    smoothed_probs[0, action_idx] = boosted_prob
            
            # Renormalize
            smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=1, keepdim=True)
        
        # For non-deterministic selection, we apply historical smoothing
        if len(self.action_history) > 0:
            # Create historical distribution
            action_counter = {}
            for act in self.action_history:
                if act in action_counter:
                    action_counter[act] += 1
                else:
                    action_counter[act] = 1
            
            # Calculate historical probabilities
            hist_probs = torch.zeros_like(smoothed_probs)
            for act, count in action_counter.items():
                hist_probs[0, act] = count / len(self.action_history)
            
            # Blend current and historical probabilities
            smoothed_probs = self.smooth_factor * smoothed_probs + (1 - self.smooth_factor) * hist_probs
            
            # Renormalize
            smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=1, keepdim=True)
        
        return smoothed_probs
    
    def _update_action_history(self, action):
        """Update the action history with a new action.
        
        Args:
            action: New action to add to history
        """
        self.action_history.append(action)
        if len(self.action_history) > self.action_memory_size:
            self.action_history.pop(0)
        
        # Track continuous actions separately
        if action in self.continuous_actions:
            self.last_continuous_action = action