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
        # Store state and action dimensions
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Hardware setup
        if config is None:
            config = HardwareConfig()
        
        self.device = torch.device(config.device if hasattr(config, "device") else "cpu")
        self.use_amp = use_amp and 'cuda' in str(self.device)
        
        # Initialize GradScaler if using AMP
        self.scaler = None
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Using automatic mixed precision (AMP) with GradScaler")
        
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
            frames_to_stack=config.frame_stack if hasattr(config, 'frame_stack') else 1,
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
    
    def update(self) -> Dict[str, float]:
        """Update policy and value function using data in self.memory."""
        if not self.training:
            logger.warning("Update called while agent is not in training mode. Skipping.")
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
            
        # Check if enough experiences are stored in memory
        updater_batch_size = getattr(self.updater, 'batch_size', 64) # Default if not found
        if self.memory.size() < updater_batch_size:
            logger.warning(f"Not enough experiences for update: {self.memory.size()} < {updater_batch_size}")
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0}
            
        # Compute returns and advantages using memory data
        logger.critical("Computing returns and advantages...")
        self.memory.compute_returns(self.gamma, self.gae_lambda)
        logger.critical("Returns and advantages computed.")
        
        logger.critical("Calling PPOUpdater.update...")
        update_stats = self.updater.update(self.memory, scaler=self.scaler if self.use_amp else None)
        logger.critical("PPOUpdater.update completed.")
            
        # Clear memory after update
        logger.critical("Clearing agent memory after update.")
        self.memory.clear()
        
        return update_stats
    
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
        
        # Create state dict using the dedicated method
        agent_state_dict = self.state_dict()
        
        # Save state dict
        model_path = os.path.join(path, 'agent.pth')
        torch.save(agent_state_dict, model_path)
        
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
        
        # Load network, policy, value_function, updater states
        if 'network' in state_dict:
            self.network.load_state_dict(state_dict['network'])
        if 'policy' in state_dict:
            self.policy.load_state_dict(state_dict['policy'])
        if 'value_function' in state_dict:
            self.value_function.load_state_dict(state_dict['value_function'])
        if 'updater' in state_dict:
            # Make sure updater has a load_state_dict method
            if hasattr(self.updater, 'load_state_dict') and callable(self.updater.load_state_dict):
                self.updater.load_state_dict(state_dict['updater'])
            else:
                logger.warning("Updater does not support loading state dict, skipping.")
                
        # Restore training parameters
        self.gamma = state_dict.get('gamma', self.gamma)
        self.gae_lambda = state_dict.get('gae_lambda', self.gae_lambda)
        self.steps_taken = state_dict.get('steps_taken', 0)
        self.episodes_completed = state_dict.get('episodes_completed', 0)
        
        logger.info(f"Loaded agent state from {model_path}")
        
        # Load optimizer state
        optimizer_path = os.path.join(path, 'optimizer.pth')
        if os.path.exists(optimizer_path):
            try:
                optim_state = torch.load(optimizer_path, map_location=self.device)
                if 'optimizer' in optim_state:
                    self.updater.optimizer.load_state_dict(optim_state['optimizer'])
                    logger.info(f"Loaded optimizer state from {optimizer_path}")
                else:
                    logger.warning(f"Optimizer state not found in {optimizer_path}")
            except Exception as e:
                logger.error(f"Error loading optimizer state from {optimizer_path}: {e}")
        else:
            logger.warning(f"Optimizer state file not found at {optimizer_path}")
            
        # Load stats (optional, mainly for info)
        stats_path = os.path.join(path, 'stats.pth')
        if os.path.exists(stats_path):
            try:
                stats = torch.load(stats_path, map_location=self.device)
                # You might want to store/use these stats if needed, e.g.:
                # self.loaded_episode_rewards = stats.get('episode_rewards', [])
                # self.loaded_episode_lengths = stats.get('episode_lengths', [])
                logger.info(f"Loaded stats from {stats_path}")
            except Exception as e:
                logger.error(f"Error loading stats from {stats_path}: {e}")
    
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