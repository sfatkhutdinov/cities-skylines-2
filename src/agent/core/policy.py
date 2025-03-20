"""
Policy module for PPO agent in Cities: Skylines 2.

This module implements the policy component of the PPO agent,
handling action selection, probability distributions, and entropy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

logger = logging.getLogger(__name__)

class Policy:
    """Implements the policy component for the PPO agent."""
    
    def __init__(self, network, action_dim: int, device: torch.device):
        """Initialize policy component.
        
        Args:
            network: Neural network that outputs action probabilities
            action_dim: Dimension of the action space
            device: Device to run computations on
        """
        self.network = network
        self.action_dim = action_dim
        self.device = device
        
        # Action selection parameters
        self.exploration_enabled = True
        self.temperature = 1.0  # Softmax temperature for exploration
        
        # Menu action tracking to penalize problematic actions
        self.menu_action_indices = []
        self.menu_action_penalties = {}
        self.menu_penalty_decay = 0.98
        
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Select an action based on the current state.
        
        Args:
            state: Current observation
            deterministic: Whether to select action deterministically
            
        Returns:
            Tuple containing:
                - Selected action tensor
                - Log probability tensor
                - Info dictionary with additional data
        """
        # Ensure state is on the correct device and has correct shape
        if state.device != self.device:
            state = state.to(self.device)
            
        # Get action probabilities from network
        with torch.no_grad():
            # Add batch dimension if needed
            if len(state.shape) == 3:  # Single state with shape [C, H, W]
                state_batch = state.unsqueeze(0)
            else:
                state_batch = state
                
            # Forward pass through network
            action_probs, value = self.network(state_batch)
            
            # Remove batch dimension if needed
            if len(state.shape) == 3:
                action_probs = action_probs.squeeze(0)
                value = value.squeeze(0) if value.dim() > 0 else value
            
            # Apply penalties to menu-opening actions
            action_probs = self._apply_action_penalties(action_probs)
            
            # Apply temperature for exploration
            if not deterministic and self.exploration_enabled:
                # Apply temperature scaling
                if self.temperature != 1.0:
                    action_probs = self._apply_temperature(action_probs)
            
        # Select action (deterministic or stochastic)
        if deterministic:
            action = action_probs.argmax().unsqueeze(0)
        else:
            action = torch.multinomial(action_probs, 1)
        
        # Get log probability of the chosen action
        log_prob = torch.log(action_probs[action.item()].clamp(min=1e-10))
        
        # Make sure log_prob is a tensor with proper dimensions
        if not isinstance(log_prob, torch.Tensor):
            log_prob = torch.tensor(log_prob, device=self.device)
        elif log_prob.dim() == 0:
            log_prob = log_prob.unsqueeze(0)
            
        # Create info dict
        info = {
            'action_probs': action_probs,
            'value': value,
            'entropy': self._compute_entropy(action_probs)
        }
        
        return action, log_prob, info
    
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for given states.
        
        Args:
            states: Batch of states
            actions: Batch of actions to evaluate
            
        Returns:
            Tuple containing:
                - Log probabilities of actions
                - State values
                - Entropy of action distributions
        """
        # Ensure inputs are on the correct device
        if states.device != self.device:
            states = states.to(self.device)
        if actions.device != self.device:
            actions = actions.to(self.device)
            
        # Forward pass through network
        action_probs, values = self.network(states)
        
        # Get log probabilities of actions
        log_probs = torch.log(torch.gather(action_probs, 1, actions.unsqueeze(1)).squeeze(1).clamp(min=1e-10))
        
        # Compute entropy of distributions
        entropy = -torch.sum(action_probs * torch.log(action_probs.clamp(min=1e-10)), dim=1)
        
        return log_probs, values, entropy
    
    def _apply_action_penalties(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Apply penalties to problematic actions.
        
        Args:
            action_probs: Action probability distribution
            
        Returns:
            Modified action probability distribution
        """
        if not self.menu_action_indices and not self.menu_action_penalties:
            return action_probs
            
        # Create a penalty mask (default 1.0 for all actions)
        penalty_mask = torch.ones_like(action_probs)
        
        # Apply specific penalties to known menu actions
        for action_idx, penalty in self.menu_action_penalties.items():
            if 0 <= action_idx < len(penalty_mask):
                # Reduce probability of this action by the penalty factor
                penalty_mask[action_idx] = max(0.05, 1.0 - penalty)  # Ensure minimal probability
        
        # Apply the mask to reduce probabilities of problematic actions
        action_probs = action_probs * penalty_mask
        
        # Renormalize probabilities
        if action_probs.sum() > 0:
            action_probs = action_probs / action_probs.sum()
        else:
            # Safety fallback if all actions are severely penalized
            action_probs = torch.ones_like(action_probs) / action_probs.size(0)
            
        return action_probs
    
    def _apply_temperature(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to action probabilities.
        
        Args:
            action_probs: Action probability distribution
            
        Returns:
            Modified action probability distribution
        """
        # Convert probs to logits
        logits = torch.log(action_probs.clamp(min=1e-10))
        
        # Apply temperature
        scaled_logits = logits / self.temperature
        
        # Convert back to probabilities
        scaled_probs = F.softmax(scaled_logits, dim=-1)
        
        return scaled_probs
    
    def _compute_entropy(self, action_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of action probability distribution.
        
        Args:
            action_probs: Action probability distribution
            
        Returns:
            Entropy value
        """
        return -torch.sum(action_probs * torch.log(action_probs.clamp(min=1e-10)))
    
    def register_menu_action(self, action_idx: int, penalty: float = 0.5) -> None:
        """Register an action as a menu action to be penalized.
        
        Args:
            action_idx: Index of action to register
            penalty: Penalty factor (0 to 1)
        """
        if action_idx not in self.menu_action_indices:
            self.menu_action_indices.append(action_idx)
            
        self.menu_action_penalties[action_idx] = min(1.0, max(0.0, penalty))
        logger.info(f"Registered menu action {action_idx} with penalty {penalty:.2f}")
        
    def update_penalty(self, action_idx: int, reward: float) -> None:
        """Update penalty for an action based on reward.
        
        Args:
            action_idx: Index of action to update
            reward: Reward received after taking action
        """
        # Only update if the action is already registered or has a penalty
        if action_idx in self.menu_action_indices or action_idx in self.menu_action_penalties:
            # If large negative reward, increase penalty
            if reward < -0.5:
                current_penalty = self.menu_action_penalties.get(action_idx, 0.0)
                # Increase penalty proportionally to negative reward
                new_penalty = min(1.0, current_penalty + abs(min(-0.1, reward)) * 0.1)
                self.menu_action_penalties[action_idx] = new_penalty
                
                if action_idx not in self.menu_action_indices:
                    self.menu_action_indices.append(action_idx)
                    
                logger.debug(f"Increased penalty for action {action_idx} to {new_penalty:.2f} due to reward {reward:.2f}")
    
    def decay_penalties(self) -> None:
        """Decay penalties for registered menu actions over time."""
        if not self.menu_action_penalties:
            return
            
        # Decay all penalties
        for action_idx in list(self.menu_action_penalties.keys()):
            # Decay penalty
            current_penalty = self.menu_action_penalties[action_idx]
            decayed_penalty = current_penalty * self.menu_penalty_decay
            
            # Remove penalty if it drops below threshold
            if decayed_penalty < 0.05:
                del self.menu_action_penalties[action_idx]
                if action_idx in self.menu_action_indices:
                    self.menu_action_indices.remove(action_idx)
            else:
                self.menu_action_penalties[action_idx] = decayed_penalty
    
    def set_exploration(self, enabled: bool, temperature: Optional[float] = None) -> None:
        """Set exploration parameters.
        
        Args:
            enabled: Whether exploration is enabled
            temperature: Optional temperature parameter
        """
        self.exploration_enabled = enabled
        if temperature is not None:
            self.temperature = max(0.01, temperature)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization.
        
        Returns:
            Dict containing state information
        """
        return {
            'menu_action_indices': self.menu_action_indices,
            'menu_action_penalties': self.menu_action_penalties,
            'menu_penalty_decay': self.menu_penalty_decay,
            'exploration_enabled': self.exploration_enabled,
            'temperature': self.temperature
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        self.menu_action_indices = state_dict.get('menu_action_indices', [])
        self.menu_action_penalties = state_dict.get('menu_action_penalties', {})
        self.menu_penalty_decay = state_dict.get('menu_penalty_decay', 0.98)
        self.exploration_enabled = state_dict.get('exploration_enabled', True)
        self.temperature = state_dict.get('temperature', 1.0) 