"""
Value function module for PPO agent in Cities: Skylines 2.

This module implements the value function component of the PPO agent,
handling state value estimation and advantage calculation.
"""

import torch
import torch.nn as nn
import logging
import numpy as np
from typing import Dict, Tuple, List, Any, Optional

logger = logging.getLogger(__name__)

class ValueFunction:
    """Implements the value function component for the PPO agent."""
    
    def __init__(self, network, device: torch.device):
        """Initialize value function component.
        
        Args:
            network: Neural network that outputs state values
            device: Device to run computations on
        """
        self.network = network
        self.device = device
        
        # Advantage computation parameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.normalize_advantages = True
        
    def estimate_value(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate value of a state.
        
        Args:
            state: Current observation
            
        Returns:
            Estimated state value
        """
        # Ensure state is on the correct device and has correct shape
        if state.device != self.device:
            state = state.to(self.device)
            
        # Get value from network
        with torch.no_grad():
            # Add batch dimension if needed
            if len(state.shape) == 3:  # Single state with shape [C, H, W]
                state_batch = state.unsqueeze(0)
            else:
                state_batch = state
                
            # Forward pass through network
            _, value = self.network(state_batch)
            
            # Remove batch dimension if needed
            if len(state.shape) == 3:
                value = value.squeeze(0) if value.dim() > 0 else value
                
        return value
    
    def compute_returns(self, rewards: torch.Tensor, 
                        dones: torch.Tensor, 
                        values: torch.Tensor, 
                        next_value: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute discounted returns.
        
        Args:
            rewards: Tensor of rewards
            dones: Tensor of done flags
            values: Tensor of state values
            next_value: Value of next state after last reward
            
        Returns:
            Tensor of discounted returns
        """
        # Ensure inputs are on the correct device
        if rewards.device != self.device:
            rewards = rewards.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
        if values.device != self.device:
            values = values.to(self.device)
            
        returns = torch.zeros_like(rewards)
        
        # If next_value is not provided, assume zero
        if next_value is None:
            next_value = torch.zeros(1, device=self.device)
        
        # Initialize with next value
        next_return = next_value
        
        # Compute returns backwards
        for t in reversed(range(len(rewards))):
            # If done, next return is zero, otherwise it's discounted next return
            next_return = rewards[t] + self.gamma * next_return * (1 - dones[t])
            returns[t] = next_return
            
        return returns
    
    def compute_gae(self, rewards: torch.Tensor, 
                   values: torch.Tensor, 
                   dones: torch.Tensor, 
                   next_value: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            rewards: Tensor of rewards
            values: Tensor of state values
            dones: Tensor of done flags
            next_value: Value of next state after last reward
            
        Returns:
            Tuple of tensors (returns, advantages)
        """
        # Ensure inputs are on the correct device
        if rewards.device != self.device:
            rewards = rewards.to(self.device)
        if values.device != self.device:
            values = values.to(self.device)
        if dones.device != self.device:
            dones = dones.to(self.device)
            
        # If next_value is not provided, assume zero
        if next_value is None:
            next_value = torch.zeros(1, device=self.device)
        
        # Compute advantages using GAE
        batch_size = len(rewards)
        returns = torch.zeros_like(rewards)
        advantages = torch.zeros_like(rewards)
        
        # Initialize with next value
        next_value = next_value
        next_advantage = 0.0
        
        # Compute advantages and returns backwards
        for t in reversed(range(batch_size)):
            # Compute delta (TD error)
            if t == batch_size - 1:
                # For last step, use next_value for bootstrapping
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                # For other steps, use value of next timestep
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
                
            # Update advantage (exponentially weighted sum of TD errors)
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1 - dones[t])
            next_advantage = advantages[t]
            
            # Compute returns (value + advantage)
            returns[t] = advantages[t] + values[t]
            
        # Normalize advantages
        if self.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        return returns, advantages
    
    def set_params(self, gamma: Optional[float] = None, 
                  gae_lambda: Optional[float] = None, 
                  normalize_advantages: Optional[bool] = None) -> None:
        """Set parameters for advantage calculation.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            normalize_advantages: Whether to normalize advantages
        """
        if gamma is not None:
            self.gamma = gamma
        if gae_lambda is not None:
            self.gae_lambda = gae_lambda
        if normalize_advantages is not None:
            self.normalize_advantages = normalize_advantages
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization.
        
        Returns:
            Dict containing state information
        """
        return {
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'normalize_advantages': self.normalize_advantages
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary.
        
        Args:
            state_dict: State dictionary to load
        """
        self.gamma = state_dict.get('gamma', 0.99)
        self.gae_lambda = state_dict.get('gae_lambda', 0.95)
        self.normalize_advantages = state_dict.get('normalize_advantages', True) 