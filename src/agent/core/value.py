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