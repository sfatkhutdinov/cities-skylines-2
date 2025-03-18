"""
Optimized neural network for Cities: Skylines 2 agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    """Optimized convolutional block with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.SiLU()  # SiLU (Swish) activation for better gradient flow
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bn(self.conv(x)))

class OptimizedNetwork(nn.Module):
    """Optimized neural network for the PPO agent."""
    
    def __init__(self, input_shape, num_actions, device=None):
        """Initialize the network.
        
        Args:
            input_shape: Shape of the input state (channels, height, width) or flattened size
            num_actions: Number of actions in the action space
            device: Computation device
        """
        super(OptimizedNetwork, self).__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing network on device: {self.device}")
        
        # Handle different types of input shapes
        if isinstance(input_shape, tuple) and len(input_shape) >= 3:
            # It's an image with shape (channels, height, width)
            in_channels, height, width = input_shape
            self.is_visual_input = True
        elif isinstance(input_shape, tuple) and len(input_shape) == 1:
            # It's a flattened vector with shape (n,)
            in_channels, height, width = 1, 1, input_shape[0]
            self.is_visual_input = False
        elif isinstance(input_shape, int):
            # It's a flattened vector with size n
            in_channels, height, width = 1, 1, input_shape
            self.is_visual_input = False
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")
            
        # Define network architecture based on input type
        if self.is_visual_input:
            # Convolutional network for visual inputs
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            # Create a dummy input tensor to calculate output size
            with torch.no_grad():
                dummy_input = torch.zeros(1, in_channels, height, width)
                dummy_output = self.conv_layers(dummy_input)
                conv_output_size = int(np.prod(dummy_output.shape))
                logger.info(f"Conv output size: {conv_output_size}")
                
            # Shared feature layers
            self.shared_layers = nn.Sequential(
                nn.Linear(conv_output_size, 512),
                nn.ReLU()
            )
        else:
            # For vector inputs, we won't use conv layers
            self.conv_layers = nn.Identity()
            
            # Fully connected network for vector inputs
            self.shared_layers = nn.Sequential(
                nn.Linear(width, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU()
            )
            
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights before moving to device
        self.apply(self._init_weights)
        
        # Move all components to the specified device after creating all layers
        self.to(self.device)
        logger.info(f"Network moved to device: {self.device}")
        
        # Verify that all components are on the correct device
        for name, module in self.named_children():
            for param_name, param in module.named_parameters():
                if param.device != self.device:
                    logger.warning(f"Parameter {name}.{param_name} is on {param.device}, not {self.device}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action_probs, value)
        """
        # Ensure input is on the correct device 
        if x.device != self.device:
            x = x.to(self.device)
            
        # Handle batch dimension for visual inputs
        if self.is_visual_input:
            # Check if batch dimension is missing
            if len(x.shape) == 3:  # [C, H, W]
                x = x.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
                
            # Pass through convolutional layers
            x = self.conv_layers(x)
            
            # Flatten the features but keep batch dimension
            x = x.reshape(x.size(0), -1)
        else:
            # For vector inputs
            if len(x.shape) == 1:  # [D]
                x = x.unsqueeze(0)  # Add batch dimension -> [1, D]
                
        # Pass through shared layers
        features = self.shared_layers(x)
        
        # Get policy (action probabilities) and value
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        
        # Apply softmax to get probabilities
        action_probs = torch.softmax(action_logits, dim=1)
        
        # Squeeze value if needed
        value = value.squeeze(-1)
        
        return action_probs, value
        
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities for a state.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Action probabilities
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Process based on input type
        if self.is_visual_input:
            # Check if we need to add batch dimension
            if len(x.shape) == 3:  # [C, H, W]
                x = x.unsqueeze(0)  # Add batch dimension
                
            # Process through convolutional layers
            features = self.conv_layers(x)
            features = features.reshape(x.size(0), -1)
        else:
            # For vector inputs
            if len(x.shape) == 1:  # [D]
                x = x.unsqueeze(0)  # Add batch dimension
            features = x
            
        # Process through shared layers and policy head
        fc_features = self.shared_layers(features)
        logits = self.policy_head(fc_features)
        
        # Apply softmax to get probabilities
        return torch.softmax(logits, dim=1)
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value for a state.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            torch.Tensor: Value prediction
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Process based on input type
        if self.is_visual_input:
            # Check if we need to add batch dimension
            if len(x.shape) == 3:  # [C, H, W]
                x = x.unsqueeze(0)  # Add batch dimension
                
            # Process through convolutional layers
            features = self.conv_layers(x)
            features = features.reshape(x.size(0), -1)
        else:
            # For vector inputs
            if len(x.shape) == 1:  # [D]
                x = x.unsqueeze(0)  # Add batch dimension
            features = x
            
        # Process through shared layers and value head
        fc_features = self.shared_layers(features)
        value = self.value_head(fc_features)
        
        return value.squeeze(-1)

    def _init_weights(self, m):
        """Initialize network weights.
        
        Args:
            m: Module to initialize
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias) 