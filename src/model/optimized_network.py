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
        logger.critical(f"Initializing network with input_shape={input_shape}, num_actions={num_actions}, device={self.device}")
        
        # Handle different types of input shapes
        if isinstance(input_shape, tuple) and len(input_shape) >= 3:
            # It's an image with shape (channels, height, width)
            in_channels, height, width = input_shape
            self.is_visual_input = True
            logger.critical(f"Visual input detected: channels={in_channels}, height={height}, width={width}")
        elif isinstance(input_shape, tuple) and len(input_shape) == 1:
            # It's a flattened vector with shape (n,)
            in_channels, height, width = 1, 1, input_shape[0]
            self.is_visual_input = False
            logger.critical(f"Vector input detected (tuple): size={input_shape[0]}")
        elif isinstance(input_shape, int):
            # It's a flattened vector with size n
            in_channels, height, width = 1, 1, input_shape
            self.is_visual_input = False
            logger.critical(f"Vector input detected (int): size={input_shape}")
        else:
            error_msg = f"Unsupported input shape: {input_shape}, type: {type(input_shape)}"
            logger.critical(error_msg)
            raise ValueError(error_msg)
            
        # Define network architecture based on input type
        if self.is_visual_input:
            # Convolutional network for visual inputs
            logger.critical("Creating convolutional network for visual input")
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU()
            )
            
            # Create a dummy input tensor to calculate output size
            logger.critical("Calculating convolutional output size...")
            try:
                with torch.no_grad():
                    dummy_input = torch.zeros(1, in_channels, height, width)
                    dummy_output = self.conv_layers(dummy_input)
                    conv_output_size = int(np.prod(dummy_output.shape))
                    logger.critical(f"Conv dummy input shape: {dummy_input.shape}")
                    logger.critical(f"Conv dummy output shape: {dummy_output.shape}")
                    logger.critical(f"Conv output size: {conv_output_size}")
            except Exception as e:
                logger.critical(f"Error calculating conv output size: {e}")
                import traceback
                logger.critical(f"Traceback: {traceback.format_exc()}")
                # Use a conservative estimate
                conv_output_size = 1024
                logger.critical(f"Using fallback conv output size: {conv_output_size}")
                
            # Shared feature layers
            logger.critical(f"Creating shared layers with input size {conv_output_size}")
            self.shared_layers = nn.Sequential(
                nn.Linear(conv_output_size, 512),
                nn.ReLU()
            )
        else:
            # For vector inputs, we won't use conv layers
            logger.critical("Creating linear network for vector input")
            self.conv_layers = nn.Identity()
            
            # Fully connected network for vector inputs
            logger.critical(f"Creating shared layers with input size {width}")
            self.shared_layers = nn.Sequential(
                nn.Linear(width, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU()
            )
            
        # Policy head
        logger.critical("Creating policy head")
        self.policy_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Value head
        logger.critical("Creating value head")
        self.value_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Initialize weights before moving to device
        logger.critical("Initializing network weights")
        self.apply(self._init_weights)
        
        # Move all components to the specified device after creating all layers
        try:
            logger.critical(f"Moving network to device: {self.device}")
            self.to(self.device)
            logger.critical(f"Network successfully moved to device: {self.device}")
        except Exception as e:
            logger.critical(f"Error moving network to device: {e}")
            import traceback
            logger.critical(f"Traceback: {traceback.format_exc()}")
        
        # Verify that all components are on the correct device
        device_issues = False
        for name, module in self.named_children():
            for param_name, param in module.named_parameters():
                if param.device != self.device:
                    logger.critical(f"Parameter {name}.{param_name} is on {param.device}, not {self.device}")
                    device_issues = True
        if not device_issues:
            logger.critical("All network parameters confirmed on correct device")
        
        # Log model summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.critical(f"Network initialized with {total_params} total parameters ({trainable_params} trainable)")
        logger.critical("Network initialization complete")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action_probs, value)
        """
        try:
            logger.critical(f"Network forward called with input shape: {x.shape}, device: {x.device}, dtype: {x.dtype}")
            if torch.isnan(x).any():
                logger.critical("WARNING: Input contains NaN values!")
            if torch.isinf(x).any():
                logger.critical("WARNING: Input contains Inf values!")
                
            # Ensure input is on the correct device 
            if x.device != self.device:
                logger.critical(f"Moving input from {x.device} to {self.device}")
                x = x.to(self.device)
                
            # Handle batch dimension for visual inputs
            if self.is_visual_input:
                # Check if batch dimension is missing
                if len(x.shape) == 3:  # [C, H, W]
                    logger.critical("Adding batch dimension to input")
                    x = x.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
                    
                # Pass through convolutional layers
                logger.critical(f"Passing through conv layers, input shape: {x.shape}")
                x = self.conv_layers(x)
                logger.critical(f"Conv output shape: {x.shape}")
                
                # Flatten the features but keep batch dimension
                x = x.reshape(x.size(0), -1)
                logger.critical(f"Flattened shape: {x.shape}")
            else:
                # For vector inputs
                if len(x.shape) == 1:  # [D]
                    logger.critical("Adding batch dimension to vector input")
                    x = x.unsqueeze(0)  # Add batch dimension -> [1, D]
                    
            # Pass through shared layers
            logger.critical("Passing through shared layers")
            features = self.shared_layers(x)
            logger.critical(f"Shared features shape: {features.shape}")
            
            # Get policy (action probabilities) and value
            logger.critical("Computing action logits")
            action_logits = self.policy_head(features)
            logger.critical(f"Action logits shape: {action_logits.shape}")
            
            logger.critical("Computing value")
            value = self.value_head(features)
            logger.critical(f"Value shape: {value.shape}")
            
            # Apply softmax to get probabilities
            logger.critical("Applying softmax to get action probabilities")
            action_probs = torch.softmax(action_logits, dim=1)
            logger.critical(f"Action probabilities shape: {action_probs.shape}")
            
            # Check for NaN/Inf in outputs
            if torch.isnan(action_probs).any():
                logger.critical("WARNING: action_probs contains NaN values!")
            if torch.isinf(action_probs).any():
                logger.critical("WARNING: action_probs contains Inf values!")
            if torch.isnan(value).any():
                logger.critical("WARNING: value contains NaN values!")
            if torch.isinf(value).any():
                logger.critical("WARNING: value contains Inf values!")
            
            # Squeeze value if needed
            value = value.squeeze(-1)
            
            logger.critical("Forward pass completed successfully")
            return action_probs, value
            
        except Exception as e:
            import traceback
            logger.critical(f"ERROR in network forward pass: {e}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
            
            # Try to return usable values
            batch_size = x.size(0) if len(x.shape) > 1 else 1
            uniform_probs = torch.ones(batch_size, self.policy_head[-1].out_features, device=self.device) / self.policy_head[-1].out_features
            zero_value = torch.zeros(batch_size, device=self.device)
            
            logger.critical(f"Returning fallback values with shapes: probs={uniform_probs.shape}, value={zero_value.shape}")
            return uniform_probs, zero_value
        
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