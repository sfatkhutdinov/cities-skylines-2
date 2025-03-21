"""
Optimized neural network for Cities: Skylines 2 agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
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
    
    def __init__(self, input_shape, num_actions, device=None, use_lstm=True, lstm_hidden_size=256):
        """Initialize the network.
        
        Args:
            input_shape: Shape of the input state (channels, height, width) or flattened size
            num_actions: Number of actions in the action space
            device: Computation device
            use_lstm: Whether to use LSTM for temporal reasoning
            lstm_hidden_size: Size of LSTM hidden state
        """
        super(OptimizedNetwork, self).__init__()
        
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.critical(f"Initializing network with input_shape={input_shape}, num_actions={num_actions}, device={self.device}")
        
        # LSTM configuration
        self.use_lstm = use_lstm
        self.lstm_hidden_size = lstm_hidden_size
        logger.critical(f"LSTM enabled: {self.use_lstm}, hidden size: {self.lstm_hidden_size}")
        
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
            
            # Track the feature size for LSTM
            self.feature_size = 512
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
            
            # Track the feature size for LSTM
            self.feature_size = 512
            
        # LSTM layer for temporal reasoning if enabled
        if self.use_lstm:
            logger.critical(f"Creating LSTM layer with hidden size {self.lstm_hidden_size}")
            self.lstm = nn.LSTM(
                input_size=self.feature_size,
                hidden_size=self.lstm_hidden_size,
                num_layers=1,
                batch_first=True
            )
            
            # Update feature size for heads
            self.output_size = self.lstm_hidden_size
        else:
            # No LSTM
            self.lstm = None
            self.output_size = self.feature_size
            
        # Policy head
        logger.critical(f"Creating policy head with input size {self.output_size}")
        self.policy_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        
        # Value head
        logger.critical(f"Creating value head with input size {self.output_size}")
        self.value_head = nn.Sequential(
            nn.Linear(self.output_size, 256),
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

    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            hidden_state: Optional hidden state for LSTM (h, c)
            
        Returns:
            Tuple: (action_probs, value, next_hidden_state)
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
            
            # Process through LSTM if enabled
            next_hidden = None
            if self.use_lstm:
                if hidden_state is None:
                    # Initialize hidden state if not provided
                    h0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                    c0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                    hidden_state = (h0, c0)
                else:
                    # Ensure hidden state is on the correct device
                    h, c = hidden_state
                    if h.device != self.device:
                        h = h.to(self.device)
                    if c.device != self.device:
                        c = c.to(self.device)
                    hidden_state = (h, c)
                
                logger.critical("Processing through LSTM")
                # Add time dimension for LSTM if needed
                if len(features.shape) == 2:  # [B, F]
                    features = features.unsqueeze(1)  # [B, 1, F]
                
                features, next_hidden = self.lstm(features, hidden_state)
                
                # Remove time dimension if added
                if features.size(1) == 1:
                    features = features.squeeze(1)
                
                logger.critical(f"LSTM output shape: {features.shape}")
            
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
            return action_probs, value, next_hidden
            
        except Exception as e:
            import traceback
            logger.critical(f"ERROR in network forward pass: {e}")
            logger.critical(f"Traceback: {traceback.format_exc()}")
            
            # Try to return usable values
            batch_size = x.size(0) if len(x.shape) > 1 else 1
            uniform_probs = torch.ones(batch_size, self.policy_head[-1].out_features, device=self.device) / self.policy_head[-1].out_features
            zero_value = torch.zeros(batch_size, device=self.device)
            
            logger.critical(f"Returning fallback values with shapes: probs={uniform_probs.shape}, value={zero_value.shape}")
            return uniform_probs, zero_value, None
        
    def get_action_probs(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get action probabilities for a state.
        
        Args:
            x (torch.Tensor): Input state tensor
            hidden_state: Optional hidden state for LSTM
            
        Returns:
            Tuple: (action_probs, next_hidden_state)
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
            
        # Process through shared layers
        features = self.shared_layers(features)
        
        # Process through LSTM if enabled
        next_hidden = None
        if self.use_lstm:
            if hidden_state is None:
                # Initialize hidden state if not provided
                h0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                c0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                hidden_state = (h0, c0)
            
            # Add time dimension for LSTM
            if len(features.shape) == 2:  # [B, F]
                features = features.unsqueeze(1)  # [B, 1, F]
            
            features, next_hidden = self.lstm(features, hidden_state)
            
            # Remove time dimension if added
            if features.size(1) == 1:
                features = features.squeeze(1)
        
        # Process through policy head
        logits = self.policy_head(features)
        
        # Apply softmax to get probabilities
        return torch.softmax(logits, dim=1), next_hidden
    
    def get_value(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Get value for a state.
        
        Args:
            x (torch.Tensor): Input state tensor
            hidden_state: Optional hidden state for LSTM
            
        Returns:
            Tuple: (value, next_hidden_state)
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
            
        # Process through shared layers
        features = self.shared_layers(features)
        
        # Process through LSTM if enabled
        next_hidden = None
        if self.use_lstm:
            if hidden_state is None:
                # Initialize hidden state if not provided
                h0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                c0 = torch.zeros(1, x.size(0), self.lstm_hidden_size, device=self.device)
                hidden_state = (h0, c0)
            
            # Add time dimension for LSTM
            if len(features.shape) == 2:  # [B, F]
                features = features.unsqueeze(1)  # [B, 1, F]
            
            features, next_hidden = self.lstm(features, hidden_state)
            
            # Remove time dimension if added
            if features.size(1) == 1:
                features = features.squeeze(1)
        
        # Process through value head
        value = self.value_head(features)
        
        return value.squeeze(-1), next_hidden
    
    def _init_weights(self, module):
        """Initialize network weights.
        
        Args:
            module: Module to initialize weights for
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param, gain=1.0)
                elif 'bias' in name:
                    nn.init.constant_(param, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0) 