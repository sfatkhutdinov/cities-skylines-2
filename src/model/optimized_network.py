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
    
    def __init__(self, config):
        """Initialize the network.
        
        Args:
            config: Hardware configuration
        """
        super(OptimizedNetwork, self).__init__()
        
        self.config = config
        
        # Store expected dimensions as class attributes - Fixed order to width, height
        width, height = getattr(config, 'resolution', (1920, 1080))
        
        # Override with our optimized processing resolution
        self.expected_width = 480  # Increased from 320
        self.expected_height = 270  # Increased from 240
        
        # Input dimensions will be 3-channel (RGB) image with config resolution
        in_channels = 3
        # Check if frame_stack exists and is greater than 1
        # Set a default of 1 if not specified
        frame_stack = getattr(config, 'frame_stack', 1)
        if frame_stack > 1:
            in_channels *= frame_stack
            
        # Define network sizes - now with more capacity for UI recognition
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            # Add an attention mechanism to focus on UI elements
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        
        # Define the size of the flattened features
        # Use the stored expected dimensions
        
        # Calculate the output size of the conv layers
        conv_output_size = self._calculate_conv_output_size(in_channels, self.expected_height, self.expected_width)
        
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Define UI features extraction to identify interface elements
        self.ui_features = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Separate policy and value heads for better specialization
        # Policy head is now MUCH larger to handle the expanded action space (51+ actions)
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            # Increase output size to handle additional UI position actions
            nn.Linear(64, 51)
        )
        
        self.value_head = nn.Linear(256, 1)
        
        # Move the model to the appropriate device
        self.to(config.get_device())
        
        # Apply PyTorch 2.0 compile optimization if available
        try:
            # Check for proper PyTorch 2.0+ and CUDA
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            supports_compile = torch_version >= (2, 0) and hasattr(torch, 'compile')
            if supports_compile and torch.cuda.is_available():
                logger.info("Applying PyTorch 2.0 model compilation for hardware optimization")
                # Wrap compilation in try-except block
                try:
                    # Selective compile for key modules
                    self.policy_head = torch.compile(self.policy_head)
                    self.value_head = torch.compile(self.value_head)
                    logger.info("Model compilation successful - this should improve performance")
                except Exception as e:
                    logger.warning(f"Module compilation failed, continuing with uncompiled version: {e}")
            else:
                logger.info("PyTorch 2.0 compilation not available (requires PyTorch 2.0+ and CUDA)")
        except Exception as e:
            logger.warning(f"Could not check for PyTorch compilation support: {e}")
        
    def _calculate_conv_output_size(self, in_channels, height, width):
        """Calculate the size of the flattened features after convolution."""
        # Use a dummy tensor to calculate the output size
        dummy_input = torch.zeros(1, in_channels, height, width)
        dummy_output = self.conv_layers(dummy_input)
        return int(np.prod(dummy_output.shape))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input state tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (action_probs, value)
        """
        # Log input shape for debugging
        input_shape = x.shape
        logger.debug(f"Network input shape: {input_shape}")
        
        # Check for channel dimension mismatch
        expected_channels = 3 * getattr(self.config, 'frame_stack', 4)  # Default to 4 frames if not specified
        if input_shape[1] != expected_channels:
            logger.warning(f"Input channel mismatch: got {input_shape[1]}, expected {expected_channels}")
            
            # Try to automatically determine if this is a frame stacking issue
            if input_shape[1] == 3 and expected_channels > 3:
                logger.warning(f"Input appears to be single frame (3 channels) but model expects {expected_channels} channels")
                logger.warning("This might be a frame stacking issue. Make sure to stack frames if the model was trained with them")
        
        # Ensure input is on the correct device
        x = x.to(self.config.get_device())
        
        # Get batch size and reshape if necessary
        if len(x.shape) == 3:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # Handle incorrect input shape
        if x.shape[2] != self.expected_height or x.shape[3] != self.expected_width:
            logger.debug(f"Resizing input from {x.shape[2]}x{x.shape[3]} to {self.expected_height}x{self.expected_width}")
            x = F.interpolate(x, size=(self.expected_height, self.expected_width), mode='bilinear', align_corners=False)
        
        try:
            # Pass through convolutional layers
            x = self.conv_layers(x)
            
            # Log shape after convolution for debugging
            logger.debug(f"After conv shape: {x.shape}")
            
            # Flatten
            x = x.view(x.size(0), -1)
            
            # Pass through fully connected layers
            features = self.fc_layers(x)
            
            # Extract UI features for better interface recognition
            ui_features = self.ui_features(features)
            
            # Get policy (action probabilities) and value
            action_logits = self.policy_head(features)
            value = self.value_head(features)
            
            # Apply softmax to get probabilities
            action_probs = torch.softmax(action_logits, dim=1)
            
            return action_probs, value.squeeze(-1)
        except RuntimeError as e:
            # Enhanced error reporting
            logger.error(f"Error during forward pass with input shape {input_shape}: {str(e)}")
            logger.error(f"Model expected height: {self.expected_height}, width: {self.expected_width}, channels: {expected_channels}")
            raise
        
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities only.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Action probabilities
        """
        # Ensure input is in the correct format and device
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
            
        # Ensure input has batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # Handle incorrect input shape
        if x.shape[2] != self.expected_height or x.shape[3] != self.expected_width:
            x = F.interpolate(x, size=(self.expected_height, self.expected_width), mode='bilinear', align_corners=False)
            
        features = self.conv_layers(x)
        features = features.reshape(x.size(0), -1)
        return self.policy_head(features)
        
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get value prediction only.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Value prediction
        """
        # Ensure input is in the correct format and device
        device = next(self.parameters()).device
        if x.device != device:
            x = x.to(device)
            
        # Ensure input has batch dimension
        if x.dim() == 3:
            x = x.unsqueeze(0)
            
        # Handle incorrect input shape
        if x.shape[2] != self.expected_height or x.shape[3] != self.expected_width:
            x = F.interpolate(x, size=(self.expected_height, self.expected_width), mode='bilinear', align_corners=False)
            
        features = self.conv_layers(x)
        features = features.reshape(x.size(0), -1)
        return self.value_head(features) 