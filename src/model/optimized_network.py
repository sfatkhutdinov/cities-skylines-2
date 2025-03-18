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
        
        # Get frame_stack from config, default to 4 if not specified
        frame_stack = getattr(config, 'frame_stack', 4)
        
        # Set expected resolution dimensions
        # Start with lower resolution for training
        self.expected_width = 320
        self.expected_height = 180
        
        # Log the configuration
        logger.info(f"Creating network with input resolution {self.expected_width}x{self.expected_height}, "
                    f"frame_stack={frame_stack}")
        
        # Input dimensions will be 3-channel (RGB) image, stacked for temporal information
        in_channels = 3 * frame_stack
            
        # Define network architecture - slightly modified for lighter model
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Calculate the output size of conv layers for our exact dimensions
        conv_output_size = self._calculate_conv_output_size(in_channels, self.expected_height, self.expected_width)
        logger.info(f"Convolution output size: {conv_output_size}")
        
        # Define fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        
        # Define UI features extraction
        self.ui_features = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Action space size
        action_space_size = 51  # Adjust based on your action space
        
        # Policy and value heads
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_size)
        )
        
        self.value_head = nn.Linear(256, 1)
        
        # Move the model to the appropriate device
        self.to(config.get_device())
        
        try:
            # Check for proper PyTorch 2.0+ and CUDA
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            supports_compile = torch_version >= (2, 0) and hasattr(torch, 'compile')
            if supports_compile and torch.cuda.is_available():
                logger.info("Applying PyTorch 2.0 model compilation")
                try:
                    # Only compile the policy head to avoid errors
                    self.policy_head = torch.compile(self.policy_head)
                    logger.info("Model compilation successful")
                except Exception as e:
                    logger.warning(f"Module compilation failed: {e}")
            else:
                logger.info("PyTorch 2.0 compilation not available")
        except Exception as e:
            logger.warning(f"Error checking PyTorch compilation support: {e}")
        
    def _calculate_conv_output_size(self, in_channels, height, width):
        """Calculate the size of the flattened features after convolution."""
        # Create a dummy input and pass it through the conv layers
        dummy_input = torch.zeros(1, in_channels, height, width)
        with torch.no_grad():
            dummy_output = self.conv_layers(dummy_input)
            
        # Get the flattened size from the output shape
        flattened_size = dummy_output.numel() // dummy_output.size(0)
        
        return flattened_size
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for the network.
        
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
                
                # Try to fix it by duplicating the input frame to match expected channels
                if getattr(self.config, 'auto_fix_frame_stack', False):
                    logger.warning(f"Attempting to auto-fix by duplicating frame {self.config.frame_stack} times")
                    # Duplicate the frame to match expected channel count
                    x_fixed = x.repeat(1, self.config.frame_stack, 1, 1)
                    logger.warning(f"Auto-fixed input shape: {x_fixed.shape}")
                    x = x_fixed
        
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
            # Final input validation
            if x.shape[1] != expected_channels:
                error_msg = f"Channel mismatch after processing: got {x.shape[1]}, expected {expected_channels}"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
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