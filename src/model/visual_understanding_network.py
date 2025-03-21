"""
Visual Understanding Network for Cities: Skylines 2 agent.
Processes raw pixel input to extract game scene information.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class VisualUnderstandingNetwork(nn.Module):
    """Visual Understanding Network for scene parsing from raw pixels."""
    
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 feature_dim: int = 512,
                 device=None):
        """Initialize the Visual Understanding Network.
        
        Args:
            input_shape: Shape of the input images (C, H, W)
            feature_dim: Output feature dimension
            device: Computation device
        """
        super(VisualUnderstandingNetwork, self).__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_shape = input_shape
        self.feature_dim = feature_dim
        
        channels, height, width = input_shape
        
        # Enhanced CNN architecture with residual connections and attention
        self.conv_layers = nn.Sequential(
            # First block
            nn.Conv2d(channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Residual block 1
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            
            # Residual block 2
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
            
            # Residual block 3
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        ).to(self.device)
        
        # Calculate the output dimensions after convolutions
        conv_output_size = self._get_conv_output_size(input_shape)
        
        # Spatial attention after convolutions
        self.spatial_attention = SpatialAttention().to(self.device)
        
        # Final layers for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, feature_dim)
        ).to(self.device)
        
        # Scene classifier branches (optional, helps with representation learning)
        self.scene_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 10)  # 10 possible scene types (abstract representation)
        ).to(self.device)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.critical(f"Initialized Visual Understanding Network on device {self.device}")
    
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def _get_conv_output_size(self, shape):
        """Calculate the output size of the convolutional layers."""
        batch_size = 1
        input = torch.rand(batch_size, *shape).to(self.device)
        output = self.conv_layers(input)
        return int(np.prod(output.shape[1:]))
    
    def forward(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, C, H, W)
            
        Returns:
            Tuple of (features, scene_logits)
        """
        # Ensure x is on the correct device and has batch dimension
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.to(self.device)
        
        # Pass through convolutional layers
        conv_features = self.conv_layers(x)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(conv_features)
        
        # Flatten and pass through feature extractor
        flattened = attended_features.view(x.size(0), -1)
        features = self.feature_extractor(flattened)
        
        # Get scene classification logits
        scene_logits = self.scene_classifier(features)
        
        return features, scene_logits
    
    def extract_visual_features(self, x):
        """Extract visual features only (no classification).
        
        Args:
            x: Input tensor of shape (batch_size, C, H, W)
            
        Returns:
            Tensor of shape (batch_size, feature_dim)
        """
        features, _ = self.forward(x)
        return features


class ResidualBlock(nn.Module):
    """Residual block with bottleneck structure."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Bottleneck architecture
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels//4)
        
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels//4)
        
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on important regions."""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        # Spatial attention layers
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Calculate spatial attention weights
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(attention_input))
        
        # Apply attention weights
        return x * attention_map 