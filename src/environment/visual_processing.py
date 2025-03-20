"""
Enhanced visual processing module with attention mechanisms and GPU optimization.
Improves feature extraction without introducing domain knowledge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import cv2
from typing import Dict, List, Tuple, Any, Optional, Union
from src.learning.advanced_learning import SpatialAttention

logger = logging.getLogger(__name__)

class EnhancedVisualProcessor(nn.Module):
    """Enhanced visual processor with attention mechanisms and GPU optimization."""
    
    def __init__(self, input_channels=3, feature_dim=512, device=None):
        """Initialize enhanced visual processor.
        
        Args:
            input_channels: Number of input channels (typically 3 for RGB)
            feature_dim: Dimension of output features
            device: Device for tensor operations
        """
        super(EnhancedVisualProcessor, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Attention mechanism for focusing on important regions
        self.spatial_attention = SpatialAttention(input_channels)
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Feature projector (calculate output size dynamically)
        with torch.no_grad():
            # Create a dummy input to compute output shape
            dummy_input = torch.zeros(1, input_channels, 240, 320, device=self.device)
            x = self.conv1(dummy_input)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)
            flattened_size = x.numel() // x.size(0)
            
        # Feature projection layers
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, feature_dim),
            nn.ReLU()
        )
        
        # Move to device
        self.to(self.device)
        
    @torch.cuda.amp.autocast(enabled=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the visual processor.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Feature representation
        """
        # Apply spatial attention
        attended, attention_map = self.spatial_attention(x)
        
        # Feature extraction
        x = self.conv1(attended)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        
        # Project to feature space
        features = self.projection(x)
        
        return features
        
    def extract_features(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract features from a frame with GPU acceleration.
        
        Args:
            frame: Input frame tensor [C, H, W]
            
        Returns:
            torch.Tensor: Feature representation
        """
        # Ensure frame is on the correct device
        if frame.device != self.device:
            frame = frame.to(self.device)
            
        # Add batch dimension if needed
        if len(frame.shape) == 3:
            frame = frame.unsqueeze(0)
            
        # Normalize if needed
        if frame.max() > 1.0:
            frame = frame / 255.0
            
        # Enable mixed precision for faster computation
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                features = self.forward(frame)
                
        return features.squeeze(0)  # Remove batch dimension
        
    def process_batch(self, frames: List[torch.Tensor]) -> List[torch.Tensor]:
        """Process a batch of frames efficiently.
        
        Args:
            frames: List of frame tensors
            
        Returns:
            List[torch.Tensor]: List of feature representations
        """
        if not frames:
            return []
            
        # Stack frames into a batch
        batch = torch.stack([f.to(self.device) for f in frames])
        
        # Normalize if needed
        if batch.max() > 1.0:
            batch = batch / 255.0
            
        # Process batch efficiently
        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                features = self.forward(batch)
                
        # Split batch back into list
        return list(features)
        
    def extract_visual_change_features(self, prev_frame: torch.Tensor, curr_frame: torch.Tensor) -> torch.Tensor:
        """Extract features specifically focused on changes between frames.
        
        Args:
            prev_frame: Previous frame tensor
            curr_frame: Current frame tensor
            
        Returns:
            torch.Tensor: Change-focused features
        """
        # Extract features from both frames
        prev_features = self.extract_features(prev_frame)
        curr_features = self.extract_features(curr_frame)
        
        # Compute feature difference
        change_features = curr_features - prev_features
        
        return change_features

class VisualStreamProcessor:
    """Processes streams of visual data efficiently with attention to important changes."""
    
    def __init__(self, feature_dim=512, history_length=10, device=None):
        """Initialize visual stream processor.
        
        Args:
            feature_dim: Dimension of feature representations
            history_length: Number of frames to keep in history
            device: Device for tensor operations
        """
        self.feature_dim = feature_dim
        self.history_length = history_length
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize visual processor
        self.visual_processor = EnhancedVisualProcessor(
            input_channels=3,
            feature_dim=feature_dim,
            device=self.device
        )
        
        # Frame history
        self.frame_history = []
        self.feature_history = []
        
        # Change detection threshold
        self.change_threshold = 0.1
        
    def process_frame(self, frame: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Process a single frame and detect changes.
        
        Args:
            frame: Input frame tensor
            
        Returns:
            Tuple[torch.Tensor, float]: Features and change score
        """
        # Extract features
        features = self.visual_processor.extract_features(frame)
        
        # Calculate change score if we have history
        change_score = 0.0
        if self.feature_history:
            prev_features = self.feature_history[-1]
            feature_diff = F.mse_loss(features, prev_features)
            change_score = float(feature_diff.item())
            
        # Update history
        self.frame_history.append(frame.detach().cpu())
        self.feature_history.append(features.detach().cpu())
        
        # Keep history within size limit
        if len(self.frame_history) > self.history_length:
            self.frame_history.pop(0)
            self.feature_history.pop(0)
            
        return features, change_score
        
    def detect_significant_change(self, change_score: float) -> bool:
        """Detect if a change is significant based on score.
        
        Args:
            change_score: Change score from process_frame
            
        Returns:
            bool: True if change is significant
        """
        return change_score > self.change_threshold
        
    def batch_process_history(self) -> List[torch.Tensor]:
        """Process the entire history as a batch for efficiency.
        
        Returns:
            List[torch.Tensor]: Updated feature history
        """
        if not self.frame_history:
            return []
            
        # Process frames in a batch
        features = self.visual_processor.process_batch(self.frame_history)
        
        # Update feature history
        self.feature_history = [f.detach().cpu() for f in features]
        
        return features 