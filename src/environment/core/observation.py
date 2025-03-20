"""
Observation management for Cities: Skylines 2 environment.

This module handles screen capture, observation processing, and
frame history management.
"""

import torch
import numpy as np
import logging
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

from src.config.hardware_config import HardwareConfig
from src.environment.optimized_capture import OptimizedScreenCapture
from src.environment.visual_metrics import VisualMetricsEstimator

logger = logging.getLogger(__name__)

class ObservationManager:
    """Manages observations from the game environment."""
    
    def __init__(self, 
                 config: Optional[HardwareConfig] = None, 
                 mock_mode: bool = False,
                 frame_history_length: int = 4):
        """Initialize observation manager.
        
        Args:
            config: Hardware configuration
            mock_mode: Whether to use mock mode (no game capture)
            frame_history_length: Number of frames to keep in history
        """
        # Basic setup
        self.config = config or HardwareConfig()
        self.device = self.config.get_device()
        self.dtype = self.config.get_dtype()
        self.mock_mode = mock_mode
        self.frame_history_length = frame_history_length
        
        # Initialize components
        self.screen_capture = OptimizedScreenCapture(config=self.config)
        self.screen_capture.use_mock = mock_mode
        
        # Initialize visual metrics for processing observations
        self.visual_metrics = VisualMetricsEstimator(config=self.config)
        
        # Frame history
        self.frame_history = deque(maxlen=frame_history_length)
        
        # Default target resolution for processed frames
        self.target_resolution = (84, 84)  # Width, height
        
        # Processing options
        self.grayscale = True
        self.normalize = True
        
        # Capture timing
        self.last_capture_time = time.time()
        self.min_capture_interval = 0.05  # Minimum time between captures
        
        logger.info(f"Observation manager initialized with target resolution {self.target_resolution}")
    
    def reset(self) -> None:
        """Reset observation manager state."""
        # Clear frame history
        self.frame_history.clear()
        
        # Reset capture timing
        self.last_capture_time = time.time()
        
    def get_observation(self) -> torch.Tensor:
        """Get current observation as processed frame.
        
        Returns:
            torch.Tensor: Processed observation
        """
        # Rate limit capturing
        current_time = time.time()
        elapsed = current_time - self.last_capture_time
        if elapsed < self.min_capture_interval:
            time.sleep(self.min_capture_interval - elapsed)
        
        # Capture raw frame
        raw_frame = self.screen_capture.capture_frame()
        
        # Process frame
        processed_frame = self._process_frame(raw_frame)
        
        # Update history
        self.frame_history.append(processed_frame)
        
        # Update timing
        self.last_capture_time = time.time()
        
        return processed_frame
    
    def get_frame_stack(self, stack_size: int = 4) -> torch.Tensor:
        """Get a stack of frames for temporal processing.
        
        Args:
            stack_size: Number of frames to stack
            
        Returns:
            torch.Tensor: Stacked frames [stack_size, C, H, W]
        """
        # Ensure we have enough frames
        if len(self.frame_history) < stack_size:
            # Fill with copies of the latest frame if not enough history
            if len(self.frame_history) > 0:
                latest_frame = self.frame_history[-1]
                while len(self.frame_history) < stack_size:
                    self.frame_history.appendleft(latest_frame)
            else:
                # Get observation if no frames available
                observation = self.get_observation()
                while len(self.frame_history) < stack_size:
                    self.frame_history.appendleft(observation)
        
        # Get most recent frames
        frames = list(self.frame_history)[-stack_size:]
        
        # Stack frames along new dimension
        stacked = torch.stack(frames)
        
        return stacked
    
    def get_current_frame(self) -> Optional[torch.Tensor]:
        """Get the most recent frame.
        
        Returns:
            torch.Tensor: Most recent frame or None if no frames
        """
        if not self.frame_history:
            return None
        return self.frame_history[-1]
    
    def get_previous_frame(self) -> Optional[torch.Tensor]:
        """Get the second most recent frame.
        
        Returns:
            torch.Tensor: Second most recent frame or None if not enough frames
        """
        if len(self.frame_history) < 2:
            return None
        return self.frame_history[-2]
    
    def compute_frame_difference(self) -> Optional[torch.Tensor]:
        """Compute difference between the two most recent frames.
        
        Returns:
            torch.Tensor: Difference between frames or None if not enough frames
        """
        if len(self.frame_history) < 2:
            return None
        
        current = self.frame_history[-1]
        previous = self.frame_history[-2]
        
        return torch.abs(current - previous)
    
    def _process_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Process raw frame for neural network input.
        
        Args:
            frame: Raw frame from screen capture
            
        Returns:
            torch.Tensor: Processed frame
        """
        import torch.nn.functional as F
        
        # Convert to float if needed
        if frame.dtype != torch.float32:
            frame = frame.float()
            
        # Normalize if needed
        if self.normalize and frame.max() > 1.0:
            frame = frame / 255.0
            
        # Convert to grayscale if needed
        if self.grayscale and frame.shape[0] == 3:
            # RGB to grayscale conversion weights
            weights = torch.tensor([0.299, 0.587, 0.114], device=frame.device).view(3, 1, 1)
            frame = torch.sum(frame * weights, dim=0, keepdim=True)
            
        # Resize if needed
        if (frame.shape[-2] != self.target_resolution[1] or 
            frame.shape[-1] != self.target_resolution[0]):
            frame = F.interpolate(
                frame.unsqueeze(0), 
                size=self.target_resolution, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
        # Ensure frame is on the correct device
        if frame.device != self.device:
            frame = frame.to(self.device)
            
        return frame
    
    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'screen_capture') and self.screen_capture:
            self.screen_capture.close()
        self.frame_history.clear()
    
    def get_observation_shape(self) -> Tuple[int, int, int]:
        """Get the shape of observations.
        
        Returns:
            Tuple[int, int, int]: Shape of observations (channels, height, width)
        """
        channels = 1 if self.grayscale else 3
        return (channels, self.target_resolution[1], self.target_resolution[0]) 