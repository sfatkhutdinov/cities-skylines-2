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
        self.grayscale = False  # Using RGB (3 channels) instead of grayscale
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
            logger.critical(f"Rate limiting: elapsed={elapsed:.4f}s, sleeping for {self.min_capture_interval - elapsed:.4f}s")
            time.sleep(self.min_capture_interval - elapsed)
        
        # Capture raw frame
        logger.critical("===== OBSERVATION CAPTURE ATTEMPT =====")
        try:
            logger.critical("Calling screen_capture.capture_frame()")
            capture_start = time.time()
            raw_frame = self.screen_capture.capture_frame()
            capture_duration = time.time() - capture_start
            logger.critical(f"Screen capture completed in {capture_duration:.4f}s")
            
            if raw_frame is None:
                logger.critical("Failed to capture frame: raw_frame is None")
                # If in mock mode, generate a random frame for testing
                if self.mock_mode:
                    logger.critical("In mock mode, generating random frame")
                    raw_frame = np.random.randint(0, 255, (84, 84, 3), dtype=np.uint8)
                else:
                    # Try again with a short delay
                    logger.critical("Waiting 0.5s and trying capture again")
                    time.sleep(0.5)
                    raw_frame = self.screen_capture.capture_frame()
                    if raw_frame is None:
                        logger.critical("Second capture attempt also failed")
                        raise ValueError("Failed to capture frame after retry")
            else:
                # Log basic frame info
                logger.critical(f"Raw frame captured: shape={raw_frame.shape}, type={type(raw_frame)}")
                logger.critical(f"Frame values - min: {np.min(raw_frame)}, max: {np.max(raw_frame)}, mean: {np.mean(raw_frame):.2f}")
                
                # Check for all-black or all-white frames which might indicate issues
                if np.mean(raw_frame) < 10:
                    logger.critical("WARNING: Frame appears to be mostly black!")
                elif np.mean(raw_frame) > 245:
                    logger.critical("WARNING: Frame appears to be mostly white!")
                    
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.critical(f"Error capturing frame: {e}")
            logger.critical(f"Error traceback: {error_trace}")
            
            # Try to diagnose screen capture issues
            logger.critical("Diagnosing screen capture issues:")
            try:
                if hasattr(self.screen_capture, 'check_window_handle'):
                    handle_ok = self.screen_capture.check_window_handle()
                    logger.critical(f"Window handle check: {'OK' if handle_ok else 'FAILED'}")
            except Exception as diag_error:
                logger.critical(f"Diagnostics error: {diag_error}")
                
            raise  # Re-raise to be handled by the Environment class
        
        # Process frame
        logger.critical("Processing captured frame")
        try:
            processed_frame = self._process_frame(raw_frame)
            logger.critical(f"Frame processed: shape={processed_frame.shape}, device={processed_frame.device}, dtype={processed_frame.dtype}")
            
            # Check for NaN or Inf values
            if torch.isnan(processed_frame).any():
                logger.critical("WARNING: Processed frame contains NaN values!")
            if torch.isinf(processed_frame).any():
                logger.critical("WARNING: Processed frame contains Inf values!")
                
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.critical(f"Error processing frame: {e}")
            logger.critical(f"Error traceback: {error_trace}")
            raise  # Re-raise to be handled by the Environment class
        
        # Update history
        self.frame_history.append(processed_frame)
        logger.critical(f"Frame added to history, current history size: {len(self.frame_history)}")
        
        # Update timing
        self.last_capture_time = time.time()
        logger.critical("===== OBSERVATION CAPTURE COMPLETE =====")
        
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
    
    def get_latest_frame(self) -> Optional[torch.Tensor]:
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
    
    def _process_frame(self, frame) -> torch.Tensor:
        """Process raw frame for neural network input.
        
        Args:
            frame: Raw frame from screen capture (numpy.ndarray or torch.Tensor)
            
        Returns:
            torch.Tensor: Processed frame
        """
        import torch.nn.functional as F
        
        # Convert numpy array to torch tensor if needed
        if not isinstance(frame, torch.Tensor):
            try:
                # Make a copy of the array to avoid negative stride issues
                if isinstance(frame, np.ndarray):
                    frame = frame.copy()
                
                # Convert from HWC to CHW format
                frame = torch.from_numpy(frame).permute(2, 0, 1) if frame.ndim == 3 else torch.from_numpy(frame)
            except Exception as e:
                logger.error(f"Error converting numpy array to tensor: {e}")
                # Return a blank frame as fallback
                blank = torch.zeros((3, self.target_resolution[1], self.target_resolution[0]), 
                                     dtype=torch.float32, device=self.device)
                return blank
        
        # Convert to float if needed
        if frame.dtype != torch.float32:
            frame = frame.float()
            
        # Normalize if needed
        if self.normalize and frame.max() > 1.0:
            frame = frame / 255.0
            
        # Convert to grayscale if needed (for grayscale mode)
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
            
        # Ensure correct number of channels (1 for grayscale, 3 for RGB)
        if self.grayscale and frame.shape[0] != 1:
            logger.warning(f"Expected grayscale image with 1 channel, but got {frame.shape[0]} channels. Reshaping.")
            frame = frame[0:1, :, :]
        elif not self.grayscale and frame.shape[0] != 3:
            logger.warning(f"Expected RGB image with 3 channels, but got {frame.shape[0]} channels. Reshaping.")
            frame = frame.expand(3, -1, -1)
            
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
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the screen.
        
        Returns:
            numpy.ndarray: Captured frame as numpy array in RGB format,
                          or None if capture failed
        """
        if hasattr(self, 'screen_capture') and self.screen_capture:
            return self.screen_capture.capture_frame()
        return None 