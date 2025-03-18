"""
Visual detection system that identifies UI elements without relying on game metrics.
This focuses on pure computer vision approaches without knowledge of game semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from src.config.hardware_config import HardwareConfig
import cv2
import os
import logging
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)

class VisualMetricsEstimator:
    def __init__(self, config: HardwareConfig):
        """Initialize visual detection system.
        
        Args:
            config (HardwareConfig): Hardware and training configuration
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # For menu detection
        self.menu_reference = None
        self.menu_matcher = cv2.SIFT_create()
        self.menu_flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
        
        # Feature extraction for general use
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device, dtype=self.dtype)
        
        # Global UI patterns (these are learned from experience without game knowledge)
        self.ui_patterns = {
            'menu': [],  # Stores feature vectors of known menu screens
            'normal_gameplay': []  # Stores feature vectors of normal gameplay
        }
        self.max_patterns = 50  # Maximum number of patterns to store
    
    def initialize_menu_detection(self, menu_reference_path):
        """Initialize menu detection with reference image."""
        if menu_reference_path and os.path.exists(menu_reference_path):
            try:
                self.menu_reference = cv2.imread(menu_reference_path, cv2.IMREAD_GRAYSCALE)
                if self.menu_reference is not None:
                    logger.info(f"Loaded menu reference image from {menu_reference_path}")
                    # Pre-compute keypoints and descriptors
                    self.menu_kp, self.menu_desc = self.menu_matcher.detectAndCompute(self.menu_reference, None)
                    return True
                else:
                    logger.warning(f"Failed to load menu reference image from {menu_reference_path}")
            except Exception as e:
                logger.error(f"Error loading menu reference: {e}")
        
        logger.warning("No valid menu reference image available")
        return False
    
    def setup_fallback_menu_detection(self):
        """Set up fallback menu detection based on color patterns and UI layout analysis.
        This method is used when no menu reference image is available.
        """
        logger.info("Setting up fallback menu detection system")
        
        # Common menu UI characteristics
        self.menu_color_ranges = [
            # Dark semi-transparent overlays (common in menus)
            ((0, 0, 0), (50, 50, 50)),
            # Blue UI elements (common in Cities Skylines UI)
            ((100, 50, 0), (255, 150, 50)),
            # White text
            ((200, 200, 200), (255, 255, 255))
        ]
        
        # Initialize UI pattern detection
        self.ui_pattern_threshold = 0.4  # Threshold for UI element detection
        self.menu_detection_initialized = True
        logger.info("Fallback menu detection initialized")
        
    def detect_menu_fallback(self, frame: torch.Tensor) -> bool:
        """Detect if current frame shows a menu using fallback method.
        
        Args:
            frame (torch.Tensor): Current frame observation
            
        Returns:
            bool: True if menu is detected, False otherwise
        """
        # Convert tensor to numpy for OpenCV processing
        if frame.device != torch.device('cpu'):
            frame_np = frame.cpu().numpy()
        else:
            frame_np = frame.numpy()
            
        # Convert from CHW to HWC format and scale to 0-255 range
        if frame_np.shape[0] == 3:  # CHW format
            frame_np = np.transpose(frame_np, (1, 2, 0))
        frame_np = (frame_np * 255).astype(np.uint8)
        
        # Convert to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        # Detect UI elements using color thresholding
        ui_mask = np.zeros((frame_bgr.shape[0], frame_bgr.shape[1]), dtype=np.uint8)
        
        for lower, upper in self.menu_color_ranges:
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(frame_bgr, lower, upper)
            ui_mask = cv2.bitwise_or(ui_mask, mask)
        
        # Calculate percentage of UI elements
        ui_percentage = np.sum(ui_mask > 0) / (ui_mask.shape[0] * ui_mask.shape[1])
        
        # Check for UI layout patterns (horizontal lines, grids, etc.)
        edges = cv2.Canny(frame_bgr, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=100, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Almost horizontal lines
                    horizontal_lines += 1
        
        # Decision based on combined factors
        is_menu = (ui_percentage > self.ui_pattern_threshold) or (horizontal_lines > 5)
        
        return is_menu
        
    def detect_main_menu(self, frame: torch.Tensor) -> bool:
        """Detect if current frame shows the main menu.
        
        Args:
            frame (torch.Tensor): Current frame observation
            
        Returns:
            bool: True if main menu is detected, False otherwise
        """
        # First try standard detection if reference image is available
        if self.menu_reference is not None:
            try:
                # Use existing feature matching method
                # Check if frame is None or empty
                if frame is None:
                    return False
                    
                # Convert to numpy for OpenCV processing
                if isinstance(frame, torch.Tensor):
                    frame_np = frame.detach().cpu().numpy()
                    if len(frame_np.shape) == 3:  # CHW format
                        frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
                else:
                    frame_np = frame
                    
                # Check if frame is empty
                if frame_np is None or frame_np.size == 0 or np.all(frame_np == 0):
                    return False
                    
                # Convert to grayscale for feature matching
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                else:
                    # If already grayscale, ensure it's the correct type
                    if len(frame_np.shape) == 2:
                        frame_gray = frame_np
                    else:
                        # Unexpected format, return false
                        return False
                        
                # Ensure frame is 8-bit unsigned integer (CV_8U)
                if frame_gray.dtype != np.uint8:
                    frame_gray = frame_gray.astype(np.uint8)
                    
                # If we have a reference image, use feature matching
                if self.menu_reference is not None and hasattr(self, 'menu_kp') and hasattr(self, 'menu_desc'):
                    try:
                        # Extract features from current frame
                        kp, desc = self.menu_matcher.detectAndCompute(frame_gray, None)
                        
                        # Not enough features for matching
                        if desc is None or len(kp) < 10:
                            return False
                            
                        # Match features
                        matches = self.menu_flann.knnMatch(self.menu_desc, desc, k=2)
                        
                        # Apply ratio test
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                                
                        # If enough matches, consider it a menu
                        match_threshold = 10  # Minimum number of matches to consider it a menu
                        return len(good_matches) >= match_threshold
                    except cv2.error:
                        # If OpenCV processing fails, assume no menu
                        return False
            except Exception as e:
                logger.warning(f"Standard menu detection failed: {e}, falling back to pattern-based detection")
                return self.detect_menu_fallback(frame)
        else:
            # Use fallback method
            return self.detect_menu_fallback(frame)
    
    def save_current_frame_as_menu_reference(self, frame: torch.Tensor, save_path: str) -> bool:
        """Save current frame as a menu reference.
        
        Args:
            frame (torch.Tensor): Current frame
            save_path (str): Path to save the reference image
            
        Returns:
            bool: Success flag
        """
        try:
            # Convert to numpy
            if isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().numpy()
                if len(frame_np.shape) == 3:  # CHW format
                    frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
            else:
                frame_np = frame
                
            # Convert to grayscale if needed
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            else:
                frame_gray = frame_np
            
            # Save the grayscale image
            cv2.imwrite(save_path, frame_gray)
            
            # Update the reference
            self.menu_reference = frame_gray
            self.menu_kp, self.menu_desc = self.menu_matcher.detectAndCompute(self.menu_reference, None)
            
            logger.info(f"Saved menu reference to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving menu reference: {e}")
            return False
            
    def update_model(self, frame: torch.Tensor, reward: float):
        """Update any internal models based on rewards (placeholder).
        
        Args:
            frame (torch.Tensor): Current frame
            reward (float): The reward received
        """
        # This is just a placeholder - we don't need implementation since
        # our autonomous system handles learning separately
        pass
        
    def calculate_reward(self, frame):
        """Calculate a simple placeholder reward (for compatibility).
        
        Args:
            frame: The current observation frame
            
        Returns:
            float: A small constant reward
        """
        # This is just a placeholder - we only keep it for backward compatibility
        # The actual rewards come from the autonomous reward system
        return 0.01 