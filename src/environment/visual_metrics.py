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
    
    def detect_main_menu(self, frame: torch.Tensor) -> bool:
        """Detect if the current frame shows a main menu or popup.
        
        Args:
            frame (torch.Tensor): Current frame
            
        Returns:
            bool: True if menu detected
        """
        # Convert to numpy for OpenCV processing
        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().cpu().numpy()
            if len(frame_np.shape) == 3:  # CHW format
                frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
        else:
            frame_np = frame
            
        # Convert to grayscale for feature matching
        if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
            frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
        else:
            frame_gray = frame_np
        
        # If we have a reference image, use feature matching
        if self.menu_reference is not None and hasattr(self, 'menu_kp') and hasattr(self, 'menu_desc'):
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
        
        # Fallback: Use generic menu detection based on UI patterns
        # This is a simple heuristic based on common menu characteristics
        
        # 1. Look for large solid-colored rectangles (common in menus)
        edges = cv2.Canny(frame_gray, 100, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        large_rectangles = 0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Look for rectangle-like contours with reasonable aspect ratios
            if 0.2 < aspect_ratio < 5 and w*h > (frame_gray.shape[0] * frame_gray.shape[1] * 0.05):
                large_rectangles += 1
        
        # 2. Check for text-like structures (menus often have text)
        # Simplified text detection using horizontal projection profiles
        # (real implementation would use OCR or text detection)
        horizontal_profile = np.sum(edges, axis=1)
        text_like_rows = np.sum(horizontal_profile > frame_gray.shape[1] * 0.1)
        
        # 3. Reduced edge density in the game area
        # (menus often dim or blur the game background)
        edge_density = np.sum(edges > 0) / (frame_gray.shape[0] * frame_gray.shape[1])
        
        # Combine heuristics
        menu_score = (large_rectangles >= 3) + (text_like_rows >= 5) + (edge_density < 0.05)
        
        return menu_score >= 2  # At least 2 of 3 heuristics suggest a menu
    
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