"""
Visual metrics estimation from raw game screenshots.
Uses only visual information to estimate game metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List, Any
from ..config.hardware_config import HardwareConfig
import cv2
import os
import logging
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)

class VisualMetricsEstimator:
    def __init__(self, config: HardwareConfig):
        """Initialize visual metrics estimator.
        
        Args:
            config (HardwareConfig): Hardware and training configuration
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Initialize CNN for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device, dtype=self.dtype)
        
        # Moving averages for stability
        self.moving_averages = {
            'building_density': [],
            'residential_areas': [],
            'traffic_density': [],
            'window_lights': []
        }
        self.ma_window = 10
        
        # Initialize menu detection variables
        self.menu_reference = None
        self.menu_reference_features = None
        self.fallback_menu_detection_initialized = False
        self.menu_color_samples = {}
        
    def estimate_population(self, frame: torch.Tensor) -> Tuple[int, Dict[str, float]]:
        """
        Estimate population from visual cues in the frame.
        Uses multiple visual indicators:
        - Building density
        - Residential area coverage
        - Traffic density
        - Window lights (night time activity)
        """
        # Ensure frame is on the correct device and dtype
        if frame.device != self.device:
            frame = frame.to(self.device)
        if frame.dtype != self.dtype:
            frame = frame.to(dtype=self.dtype)
        
        # Extract visual features
        features = self._extract_features(frame)
        
        # Compute various density metrics
        metrics = {
            'building_density': self._compute_building_density(frame),
            'residential_areas': self._compute_residential_areas(frame),
            'traffic_density': self._compute_traffic_density(frame),
            'window_lights': self._compute_window_lights(frame)
        }
        
        # Update moving averages
        for key, value in metrics.items():
            self.moving_averages[key].append(value)
            if len(self.moving_averages[key]) > self.ma_window:
                self.moving_averages[key].pop(0)
        
        # Compute smoothed metrics
        smoothed_metrics = {
            k: np.mean(v) for k, v in self.moving_averages.items()
        }
        
        # Estimate population based on smoothed metrics
        # This is a simplified model that learns to correlate visual patterns with population
        estimated_population = self._combine_metrics_for_population(smoothed_metrics)
        
        return estimated_population, smoothed_metrics
    
    def _extract_features(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract visual features from frame using CNN."""
        with torch.no_grad():
            return self.feature_extractor(frame.unsqueeze(0))
    
    def _compute_building_density(self, frame: torch.Tensor) -> float:
        """
        Estimate building density from frame.
        Uses edge detection and color clustering to identify buildings.
        """
        # Convert to grayscale for edge detection
        gray = frame.mean(dim=0)
        
        # Simple Sobel edge detection
        edges_x = F.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                          torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().to(self.device, dtype=self.dtype).unsqueeze(0),
                          padding=1)
        edges_y = F.conv2d(gray.unsqueeze(0).unsqueeze(0),
                          torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().to(self.device, dtype=self.dtype).unsqueeze(0),
                          padding=1)
        
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return float(edges.mean().item())
    
    def _compute_residential_areas(self, frame: torch.Tensor) -> float:
        """
        Estimate residential area coverage.
        Uses color patterns typical of residential zones.
        """
        # Simple color-based detection (assuming residential areas have certain color patterns)
        # This would need to be refined based on actual game visuals
        residential_mask = ((frame[0] > 0.4) & (frame[1] > 0.4) & (frame[2] > 0.4))
        return float(residential_mask.float().mean().item())
    
    def _compute_traffic_density(self, frame: torch.Tensor) -> float:
        """
        Estimate traffic density from movement patterns.
        Uses temporal differences between frames for movement detection.
        """
        # Simplified version - would need frame history for actual implementation
        # Currently returns a proxy based on road-like patterns
        road_mask = ((frame[0] < 0.3) & (frame[1] < 0.3) & (frame[2] < 0.3))
        return float(road_mask.float().mean().item())
    
    def _compute_window_lights(self, frame: torch.Tensor) -> float:
        """
        Estimate active buildings through window lights.
        Particularly useful for night-time population estimation.
        """
        # Look for bright spots in darker areas
        brightness = frame.mean(dim=0)
        light_spots = (brightness > 0.7).float()
        return float(light_spots.mean().item())
    
    def _combine_metrics_for_population(self, metrics: Dict[str, float]) -> int:
        """
        Combine various metrics to estimate population.
        Uses a weighted combination of different visual indicators.
        """
        # Simplified model to convert visual metrics to population estimate
        # In a full implementation, this would be a learned model
        weights = {
            'building_density': 50000,
            'residential_areas': 80000,
            'traffic_density': 30000,
            'window_lights': 40000
        }
        
        estimated_population = sum(weights[k] * v for k, v in metrics.items())
        return int(estimated_population)
    
    def update_model(self, frame: torch.Tensor, reward: float):
        """Update visual metrics model based on reward feedback.
        
        Args:
            frame (torch.Tensor): Current frame
            reward (float): Reward received
        """
        # In a complete implementation, this would update the model weights
        # to improve future population estimates
        pass
        
    def initialize_menu_detection(self, menu_reference_path):
        """Initialize menu detection using a reference screenshot.
        
        Args:
            menu_reference_path (str): Path to menu reference image
        """
        if not os.path.exists(menu_reference_path):
            logger.error(f"Menu reference image not found at {menu_reference_path}")
            return False
            
        try:
            # Load reference image
            self.menu_reference = cv2.imread(menu_reference_path)
            if self.menu_reference is None:
                logger.error(f"Failed to load menu reference image from {menu_reference_path}")
                return False
                
            # Convert to grayscale for feature extraction
            menu_gray = cv2.cvtColor(self.menu_reference, cv2.COLOR_BGR2GRAY)
            
            # Extract ORB features
            orb = cv2.ORB_create(nfeatures=1000)
            self.menu_reference_keypoints, self.menu_reference_descriptors = orb.detectAndCompute(menu_gray, None)
            
            if self.menu_reference_keypoints is None or len(self.menu_reference_keypoints) == 0:
                logger.warning("No keypoints detected in menu reference image. Using color pattern detection instead.")
                self.setup_fallback_menu_detection()
            else:
                logger.info(f"Initialized menu detection with {len(self.menu_reference_keypoints)} keypoints from reference image")
                return True
                
        except Exception as e:
            logger.error(f"Error initializing menu detection: {e}")
            self.setup_fallback_menu_detection()
            return False
            
    def setup_fallback_menu_detection(self):
        """Setup fallback menu detection based on color patterns."""
        logger.info("Setting up fallback menu detection based on color patterns")
        
        # Define UI color patterns typical of menu screens
        # These are common UI colors in darker menu overlays
        self.menu_color_samples = {
            'dark_overlay': [(0, 0, 0), (10, 10, 10), (20, 20, 20)],  # Dark semi-transparent overlay
            'ui_highlight': [(200, 200, 200), (220, 220, 220), (240, 240, 240)],  # White UI elements
            'button_blue': [(50, 100, 200), (70, 120, 220), (90, 140, 240)],  # Blue button colors
            'text_color': [(230, 230, 230), (240, 240, 240), (250, 250, 250)]  # White text
        }
        
        self.fallback_menu_detection_initialized = True
        logger.info("Fallback menu detection initialized")
        return True
            
    def detect_menu_fallback(self, frame: torch.Tensor) -> bool:
        """Detect menu using fallback color and pattern detection.
        
        Args:
            frame (torch.Tensor): Current frame [C, H, W]
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        # Convert tensor to numpy array for OpenCV processing
        if isinstance(frame, torch.Tensor):
            # Ensure frame is on CPU and convert to numpy
            if frame.device != torch.device('cpu'):
                frame = frame.cpu()
            
            # Convert from [C, H, W] to [H, W, C] and normalize to 0-255
            frame_np = frame.permute(1, 2, 0).numpy() * 255
            frame_np = frame_np.astype(np.uint8)
            
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            # Handle case where frame is already a numpy array
            frame_bgr = frame
            
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # If we have a reference image, try to match with it first
        if hasattr(self, 'menu_reference') and self.menu_reference is not None:
            menu_detected = self._detect_menu_with_reference(frame_bgr, frame_gray)
            if menu_detected:
                return True
                
        # If reference matching fails or unavailable, use fallback detection
        if not hasattr(self, 'fallback_menu_detection_initialized') or not self.fallback_menu_detection_initialized:
            self.setup_fallback_menu_detection()
            
        # Use pattern and color detection as fallback
        return self._detect_menu_with_heuristics(frame_bgr, frame_gray)
        
    def _detect_menu_with_reference(self, frame_bgr, frame_gray):
        """Detect menu by matching against reference image.
        
        Args:
            frame_bgr: BGR frame
            frame_gray: Grayscale frame
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        # Ensure we have reference keypoints
        if not hasattr(self, 'menu_reference_keypoints') or self.menu_reference_keypoints is None:
            return False
            
        try:
            # Extract features from current frame
            orb = cv2.ORB_create(nfeatures=1000)
            keypoints, descriptors = orb.detectAndCompute(frame_gray, None)
            
            # Check if we found any keypoints
            if keypoints is None or len(keypoints) == 0 or descriptors is None:
                return False
                
            # Match features between reference and current frame
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(self.menu_reference_descriptors, descriptors)
            
            # Sort matches by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate match quality
            good_matches = [m for m in matches if m.distance < 50]  # Lower distance is better
            match_ratio = len(good_matches) / max(1, len(self.menu_reference_keypoints))
            
            # Consider a menu detected if match quality is high enough
            return match_ratio > 0.25  # At least 25% of keypoints matched
            
        except Exception as e:
            logger.error(f"Error in reference-based menu detection: {e}")
            return False
            
    def _detect_menu_with_heuristics(self, frame_bgr, frame_gray):
        """Detect menu using heuristics based on UI patterns.
        
        Args:
            frame_bgr: BGR frame
            frame_gray: Grayscale frame
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        # Several heuristics to detect menu screens
        
        # 1. Check for darkened overlay (common in pause menus)
        dark_pixel_ratio = np.mean(frame_gray < 50) / np.mean(frame_gray < 200)
        
        # 2. Check for horizontal lines (UI elements like menus often have horizontal separators)
        edges = cv2.Canny(frame_gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # Consider lines within 10 degrees of horizontal
                    horizontal_lines += 1
                    
        horizontal_line_score = horizontal_lines / max(1, len(lines) if lines is not None else 1)
        
        # 3. Check for UI color patterns
        ui_color_score = 0
        for color_name, color_samples in self.menu_color_samples.items():
            for color in color_samples:
                # Create a mask for pixels close to this color
                lower_bound = np.array([max(0, c - 20) for c in color])
                upper_bound = np.array([min(255, c + 20) for c in color])
                mask = cv2.inRange(frame_bgr, lower_bound, upper_bound)
                
                # Calculate percentage of pixels that match this color
                color_ratio = np.count_nonzero(mask) / (mask.shape[0] * mask.shape[1])
                ui_color_score += color_ratio
                
        # 4. Check for text-like patterns (high frequency content in grayscale)
        text_pattern_score = 0
        laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F).var()
        text_pattern_score = min(1.0, laplacian / 1000)  # Normalize
        
        # Combine scores with different weights
        menu_score = (
            dark_pixel_ratio * 0.3 +
            horizontal_line_score * 0.2 +
            ui_color_score * 0.3 +
            text_pattern_score * 0.2
        )
        
        # Define threshold for menu detection
        menu_threshold = 0.25
        
        # Return detection result
        return menu_score > menu_threshold
    
    def detect_main_menu(self, frame: torch.Tensor) -> bool:
        """Detect if the current frame shows a menu.
        
        Args:
            frame (torch.Tensor): Current frame [C, H, W]
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        return self.detect_menu_fallback(frame)
        
    def save_current_frame_as_menu_reference(self, frame: torch.Tensor, save_path: str) -> bool:
        """Save current frame as menu reference image.
        
        Args:
            frame (torch.Tensor): Current frame [C, H, W]
            save_path (str): Path to save the reference image
            
        Returns:
            bool: Success status
        """
        try:
            # Convert tensor to numpy array
            if frame.device != torch.device('cpu'):
                frame = frame.cpu()
                
            frame_np = frame.permute(1, 2, 0).numpy() * 255
            frame_np = frame_np.astype(np.uint8)
            
            # Convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(save_path, frame_bgr)
            
            # Also initialize menu detection with this image
            self.menu_reference = frame_bgr
            
            # Extract features
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            orb = cv2.ORB_create(nfeatures=1000)
            self.menu_reference_keypoints, self.menu_reference_descriptors = orb.detectAndCompute(frame_gray, None)
            
            logger.info(f"Saved menu reference image to {save_path} and initialized detection with {len(self.menu_reference_keypoints)} keypoints")
            return True
            
        except Exception as e:
            logger.error(f"Error saving menu reference image: {e}")
            return False
            
    def update_model(self, frame: torch.Tensor, reward: float):
        """Update the model based on reward feedback.
        
        Args:
            frame (torch.Tensor): Current frame
            reward (float): Reward received
        """
        # Would update model parameters based on reward signal
        pass
        
    def calculate_reward(self, frame):
        """Calculate a reward signal based on visual assessment.
        
        Args:
            frame (torch.Tensor): Current frame
            
        Returns:
            float: Reward value
        """
        # Estimate population
        population, metrics = self.estimate_population(frame)
        
        # Reward is based on population and visual quality
        reward = 0.001 * population
        
        # Add bonus for balanced metrics
        variance = np.var(list(metrics.values()))
        balance_bonus = 1.0 / (1.0 + variance)
        
        reward += 10.0 * balance_bonus
        
        return reward