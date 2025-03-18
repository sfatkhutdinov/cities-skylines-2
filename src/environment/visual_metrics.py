"""
Visual metrics estimation from raw game screenshots.
Uses only visual information to estimate game metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
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
        weights = {
            'building_density': 0.4,
            'residential_areas': 0.3,
            'traffic_density': 0.2,
            'window_lights': 0.1
        }
        
        # Compute weighted sum and scale to reasonable population range
        weighted_sum = sum(metrics[k] * weights[k] for k in weights)
        
        # Scale to population (this would need calibration)
        # Assuming maximum reasonable starting city population around 10000
        estimated_population = int(weighted_sum * 10000)
        
        return max(0, estimated_population)  # Ensure non-negative
        
    def update_model(self, frame: torch.Tensor, reward: float):
        """
        Update the estimation model based on rewards.
        This allows the model to learn from the success/failure of its estimates.
        """
        # This would be expanded to actually update the feature extractor
        # based on reward signals from the environment
        pass

    def initialize_menu_detection(self, menu_reference_path):
        """Load and prepare a reference screenshot of the menu for detection.
        
        Args:
            menu_reference_path: Path to a screenshot of the menu
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load reference image
            logger.info(f"Loading menu reference image from: {menu_reference_path}")
            self.menu_reference = cv2.imread(menu_reference_path)
            
            if self.menu_reference is None:
                logger.error(f"Failed to load menu reference image from {menu_reference_path}")
                return False
                
            # Convert to RGB if needed
            if len(self.menu_reference.shape) == 3 and self.menu_reference.shape[2] == 3:
                # Already a color image
                self.menu_reference_gray = cv2.cvtColor(self.menu_reference, cv2.COLOR_BGR2GRAY)
            else:
                # Grayscale image
                self.menu_reference_gray = self.menu_reference
                
            # Calculate histogram of reference image for comparison
            self.menu_reference_hist = cv2.calcHist([self.menu_reference_gray], [0], None, [256], [0, 256])
            cv2.normalize(self.menu_reference_hist, self.menu_reference_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Set flag to indicate we have a reference
            self.has_menu_reference = True
            logger.info("Menu reference image loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing menu detection: {e}")
            self.has_menu_reference = False
            return False
            
    def detect_main_menu(self, frame):
        """Detect if the main menu is visible in the current frame.
        
        This method uses a combination of approaches:
        1. If a reference image is available, it compares the current frame to it
        2. If no reference image is available, it falls back to heuristic detection
        
        Args:
            frame: The current frame to analyze
            
        Returns:
            bool: True if menu is detected, False otherwise
        """
        # Ensure frame is a numpy array
        if isinstance(frame, torch.Tensor):
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            # If normalized tensor (0-1), convert to 0-255 range
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
        else:
            frame_np = frame
            
        # Convert to BGR if needed (OpenCV format)
        if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
            frame_bgr = frame_np
        else:
            # Unexpected format, try to convert or use as is
            logger.warning(f"Unexpected frame format: {frame_np.shape}")
            frame_bgr = frame_np
            
        # Convert to grayscale for processing
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if len(frame_bgr.shape) == 3 else frame_bgr
        
        # If we have a reference image, use image comparison method
        if hasattr(self, 'has_menu_reference') and self.has_menu_reference:
            return self._detect_menu_with_reference(frame_bgr, frame_gray)
        else:
            # Fall back to heuristic detection
            return self._detect_menu_with_heuristics(frame_bgr, frame_gray)
            
    def _detect_menu_with_reference(self, frame_bgr, frame_gray):
        """Detect menu using reference image comparison.
        
        Args:
            frame_bgr: BGR color frame
            frame_gray: Grayscale frame
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        # Resize frame to match reference if they're different sizes
        if frame_gray.shape != self.menu_reference_gray.shape:
            frame_resized = cv2.resize(frame_gray, (self.menu_reference_gray.shape[1], self.menu_reference_gray.shape[0]))
        else:
            frame_resized = frame_gray
            
        # Method 1: Structural similarity index (SSIM)
        try:
            ssim_score = structural_similarity(frame_resized, self.menu_reference_gray)
            logger.debug(f"SSIM score: {ssim_score}")
            
            # Method 2: Histogram comparison
            frame_hist = cv2.calcHist([frame_resized], [0], None, [256], [0, 256])
            cv2.normalize(frame_hist, frame_hist, 0, 1, cv2.NORM_MINMAX)
            hist_match = cv2.compareHist(self.menu_reference_hist, frame_hist, cv2.HISTCMP_CORREL)
            logger.debug(f"Histogram match: {hist_match}")
            
            # Detect menu based on combined scores with more stringent thresholds
            is_menu = (ssim_score > 0.65) or (hist_match > 0.95)
            
            if is_menu:
                logger.info(f"Menu detected (SSIM: {ssim_score:.2f}, Hist: {hist_match:.2f})")
            
            return is_menu
            
        except Exception as e:
            logger.error(f"Error in reference-based menu detection: {e}")
            # Fall back to heuristic approach
            return self._detect_menu_with_heuristics(frame_bgr, frame_gray)
            
    def _detect_menu_with_heuristics(self, frame_bgr, frame_gray):
        """Detect menu using heuristic methods when no reference image is available.
        
        Args:
            frame_bgr: BGR color frame
            frame_gray: Grayscale frame
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        # Extract frame dimensions
        height, width = frame_gray.shape
        
        # Create regions of interest for menu detection
        left_region = frame_gray[:, :int(width * 0.3)]  # Left 30% - where menu items appear
        top_region = frame_gray[:int(height * 0.1), :]  # Top 10% - where logo appears
        full_frame = frame_gray.copy()
        
        # Initialize menu indicators with confidence scores
        menu_indicators = []
        
        # 1. Check for horizontal lines typical in menu UI
        # Apply edge detection
        edges = cv2.Canny(full_frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=width*0.2, maxLineGap=20)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if the line is horizontal
                if abs(y2 - y1) < 10:  # Small vertical difference = horizontal line
                    horizontal_lines += 1
        
        if horizontal_lines >= 3:
            score = min(1.0, horizontal_lines / 10.0)  # Score based on number of lines, max 1.0
            menu_indicators.append(("horizontal_lines", score))
            logger.debug(f"Menu indicator: {horizontal_lines} horizontal lines found (score: {score:.2f})")
        
        # 2. Check for menu button text (bright text in left region)
        _, left_thresh = cv2.threshold(left_region, 200, 255, cv2.THRESH_BINARY)
        white_pixels = np.sum(left_thresh == 255)
        white_ratio = white_pixels / (left_region.shape[0] * left_region.shape[1])
        
        if white_ratio > 0.01:  # At least 1% bright pixels
            score = min(1.0, white_ratio * 20)  # Scale ratio to score
            menu_indicators.append(("bright_text", score))
            logger.debug(f"Menu indicator: bright text detected (ratio: {white_ratio:.3f}, score: {score:.2f})")
        
        # 3. Check for dark overlay (common in menu screens)
        dark_pixels = np.sum(full_frame < 50)  # Count very dark pixels
        dark_ratio = dark_pixels / (height * width)
        
        if dark_ratio > 0.3:  # More than 30% of the screen is very dark
            score = min(1.0, dark_ratio)  # Use ratio directly as score
            menu_indicators.append(("dark_overlay", score))
            logger.debug(f"Menu indicator: dark overlay detected (ratio: {dark_ratio:.3f}, score: {score:.2f})")
        
        # 4. Check for aligned menu button arrangement
        # Look for clusters of white pixels in the left region at regular vertical intervals
        row_sums = np.sum(left_thresh, axis=1)
        peaks = []
        
        for i in range(1, len(row_sums) - 1):
            if row_sums[i] > row_sums[i-1] and row_sums[i] > row_sums[i+1] and row_sums[i] > 100:
                peaks.append(i)
        
        # Check if we have multiple peaks with somewhat regular spacing
        if len(peaks) >= 3:
            intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks) - 1)]
            avg_interval = sum(intervals) / len(intervals)
            interval_diffs = [abs(interval - avg_interval) for interval in intervals]
            avg_diff = sum(interval_diffs) / len(interval_diffs)
            
            if avg_diff < 15:  # Fairly regular spacing
                score = min(1.0, len(peaks) / 10.0)  # Score based on number of peaks
                menu_indicators.append(("aligned_buttons", score))
                logger.debug(f"Menu indicator: aligned menu buttons detected ({len(peaks)} buttons, score: {score:.2f})")
        
        # 5. NEW: Check for a bordered rectangle (menu container)
        # Apply edge detection again with different parameters
        edges2 = cv2.Canny(full_frame, 30, 200)
        contours, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for large rectangular contours
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > (width * height * 0.1):  # At least 10% of screen
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                # If it's a reasonable rectangle shape (not too extreme in aspect ratio)
                if 0.2 < aspect_ratio < 5.0:
                    rect_score = min(1.0, area / (width * height * 0.5))  # Score based on area
                    menu_indicators.append(("menu_container", rect_score))
                    logger.debug(f"Menu indicator: potential menu container found (area: {area}, score: {rect_score:.2f})")
                    break
        
        # Determine if menu is present based on weighted indicators
        if len(menu_indicators) >= 2:
            # Calculate overall confidence score
            total_score = sum(score for _, score in menu_indicators)
            avg_score = total_score / len(menu_indicators)
            confidence = avg_score * (1.0 + (len(menu_indicators) - 2) * 0.2)  # Boost for more indicators
            
            # Consider it a menu if we have strong confidence
            is_menu = confidence > 0.5
            
            if is_menu:
                logger.info(f"Menu detected with {len(menu_indicators)} indicators, confidence: {confidence:.2f}")
                # Log all indicators
                for name, score in menu_indicators:
                    logger.info(f"  - {name}: {score:.2f}")
            
            return is_menu
        
        return False  # Not enough indicators

    def save_current_frame_as_menu_reference(self, frame: torch.Tensor, save_path: str = "menu_reference.png"):
        """Save the current frame as a menu reference image.
        
        Args:
            frame (torch.Tensor): Current frame to save as reference
            save_path (str): Path to save the reference image
        
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            # Convert frame to numpy for OpenCV processing if it's a tensor
            if isinstance(frame, torch.Tensor):
                frame_np = frame.permute(1, 2, 0).cpu().numpy()  # [C,H,W] -> [H,W,C]
                
                # If normalized, convert to 0-255 range
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame.copy()
                
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            
            # Save image
            cv2.imwrite(save_path, frame_bgr)
            
            # Also update our reference image
            self.menu_reference_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            print(f"Saved menu reference image to {save_path}")
            return True
        except Exception as e:
            print(f"Error saving menu reference image: {e}")
            return False

    def calculate_reward(self, frame):
        """Calculate a reward based on visual metrics from the given frame.
        
        Args:
            frame: The current observation frame
            
        Returns:
            float: The calculated reward
        """
        # This is a placeholder implementation
        # In a real environment, you would implement more sophisticated reward calculations
        # based on visual features like population numbers, budget, or other game metrics
        
        # For now, we'll return a small positive reward for each valid step
        # This encourages exploration
        return 0.01 