"""
Menu detector for Cities: Skylines 2.

This module handles the detection of in-game menus.
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import torch
import os

from src.utils.image_utils import ImageUtils
from src.environment.core.observation import ObservationManager
from ..core.performance import PerformanceMonitor

logger = logging.getLogger(__name__)

class MenuDetector:
    """Handles detection of in-game menus."""
    
    # Menu types and their characteristics
    MENU_TYPES = {
        "main_menu": {"threshold": 0.8, "signature_regions": [(0.1, 0.05, 0.5, 0.15)]},
        "pause_menu": {"threshold": 0.75, "signature_regions": [(0.4, 0.35, 0.6, 0.65)]},
        "settings_menu": {"threshold": 0.7, "signature_regions": [(0.15, 0.1, 0.85, 0.9)]},
        "dialog": {"threshold": 0.85, "signature_regions": [(0.3, 0.4, 0.7, 0.6)]},
        "notification": {"threshold": 0.9, "signature_regions": [(0.7, 0.1, 0.95, 0.3)]},
    }
    
    # Regions that represent normal game UI, not menus (normalized coordinates)
    GAME_UI_REGIONS = [
        (0.0, 0.0, 0.25, 0.1),    # Top-left UI bar 
        (0.75, 0.0, 1.0, 0.1),    # Top-right UI bar
        (0.0, 0.85, 0.3, 1.0),    # Bottom-left control panel
        (0.3, 0.85, 1.0, 1.0),    # Bottom info panel
        (0.9, 0.1, 1.0, 0.3),     # Side tool panel
    ]
    
    # CS2 Logo region (normalized coordinates based on 1920x1080 resolution)
    CS2_LOGO_REGION = (0.435, 0.059, 0.568, 0.407)  # Corresponds to (835,64) to (1090, 440)
    
    def __init__(
        self,
        observation_manager: ObservationManager,
        performance_monitor: Optional[PerformanceMonitor] = None,
    ):
        """Initialize menu detector.
        
        Args:
            observation_manager: Observation management module
            performance_monitor: Optional performance monitoring module
        """
        self.observation_manager = observation_manager
        self.performance_monitor = performance_monitor
        self.image_utils = ImageUtils()
        
        # Menu state
        self.in_menu = False
        self.menu_type = None
        self.menu_detection_confidence = 0.0
        self.last_menu_check_time = 0
        self.menu_check_interval = 0.5  # seconds
        self.menu_templates = self._load_menu_templates()
        
        # Menu toggling prevention
        self.last_menu_exit_time = 0
        self.menu_exit_cooldown = 1.0  # seconds to wait after exiting a menu before detecting again
        self.consecutive_menu_detections = 0
        self.detection_threshold_adjustment = 0.0  # dynamic adjustment to threshold
        
        # Menu detection refinement
        self.last_detection_result = (False, None, 0.0)
        self.detection_history = []
        self.max_history_size = 5
        self.detection_stability_count = 0
        self.min_stability_count = 2  # Minimum number of consistent detections needed
        
        # Load CS2 logo template for reliable menu detection
        self.cs2_logo_template = self._load_cs2_logo_template()
        self.cs2_logo_detection_threshold = 0.6
        self.use_logo_detection = True  # Flag to enable/disable logo detection
        
        logger.info("Menu detector initialized")
    
    def _load_cs2_logo_template(self) -> Optional[np.ndarray]:
        """Load the CS2 logo template for menu detection.
        
        Returns:
            Optional[np.ndarray]: The logo template or None if not found
        """
        try:
            # Try several possible paths for the logo template
            possible_paths = [
                "menu_templates/cs2_logo.png",                                    # From project root
                os.path.join(os.getcwd(), "menu_templates/cs2_logo.png"),         # From current working directory
                os.path.abspath("menu_templates/cs2_logo.png"),                   # Absolute path from relative
                "C:/Users/stani/Desktop/cities skylines 2/menu_templates/cs2_logo.png",  # Direct absolute path
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../menu_templates/cs2_logo.png"),  # From script location
            ]
            
            for path in possible_paths:
                logger.debug(f"Trying to load CS2 logo from: {path}")
                if os.path.exists(path):
                    logger.info(f"Loading CS2 logo template from: {path}")
                    logo = cv2.imread(path)
                    
                    if logo is not None:
                        logger.info(f"Successfully loaded CS2 logo template, shape: {logo.shape}")
                        return logo
            
            logger.warning("CS2 logo template not found in any of the possible locations")
            logger.warning(f"Current working directory: {os.getcwd()}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading CS2 logo template: {e}")
            return None

    def _load_menu_templates(self) -> Dict:
        """Load menu template images for detection.
        
        Returns:
            Dict: Dictionary of menu templates
        """
        # In a real implementation, this would load actual template images
        # For now, we'll return an empty dict and rely on visual change detection
        return {}
    
    def _extract_menu_signatures(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract signature regions from frame for each menu type.
        
        Args:
            frame: The current frame
            
        Returns:
            Dictionary mapping menu types to signature region arrays
        """
        signatures = {}
        
        for menu_type, config in self.MENU_TYPES.items():
            regions = []
            
            for region in config["signature_regions"]:
                x1, y1, x2, y2 = region
                h, w = frame.shape[:2]
                
                # Convert normalized coordinates to pixel values
                x1_px, y1_px = int(x1 * w), int(y1 * h)
                x2_px, y2_px = int(x2 * w), int(y2 * h)
                
                # Extract the region
                region_data = frame[y1_px:y2_px, x1_px:x2_px]
                regions.append(region_data)
                
            # Combine regions if there are multiple
            if regions:
                signatures[menu_type] = regions
                
        return signatures
    
    def detect_cs2_logo(self, frame=None):
        """Detect the CS2 logo in the current frame.
        
        Args:
            frame: Optional frame to detect in, otherwise captures current frame
            
        Returns:
            Tuple of (is_detected, confidence, location)
        """
        # Use provided frame or capture new one
        if frame is None:
            frame = self.observation_manager.capture_frame()
            if frame is None:
                return False, 0.0, None
        
        if self.cs2_logo_template is None:
            self.load_cs2_logo_template()
            if self.cs2_logo_template is None:
                return False, 0.0, None
        
        # Extract region of interest
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.CS2_LOGO_REGION
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)
        
        if x1_px >= x2_px or y1_px >= y2_px or x2_px > w or y2_px > h:
            return False, 0.0, None
            
        roi = frame[y1_px:y2_px, x1_px:x2_px]
        if roi.size == 0:
            return False, 0.0, None
        
        # Resize ROI to match template dimensions if needed
        template_h, template_w = self.cs2_logo_template.shape[:2]
        if roi.shape[0] != template_h or roi.shape[1] != template_w:
            roi = cv2.resize(roi, (template_w, template_h))
        
        # Perform template matching
        try:
            result = cv2.matchTemplate(roi, self.cs2_logo_template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, location = cv2.minMaxLoc(result)
        except Exception as e:
            logger.error(f"Error during logo template matching: {e}")
            return False, 0.0, None
        
        is_detected = confidence >= self.cs2_logo_detection_threshold
        return is_detected, confidence, location

    def get_logo_embedding(self, frame=None, embedding_size=32):
        """Generate a compact embedding of the CS2 logo region for RL agent.
        
        This method extracts the logo region, downsamples it to a small size,
        and converts it to a flattened numeric vector for efficient processing.
        
        Args:
            frame: Optional frame to use, otherwise captures current frame
            embedding_size: Size of the square embedding (default: 32x32)
            
        Returns:
            dict: Embedding dictionary with vector, detection flag, and confidence
        """
        # Use provided frame or capture new one
        if frame is None:
            frame = self.observation_manager.capture_frame()
            if frame is None:
                return {"detected": False, "confidence": 0.0, "vector": np.zeros(embedding_size * embedding_size)}
        
        # First run standard logo detection
        is_detected, confidence, _ = self.detect_cs2_logo(frame)
        
        # Extract region of interest
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.CS2_LOGO_REGION
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)
        
        # Handle edge cases to prevent crashes
        if x1_px >= x2_px or y1_px >= y2_px or x2_px > w or y2_px > h:
            return {"detected": False, "confidence": 0.0, "vector": np.zeros(embedding_size * embedding_size)}
            
        roi = frame[y1_px:y2_px, x1_px:x2_px]
        if roi.size == 0:
            return {"detected": False, "confidence": 0.0, "vector": np.zeros(embedding_size * embedding_size)}
        
        # Convert to grayscale to reduce dimensions
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        
        # Resize to small fixed size for efficiency
        small_roi = cv2.resize(gray_roi, (embedding_size, embedding_size))
        
        # Normalize pixel values to 0-1 range
        normalized_roi = small_roi.astype(np.float32) / 255.0
        
        # Flatten to 1D array
        flat_vector = normalized_roi.flatten()
        
        # Return embedding dictionary with detection info
        return {
            "detected": is_detected,
            "confidence": float(confidence),
            "vector": flat_vector
        }
    
    def detect_menu(self, frame=None):
        """Detect if the current frame shows a menu.
        
        Args:
            frame: Optional frame to detect in, otherwise captures current frame
            
        Returns:
            Tuple of (is_in_menu, menu_type, confidence)
        """
        # Use provided frame or capture new one
        if frame is None:
            frame = self.observation_manager.capture_frame()
            if frame is None:
                return False, None, 0.0
        
        # First check for CS2 logo as a reliable indicator
        logo_detected, logo_confidence, _ = self.detect_cs2_logo(frame)
        if logo_detected:
            # Store the logo location for use in navigation
            self.last_menu_type = "logo_menu"
            self.last_detection_time = time.time()
            self.last_menu_confidence = logo_confidence
            return True, "logo_menu", logo_confidence
        
        # Fallback to regular menu detection methods
        # Get screen layout hash
        layout_hash = self._compute_layout_hash(frame)
        
        # Check for match with known menu templates
        best_match = None
        best_confidence = 0.0
        
        # Find best template match
        for template_name, template in self.templates.items():
            # Get template image
            template_img = template.get("image")
            if template_img is None:
                continue
                
            # Do template matching
            match, confidence = self._match_template(frame, template_img)
            
            logger.debug(f"Template {template_name}: matched={match}, confidence={confidence:.4f}")
            
            if match and confidence > best_confidence:
                best_match = template_name
                best_confidence = confidence
        
        # Store list of menu scores for debugging
        self.last_menu_scores = [(name, self._match_template(frame, template.get("image"))[1]) 
                                 for name, template in self.templates.items() 
                                 if template.get("image") is not None]
        self.last_menu_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Check if confidence exceeds threshold
        if best_match and best_confidence >= self.detection_threshold:
            self.last_menu_type = best_match
            self.last_detection_time = time.time()
            self.last_menu_confidence = best_confidence
            return True, best_match, best_confidence
        
        # Check layout hash against known menu hashes
        if layout_hash in self.menu_hashes:
            menu_type = self.menu_hashes[layout_hash]
            logger.debug(f"Layout hash match for menu type: {menu_type}")
            self.last_menu_type = menu_type
            self.last_detection_time = time.time()
            self.last_menu_confidence = 0.8  # Default confidence for hash match
            return True, menu_type, 0.8
            
        # No menu detected
        return False, None, 0.0
    
    def check_menu_state(self) -> bool:
        """Check if we're currently in a menu state.
        
        Returns:
            Whether we're currently in a menu
        """
        current_time = time.time()
        
        # Only check periodically to improve performance
        if current_time - self.last_menu_check_time < self.menu_check_interval:
            return self.in_menu
            
        self.last_menu_check_time = current_time
        
        try:
            # Capture current frame
            current_frame = self.observation_manager.capture_frame()
            if current_frame is None:
                logger.warning("Failed to capture frame for menu detection")
                return self.in_menu  # Return previous state
            
            # Convert to array if it's a tensor
            if isinstance(current_frame, torch.Tensor):
                current_frame = current_frame.cpu().numpy()
                
                # Convert from CxHxW to HxWxC format if needed
                if current_frame.shape[0] == 3 and current_frame.ndim == 3:
                    current_frame = np.transpose(current_frame, (1, 2, 0))
                
                # Convert to uint8 if needed
                if current_frame.dtype != np.uint8:
                    current_frame = (current_frame * 255).astype(np.uint8)
            
            # Detect menu
            menu_detected, menu_type, confidence = self.detect_menu(current_frame)
            
            # Store results
            self.detection_history.append((menu_detected, menu_type, confidence))
            if len(self.detection_history) > self.max_history_size:
                self.detection_history.pop(0)
            
            # Check for detection stability
            if self.detection_history:
                # All entries agree on menu detection
                agree = all(entry[0] == self.detection_history[0][0] for entry in self.detection_history)
                
                if agree:
                    self.detection_stability_count += 1
                else:
                    self.detection_stability_count = 0
                
                # If we have stable menu detection
                if self.detection_stability_count >= self.min_stability_count:
                    # Update menu state
                    self.in_menu = menu_detected
                    self.menu_type = menu_type
                    self.menu_detection_confidence = confidence
                    
                    # Log transition
                    if menu_detected != self.last_detection_result[0]:
                        if menu_detected:
                            logger.info(f"Menu detected: {menu_type} (confidence: {confidence:.2f})")
                            # Record when we entered a menu
                        else:
                            logger.info(f"Exited menu (confidence: {confidence:.2f})")
                            self.last_menu_exit_time = current_time
                    
                    self.last_detection_result = (menu_detected, menu_type, confidence)
            
            return self.in_menu
            
        except Exception as e:
            logger.error(f"Error checking menu state: {e}")
            return self.in_menu  # Return previous state
    
    def is_in_menu(self) -> bool:
        """Check if we're currently in a menu.
        
        Returns:
            Whether we're in a menu
        """
        return self.in_menu
    
    def get_menu_type(self) -> Optional[str]:
        """Get the current menu type.
        
        Returns:
            Current menu type or None if not in a menu
        """
        return self.menu_type if self.in_menu else None
    
    def get_detection_confidence(self) -> float:
        """Get the confidence of the menu detection.
        
        Returns:
            Detection confidence
        """
        return self.menu_detection_confidence 