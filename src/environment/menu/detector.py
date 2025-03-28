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
        
        logger.info("Menu detector initialized")
    
    def _load_menu_templates(self) -> Dict:
        """Load menu template images for detection.
        
        Returns:
            Dict: Dictionary of menu templates
        """
        # In a real implementation, this would load actual template images
        # For now, we'll return an empty dict and rely on visual change detection
        templates = {}
        
        # Try to load logo template for main menu detection
        try:
            # Check for template directory
            template_paths = [
                os.path.join("menu_templates", "logo_template.png"),  # Default location
                os.path.join("assets", "templates", "logo_template.png"),  # Alternative location
                "logo_template.png"  # Fallback to root directory
            ]
            
            for path in template_paths:
                if os.path.exists(path):
                    logger.info(f"Loading logo template from {path}")
                    logo_template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    if logo_template is not None:
                        # Preprocess the template (invert if it's black on white)
                        logo_template = self._preprocess_template(logo_template)
                        templates["main_menu_logo"] = logo_template
                        logger.info(f"Logo template loaded with shape {logo_template.shape}")
                        break
        except Exception as e:
            logger.warning(f"Failed to load logo template: {e}")
        
        return templates
    
    def _preprocess_template(self, template: np.ndarray) -> np.ndarray:
        """Preprocess a template image for better matching.
        
        Args:
            template: The template image
            
        Returns:
            Preprocessed template
        """
        # Check if template is predominantly dark (logo is black on white)
        if template.mean() > 127:  # If average is bright (white background)
            # Invert the template to make it white on black for better matching
            template = cv2.bitwise_not(template)
            logger.debug("Inverted template for better matching")
        
        return template
    
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
    
    def detect_menu(self, current_frame: np.ndarray) -> Tuple[bool, Optional[str], float]:
        """Detect if the current frame shows a menu and which type.
        
        Args:
            current_frame: Current frame to analyze
            
        Returns:
            Tuple of (menu_detected, menu_type, confidence)
        """
        # Skip detection if we recently exited a menu (prevents flapping)
        if time.time() - self.last_menu_exit_time < self.menu_exit_cooldown:
            return False, None, 0.0
        
        # Use visual metrics if available
        if self.performance_monitor is not None and hasattr(self.performance_monitor, 'detect_menu'):
            try:
                menu_detected, menu_type, confidence = self.performance_monitor.detect_menu(current_frame)
                
                # Apply dynamic threshold adjustment based on history
                adjusted_confidence = confidence - self.detection_threshold_adjustment
                threshold = 0.75  # Base threshold
                
                if adjusted_confidence > threshold:
                    self.consecutive_menu_detections += 1
                    # Increase threshold slightly to prevent false positives
                    if self.consecutive_menu_detections > 3:
                        self.detection_threshold_adjustment = min(0.1, self.detection_threshold_adjustment + 0.02)
                else:
                    self.consecutive_menu_detections = 0
                    # Decrease threshold adjustment over time to recover sensitivity
                    self.detection_threshold_adjustment = max(0, self.detection_threshold_adjustment - 0.01)
                
                # Return adjusted result
                if adjusted_confidence > threshold:
                    return True, menu_type, adjusted_confidence
                else:
                    return False, None, adjusted_confidence
                    
            except Exception as e:
                logger.error(f"Error in visual metrics menu detection: {e}")
                # Fall through to the fallback implementation
        
        # Fallback implementation: template matching and visual analysis
        try:
            # Try template matching if we have templates
            if self.menu_templates:
                # Convert current frame to grayscale if needed for template matching
                current_frame_gray = current_frame
                if len(current_frame.shape) == 3 and current_frame.shape[2] == 3:
                    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                
                for menu_type, template in self.menu_templates.items():
                    try:
                        # If "logo" in menu_type, we're looking for a logo specifically
                        if "logo" in menu_type.lower():
                            # Make sure template and frame have same number of channels
                            if len(template.shape) != len(current_frame_gray.shape):
                                if len(template.shape) == 2:  # Grayscale template
                                    if len(current_frame_gray.shape) == 3:
                                        current_frame_gray = cv2.cvtColor(current_frame_gray, cv2.COLOR_BGR2GRAY)
                                else:  # Color template with grayscale frame
                                    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                            
                            # Apply template matching
                            result = cv2.matchTemplate(current_frame_gray, template, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(result)
                            
                            # Use a slightly higher threshold for logo detection (0.8)
                            logo_threshold = 0.8
                            if max_val > logo_threshold:
                                logger.info(f"Logo detected with confidence {max_val:.2f}")
                                return True, "main_menu", max_val
                        else:
                            # Standard template matching for other templates
                            result = cv2.matchTemplate(current_frame, template, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, _ = cv2.minMaxLoc(result)
                            
                            threshold = self.MENU_TYPES.get(menu_type, {"threshold": 0.75})["threshold"]
                            if max_val > threshold:
                                return True, menu_type, max_val
                    except Exception as e:
                        logger.warning(f"Error matching template {menu_type}: {e}")
                        continue
            
            # Visual analysis approach
            # Extract regions that are characteristic of menus
            signatures = self._extract_menu_signatures(current_frame)
            
            # Check each menu type
            for menu_type, regions in signatures.items():
                # Analyze the signature regions
                # This is a simplified placeholder for more complex analysis
                # In a real implementation, this would analyze colors, patterns, etc.
                
                # Example: Check for uniform bright regions which might indicate a menu
                confidence = 0
                for region in regions:
                    if region.mean() > 160:  # Bright region
                        confidence += 0.5
                    if np.std(region) < 30:  # Uniform region
                        confidence += 0.3
                
                threshold = self.MENU_TYPES[menu_type]["threshold"]
                if confidence > threshold:
                    return True, menu_type, confidence
            
            # No menu detected
            return False, None, 0.0
            
        except Exception as e:
            logger.error(f"Error in fallback menu detection: {e}")
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
    
    def detect_logo(self, frame: np.ndarray) -> Tuple[bool, float]:
        """Check if logo is detected in the frame.
        
        Args:
            frame: Frame to check
            
        Returns:
            Tuple of (logo_detected, confidence)
        """
        try:
            # Convert frame to grayscale if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frame_gray = frame
            
            # Get the main menu logo template
            logo_template = self.menu_templates.get("main_menu_logo")
            if logo_template is None:
                logger.debug("No logo template available for detection")
                return False, 0.0
            
            # Match template
            result = cv2.matchTemplate(frame_gray, logo_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Logo is detected if confidence exceeds threshold
            logo_threshold = 0.8
            logo_detected = max_val > logo_threshold
            
            if logo_detected:
                logger.debug(f"Logo detected with confidence {max_val:.2f} at {max_loc}")
            
            return logo_detected, max_val
            
        except Exception as e:
            logger.error(f"Error in logo detection: {e}")
            return False, 0.0 