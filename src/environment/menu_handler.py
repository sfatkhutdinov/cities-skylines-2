import logging
import time
import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional, Union
import torch

from .input_simulator import InputSimulator
from .optimized_capture import OptimizedScreenCapture
from .visual_metrics import VisualMetricsEstimator
from src.utils.image_utils import ImageUtils

logger = logging.getLogger(__name__)

class MenuHandler:
    """Handles detection and navigation of in-game menus."""
    
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
    
    # Gear icon position (normalized)
    GEAR_ICON_POSITION = (0.95, 0.05)  # Top-right corner
    
    def __init__(
        self,
        screen_capture: OptimizedScreenCapture,
        input_simulator: InputSimulator,
        visual_metrics: Optional[VisualMetricsEstimator] = None,
    ):
        """Initialize menu handler.
        
        Args:
            screen_capture: Screen capture module
            input_simulator: Input simulator for keyboard/mouse actions
            visual_metrics: Optional VisualMetricsEstimator instance for menu detection
        """
        self.screen_capture = screen_capture
        self.input_simulator = input_simulator
        self.visual_metrics = visual_metrics
        self.image_utils = ImageUtils()
        
        # Menu state
        self.in_menu = False
        self.menu_type = None
        self.menu_detection_confidence = 0.0
        self.last_menu_check_time = 0
        self.menu_check_interval = 0.5  # seconds
        self.menu_templates = self._load_menu_templates()
        self.menu_recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        # Menu toggling prevention
        self.last_menu_exit_time = 0
        self.menu_exit_cooldown = 1.0  # seconds to wait after exiting a menu before detecting again
        self.last_menu_action_time = 0
        self.consecutive_menu_detections = 0
        self.detection_threshold_adjustment = 0.0  # dynamic adjustment to threshold
        
        # Menu stuck detection
        self.menu_stuck_time = None
        self.max_menu_stuck_time = 60.0  # seconds to be stuck in menu before drastic action
        self.last_menu_reset_time = 0
        self.menu_reset_cooldown = 120.0  # seconds between forced resets
        
        # Navigation paths for different menus (button sequence to exit)
        self.menu_exit_paths = {
            "main_menu": ["play_button"],
            "pause_menu": ["resume_button"],
            "settings_menu": ["back_button", "resume_button"],
            "dialog": ["ok_button", "close_button"],
            "notification": ["close_button"]
        }
        
        # Menu button positions (normalized coordinates)
        self.menu_button_positions = {
            "play_button": (0.5, 0.4),
            "resume_button": (0.5, 0.35),
            "back_button": (0.2, 0.9),
            "ok_button": (0.5, 0.65),
            "close_button": (0.9, 0.1),
            "settings_button": (0.5, 0.5),
            "exit_button": (0.5, 0.7)
        }
        
        # Action tracking for menu correlation
        self.recent_actions = []
        self.max_action_history = 10
        self.last_esc_press_time = 0
        self.last_gear_click_time = 0
        self.menu_prone_actions = {
            "esc_key": {"time": 0, "active": False},
            "gear_click": {"time": 0, "active": False}
        }
        self.action_vigilance_timeout = 2.0  # Seconds to maintain vigilance after menu-prone action
        
        logger.info("Menu handler initialized")
    
    def _load_menu_templates(self) -> Dict:
        """Load menu template images for detection.
        
        Returns:
            Dict: Dictionary of menu templates
        """
        # In a real implementation, this would load actual template images
        # For now, we'll return an empty dict and rely on visual change detection
        return {}
    
    # Track actions that might trigger menus
    def track_action(self, action_type: str) -> None:
        """Track actions that might lead to menu screens.
        
        Args:
            action_type: Type of action (e.g., "esc_key", "gear_click")
        """
        current_time = time.time()
        self.recent_actions.append((action_type, current_time))
        
        # Trim history if needed
        if len(self.recent_actions) > self.max_action_history:
            self.recent_actions = self.recent_actions[-self.max_action_history:]
        
        # Mark specific actions for vigilance
        if action_type == "esc_key":
            self.menu_prone_actions["esc_key"]["time"] = current_time
            self.menu_prone_actions["esc_key"]["active"] = True
            self.last_esc_press_time = current_time
            
        elif action_type == "gear_click":
            self.menu_prone_actions["gear_click"]["time"] = current_time
            self.menu_prone_actions["gear_click"]["active"] = True
            self.last_gear_click_time = current_time

    def detect_menu(self, current_frame: np.ndarray) -> Tuple[bool, Optional[str], float]:
        """Detect if the current frame shows a menu and which type.
        
        Args:
            current_frame: Current frame to analyze
            
        Returns:
            Tuple[bool, str, float]: Menu detected, menu type, confidence
        """
        # Skip detection if too soon since last check
        current_time = time.time()
        
        # Reduce the check interval to be more responsive to manual menu opening
        self.menu_check_interval = 0.1  # Reduced from 0.5 to 0.1 seconds
        
        if current_time - self.last_menu_check_time < self.menu_check_interval:
            return self.in_menu, self.menu_type, self.menu_detection_confidence
            
        # Update last check time
        self.last_menu_check_time = current_time
        
        # Safety check for None frame
        if current_frame is None:
            logger.warning("Received None frame in detect_menu")
            return False, None, 0.0
            
        # Ensure the frame is valid
        if isinstance(current_frame, np.ndarray) and current_frame.size == 0:
            logger.warning("Received empty frame in detect_menu")
            return False, None, 0.0
            
        # Force a specific result for testing purposes
        if hasattr(self, '_force_menu_detection_result') and self._force_menu_detection_result is not None:
            forced_result, forced_type, forced_confidence = self._force_menu_detection_result
            logger.debug(f"Forcing menu detection result: {forced_result}, {forced_type}, {forced_confidence}")
            return forced_result, forced_type, forced_confidence
        
        # When user manually opens a menu, the cooldown should be bypassed
        # so reduce the threshold for the menu exit cooldown
        menu_exit_cooldown_check = (not self.in_menu and 
                         current_time - self.last_menu_exit_time < self.menu_exit_cooldown / 2)
        
        if menu_exit_cooldown_check:
            logger.debug(f"In menu exit cooldown period (remaining: {self.menu_exit_cooldown - (current_time - self.last_menu_exit_time):.1f}s)")
            
            # Even if in cooldown, do a minimal check for sudden visual changes 
            # that might indicate a manually opened menu
            if hasattr(self.visual_metrics, 'visual_change_analyzer'):
                change_score = self.visual_metrics.visual_change_analyzer.get_recent_change_score()
                if change_score > 0.3:  # If significant visual change detected
                    logger.info(f"Significant visual change detected ({change_score:.2f}), checking for menu despite cooldown")
                    # Continue with detection despite cooldown
                else:
                    return False, None, 0.0
            else:
                return False, None, 0.0
            
        # Check if input suggests we're in a menu-prone state
        increased_vigilance = self._check_menu_prone_actions()
        
        # Use visual metrics if available
        if self.visual_metrics:
            try:
                # 1. Combined approach using multiple detection methods
                detection_scores = {}
                
                # A. Direct menu detection (template matching)
                try:
                    direct_menu_detected = self.visual_metrics.detect_main_menu(current_frame)
                    detection_scores["direct"] = 0.8 if direct_menu_detected else 0.0
                except Exception as e:
                    logger.error(f"Error in direct menu detection: {e}")
                    detection_scores["direct"] = 0.0
                
                # B. UI element density analysis
                try:
                    ui_density = self.visual_metrics.calculate_ui_density(current_frame)
                    detection_scores["density"] = ui_density
                except Exception as e:
                    logger.error(f"Error in UI density analysis: {e}")
                    detection_scores["density"] = 0.0
                
                # C. Contrast pattern analysis
                try:
                    contrast_score = self.visual_metrics.detect_menu_contrast_pattern(current_frame)
                    detection_scores["contrast"] = contrast_score
                except Exception as e:
                    logger.error(f"Error in contrast pattern analysis: {e}")
                    detection_scores["contrast"] = 0.0
                
                # D. Visual change score (game paused/menu appeared)
                try:
                    visual_change_score = self.visual_metrics.get_visual_change_score(current_frame)
                    # FIX: Handle PyTorch tensor conversion correctly
                    if hasattr(visual_change_score, 'item'):
                        visual_change_score = visual_change_score.item()
                    # Invert: lower change = higher score
                    detection_scores["change"] = max(0, 1.0 - visual_change_score * 5.0)  
                except Exception as e:
                    logger.error(f"Error in visual change score: {e}")
                    detection_scores["change"] = 0.0
                
                # E. UI element analysis
                try:
                    ui_elements = self.visual_metrics.detect_ui_elements(current_frame)
                    if ui_elements is None:
                        ui_elements = []
                    
                    # Filter out UI elements that are in normal game UI regions
                    filtered_ui_elements = self._filter_normal_ui_elements(ui_elements, current_frame.shape)
                    
                    # Calculate what percentage of UI elements are in potential menu regions
                    ui_elements_ratio = len(filtered_ui_elements) / max(1, len(ui_elements))
                    logger.debug(f"UI elements: {len(ui_elements)} total, {len(filtered_ui_elements)} filtered, ratio: {ui_elements_ratio:.2f}")
                    
                    # Only count UI elements outside normal UI regions
                    ui_score = min(1.0, len(filtered_ui_elements) / 8.0)  # Normalize, assuming 8+ elements is definitely a menu
                    detection_scores["ui_elements"] = ui_score
                    
                    # Add a penalty if most UI elements are in normal UI regions
                    if ui_elements_ratio < 0.4 and len(ui_elements) > 5:
                        # If less than 40% of UI elements are outside normal UI regions,
                        # this is likely just normal game UI, not a menu
                        detection_scores["normal_ui_penalty"] = -0.5
                        logger.debug("Added normal UI penalty (-0.5) due to low UI element ratio")
                    else:
                        detection_scores["normal_ui_penalty"] = 0.0
                except Exception as e:
                    # Handle UI element detection errors
                    logger.error(f"Error in UI element detection: {e}")
                    detection_scores["ui_elements"] = 0.0
                    detection_scores["normal_ui_penalty"] = 0.0
                
                # F. Check for version string in bottom left (very reliable indicator)
                try:
                    if hasattr(self.visual_metrics, 'detect_version_string'):
                        # Handle tensor conversion properly
                        frame_np = current_frame
                        if hasattr(current_frame, 'detach'):
                            frame_np = current_frame.detach().cpu().numpy()
                            if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:  # CHW format
                                frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
                        
                        # Ensure data is uint8 for OpenCV
                        if frame_np.dtype != np.uint8:
                            if frame_np.max() <= 1.0:
                                frame_np = (frame_np * 255).astype(np.uint8)
                            else:
                                frame_np = frame_np.astype(np.uint8)
                        
                        version_string_detected = self.visual_metrics.detect_version_string(frame_np)
                        detection_scores["version_string"] = 1.0 if version_string_detected else 0.0
                        if version_string_detected:
                            logger.debug("Version string detected in bottom left - strong indicator of menu")
                    else:
                        detection_scores["version_string"] = 0.0
                except Exception as e:
                    logger.error(f"Error checking for version string: {e}")
                    detection_scores["version_string"] = 0.0
                
                # G. Check if mouse freedom test indicated a menu
                if hasattr(self.input_simulator, 'game_env') and hasattr(self.input_simulator.game_env, 'likely_in_menu_from_mouse_test'):
                    mouse_test_menu = self.input_simulator.game_env.likely_in_menu_from_mouse_test
                    mouse_test_change = getattr(self.input_simulator.game_env, 'mouse_freedom_visual_change', 1.0)
                    
                    # If mouse test suggests a menu, add significant confidence
                    if mouse_test_menu:
                        detection_scores["mouse_freedom"] = 0.8
                        logger.debug(f"Mouse freedom test indicates menu (visual change: {mouse_test_change:.6f})")
                    else:
                        detection_scores["mouse_freedom"] = 0.0
                else:
                    detection_scores["mouse_freedom"] = 0.0
                
                # Add the recent action vigilance bonus (if applicable)
                if increased_vigilance:
                    detection_scores["action_vigilance"] = 0.3
                    logger.debug("Increased menu detection vigilance due to recent menu-prone action")
                else:
                    detection_scores["action_vigilance"] = 0.0

                # H. Region-based analysis using specialized methods
                try:
                    # Get UI elements if none detected yet
                    if not 'ui_elements' in locals() or not ui_elements:
                        ui_elements = self.visual_metrics.detect_ui_elements(current_frame)
                        
                    # Use specialized region analysis
                    region_scores = self._analyze_menu_regions(current_frame, ui_elements)
                    
                    # If region_scores is valid, use it for menu type determination
                    if region_scores and isinstance(region_scores, dict):
                        for menu_type, score in region_scores.items():
                            # Add region scores to detection_scores with a specific key
                            detection_scores[f"region_{menu_type}"] = score
                            
                            # If this is the highest menu score so far, update menu_type
                            if score > 0.5 and (not menu_type or score > region_scores.get(menu_type, 0)):
                                menu_type = menu_type
                                detection_scores["region_analysis"] = score
                    
                    # Check if any region score is very high - strong indicator
                    max_region_score = max(region_scores.values()) if region_scores else 0
                    if max_region_score > 0.7:
                        detection_scores["strong_region_signal"] = max_region_score
                except Exception as e:
                    logger.error(f"Error in region-based analysis: {e}")
                    detection_scores["region_analysis"] = 0.0
                    # Don't fail completely if region analysis fails
                    
                # Aggregate scores with weights
                weights = {
                    "direct": 1.0,
                    "density": 0.8,
                    "contrast": 0.7,
                    "change": 0.5,
                    "ui_elements": 0.9,
                    "action_vigilance": 0.6,
                    "mouse_freedom": 1.0,  # High weight because this is a very reliable signal
                    "normal_ui_penalty": 0.5,
                    "version_string": 1.5,  # Version string is an extremely reliable indicator
                    "region_analysis": 0.8,  # General region analysis score
                    "region_main_menu": 0.9,  # More specific region scores
                    "region_pause_menu": 0.9,
                    "region_settings_menu": 0.8,
                    "region_dialog": 0.7,
                    "region_notification": 0.6,
                    "strong_region_signal": 1.2  # Very high confidence when a region has a strong signal
                }
                
                # Compute weighted score
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for key, weight in weights.items():
                    if key in detection_scores:
                        weighted_sum += detection_scores[key] * weight
                        weight_sum += weight
                
                weighted_score = weighted_sum / max(1e-6, weight_sum)  # Avoid division by zero
                
                # Check direct signal from visual analysis in specific regions
                menu_type = None
                menu_scores = {}
                
                # FIXED: Handle PyTorch tensor operations properly
                # Properly evaluate menu region scores
                try:
                    for mtype, config in self.MENU_TYPES.items():
                        thresh = config["threshold"]
                        regions = config["signature_regions"]
                        
                        # Analyze each signature region for this menu type
                        region_scores = []
                        
                        # Extract region features and scores
                        for region in regions:
                            x1, y1, x2, y2 = region
                            h, w = current_frame.shape[:2]
                            rx1, ry1 = int(x1 * w), int(y1 * h)
                            rx2, ry2 = int(x2 * w), int(y2 * h)
                            
                            # Safety check for valid coordinates
                            rx1 = max(0, min(rx1, w-1))
                            ry1 = max(0, min(ry1, h-1))
                            rx2 = max(rx1+1, min(rx2, w))
                            ry2 = max(ry1+1, min(ry2, h))
                            
                            if rx2 <= rx1 or ry2 <= ry1:
                                continue
                                
                            # Extract region and analyze
                            try:
                                region_data = current_frame[ry1:ry2, rx1:rx2]
                                if region_data.size == 0:
                                    continue
                                    
                                # Calculate characteristics like mean color, contrast, edge density
                                if isinstance(region_data, torch.Tensor):
                                    try:
                                        # Use proper torch syntax for mean calculation
                                        region_mean = torch.mean(region_data).item()
                                        if len(region_data.shape) == 3:
                                            # Get color channel means for RGB separation analysis
                                            # Fix: Using dim=(0,1) instead of axis parameter
                                            channel_means = torch.mean(region_data, dim=(0, 1)).tolist()
                                        else:
                                            channel_means = [region_mean, region_mean, region_mean]
                                        
                                        # Calculate contrast (standard deviation of pixel values)
                                        contrast = torch.std(region_data).item()
                                    except TypeError as e:
                                        logger.warning(f"TypeError in tensor mean calculation: {e}. Falling back to numpy.")
                                        # Convert tensor to numpy and use numpy operations instead
                                        region_np = region_data.detach().cpu().numpy()
                                        region_mean = np.mean(region_np)
                                        if len(region_np.shape) == 3:
                                            channel_means = np.mean(region_np, axis=(0, 1)).tolist()
                                        else:
                                            channel_means = [region_mean, region_mean, region_mean]
                                        contrast = np.std(region_np)
                                    except Exception as e:
                                        logger.error(f"Error in tensor mean calculation: {e}")
                                        # Provide default values if calculation fails
                                        region_mean = 0.5
                                        channel_means = [0.5, 0.5, 0.5]
                                        contrast = 0.1
                                else:
                                    try:
                                        # Fallback to numpy
                                        region_mean = np.mean(region_data)
                                        if len(region_data.shape) == 3:
                                            channel_means = np.mean(region_data, axis=(0, 1)).tolist()
                                        else:
                                            channel_means = [region_mean, region_mean, region_mean]
                                        
                                        # Calculate contrast (standard deviation of pixel values)
                                        contrast = np.std(region_data)
                                    except Exception as e:
                                        logger.error(f"Error in numpy mean calculation: {e}")
                                        # Provide default values if calculation fails
                                        region_mean = 0.5
                                        channel_means = [0.5, 0.5, 0.5]
                                        contrast = 0.1
                                
                                # Count UI elements in the region
                                elements_in_region = 0
                                for element in ui_elements:
                                    if len(element) >= 4:
                                        ex, ey, ew, eh = element
                                        element_center_x = ex + ew/2
                                        element_center_y = ey + eh/2
                                        
                                        if (rx1 <= element_center_x <= rx2 and 
                                            ry1 <= element_center_y <= ry2):
                                            elements_in_region += 1
                                
                                # Calculate UI density
                                region_area = (rx2 - rx1) * (ry2 - ry1)
                                ui_density = elements_in_region / max(1, region_area) * 1000
                                
                                # Score based on menu type
                                if mtype == "main_menu":
                                    # Main menu usually has characteristic colors
                                    color_score = abs(channel_means[2] - channel_means[0]) * 2.0  # Blue-red difference
                                    score = contrast * 2.0 + color_score + ui_density
                                elif mtype == "pause_menu":
                                    # Pause menus often have higher contrast
                                    score = contrast * 3.0 + ui_density
                                elif mtype == "settings_menu":
                                    # Settings menus have many UI elements
                                    score = ui_density * 2.0 + contrast
                                elif mtype == "dialog":
                                    # Dialogs typically have high contrast in center
                                    score = contrast * 4.0 + ui_density
                                else:  # notification
                                    # Notifications have small, focused regions
                                    score = (contrast * 2.0 + ui_density) * 1.5
                                    
                                region_scores.append(score)
                            except Exception as e:
                                logger.error(f"Error analyzing region for {mtype}: {e}")
                                
                        # Compute average score for this menu type
                        if region_scores:
                            avg_score = sum(region_scores) / len(region_scores)
                            menu_scores[mtype] = avg_score
                            
                            # Check if this is potentially the highest scoring menu
                            if avg_score > thresh and (menu_type is None or avg_score > menu_scores.get(menu_type, 0)):
                                menu_type = mtype
                except Exception as e:
                    logger.error(f"Error in menu region analysis: {e}")
                
                # Determine final menu detection result
                # Lower the threshold to be more sensitive to manually opened menus
                detection_threshold = 0.35  # Reduced from 0.45 to 0.35
                
                # Apply dynamic adjustment based on consecutive detections
                detection_threshold += self.detection_threshold_adjustment
                
                logger.debug(f"Menu detection weighted score: {weighted_score:.2f}, threshold: {detection_threshold:.2f}")
                
                # Update menu state based on final detection
                menu_detected = weighted_score > detection_threshold
                
                # Update menu stuck detection
                was_in_menu = self.in_menu
                previous_menu_type = self.menu_type
                
                # Update menu state based on final detection
                if menu_detected:
                    self.in_menu = True
                    self.menu_type = menu_type if menu_type is not None else "unknown"
                    self.menu_detection_confidence = weighted_score
                    
                    # Start tracking stuck time if we weren't in a menu before
                    if not was_in_menu:
                        self.menu_stuck_time = current_time
                        logger.info(f"Entered menu: {self.menu_type}")
                    
                    # Check if we've been stuck in menus for too long
                    self.check_menu_stuck_status()
                    
                    # Increase consecutive detections counter
                    self.consecutive_menu_detections += 1
                    
                    # Adjust detection threshold upward to require more evidence to exit menu state
                    self.detection_threshold_adjustment = min(0.1, self.consecutive_menu_detections * 0.02)
                else:
                    # Reset consecutive detections counter
                    self.consecutive_menu_detections = 0
                    
                    # Reset detection threshold adjustment
                    self.detection_threshold_adjustment = 0.0
                    
                    self.in_menu = False
                    self.menu_type = None
                    self.menu_detection_confidence = 0.0
                    
                    # Reset stuck time if we're no longer in a menu
                    self.menu_stuck_time = None
                    
                    # If we were in a menu before but aren't now, record the exit time
                    if was_in_menu:
                        self.last_menu_exit_time = current_time
                        logger.info(f"Exited menu: {previous_menu_type}")
                
                return menu_detected, self.menu_type, weighted_score
                
            except Exception as e:
                logger.error(f"Error in menu detection: {e}")
                # Return existing state on error
                return self.in_menu, self.menu_type, self.menu_detection_confidence
        
        # If no visual metrics, just return current state
        return self.in_menu, self.menu_type, self.menu_detection_confidence
    
    def recover_from_menu(self, max_attempts: int = 3) -> bool:
        """Attempt to exit any detected menu.
        
        Args:
            max_attempts: Maximum number of attempts to exit the menu
            
        Returns:
            bool: True if successfully recovered, False otherwise
        """
        if not self.in_menu:
            return True
        
        logger.info(f"Attempting to recover from menu: {self.menu_type}")
        self.menu_recovery_attempts += 1
        
        if self.menu_recovery_attempts > max_attempts:
            logger.warning(f"Exceeded maximum menu recovery attempts ({max_attempts})")
            # Force reset menu state to avoid getting permanently stuck
            logger.info("Forcing menu state reset after max recovery attempts")
            self.reset_state()
            self.menu_recovery_attempts = 0
            # Simply click in a game area and hope we exit the menu
            center_x, center_y = self.screen_capture.get_window_center()
            self.input_simulator.move_mouse(center_x, center_y)
            self.input_simulator.click_mouse()
            time.sleep(0.5)
            self.input_simulator.press_escape()  # Try ESC key as a last resort
            time.sleep(0.5)
            return True  # Return true even though we're forcing it
        
        # More aggressive exit strategy
        # First, try pressing ESC key which works for most menus
        logger.info("Trying ESC key first for menu exit")
        self.input_simulator.press_escape()
        time.sleep(0.7)  # Wait a bit longer to see if menu transitions
        
        # Check if this helped
        current_frame = self.screen_capture.capture_frame()
        if current_frame is not None:
            in_menu, menu_type, _ = self.detect_menu(current_frame)
            if not in_menu:
                logger.info("Successfully exited menu with ESC key")
                self.menu_recovery_attempts = 0
                # Record the successful exit time
                self.last_menu_exit_time = time.time()
                return True
        
        # If ESC didn't work, try the button sequence approach
        # Get the sequence of buttons to press for this menu type
        exit_path = self.menu_exit_paths.get(self.menu_type, ["close_button", "ok_button"])
        if self.menu_type == "unknown" or self.menu_type is None:
            # More comprehensive exit strategy for unknown menus
            exit_path = ["close_button", "ok_button", "back_button", "resume_button"]
        
        logger.info(f"Trying button clicks: {exit_path}")
        
        # Execute the exit sequence with longer pauses
        for button in exit_path:
            time.sleep(0.7)  # Longer delay between actions
            success = self._click_menu_button(button)
            if success:
                logger.info(f"Clicked {button} successfully")
            else:
                logger.warning(f"Failed to click {button}")
            
            # Check immediately if this button click helped
            current_frame = self.screen_capture.capture_frame()
            if current_frame is not None:
                in_menu, menu_type, _ = self.detect_menu(current_frame)
                if not in_menu:
                    logger.info(f"Successfully exited menu with {button} button")
                    self.menu_recovery_attempts = 0
                    # Record the successful exit time
                    self.last_menu_exit_time = time.time()
                    return True
        
        # If standard approaches didn't work, try clicking in different screen regions
        logger.info("Trying fallback region clicks for menu exit")
        screen_width, screen_height = self.screen_capture.get_window_size()
        
        # Define regions to try clicking (normalized coordinates)
        regions_to_try = [
            (0.5, 0.5),   # Center
            (0.9, 0.1),   # Top right (often close button)
            (0.1, 0.9),   # Bottom left 
            (0.5, 0.8)    # Bottom center
        ]
        
        for norm_x, norm_y in regions_to_try:
            x, y = int(norm_x * screen_width), int(norm_y * screen_height)
            self.input_simulator.move_mouse(x, y)
            time.sleep(0.3)
            self.input_simulator.click_mouse()
            time.sleep(0.5)
            
            # Check if the click helped
            current_frame = self.screen_capture.capture_frame()
            if current_frame is not None:
                in_menu, menu_type, _ = self.detect_menu(current_frame)
                if not in_menu:
                    logger.info(f"Successfully exited menu with click at ({norm_x}, {norm_y})")
                    self.menu_recovery_attempts = 0
                    self.last_menu_exit_time = time.time()
                    return True
        
        # Last resort: Press ESC again
        logger.info("Final attempt: pressing ESC key again")
        self.input_simulator.press_escape()
        time.sleep(0.7)
        
        # Final check
        current_frame = self.screen_capture.capture_frame()
        if current_frame is not None:
            in_menu, menu_type, _ = self.detect_menu(current_frame)
            if not in_menu:
                logger.info("Successfully exited menu with final ESC press")
                self.menu_recovery_attempts = 0
                self.last_menu_exit_time = time.time()
                return True
            else:
                logger.warning(f"Still in menu after all recovery attempts: {menu_type}")
                return False
        
        return False
    
    def _click_menu_button(self, button_name: str) -> bool:
        """Click a menu button by name.
        
        Args:
            button_name: Button identifier
            
        Returns:
            bool: True if the button was clicked, False if not found
        """
        if button_name not in self.menu_button_positions:
            logger.warning(f"Unknown menu button: {button_name}")
            return False
        
        # Get screen dimensions
        width, height = self.screen_capture.get_resolution()
        if width is None or height is None:
            logger.error("Failed to get screen resolution")
            return False
        
        # Convert normalized coordinates to actual screen position
        norm_x, norm_y = self.menu_button_positions[button_name]
        screen_x, screen_y = int(norm_x * width), int(norm_y * height)
        
        # Click the button position
        logger.debug(f"Clicking menu button '{button_name}' at ({screen_x}, {screen_y})")
        self.input_simulator.move_mouse(screen_x, screen_y)
        time.sleep(0.1)  # Brief pause
        self.input_simulator.click_mouse_left()
        
        return True
    
    def wait_for_menu_transition(self, timeout: float = 3.0) -> bool:
        """Wait for a menu transition to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if transition completed, False if timed out
        """
        start_time = time.time()
        last_frame = None
        stable_frames = 0
        
        while time.time() - start_time < timeout:
            current_frame = self.screen_capture.capture_frame()
            if current_frame is None:
                time.sleep(0.1)
                continue
                
            if last_frame is not None:
                # Calculate frame difference
                frame_diff = np.mean(np.abs(current_frame.astype(np.float32) - last_frame.astype(np.float32)))
                
                # If frames are similar, count as stable
                if frame_diff < 5.0:  # Arbitrary threshold
                    stable_frames += 1
                else:
                    stable_frames = 0
                
                # Consider transition complete after several stable frames
                if stable_frames >= 3:
                    return True
            
            last_frame = current_frame.copy()
            time.sleep(0.1)
        
        logger.warning("Menu transition timed out")
        return False
    
    def is_in_menu(self) -> bool:
        """Check if the game is currently in a menu.
        
        Returns:
            bool: True if in a menu, False otherwise
        """
        return self.in_menu
    
    def get_menu_type(self) -> Optional[str]:
        """Get the current menu type if in a menu.
        
        Returns:
            Optional[str]: Menu type or None if not in a menu
        """
        return self.menu_type if self.in_menu else None
    
    def reset_state(self) -> None:
        """Reset the menu handler state."""
        self.in_menu = False
        self.menu_type = None
        self.menu_detection_confidence = 0.0
        self.menu_recovery_attempts = 0
        self.last_menu_check_time = 0 

    def _find_button_by_image(self, button_name: str) -> Optional[Tuple[int, int]]:
        """Find a button by matching its image template.
        
        Args:
            button_name: Button identifier
            
        Returns:
            Optional[Tuple[int, int]]: Screen coordinates of the button if found, None otherwise
        """
        # Get button template
        template = self._get_button_template(button_name)
        if template is None:
            return None
            
        # Capture current screen
        current_frame = self.screen_capture.capture_frame()
        if current_frame is None:
            return None
            
        # Convert to numpy array if needed
        if isinstance(current_frame, torch.Tensor):
            current_frame = current_frame.cpu().numpy()
            if current_frame.shape[0] == 3:  # CHW to HWC
                current_frame = np.transpose(current_frame, (1, 2, 0))
            if current_frame.max() <= 1.0:
                current_frame = (current_frame * 255).astype(np.uint8)
                
        # Match template
        if self.image_utils is None:
            from src.utils.image_utils import ImageUtils
            self.image_utils = ImageUtils()
            
        match_score, match_rect = self.image_utils.template_match(current_frame, template)
        
        if match_rect:
            # Return center of matched region
            x, y, w, h = match_rect
            return (x + w // 2, y + h // 2)
            
        return None

    def _check_menu_prone_actions(self) -> bool:
        """Check if we recently performed a menu-prone action.
        
        Returns:
            bool: True if increased vigilance is needed
        """
        current_time = time.time()
        increased_vigilance = False
        
        for action_type, action_data in self.menu_prone_actions.items():
            if action_data["active"] and current_time - action_data["time"] < self.action_vigilance_timeout:
                increased_vigilance = True
            else:
                self.menu_prone_actions[action_type]["active"] = False
                
        return increased_vigilance
        
    def _analyze_menu_regions(self, frame, ui_elements: List) -> Dict[str, float]:
        """Analyze specific regions of a frame for menu elements.
        
        Args:
            frame: Current frame
            ui_elements: Detected UI elements
            
        Returns:
            Dict: Menu type -> confidence score
        """
        # Emergency fallback for None or empty frames
        if frame is None:
            logger.warning("Received None frame in _analyze_menu_regions")
            return {menu_type: 0.0 for menu_type in self.MENU_TYPES.keys()}
            
        # If it looks like a tensor but might not be valid for our operations, handle carefully
        if hasattr(frame, 'numel') and hasattr(frame, 'shape'):
            try:
                # Quick test to see if tensor operations work properly
                if frame.dim() > 0:
                    test = torch.mean(frame) if frame.numel() > 0 else 0
                # If we get here, tensor is usable
            except (RuntimeError, TypeError) as e:
                # This is likely not a usable tensor for our operations
                logger.warning(f"Tensor error in _analyze_menu_regions: {e}. Falling back to numpy processing.")
                # Convert to numpy if possible, otherwise use fallback
                try:
                    if hasattr(frame, 'detach') and hasattr(frame, 'cpu') and hasattr(frame, 'numpy'):
                        frame_np = frame.detach().cpu().numpy()
                        return self._analyze_menu_regions_numpy(frame_np, ui_elements)
                except Exception:
                    return self._analyze_menu_regions_numpy(frame, ui_elements)
                return self._analyze_menu_regions_numpy(frame, ui_elements)
        
        # Convert frame to torch tensor if needed
        if not isinstance(frame, torch.Tensor) and hasattr(torch, 'from_numpy'):
            # Convert numpy array to PyTorch tensor for easier region processing
            try:
                # Handle different input types
                if isinstance(frame, np.ndarray):
                    if frame.dtype == np.uint8:
                        # Normalize to 0-1 range for consistent processing
                        frame_tensor = torch.from_numpy(frame).float() / 255.0
                    else:
                        frame_tensor = torch.from_numpy(frame).float()
                        if frame_tensor.max() > 1.0:
                            frame_tensor = frame_tensor / 255.0
                else:
                    # Can't convert to tensor, use numpy processing instead
                    return self._analyze_menu_regions_numpy(frame, ui_elements)
                    
                # Ensure frame tensor is in proper format (H, W, C)
                if len(frame_tensor.shape) == 3 and frame_tensor.shape[0] == 3:
                    # Convert from (C, H, W) to (H, W, C)
                    frame_tensor = frame_tensor.permute(1, 2, 0)
            except Exception as e:
                logger.error(f"Error converting frame to tensor: {e}")
                # Fallback to numpy implementation
                return self._analyze_menu_regions_numpy(frame, ui_elements)
        else:
            frame_tensor = frame
        
        scores = {}
        
        # Safety check for tensor
        if frame_tensor is None:
            return {}
        
        # Make sure we have at least 2 dimensions
        if not hasattr(frame_tensor, 'shape') or len(frame_tensor.shape) < 2:
            return {}
        
        height, width = frame_tensor.shape[:2] if len(frame_tensor.shape) >= 2 else (0, 0)
        
        if height == 0 or width == 0:
            return {}
        
        # Analyze each menu type's signature regions
        for menu_type, info in self.MENU_TYPES.items():
            try:
                signature_regions = info["signature_regions"]
                region_scores = []
                
                for region_coords in signature_regions:
                    # Convert normalized coordinates to actual pixel values
                    x1, y1, x2, y2 = region_coords
                    x1_px, y1_px = int(x1 * width), int(y1 * height)
                    x2_px, y2_px = int(x2 * width), int(y2 * height)
                    
                    # Ensure coordinates are within bounds
                    x1_px, y1_px = max(0, x1_px), max(0, y1_px)
                    x2_px, y2_px = min(width, x2_px), min(height, y2_px)
                    
                    if x2_px <= x1_px or y2_px <= y1_px:
                        continue
                    
                    # Extract region
                    region = frame_tensor[y1_px:y2_px, x1_px:x2_px]
                    
                    # Skip if region is empty
                    if region.numel() == 0:
                        continue
                    
                    # Calculate characteristics like mean color, contrast, edge density
                    if isinstance(region, torch.Tensor):
                        try:
                            # Use proper torch syntax for mean calculation
                            region_mean = torch.mean(region).item()
                            if len(region.shape) == 3:
                                # Get color channel means for RGB separation analysis
                                # Fix: Using dim=(0,1) instead of axis parameter
                                channel_means = torch.mean(region, dim=(0, 1)).tolist()
                            else:
                                channel_means = [region_mean, region_mean, region_mean]
                            
                            # Calculate contrast (standard deviation of pixel values)
                            contrast = torch.std(region).item()
                        except TypeError as e:
                            logger.warning(f"TypeError in tensor mean calculation: {e}. Falling back to numpy.")
                            # Convert tensor to numpy and use numpy operations instead
                            region_np = region.detach().cpu().numpy()
                            region_mean = np.mean(region_np)
                            if len(region_np.shape) == 3:
                                channel_means = np.mean(region_np, axis=(0, 1)).tolist()
                            else:
                                channel_means = [region_mean, region_mean, region_mean]
                            contrast = np.std(region_np)
                        except Exception as e:
                            logger.error(f"Error in tensor mean calculation: {e}")
                            # Provide default values if calculation fails
                            region_mean = 0.5
                            channel_means = [0.5, 0.5, 0.5]
                            contrast = 0.1
                    else:
                        try:
                            # Fallback to numpy
                            region_mean = np.mean(region)
                            if len(region.shape) == 3:
                                channel_means = np.mean(region, axis=(0, 1)).tolist()
                            else:
                                channel_means = [region_mean, region_mean, region_mean]
                            
                            # Calculate contrast (standard deviation of pixel values)
                            contrast = np.std(region)
                        except Exception as e:
                            logger.error(f"Error in numpy mean calculation: {e}")
                            # Provide default values if calculation fails
                            region_mean = 0.5
                            channel_means = [0.5, 0.5, 0.5]
                            contrast = 0.1
                    
                    # Count UI elements in the region
                    elements_in_region = 0
                    for element in ui_elements:
                        if len(element) >= 4:
                            ex, ey, ew, eh = element
                            element_center_x = ex + ew/2
                            element_center_y = ey + eh/2
                            
                            if (x1_px <= element_center_x <= x2_px and 
                                y1_px <= element_center_y <= y2_px):
                                elements_in_region += 1
                    
                    # Calculate UI density
                    region_area = (x2_px - x1_px) * (y2_px - y1_px)
                    ui_density = elements_in_region / max(1, region_area) * 1000
                    
                    # Score based on menu type
                    if menu_type == "main_menu":
                        # Main menu usually has characteristic colors
                        color_score = abs(channel_means[2] - channel_means[0]) * 2.0  # Blue-red difference
                        score = contrast * 2.0 + color_score + ui_density
                    elif menu_type == "pause_menu":
                        # Pause menus often have higher contrast
                        score = contrast * 3.0 + ui_density
                    elif menu_type == "settings_menu":
                        # Settings menus have many UI elements
                        score = ui_density * 2.0 + contrast
                    elif menu_type == "dialog":
                        # Dialogs typically have high contrast in center
                        score = contrast * 4.0 + ui_density
                    else:  # notification
                        # Notifications have small, focused regions
                        score = (contrast * 2.0 + ui_density) * 1.5
                        
                    region_scores.append(score)
                
                # Use maximum score from regions
                if region_scores:
                    scores[menu_type] = max(min(1.0, s/10.0) for s in region_scores)  # Normalize to 0-1
                else:
                    scores[menu_type] = 0.0
                
            except Exception as e:
                logger.error(f"Error analyzing region for {menu_type}: {e}")
                scores[menu_type] = 0.0
        
        return scores
        
    def force_menu_detection_result(self, result: bool, menu_type: Optional[str] = None, confidence: float = 0.5):
        """Force a specific menu detection result (for testing).
        
        Args:
            result: Whether to detect a menu
            menu_type: Type of menu to detect
            confidence: Confidence score
        """
        self._force_menu_detection_result = (result, menu_type, confidence)
        logger.info(f"Forced menu detection result set to: {result}, {menu_type}, {confidence}")
        
    def reset_forced_menu_detection(self):
        """Reset any forced menu detection result."""
        if hasattr(self, '_force_menu_detection_result'):
            del self._force_menu_detection_result
            logger.info("Forced menu detection result cleared")

    def _filter_normal_ui_elements(self, ui_elements, frame_shape):
        """Filter out UI elements that are in normal game UI regions, not menus.
        
        Args:
            ui_elements: List of UI element coordinates
            frame_shape: Shape of the frame (height, width)
            
        Returns:
            List: Filtered UI elements not in normal game UI regions
        """
        # Fix: Handle None case explicitly
        if ui_elements is None or not ui_elements:
            return []
            
        h, w = frame_shape[:2]
        filtered_elements = []
        
        for elem in ui_elements:
            # Fix: Check if element has required dimensions
            if not (isinstance(elem, tuple) and len(elem) >= 2):
                continue
                
            x, y = elem[0], elem[1]
            in_normal_ui = False
            
            # Check if this element is in a normal UI region
            for region in self.GAME_UI_REGIONS:
                x1, y1, x2, y2 = region
                region_x1, region_y1 = int(x1 * w), int(y1 * h)
                region_x2, region_y2 = int(x2 * w), int(y2 * h)
                
                if (x >= region_x1 and x <= region_x2 and
                    y >= region_y1 and y <= region_y2):
                    in_normal_ui = True
                    break
            
            # Keep elements that are NOT in normal UI regions
            if not in_normal_ui:
                filtered_elements.append(elem)
                
        return filtered_elements

    def _check_fullscreen_menu(self, frame: np.ndarray) -> bool:
        """Check if the current frame indicates a fullscreen menu.
        
        Args:
            frame: Current frame to analyze
            
        Returns:
            bool: True if the frame indicates a fullscreen menu, False otherwise
        """
        if frame is None:
            return False
            
        h, w = frame.shape[:2]
        return (
            np.sum(frame) > 0.9 * h * w  # Check if most of the frame is occupied
        )
        
    def _get_button_template(self, button_name: str) -> Optional[np.ndarray]:
        """Get a button template image by name.
        
        Args:
            button_name: Button identifier
            
        Returns:
            Optional[np.ndarray]: Button template image if found, None otherwise
        """
        # In a real implementation, this would load actual template images
        # For now, we'll return None and rely on visual change detection
        return None

    def check_menu_stuck_status(self) -> None:
        """Check if the agent has been stuck in menus for too long and take recovery action."""
        current_time = time.time()
        
        # If we've been in a menu for too long and we haven't recently attempted a reset
        if (self.menu_stuck_time is not None and 
            current_time - self.menu_stuck_time > self.max_menu_stuck_time and
            current_time - self.last_menu_reset_time > self.menu_reset_cooldown):
            
            stuck_duration = current_time - self.menu_stuck_time
            logger.warning(f"Agent appears to be stuck in menus for {stuck_duration:.1f} seconds. Initiating forced reset.")
            
            # Update reset time
            self.last_menu_reset_time = current_time
            
            # Perform a more drastic reset
            self.force_game_reset()
    
    def force_game_reset(self) -> None:
        """Perform a forced game reset when stuck in menus for too long."""
        logger.warning("Performing forced game reset")
        
        # Reset internal menu state
        self.reset_state()
        
        # 1. First try multiple ESC presses to get out of nested menus
        for _ in range(3):
            self.input_simulator.press_escape()
            time.sleep(0.7)
        
        # 2. Try clicking at known menu button positions
        screen_width, screen_height = self.screen_capture.get_window_size()
        
        # Try each menu button position
        for button_name, (norm_x, norm_y) in self.menu_button_positions.items():
            x, y = int(norm_x * screen_width), int(norm_y * screen_height)
            logger.info(f"Trying to click {button_name} at ({x}, {y})")
            self.input_simulator.move_mouse(x, y)
            time.sleep(0.3)
            self.input_simulator.click_mouse()
            time.sleep(0.5)
        
        # 3. Try pressing common keys that might help
        logger.info("Pressing common navigation keys")
        self.input_simulator.press_key("enter")
        time.sleep(0.5)
        self.input_simulator.press_key("space")
        time.sleep(0.5)
        
        # 4. As a last resort, try alt+F4 and then restart the game
        # This is a drastic measure and should only be used if the game is completely stuck
        # Not implementing this yet as it requires game restart logic
        
        # Reset the menu stuck time
        self.menu_stuck_time = None
        logger.info("Forced reset completed")

    def _analyze_menu_regions_numpy(self, frame: np.ndarray, ui_elements: List) -> Dict[str, float]:
        """Numpy version of region analysis for when tensor conversion fails.
        
        Args:
            frame: Current frame as numpy array
            ui_elements: Detected UI elements
            
        Returns:
            Dict: Menu type -> confidence score
        """
        scores = {}
        
        # Safety check
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            logger.warning("Invalid frame in _analyze_menu_regions_numpy")
            # Return empty scores dict instead of failing
            return {menu_type: 0.0 for menu_type in self.MENU_TYPES.keys()}
            
        # Ensure proper data type
        try:
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
        except Exception as e:
            logger.error(f"Error converting frame dtype: {e}")
            return {menu_type: 0.0 for menu_type in self.MENU_TYPES.keys()}
        
        # Get dimensions
        try:
            if len(frame.shape) == 3:
                height, width = frame.shape[:2]
            else:
                height, width = frame.shape
                frame = np.stack([frame, frame, frame], axis=2)  # Convert to 3 channels
        except Exception as e:
            logger.error(f"Error determining frame dimensions: {e}")
            return {menu_type: 0.0 for menu_type in self.MENU_TYPES.keys()}
        
        # Safety check for ui_elements
        if ui_elements is None:
            ui_elements = []
        
        # Process each menu type
        for menu_type, info in self.MENU_TYPES.items():
            try:
                signature_regions = info["signature_regions"]
                region_scores = []
                
                for region_coords in signature_regions:
                    try:
                        # Convert normalized coordinates to actual pixel values
                        x1, y1, x2, y2 = region_coords
                        x1_px, y1_px = int(x1 * width), int(y1 * height)
                        x2_px, y2_px = int(x2 * width), int(y2 * height)
                        
                        # Ensure coordinates are within bounds
                        x1_px, y1_px = max(0, x1_px), max(0, y1_px)
                        x2_px, y2_px = min(width, x2_px), min(height, y2_px)
                        
                        if x2_px <= x1_px or y2_px <= y1_px:
                            continue
                        
                        # Extract region
                        region = frame[y1_px:y2_px, x1_px:x2_px]
                        
                        if region.size == 0:
                            continue
                        
                        # Calculate region stats
                        try:
                            region_mean = np.mean(region)
                            if len(region.shape) == 3 and region.shape[2] >= 3:
                                # Get mean for each channel (up to 3 channels)
                                channel_means = [
                                    np.mean(region[:, :, 0]),
                                    np.mean(region[:, :, 1]),
                                    np.mean(region[:, :, 2 if region.shape[2] > 2 else 1])
                                ]
                            else:
                                channel_means = [region_mean, region_mean, region_mean]
                            
                            contrast = np.std(region)
                        except Exception as e:
                            logger.error(f"Error calculating region stats for {menu_type}: {e}")
                            # Provide reasonable defaults
                            region_mean = 0.5
                            channel_means = [0.5, 0.5, 0.5]
                            contrast = 0.1
                        
                        # Count UI elements in the region
                        elements_in_region = 0
                        for element in ui_elements:
                            if len(element) >= 4:  # Make sure it has x, y, w, h components
                                ex, ey, ew, eh = element
                                element_center_x = ex + ew/2
                                element_center_y = ey + eh/2
                                
                                if (x1_px <= element_center_x <= x2_px and 
                                    y1_px <= element_center_y <= y2_px):
                                    elements_in_region += 1
                        
                        # Calculate UI density
                        region_area = (x2_px - x1_px) * (y2_px - y1_px)
                        ui_density = elements_in_region / max(1, region_area) * 1000
                        
                        # Score based on menu type
                        if menu_type == "main_menu":
                            # Main menu usually has characteristic colors
                            color_score = abs(channel_means[2] - channel_means[0]) * 2.0  # Blue-red difference
                            score = contrast * 2.0 + color_score + ui_density
                        elif menu_type == "pause_menu":
                            # Pause menus often have higher contrast
                            score = contrast * 3.0 + ui_density
                        elif menu_type == "settings_menu":
                            # Settings menus have many UI elements
                            score = ui_density * 2.0 + contrast
                        elif menu_type == "dialog":
                            # Dialogs typically have high contrast in center
                            score = contrast * 4.0 + ui_density
                        else:  # notification
                            # Notifications have small, focused regions
                            score = (contrast * 2.0 + ui_density) * 1.5
                            
                        region_scores.append(score)
                    except Exception as e:
                        logger.error(f"Error processing region for {menu_type}: {e}")
                        continue
                
                # Use maximum score from regions
                if region_scores:
                    scores[menu_type] = max(min(1.0, s/10.0) for s in region_scores)  # Normalize to 0-1
                else:
                    scores[menu_type] = 0.0
                
            except Exception as e:
                logger.error(f"Error analyzing region for {menu_type}: {e}")
                scores[menu_type] = 0.0
        
        return scores 