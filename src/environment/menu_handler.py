import logging
import time
import numpy as np
from typing import Tuple, List, Dict, Optional, Union

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
            current_frame: Current game frame
            
        Returns:
            Tuple[bool, Optional[str], float]: 
                - Whether a menu is detected
                - Menu type (if detected)
                - Confidence score
        """
        # Skip frequent checks to reduce performance impact
        current_time = time.time()
        if current_time - self.last_menu_check_time < self.menu_check_interval:
            return self.in_menu, self.menu_type, self.menu_detection_confidence
        
        self.last_menu_check_time = current_time
        
        # Check if we recently performed a menu-prone action
        increased_vigilance = False
        for action_type, action_data in self.menu_prone_actions.items():
            if action_data["active"] and current_time - action_data["time"] < self.action_vigilance_timeout:
                increased_vigilance = True
            else:
                self.menu_prone_actions[action_type]["active"] = False
                
        # Use visual metrics if available
        if self.visual_metrics:
            try:
                # 1. Combined approach using multiple detection methods
                detection_scores = {}
                
                # A. Direct menu detection (template matching)
                direct_menu_detected = self.visual_metrics.detect_main_menu(current_frame)
                detection_scores["direct"] = 0.8 if direct_menu_detected else 0.0
                
                # B. UI element density analysis
                ui_density = self.visual_metrics.calculate_ui_density(current_frame)
                detection_scores["density"] = ui_density
                
                # C. Contrast pattern analysis
                contrast_score = self.visual_metrics.detect_menu_contrast_pattern(current_frame)
                detection_scores["contrast"] = contrast_score
                
                # D. Visual change score (game paused/menu appeared)
                visual_change_score = self.visual_metrics.get_visual_change_score(current_frame)
                detection_scores["change"] = max(0, 1.0 - visual_change_score * 5.0)  # Invert: lower change = higher score
                
                # E. UI element analysis
                ui_elements = self.visual_metrics.detect_ui_elements(current_frame)
                ui_score = min(1.0, len(ui_elements) / 10.0)  # Normalize, assuming 10+ elements is definitely a menu
                detection_scores["ui_elements"] = ui_score
                
                # Add the recent action vigilance bonus (if applicable)
                if increased_vigilance:
                    detection_scores["action_vigilance"] = 0.3
                    logger.debug("Increased menu detection vigilance due to recent menu-prone action")
                else:
                    detection_scores["action_vigilance"] = 0.0
                
                # Calculate weighted score
                weighted_scores = {
                    "direct": 0.3,            # Strong indicator if template match
                    "density": 0.15,          # Good indicator of UI concentration
                    "contrast": 0.15,         # Good indicator of menu-like appearance
                    "change": 0.1,            # Weak indicator (game might just be idle)
                    "ui_elements": 0.15,      # Medium indicator of UI presence
                    "action_vigilance": 0.15  # Context from recent actions
                }
                
                total_score = sum(detection_scores[k] * weighted_scores[k] for k in detection_scores)
                
                # Determine confidence thresholds
                base_threshold = 0.4  # Default detection threshold
                high_confidence_threshold = 0.6  # For confident detection
                
                # Adjust thresholds based on recent actions
                if increased_vigilance:
                    base_threshold *= 0.8  # Lower threshold (more sensitive) after ESC press or gear click
                    
                # Store detailed results for logging
                detection_details = {
                    "scores": detection_scores,
                    "total": total_score,
                    "threshold": base_threshold,
                    "high_threshold": high_confidence_threshold,
                    "vigilance_active": increased_vigilance
                }
                
                # Make detection decision
                if total_score > high_confidence_threshold:
                    # High confidence menu detection
                    result = True
                    confidence = total_score
                    
                    # Try to identify menu type through signature regions
                    menu_type = "unknown"
                    best_type_score = 0
                    
                    for menu_type_name, properties in self.MENU_TYPES.items():
                        signature_regions = properties["signature_regions"]
                        region_match_count = 0
                        
                        # Check signature regions for UI element density
                        for region in signature_regions:
                            x1, y1, x2, y2 = region
                            h, w = current_frame.shape[:2]
                            region_x1, region_y1 = int(x1 * w), int(y1 * h)
                            region_x2, region_y2 = int(x2 * w), int(y2 * h)
                            
                            # Count UI elements in this region
                            elements_in_region = [
                                elem for elem in ui_elements
                                if (elem[0] >= region_x1 and elem[0] <= region_x2 and
                                    elem[1] >= region_y1 and elem[1] <= region_y2)
                            ]
                            
                            # Be more lenient - accept even a single UI element in region
                            if len(elements_in_region) >= 1:
                                region_match_count += 1
                        
                        # Calculate type match score
                        type_score = region_match_count / max(1, len(signature_regions))
                        if type_score > best_type_score:
                            best_type_score = type_score
                            menu_type = menu_type_name
                            
                    logger.debug(f"Menu detected with high confidence: {menu_type} ({total_score:.2f})")
                    
                    self.in_menu = True
                    self.menu_type = menu_type
                    self.menu_detection_confidence = confidence
                    return True, menu_type, confidence
                    
                elif total_score > base_threshold:
                    # Lower confidence menu detection
                    logger.debug(f"Menu detected with moderate confidence: unknown ({total_score:.2f})")
                    self.in_menu = True
                    self.menu_type = "unknown"
                    self.menu_detection_confidence = total_score
                    return True, "unknown", total_score
                
                # No menu detected
                logger.debug(f"No menu detected (score: {total_score:.2f}, threshold: {base_threshold:.2f})")
                self.in_menu = False
                self.menu_type = None
                self.menu_detection_confidence = 0.0
                return False, None, 0.0
                    
            except Exception as e:
                logger.warning(f"Error in enhanced menu detection: {e}")
                # Fall through to legacy detection methods
        
        # Legacy template matching fallback (if visual metrics not available)
        for menu_type, template in self.menu_templates.items():
            if template is not None:
                # Simple template matching (would be implemented with actual images)
                # match_score = self.image_utils.template_match(current_frame, template)
                match_score = 0.0  # Placeholder
                
                if match_score > self.MENU_TYPES[menu_type]["threshold"]:
                    self.in_menu = True
                    self.menu_type = menu_type
                    self.menu_detection_confidence = match_score
                    return True, menu_type, match_score
        
        # No menu detected
        self.in_menu = False
        self.menu_type = None
        self.menu_detection_confidence = 0.0
        return False, None, 0.0
    
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
            self.menu_recovery_attempts = 0
            return False
        
        # Get the sequence of buttons to press for this menu type
        exit_path = self.menu_exit_paths.get(self.menu_type, ["close_button"])
        if self.menu_type == "unknown":
            # Try common exit strategies for unknown menus
            exit_path = ["close_button", "ok_button", "back_button"]
        
        # Execute the exit sequence
        for button in exit_path:
            time.sleep(0.5)  # Brief delay between actions
            self._click_menu_button(button)
        
        # Special case for ESC key to close menus
        if self.menu_type in ["pause_menu", "settings_menu"]:
            self.input_simulator.press_escape()
            time.sleep(0.5)
        
        # Check if we're still in a menu
        current_frame = self.screen_capture.capture_frame()
        if current_frame is not None:
            self.detect_menu(current_frame)
            
            if not self.in_menu:
                logger.info("Successfully exited menu")
                self.menu_recovery_attempts = 0
                return True
            else:
                logger.warning(f"Still in menu after recovery attempt: {self.menu_type}")
                # Try a different strategy on the next attempt
                return self.recover_from_menu(max_attempts)
        
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

    def force_menu_detection_result(self, is_menu: bool, menu_type: Optional[str] = None) -> None:
        """Force a specific menu detection result for testing purposes.
        
        Args:
            is_menu: Whether to treat the current state as a menu
            menu_type: The type of menu to set (if is_menu is True)
        """
        self.in_menu = is_menu
        self.menu_type = menu_type if is_menu else None
        self.menu_detection_confidence = 0.9 if is_menu else 0.0
        logger.info(f"Forced menu detection state: in_menu={is_menu}, type={menu_type}") 