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
            current_frame: Current frame to analyze
            
        Returns:
            Tuple[bool, str, float]: Menu detected, menu type, confidence
        """
        # Skip detection if too soon since last check
        current_time = time.time()
        if current_time - self.last_menu_check_time < self.menu_check_interval:
            return self.in_menu, self.menu_type, self.menu_detection_confidence
            
        # Update last check time
        self.last_menu_check_time = current_time
        
        # Force a specific result for testing purposes
        if hasattr(self, '_force_menu_detection_result') and self._force_menu_detection_result is not None:
            forced_result, forced_type, forced_confidence = self._force_menu_detection_result
            logger.debug(f"Forcing menu detection result: {forced_result}, {forced_type}, {forced_confidence}")
            return forced_result, forced_type, forced_confidence
        
        # Check if input suggests we're in a menu-prone state
        increased_vigilance = self._check_menu_prone_actions()
        
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
                
                # F. Check if mouse freedom test indicated a menu
                # Access game_env's mouse freedom test results if available
                if hasattr(self.input_simulator, 'game_env') and hasattr(self.input_simulator.game_env, 'likely_in_menu_from_mouse_test'):
                    mouse_test_menu = self.input_simulator.game_env.likely_in_menu_from_mouse_test
                    mouse_test_change = getattr(self.input_simulator.game_env, 'mouse_freedom_visual_change', 1.0)
                    
                    # If mouse test suggests a menu, add significant confidence
                    if mouse_test_menu:
                        detection_scores["mouse_freedom"] = 0.8
                        logger.debug(f"Mouse freedom test indicates menu (visual change: {mouse_test_change:.6f})")
                    else:
                        detection_scores["mouse_freedom"] = 0.0
                
                # Add the recent action vigilance bonus (if applicable)
                if increased_vigilance:
                    detection_scores["action_vigilance"] = 0.3
                    logger.debug("Increased menu detection vigilance due to recent menu-prone action")
                else:
                    detection_scores["action_vigilance"] = 0.0

                # Aggregate scores with weights
                weights = {
                    "direct": 1.0,
                    "density": 0.8,
                    "contrast": 0.7,
                    "change": 0.5,
                    "ui_elements": 0.9,
                    "action_vigilance": 0.6,
                    "mouse_freedom": 1.0  # High weight because this is a very reliable signal
                }
                
                # Calculate weighted average
                total_weight = sum(weights.get(k, 0.0) for k in detection_scores.keys())
                if total_weight > 0:
                    weighted_score = sum(detection_scores[k] * weights.get(k, 0.0) for k in detection_scores.keys()) / total_weight
                else:
                    weighted_score = 0.0
                
                # Decision threshold
                menu_detected = weighted_score >= 0.35  # Lower threshold for more sensitivity
                
                # Determine menu type based on UI signature regions
                menu_type = "unknown"
                if menu_detected:
                    # Get signature regions for different menu types
                    menu_signatures = self._analyze_menu_regions(current_frame, ui_elements)
                    if menu_signatures:
                        # Use the type with highest confidence
                        menu_type, _ = max(menu_signatures.items(), key=lambda x: x[1])
                
                # Log detailed scores for debugging
                logger.debug(f"Menu detection scores: {detection_scores}, weighted: {weighted_score:.2f}")
                
                # Update menu state
                self.in_menu = menu_detected
                self.menu_type = menu_type if menu_detected else None
                self.menu_detection_confidence = weighted_score
                
                return menu_detected, menu_type if menu_detected else None, weighted_score
                
            except Exception as e:
                logger.error(f"Error in menu detection: {e}")
                # Fall through to fallback methods
        
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
        
    def _analyze_menu_regions(self, frame: np.ndarray, ui_elements: List) -> Dict[str, float]:
        """Analyze regions of the frame to determine menu type.
        
        Args:
            frame: Current frame
            ui_elements: Detected UI elements
            
        Returns:
            Dict[str, float]: Menu type to confidence mapping
        """
        if frame is None or not ui_elements:
            return {}
            
        h, w = frame.shape[:2]
        menu_scores = {}
        
        # Check each menu type's signature regions
        for menu_type_name, properties in self.MENU_TYPES.items():
            signature_regions = properties.get("signature_regions", [])
            if not signature_regions:
                continue
                
            region_match_count = 0
            
            # Check signature regions for UI element density
            for region in signature_regions:
                x1, y1, x2, y2 = region
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
            if signature_regions:
                type_score = region_match_count / len(signature_regions)
                menu_scores[menu_type_name] = type_score
        
        return menu_scores
        
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