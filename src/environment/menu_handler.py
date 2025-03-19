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
        
        logger.info("Menu handler initialized")
    
    def _load_menu_templates(self) -> Dict:
        """Load menu template images for detection.
        
        Returns:
            Dict: Dictionary of menu templates
        """
        # In a real implementation, this would load actual template images
        # For now, we'll return an empty dict and rely on visual change detection
        return {}
    
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
        
        # Use visual metrics if available
        if self.visual_metrics:
            try:
                # First try direct menu detection from visual metrics
                is_menu = self.visual_metrics.detect_main_menu(current_frame)
                if is_menu:
                    self.in_menu = True
                    self.menu_type = "unknown"
                    self.menu_detection_confidence = 0.8
                    logger.debug("Menu detected by direct visual metrics")
                    return True, "unknown", 0.8
                    
                # Then try UI element detection approach
                visual_change_score = self.visual_metrics.get_visual_change_score(current_frame)
                ui_elements = self.visual_metrics.detect_ui_elements(current_frame)
                
                # More aggressive detection: lower threshold for UI elements and visual change
                if len(ui_elements) > 3 and visual_change_score < 0.3:
                    # Try to identify the specific menu type
                    for menu_type, properties in self.MENU_TYPES.items():
                        threshold = properties["threshold"] * 0.8  # Lower the threshold by 20%
                        signature_regions = properties["signature_regions"]
                        
                        # Check signature regions for UI element density
                        region_match_count = 0
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
                        
                        # If any signature regions match, consider this as a potential menu type
                        if region_match_count >= max(1, len(signature_regions) * 0.3):  # Match at least 30% of regions or 1
                            confidence = min(1.0, 0.5 + len(ui_elements) / 20.0)  # Base confidence on UI element count
                            if confidence > threshold:
                                self.in_menu = True
                                self.menu_type = menu_type
                                self.menu_detection_confidence = confidence
                                logger.debug(f"Menu detected: {menu_type} with confidence {confidence:.2f}")
                                return True, menu_type, confidence
                
                # Fallback: generic menu detection based on visual change and UI count
                if visual_change_score < 0.15 and len(ui_elements) > 2:  # More aggressive thresholds
                    self.in_menu = True
                    self.menu_type = "unknown"
                    self.menu_detection_confidence = 0.6
                    logger.debug("Menu detected by fallback method")
                    return True, "unknown", 0.6
            except Exception as e:
                logger.warning(f"Error in menu detection: {e}")
        
        # Template matching fallback (if visual metrics not available)
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