"""
Menu handler for Cities: Skylines 2 environment.

This module integrates the various menu related components.
"""

import logging
import time
from typing import Dict, List, Tuple, Optional, Union, Any

from ..optimized_capture import OptimizedScreenCapture
from ..visual_metrics import VisualMetricsEstimator
from ..input.keyboard import KeyboardInput
from ..input.mouse import MouseInput

from .detector import MenuDetector
from .navigator import MenuNavigator
from .recovery import MenuRecovery
from .templates import MenuTemplateManager

logger = logging.getLogger(__name__)

class MenuHandler:
    """Integrates menu detection, navigation, and recovery components."""
    
    def __init__(
        self,
        screen_capture: OptimizedScreenCapture,
        input_simulator = None,  # For backward compatibility
        visual_metrics: Optional[VisualMetricsEstimator] = None,
        templates_dir: str = "menu_templates"
    ):
        """Initialize the menu handler.
        
        Args:
            screen_capture: Screen capture module
            input_simulator: Legacy input simulator (for backward compatibility)
            visual_metrics: Visual metrics estimator
            templates_dir: Directory to store menu templates
        """
        self.screen_capture = screen_capture
        self.visual_metrics = visual_metrics
        
        # For backward compatibility
        self.input_simulator = input_simulator
        
        # Extract or create input components
        if input_simulator is not None and hasattr(input_simulator, 'keyboard'):
            self.keyboard_input = input_simulator.keyboard
            self.mouse_input = input_simulator.mouse
        else:
            logger.info("Creating new keyboard and mouse input instances")
            self.keyboard_input = KeyboardInput()
            self.mouse_input = MouseInput()
        
        # Initialize template manager
        self.template_manager = MenuTemplateManager(templates_dir=templates_dir)
        
        # Initialize detector
        self.detector = MenuDetector(
            screen_capture=screen_capture,
            visual_metrics=visual_metrics
        )
        
        # Initialize navigator
        self.navigator = MenuNavigator(
            screen_capture=screen_capture,
            keyboard_input=self.keyboard_input,
            mouse_input=self.mouse_input,
            menu_detector=self.detector
        )
        
        # Initialize recovery
        self.recovery = MenuRecovery(
            menu_detector=self.detector,
            menu_navigator=self.navigator,
            keyboard_input=self.keyboard_input,
            mouse_input=self.mouse_input
        )
        
        # Menu handler state
        self.menu_recovery_attempts = 0
        self.max_recovery_attempts = 3
        
        logger.info("Menu handler initialized")
    
    def check_menu_state(self) -> bool:
        """Check if we're currently in a menu state.
        
        Returns:
            Whether we're currently in a menu
        """
        return self.detector.check_menu_state()
    
    def is_in_menu(self) -> bool:
        """Check if we're currently in a menu.
        
        Returns:
            Whether we're in a menu
        """
        return self.detector.is_in_menu()
    
    def get_menu_type(self) -> Optional[str]:
        """Get the current menu type.
        
        Returns:
            Current menu type or None if not in a menu
        """
        return self.detector.get_menu_type()
    
    def handle_menu_recovery(self, retries: int = 3) -> bool:
        """Handle recovery from a stuck menu.
        
        Args:
            retries: Number of times to retry recovery
            
        Returns:
            Whether recovery was successful
        """
        if not self.detector.is_in_menu():
            return True  # Already not in a menu
            
        logger.info(f"Attempting menu recovery (retries={retries})")
        
        # Track attempts
        self.menu_recovery_attempts += 1
        
        # Check if we've exceeded max attempts for this session
        if self.menu_recovery_attempts > self.max_recovery_attempts:
            logger.warning(f"Exceeded maximum recovery attempts ({self.max_recovery_attempts})")
            # Reset counter after a while to allow future recovery attempts
            if time.time() - self.recovery.last_recovery_time > 300:  # 5 minutes
                self.menu_recovery_attempts = 0
            return False
        
        # Attempt recovery
        for attempt in range(retries):
            logger.info(f"Recovery attempt {attempt+1}/{retries}")
            
            # Check if we're still in a menu
            if not self.detector.check_menu_state():
                logger.info("No longer in a menu, recovery succeeded")
                return True
                
            # Try recovery
            if self.recovery.recover_from_menu():
                logger.info("Recovery succeeded")
                return True
                
            # Wait a bit before next attempt
            time.sleep(1.0)
        
        # Check one last time
        if not self.detector.check_menu_state():
            logger.info("No longer in a menu after all attempts, recovery succeeded")
            return True
            
        logger.warning("Failed to recover from menu after all attempts")
        return False
    
    def exit_menu(self, retries: int = 3) -> bool:
        """Attempt to exit any current menu.
        
        Args:
            retries: Number of times to retry
            
        Returns:
            Whether we successfully exited the menu
        """
        return self.navigator.exit_menu(retries=retries)
    
    def navigate_menu(self, menu_type: Optional[str] = None) -> bool:
        """Navigate through a menu based on its type.
        
        Args:
            menu_type: Type of menu to navigate, or None to use detected type
            
        Returns:
            Whether navigation was successful
        """
        return self.navigator.navigate_menu(menu_type=menu_type)
    
    def click_button(self, button_name: str) -> bool:
        """Click a menu button by name.
        
        Args:
            button_name: Name of the button to click
            
        Returns:
            Whether the click was successful
        """
        return self.navigator.click_button(button_name)
    
    def add_template(
        self,
        menu_type: str,
        template_image,
        signature_regions: Optional[List[Tuple[float, float, float, float]]] = None,
        threshold: float = 0.7
    ) -> bool:
        """Add a new template for a menu type.
        
        Args:
            menu_type: Type of menu
            template_image: Image of the menu
            signature_regions: Regions that are characteristic of this menu
            threshold: Detection threshold
            
        Returns:
            Whether the template was added successfully
        """
        return self.template_manager.add_template(
            menu_type=menu_type,
            template_image=template_image,
            signature_regions=signature_regions,
            threshold=threshold
        )
    
    def learn_from_current_frame(
        self,
        menu_type: str,
        region: Optional[Tuple[float, float, float, float]] = None,
        threshold: float = 0.7,
        signature_regions: Optional[List[Tuple[float, float, float, float]]] = None
    ) -> bool:
        """Learn a new template from the current frame.
        
        Args:
            menu_type: Type of menu
            region: Region to extract (normalized coordinates)
            threshold: Detection threshold
            signature_regions: Regions that are characteristic of this menu
            
        Returns:
            Whether the template was learned successfully
        """
        # Capture current frame
        current_frame = self.screen_capture.capture_frame()
        if current_frame is None:
            logger.warning("Failed to capture frame for template learning")
            return False
            
        # Learn from frame
        return self.template_manager.learn_from_frame(
            frame=current_frame,
            menu_type=menu_type,
            region=region,
            threshold=threshold,
            signature_regions=signature_regions
        )
    
    # Compatibility methods for backward compatibility
    
    def track_action(self, action_type: str) -> None:
        """Track actions that might lead to menu screens.
        
        Args:
            action_type: Type of action (e.g., "esc_key", "gear_click")
        """
        # This method exists for backward compatibility
        # It doesn't do anything in the new implementation
        logger.debug(f"Action tracked (compatibility): {action_type}")
    
    def detect_menu(self, current_frame) -> Tuple[bool, Optional[str], float]:
        """Detect if the current frame shows a menu and which type.
        
        Args:
            current_frame: Current frame to analyze
            
        Returns:
            Tuple of (menu_detected, menu_type, confidence)
        """
        return self.detector.detect_menu(current_frame) 