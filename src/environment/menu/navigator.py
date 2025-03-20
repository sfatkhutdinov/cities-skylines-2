"""
Menu navigator for Cities: Skylines 2.

This module handles navigation within in-game menus.
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any

from src.environment.input.keyboard import KeyboardController
from src.environment.input.mouse import MouseController
from .detector import MenuDetector
from src.environment.core.observation import ObservationManager

logger = logging.getLogger(__name__)

class MenuNavigator:
    """Handles navigation of in-game menus."""
    
    # Menu button positions (normalized coordinates)
    DEFAULT_BUTTON_POSITIONS = {
        "play_button": (0.5, 0.4),
        "resume_button": (0.5, 0.35),
        "back_button": (0.2, 0.9),
        "ok_button": (0.5, 0.65),
        "close_button": (0.9, 0.1),
        "settings_button": (0.5, 0.5),
        "exit_button": (0.5, 0.7),
        "save_button": (0.5, 0.6),
        "load_button": (0.5, 0.5),
        "confirm_button": (0.6, 0.65),
        "cancel_button": (0.4, 0.65)
    }
    
    # Navigation paths for different menus (button sequence to exit)
    DEFAULT_MENU_EXIT_PATHS = {
        "main_menu": ["play_button"],
        "pause_menu": ["resume_button"],
        "settings_menu": ["back_button", "resume_button"],
        "dialog": ["ok_button", "close_button"],
        "notification": ["close_button"],
        "save_dialog": ["cancel_button"],
        "confirm_dialog": ["cancel_button"]
    }
    
    def __init__(
        self,
        observation_manager: ObservationManager,
        keyboard_controller: KeyboardController,
        mouse_controller: MouseController,
        menu_detector: MenuDetector,
        custom_button_positions: Optional[Dict[str, Tuple[float, float]]] = None,
        custom_menu_exit_paths: Optional[Dict[str, List[str]]] = None
    ):
        """Initialize menu navigator.
        
        Args:
            observation_manager: Observation manager for screen dimensions
            keyboard_controller: Keyboard input module
            mouse_controller: Mouse input module
            menu_detector: Menu detector module
            custom_button_positions: Optional custom button positions
            custom_menu_exit_paths: Optional custom menu exit paths
        """
        self.observation_manager = observation_manager
        self.keyboard_controller = keyboard_controller
        self.mouse_controller = mouse_controller
        self.menu_detector = menu_detector
        
        # Copy the default button positions and update with any custom ones
        self.button_positions = self.DEFAULT_BUTTON_POSITIONS.copy()
        if custom_button_positions:
            self.button_positions.update(custom_button_positions)
            
        # Copy the default menu exit paths and update with any custom ones
        self.menu_exit_paths = self.DEFAULT_MENU_EXIT_PATHS.copy()
        if custom_menu_exit_paths:
            self.menu_exit_paths.update(custom_menu_exit_paths)
        
        # Navigation state
        self.last_click_time = 0
        self.click_cooldown = 0.5  # seconds
        self.navigation_attempts = 0
        self.max_navigation_attempts = 5
        self.last_navigation_success_time = 0
        
        logger.info("Menu navigator initialized")
    
    def click_button(self, button_name: str) -> bool:
        """Click a menu button by name.
        
        Args:
            button_name: Name of the button to click
            
        Returns:
            Whether the click was successful
        """
        # Check if button exists in our positions
        if button_name not in self.button_positions:
            logger.warning(f"Unknown button: {button_name}")
            return False
            
        # Get normalized position
        norm_x, norm_y = self.button_positions[button_name]
        
        # Get screen dimensions from screen_capture
        screen_width, screen_height = self.observation_manager.get_screen_dimensions()
        
        # Calculate pixel coordinates
        pixel_x = int(norm_x * screen_width)
        pixel_y = int(norm_y * screen_height)
        
        # Enforce click cooldown
        current_time = time.time()
        if current_time - self.last_click_time < self.click_cooldown:
            time_to_wait = self.click_cooldown - (current_time - self.last_click_time)
            logger.debug(f"Waiting {time_to_wait:.2f}s for click cooldown")
            time.sleep(time_to_wait)
        
        # Perform the click
        try:
            logger.info(f"Clicking button: {button_name} at ({pixel_x}, {pixel_y})")
            self.mouse_controller.move_to(pixel_x, pixel_y)
            time.sleep(0.1)  # Small delay to ensure move completes
            self.mouse_controller.click()
            self.last_click_time = time.time()
            return True
        except Exception as e:
            logger.error(f"Error clicking button {button_name}: {e}")
            return False
    
    def navigate_menu(self, menu_type: Optional[str] = None) -> bool:
        """Navigate through a menu based on its type.
        
        Args:
            menu_type: Type of menu to navigate, or None to use detected type
            
        Returns:
            Whether navigation was successful
        """
        # Get menu type if not provided
        if menu_type is None:
            menu_type = self.menu_detector.get_menu_type()
            
        if menu_type is None:
            logger.warning("Cannot navigate: no menu type detected")
            return False
            
        # Get navigation path for this menu type
        if menu_type not in self.menu_exit_paths:
            logger.warning(f"No navigation path for menu type: {menu_type}")
            return False
            
        path = self.menu_exit_paths[menu_type]
        logger.info(f"Navigating menu type: {menu_type} using path: {path}")
        
        # Follow the navigation path
        success = True
        for button_name in path:
            click_success = self.click_button(button_name)
            if not click_success:
                logger.warning(f"Failed to click button: {button_name}")
                success = False
                break
                
            # Wait a moment for the UI to respond
            time.sleep(0.3)
            
            # Check if we're still in a menu
            if not self.menu_detector.check_menu_state():
                logger.info("Successfully exited menu")
                break
        
        # Record success time if navigation succeeded
        if success:
            self.last_navigation_success_time = time.time()
        
        return success
    
    def exit_menu(self, retries: int = 3) -> bool:
        """Attempt to exit any current menu.
        
        Args:
            retries: Number of times to retry
            
        Returns:
            Whether we successfully exited the menu
        """
        # Check if we're in a menu
        if not self.menu_detector.check_menu_state():
            return True  # Already not in a menu
            
        logger.info("Attempting to exit menu")
        
        # Try standard navigation first
        menu_type = self.menu_detector.get_menu_type()
        if menu_type and menu_type in self.menu_exit_paths:
            success = self.navigate_menu(menu_type)
            if success and not self.menu_detector.check_menu_state():
                return True
                
        # If standard navigation failed, try alternative approaches
        for attempt in range(retries):
            logger.info(f"Exit menu retry {attempt+1}/{retries}")
            
            # Try escape key
            logger.info("Trying escape key")
            self.keyboard_controller.press_key("escape")
            time.sleep(0.5)
            
            # Check if we exited
            if not self.menu_detector.check_menu_state():
                return True
                
            # Try clicking common exit buttons
            for button in ["close_button", "cancel_button", "back_button", "resume_button"]:
                logger.info(f"Trying {button}")
                self.click_button(button)
                time.sleep(0.5)
                
                # Check if we exited
                if not self.menu_detector.check_menu_state():
                    return True
        
        # All attempts failed
        logger.warning("Failed to exit menu after all attempts")
        return False
    
    def handle_specific_menu(self, menu_type: str) -> bool:
        """Handle a specific menu type with custom logic.
        
        Args:
            menu_type: Type of menu to handle
            
        Returns:
            Whether the menu was handled successfully
        """
        # Check if we have a specific handler for this menu type
        handler_method_name = f"_handle_{menu_type}_menu"
        handler = getattr(self, handler_method_name, None)
        
        if handler is not None and callable(handler):
            logger.info(f"Using specialized handler for {menu_type} menu")
            return handler()
        else:
            # Fall back to standard navigation
            logger.info(f"No specialized handler for {menu_type} menu, using standard navigation")
            return self.navigate_menu(menu_type)
    
    def _handle_pause_menu(self) -> bool:
        """Handle the pause menu.
        
        Returns:
            Whether the menu was handled successfully
        """
        logger.info("Handling pause menu")
        return self.click_button("resume_button")
    
    def _handle_dialog(self) -> bool:
        """Handle generic dialog.
        
        Returns:
            Whether the menu was handled successfully
        """
        logger.info("Handling dialog")
        
        # Try OK button first
        if self.click_button("ok_button"):
            time.sleep(0.3)
            if not self.menu_detector.check_menu_state():
                return True
                
        # Then try close button
        return self.click_button("close_button")
    
    def _handle_notification(self) -> bool:
        """Handle notification.
        
        Returns:
            Whether the menu was handled successfully
        """
        logger.info("Handling notification")
        return self.click_button("close_button")
    
    def _handle_settings_menu(self) -> bool:
        """Handle settings menu.
        
        Returns:
            Whether the menu was handled successfully
        """
        logger.info("Handling settings menu")
        
        # First click back button
        if self.click_button("back_button"):
            time.sleep(0.3)
            
            # Then check if we're still in a menu (might be in the parent menu)
            if self.menu_detector.check_menu_state():
                # Try clicking resume button
                return self.click_button("resume_button")
            else:
                # Already exited all menus
                return True
        
        return False 