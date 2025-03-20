"""
Menu recovery for Cities: Skylines 2 environment.

This module provides functionality for recovering from stuck menu situations.
"""

import logging
import time
import random
from typing import Dict, List, Tuple, Optional, Union, Any

from .detector import MenuDetector
from .navigator import MenuNavigator
from ..input.keyboard import KeyboardInput
from ..input.mouse import MouseInput

logger = logging.getLogger(__name__)

class MenuRecovery:
    """Handles recovery from stuck menu situations."""
    
    def __init__(
        self,
        menu_detector: MenuDetector,
        menu_navigator: MenuNavigator,
        keyboard_input: KeyboardInput,
        mouse_input: MouseInput
    ):
        """Initialize menu recovery.
        
        Args:
            menu_detector: Menu detector module
            menu_navigator: Menu navigator module
            keyboard_input: Keyboard input module
            mouse_input: Mouse input module
        """
        self.menu_detector = menu_detector
        self.menu_navigator = menu_navigator
        self.keyboard_input = keyboard_input
        self.mouse_input = mouse_input
        
        # Recovery state
        self.recovery_attempts = 0
        self.max_recovery_attempts = 5
        self.last_recovery_time = 0
        self.recovery_cooldown = 30.0  # seconds
        self.recovery_escalation_level = 0
        self.menu_stuck_time = None
        self.max_menu_stuck_time = 60.0  # seconds
        self.drastic_recovery_count = 0
        self.max_drastic_recoveries = 3  # Limit drastic recoveries per session
        
        logger.info("Menu recovery initialized")
    
    def can_attempt_recovery(self) -> bool:
        """Check if we can attempt recovery based on cooldown.
        
        Returns:
            Whether recovery can be attempted
        """
        # Check if we're within the cooldown period
        current_time = time.time()
        if current_time - self.last_recovery_time < self.recovery_cooldown:
            return False
            
        # Check if we've exceeded max recovery attempts
        if self.recovery_attempts >= self.max_recovery_attempts:
            # Reset counter after a longer cooldown
            if current_time - self.last_recovery_time > self.recovery_cooldown * 5:
                self.recovery_attempts = 0
            else:
                return False
                
        return True
    
    def update_stuck_time(self, in_menu: bool):
        """Update menu stuck time tracking.
        
        Args:
            in_menu: Whether we're currently in a menu
        """
        current_time = time.time()
        
        if in_menu:
            # Initialize stuck time if we just entered a menu
            if self.menu_stuck_time is None:
                self.menu_stuck_time = current_time
        else:
            # Reset stuck time if we're no longer in a menu
            self.menu_stuck_time = None
            
            # Also reset escalation level
            self.recovery_escalation_level = 0
    
    def check_if_stuck(self) -> bool:
        """Check if we're stuck in a menu for too long.
        
        Returns:
            Whether we're stuck in a menu
        """
        if self.menu_stuck_time is None:
            return False
            
        stuck_duration = time.time() - self.menu_stuck_time
        return stuck_duration > self.max_menu_stuck_time
    
    def standard_recovery(self) -> bool:
        """Perform standard recovery from menu situation.
        
        Returns:
            Whether recovery was successful
        """
        logger.info("Performing standard menu recovery")
        
        # Record recovery attempt
        self.recovery_attempts += 1
        self.last_recovery_time = time.time()
        
        # First try standard menu navigation
        menu_type = self.menu_detector.get_menu_type()
        if menu_type:
            logger.info(f"Trying standard navigation for menu type: {menu_type}")
            if self.menu_navigator.navigate_menu(menu_type):
                # Check if we're out of the menu
                if not self.menu_detector.check_menu_state():
                    logger.info("Standard navigation successful")
                    return True
        
        # Try generic exit strategies
        logger.info("Trying generic exit strategies")
        if self.menu_navigator.exit_menu(retries=2):
            logger.info("Generic exit successful")
            return True
        
        # Escape key sometimes works for menus
        logger.info("Trying escape key")
        self.keyboard_input.press_key("escape")
        time.sleep(0.5)
        
        # Check if we're out of the menu
        if not self.menu_detector.check_menu_state():
            logger.info("Escape key successful")
            return True
            
        # Try clicking in corners (sometimes closes dialogs)
        logger.info("Trying corner clicks")
        screen_width, screen_height = self.mouse_input.get_screen_dimensions()
        corners = [
            (screen_width - 10, 10),  # Top right
            (10, 10),                 # Top left
            (10, screen_height - 10),  # Bottom left
            (screen_width - 10, screen_height - 10)  # Bottom right
        ]
        
        for x, y in corners:
            self.mouse_input.move_to(x, y)
            time.sleep(0.1)
            self.mouse_input.click()
            time.sleep(0.3)
            
            # Check if we're out of the menu
            if not self.menu_detector.check_menu_state():
                logger.info("Corner click successful")
                return True
        
        logger.warning("Standard recovery failed")
        return False
    
    def escalated_recovery(self) -> bool:
        """Perform escalated recovery methods.
        
        Returns:
            Whether recovery was successful
        """
        logger.info(f"Performing escalated recovery (level {self.recovery_escalation_level})")
        
        # Record recovery attempt
        self.recovery_attempts += 1
        self.last_recovery_time = time.time()
        
        # Escalation level determines strategy
        if self.recovery_escalation_level == 1:
            # Try rapidly clicking in the center of the screen
            logger.info("Trying rapid center clicks")
            screen_width, screen_height = self.mouse_input.get_screen_dimensions()
            center_x, center_y = screen_width // 2, screen_height // 2
            
            for _ in range(5):
                self.mouse_input.move_to(center_x, center_y)
                self.mouse_input.click()
                time.sleep(0.1)
                
                # Check if we're out of the menu
                if not self.menu_detector.check_menu_state():
                    logger.info("Rapid center clicks successful")
                    return True
                    
            # Try pressing various common keys
            logger.info("Trying various keys")
            keys = ["enter", "space", "escape", "tab"]
            
            for key in keys:
                self.keyboard_input.press_key(key)
                time.sleep(0.3)
                
                # Check if we're out of the menu
                if not self.menu_detector.check_menu_state():
                    logger.info(f"{key} key successful")
                    return True
                    
        elif self.recovery_escalation_level == 2:
            # Try clicking in a grid pattern across the screen
            logger.info("Trying grid pattern clicks")
            screen_width, screen_height = self.mouse_input.get_screen_dimensions()
            
            grid_size = 5
            for x_idx in range(grid_size):
                for y_idx in range(grid_size):
                    x = int((x_idx + 0.5) * screen_width / grid_size)
                    y = int((y_idx + 0.5) * screen_height / grid_size)
                    
                    self.mouse_input.move_to(x, y)
                    time.sleep(0.1)
                    self.mouse_input.click()
                    time.sleep(0.2)
                    
                    # Check if we're out of the menu
                    if not self.menu_detector.check_menu_state():
                        logger.info(f"Grid click at ({x}, {y}) successful")
                        return True
            
            # Try key combinations
            logger.info("Trying key combinations")
            combinations = [
                ("alt", "f4"),  # Close application
                ("ctrl", "w"),  # Close window
                ("alt", "tab"),  # Switch window
            ]
            
            for mod_key, key in combinations:
                self.keyboard_input.press_key(mod_key, hold=True)
                time.sleep(0.1)
                self.keyboard_input.press_key(key)
                time.sleep(0.1)
                self.keyboard_input.release_key(mod_key)
                time.sleep(0.5)
                
                # Check if we're out of the menu
                if not self.menu_detector.check_menu_state():
                    logger.info(f"{mod_key}+{key} successful")
                    return True
                    
        elif self.recovery_escalation_level >= 3:
            # Drastic measures - may disrupt gameplay but better than being stuck
            if self.drastic_recovery_count < self.max_drastic_recoveries:
                logger.warning("Using drastic recovery measures")
                self.drastic_recovery_count += 1
                
                # Try really aggressive key presses
                for key in ["escape", "enter", "space"]:
                    # Press key multiple times rapidly
                    for _ in range(5):
                        self.keyboard_input.press_key(key)
                        time.sleep(0.1)
                    
                    # Check if we're out of the menu
                    if not self.menu_detector.check_menu_state():
                        logger.info(f"Aggressive {key} presses successful")
                        return True
                
                # As a last resort, try to click and drag in random directions
                logger.warning("Trying random mouse movements")
                screen_width, screen_height = self.mouse_input.get_screen_dimensions()
                
                start_x, start_y = screen_width // 2, screen_height // 2
                self.mouse_input.move_to(start_x, start_y)
                self.mouse_input.press_button()
                
                # Make random movements
                for _ in range(10):
                    end_x = random.randint(0, screen_width)
                    end_y = random.randint(0, screen_height)
                    self.mouse_input.move_to(end_x, end_y)
                    time.sleep(0.1)
                
                self.mouse_input.release_button()
                time.sleep(0.5)
                
                # Check if we're out of the menu
                if not self.menu_detector.check_menu_state():
                    logger.info("Random mouse movements successful")
                    return True
                
                # Absolute last resort
                logger.critical("All recovery methods failed, attempting Alt+F4")
                self.keyboard_input.press_key("alt", hold=True)
                time.sleep(0.1)
                self.keyboard_input.press_key("f4")
                time.sleep(0.1)
                self.keyboard_input.release_key("alt")
                
                # This may close the game, but at least we're not stuck
            else:
                logger.critical("Exceeded maximum drastic recoveries, giving up")
        
        logger.warning(f"Escalated recovery level {self.recovery_escalation_level} failed")
        return False
    
    def recover_from_menu(self) -> bool:
        """Attempt to recover from a stuck menu situation.
        
        Returns:
            Whether recovery was successful
        """
        # Check if we can attempt recovery
        if not self.can_attempt_recovery():
            logger.info("Recovery on cooldown, skipping")
            return False
            
        # Check if we're actually in a menu
        if not self.menu_detector.check_menu_state():
            logger.info("Not in a menu, no recovery needed")
            return True
            
        # Update stuck time tracking
        self.update_stuck_time(True)
        
        # Determine recovery strategy based on escalation level
        if self.recovery_escalation_level == 0:
            success = self.standard_recovery()
        else:
            success = self.escalated_recovery()
            
        # If we're still in a menu after recovery, escalate
        if not success or self.menu_detector.check_menu_state():
            self.recovery_escalation_level = min(3, self.recovery_escalation_level + 1)
            logger.info(f"Escalating recovery to level {self.recovery_escalation_level}")
            return False
            
        # Reset escalation if recovery was successful
        self.recovery_escalation_level = 0
        self.menu_stuck_time = None
        logger.info("Menu recovery successful")
        return True 