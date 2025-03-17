"""
Input simulator for Cities: Skylines 2.
Handles keyboard and mouse input simulation using pynput.
"""

import time
import logging
import os
import subprocess
import re
from typing import Tuple, Optional, List, Set
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class InputSimulator:
    """Simulates keyboard and mouse input for Cities: Skylines 2."""
    
    def __init__(self):
        """Initialize input simulator."""
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        logger.debug("Input simulator initialized")
        
        # List of blocked key combinations
        self.blocked_combos = [
            {'alt', 'tab'},     # Alt+Tab (window switching)
            {'alt', 'f4'},      # Alt+F4 (close window)
            {'windows'},        # Windows key
            {'ctrl', 'alt', 'delete'},  # Ctrl+Alt+Del
        ]
        
        # Game window name
        self.game_window_name = "Cities: Skylines II"
        
    def find_game_window(self) -> bool:
        """Find the Cities: Skylines II window.
        
        Returns:
            bool: True if window was found, False otherwise
        """
        logger.debug("Looking for Cities: Skylines II window")
        
        try:
            # Use powershell to find the window
            command = 'powershell -command "Get-Process | Where-Object {$_.MainWindowTitle -match \'Cities: Skylines II\'} | Select-Object Name,MainWindowTitle"'
            result = subprocess.run(command, capture_output=True, text=True)
            
            # Check if the window was found
            if "Cities: Skylines II" in result.stdout:
                logger.debug(f"Found game window: {result.stdout.strip()}")
                return True
            else:
                logger.warning("Game window not found")
                return False
        except Exception as e:
            logger.error(f"Error finding game window: {str(e)}")
            return False
        
    def ensure_game_window_focused(self) -> bool:
        """Ensure the game window is focused.
        
        Returns:
            bool: True if focus was successful
        """
        logger.debug("Attempting to focus game window...")
        
        # First check if the window exists
        if not self.find_game_window():
            logger.warning("Cannot focus game window - window not found")
            return False
            
        try:
            # Use Alt+Tab to switch windows until we find the game
            # This is not perfect but should work if the game is running
            self.keyboard.press(Key.alt)
            time.sleep(0.2)
            
            # Try up to 10 window switches
            for _ in range(10):
                self.keyboard.press(Key.tab)
                time.sleep(0.1)
                self.keyboard.release(Key.tab)
                time.sleep(0.5)  # Give time for the window to appear
                
                # We can't easily check if we're focused on the right window
                # Future improvement: use windows API for this
            
            self.keyboard.release(Key.alt)
            time.sleep(0.5)
            
            # Try to activate the window explicitly with a powershell command
            command = 'powershell -command "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate(\'Cities: Skylines II\')"'
            subprocess.run(command, capture_output=True, text=True)
            
            logger.debug("Window focus attempt completed")
            return True
        except Exception as e:
            logger.error(f"Error focusing window: {str(e)}")
            return False
        
    def is_blocked_combo(self, keys: List[str]) -> bool:
        """Check if a key combination is blocked.
        
        Args:
            keys (List[str]): List of keys in the combination
            
        Returns:
            bool: True if combination is blocked
        """
        key_set = set(k.lower() for k in keys)
        
        # Check against all blocked combinations
        for blocked in self.blocked_combos:
            if blocked.issubset(key_set):
                logger.warning(f"Blocked key combination: {keys}")
                return True
                
        return False
        
    def press_key(self, key: str, duration: float = 0.1):
        """Press and release a key."""
        # Check if it's a blocked key
        if self.is_blocked_combo([key]):
            return
            
        logger.debug(f"Pressing key: {key} for {duration} seconds")
        self.keyboard.press(key)
        time.sleep(duration)
        self.keyboard.release(key)
        logger.debug(f"Released key: {key}")
        
    def press_keys(self, keys: list, duration: float = 0.1):
        """Press and release multiple keys simultaneously."""
        # Check if it's a blocked combination
        if self.is_blocked_combo(keys):
            return
            
        logger.debug(f"Pressing keys: {keys} for {duration} seconds")
        for key in keys:
            self.keyboard.press(key)
        time.sleep(duration)
        for key in keys:
            self.keyboard.release(key)
        logger.debug(f"Released keys: {keys}")
        
    def key_combination(self, keys: List[str], duration: float = 0.1):
        """Press a combination of keys with better handling of special keys."""
        # Don't execute blocked combinations
        if self.is_blocked_combo(keys):
            return
            
        # Map key names to actual key objects
        key_objects = []
        for k in keys:
            key_obj = self._get_key_object(k)
            if key_obj:
                key_objects.append(key_obj)
                
        # Press all keys
        for key_obj in key_objects:
            self.keyboard.press(key_obj)
            
        time.sleep(duration)
        
        # Release all keys in reverse order
        for key_obj in reversed(key_objects):
            self.keyboard.release(key_obj)
            
    def _get_key_object(self, key_name: str):
        """Convert key name string to actual key object."""
        # Handle special keys
        special_keys = {
            'alt': Key.alt,
            'ctrl': Key.ctrl,
            'shift': Key.shift,
            'enter': Key.enter,
            'space': Key.space,
            'tab': Key.tab,
            'esc': Key.esc,
            'escape': Key.esc,
            'up': Key.up,
            'down': Key.down,
            'left': Key.left,
            'right': Key.right,
        }
        
        if key_name.lower() in special_keys:
            return special_keys[key_name.lower()]
        elif len(key_name) == 1:  # Single character key
            return key_name
        else:
            logger.warning(f"Unknown key: {key_name}")
            return None
            
    def move_mouse(self, x: int, y: int):
        """Move mouse to absolute coordinates."""
        logger.debug(f"Moving mouse to: ({x}, {y})")
        self.mouse.position = (x, y)
        
    def move_mouse_relative(self, dx: int, dy: int):
        """Move mouse by relative offset."""
        current_pos = self.mouse.position
        new_pos = (current_pos[0] + dx, current_pos[1] + dy)
        logger.debug(f"Moving mouse by offset ({dx}, {dy}) from {current_pos} to {new_pos}")
        self.mouse.position = new_pos
        
    def click(self, button: str = 'left'):
        """Perform mouse click."""
        logger.debug(f"Clicking {button} button")
        btn = Button.left if button == 'left' else Button.right
        self.mouse.click(btn)
        
    def double_click(self, button: str = 'left'):
        """Perform mouse double click."""
        logger.debug(f"Double clicking {button} button")
        btn = Button.left if button == 'left' else Button.right
        self.mouse.click(btn, 2)
        
    def mouse_down(self, button: str = 'left'):
        """Press and hold mouse button."""
        logger.debug(f"Pressing {button} button")
        btn = Button.left if button == 'left' else Button.right
        self.mouse.press(btn)
        
    def mouse_up(self, button: str = 'left'):
        """Release mouse button."""
        logger.debug(f"Releasing {button} button")
        btn = Button.left if button == 'left' else Button.right
        self.mouse.release(btn)
        
    def scroll(self, amount: int):
        """Scroll the mouse wheel."""
        logger.debug(f"Scrolling by {amount}")
        self.mouse.scroll(0, amount)
        
    def drag(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], button: str = 'left'):
        """Perform mouse drag operation."""
        logger.debug(f"Dragging from {start_pos} to {end_pos} with {button} button")
        self.move_mouse(*start_pos)
        time.sleep(0.1)
        self.mouse_down(button)
        time.sleep(0.1)
        self.move_mouse(*end_pos)
        time.sleep(0.1)
        self.mouse_up(button)
        
    def close(self):
        """Clean up resources."""
        logger.debug("Closing input simulator")
        # Nothing specific to clean up with pynput
        pass 