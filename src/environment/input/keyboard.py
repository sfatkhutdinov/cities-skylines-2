"""
Keyboard input simulation for Cities: Skylines 2.

This module provides keyboard input capabilities for the game environment.
"""

import time
import logging
import ctypes
from typing import Dict, Optional, Any
from pynput.keyboard import Key, Controller as KeyboardController
import pyautogui
import win32gui

logger = logging.getLogger(__name__)

class KeyboardInput:
    """Handles keyboard input simulation for the game."""
    
    def __init__(self):
        """Initialize keyboard controller and key mappings."""
        self.keyboard = KeyboardController()
        
        # Initialize virtual key code mapping for all standard keys
        self.key_map = {
            'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f', 'g': 'g', 'h': 'h',
            'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l', 'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p',
            'q': 'q', 'r': 'r', 's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x',
            'y': 'y', 'z': 'z', '0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
            '6': '6', '7': '7', '8': '8', '9': '9', 'space': Key.space, 'enter': Key.enter,
            'escape': Key.esc, 'tab': Key.tab, 'capslock': Key.caps_lock,
            'shift': Key.shift, 'ctrl': Key.ctrl, 'alt': Key.alt,
            'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
            'delete': Key.delete, 'home': Key.home, 'end': Key.end, 'pageup': Key.page_up,
            'pagedown': Key.page_down, 'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8, 'f9': Key.f9, 'f10': Key.f10,
            'f11': Key.f11, 'f12': Key.f12,
        }
        
        # Add letter keys
        for c in 'abcdefghijklmnopqrstuvwxyz0123456789':
            self.key_map[c] = c
            
        # Block escape key by default to prevent accidental menu toggling
        self.block_escape = True
        
        logger.info("Keyboard input initialized")
    
    def key_press(self, key: str, duration: float = 0.1) -> bool:
        """Press a key on the keyboard.
        
        Args:
            key: String representation of the key to press
            duration: Duration to hold the key down in seconds
            
        Returns:
            True if successful, False otherwise
        """
        key_id = str(time.time())[-6:]  # Use last 6 digits of timestamp as key ID
        logger.info(f"[KEY-{key_id}] Pressing key '{key}' for {duration} seconds")
        
        try:
            # Map string to Key object if needed
            try:
                mapped_key = self._map_key(key)
                logger.debug(f"[KEY-{key_id}] Mapped key '{key}' to {mapped_key}")
            except ValueError as e:
                logger.error(f"[KEY-{key_id}] Failed to map key '{key}': {e}")
                return False
            
            # Ensure game window is focused
            game_hwnd = win32gui.FindWindow(None, "Cities: Skylines II")
            if game_hwnd:
                # Bring window to foreground
                win32gui.SetForegroundWindow(game_hwnd)
                # Wait for window to be ready
                time.sleep(0.1)
            
            # Press key down
            logger.debug(f"[KEY-{key_id}] Pressing key down")
            self.keyboard.press(mapped_key)
            
            # Wait for specified duration
            logger.debug(f"[KEY-{key_id}] Waiting for {duration} seconds")
            time.sleep(duration)
            
            # Release key
            logger.debug(f"[KEY-{key_id}] Releasing key")
            self.keyboard.release(mapped_key)
            
            logger.info(f"[KEY-{key_id}] Successfully pressed key '{key}'")
            return True
            
        except Exception as e:
            logger.error(f"[KEY-{key_id}] Error pressing key '{key}': {e}")
            # Ensure key is released in case of error
            try:
                if 'mapped_key' in locals():
                    logger.debug(f"[KEY-{key_id}] Attempting to release key after error")
                    self.keyboard.release(mapped_key)
            except Exception as release_error:
                logger.error(f"[KEY-{key_id}] Failed to release key after error: {release_error}")
            return False
    
    def press_key(self, key: str) -> bool:
        """Press a key without releasing it.
        
        Args:
            key: Key to press down and hold
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map string key to pynput Key if needed
            mapped_key = self.key_map.get(key, key)
            
            # Block escape if needed
            if self.block_escape and ((isinstance(key, str) and key.lower() == 'escape') or mapped_key == Key.esc):
                logger.warning("Escape key press blocked (safety feature)")
                return False
            
            # Press the key
            self.keyboard.press(mapped_key)
            return True
        except Exception as e:
            logger.error(f"Error pressing key {key}: {e}")
            return False
    
    def release_key(self, key: str) -> bool:
        """Release a previously pressed key.
        
        Args:
            key: Key to release
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map string key to pynput Key if needed
            mapped_key = self.key_map.get(key, key)
            
            # Release the key
            self.keyboard.release(mapped_key)
            return True
        except Exception as e:
            logger.error(f"Error releasing key {key}: {e}")
            return False
    
    def type_text(self, text: str, interval: float = 0.05) -> bool:
        """Type a sequence of text with a delay between each character.
        
        Args:
            text: The text to type
            interval: Time between keypresses in seconds
            
        Returns:
            True if successful, False otherwise
        """
        try:
            for char in text:
                self.keyboard.press(char)
                self.keyboard.release(char)
                time.sleep(interval)
            return True
        except Exception as e:
            logger.error(f"Error typing text: {e}")
            return False
    
    def close(self) -> None:
        """Release all keys to clean up resources."""
        try:
            # Release common keyboard keys
            for key in ["shift", "ctrl", "alt", "space"]:
                try:
                    if key in self.key_map:
                        self.keyboard.release(self.key_map[key])
                except Exception as e:
                    logger.warning(f"Error releasing key {key}: {e}")
        except Exception as e:
            logger.error(f"Error in keyboard cleanup: {e}")
        
        logger.info("Keyboard input resources released")
    
    def _map_key(self, key: str) -> Any:
        """Map a string key representation to a pynput Key object.
        
        Args:
            key: String representation of the key
            
        Returns:
            pynput Key object or string character
            
        Raises:
            ValueError: If the key cannot be mapped
        """
        # First check if it's a one-character key (a-z, 0-9, etc.)
        if len(key) == 1:
            return key
            
        # Check if it's in our key_map
        if key.lower() in self.key_map:
            return self.key_map[key.lower()]
            
        # Try to find it in Key enum
        try:
            if hasattr(Key, key.lower()):
                return getattr(Key, key.lower())
        except (AttributeError, TypeError):
            pass
            
        # If we get here, we couldn't map the key
        raise ValueError(f"Could not map key: {key}") 