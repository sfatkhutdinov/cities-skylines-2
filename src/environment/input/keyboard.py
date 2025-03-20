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
    
    def key_press(self, key: Any, duration: float = 0.1, force_direct: bool = False) -> bool:
        """Press a key and hold for specified duration.
        
        Args:
            key: Key to press (string representation or pynput Key object)
            duration: How long to hold the key in seconds
            force_direct: Force using direct key press even for special keys
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to pynput Key if a string is provided
            actual_key = self.key_map.get(key, key) if isinstance(key, str) else key
            
            # Block escape if needed
            if self.block_escape and ((isinstance(key, str) and key.lower() == 'escape') or actual_key == Key.esc):
                logger.warning("Escape key press blocked (safety feature)")
                return False
                
            # Try to press the key using pynput
            try:
                # Press key normally (pynput)
                if not force_direct and isinstance(actual_key, (Key, str)):
                    self.keyboard.press(actual_key)
                    time.sleep(duration)
                    self.keyboard.release(actual_key)
                    return True
            except Exception as e:
                logger.warning(f"Error pressing key using pynput: {e}")
            
            # Fallback: try using pyautogui
            try:
                # Some special keys have different names in pyautogui
                key_mapping = {
                    'escape': 'esc',
                    Key.esc: 'esc',
                    'space': 'space',
                    Key.space: 'space',
                    'enter': 'enter',
                    Key.enter: 'enter',
                    # Add more mappings as needed
                }
                
                # Try to get the mapped key name for pyautogui
                pyautogui_key = key
                if isinstance(key, str):
                    pyautogui_key = key_mapping.get(key.lower(), key.lower())
                elif isinstance(key, Key):
                    pyautogui_key = key_mapping.get(key, str(key).replace('Key.', ''))
                
                pyautogui.keyDown(pyautogui_key)
                time.sleep(duration)
                pyautogui.keyUp(pyautogui_key)
                return True
            except Exception as e:
                logger.error(f"Error pressing key using pyautogui fallback: {e}")
                return False
        except Exception as e:
            logger.error(f"Error in key_press: {e}")
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