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
    
    def key_press(self, key: str, duration: float = 0.1, force_direct: bool = False) -> bool:
        """Press a key for the specified duration.
        
        Args:
            key: Key to press
            duration: Duration to hold the key in seconds
            force_direct: Force using direct input even for escape key
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Generate a unique ID for this key press operation
        operation_id = f"{int(time.time() * 1000) % 100000:05d}"
        logger.info(f"[KEY-{operation_id}] Pressing key '{key}' for {duration} seconds")
        
        # Special handling for escape key
        if key.lower() == 'escape' and self.block_escape and not force_direct:
            logger.warning(f"[KEY-{operation_id}] Escape key is blocked. Use force_direct=True to override.")
            return False
            
        try:
            # Map the key to a virtual key code
            logger.debug(f"[KEY-{operation_id}] Mapped key '{key}' to {self._map_key_to_vk(key)}")
            
            # First ensure key is released
            self.ensure_key_released(key)
            
            # Use the more reliable direct method
            success = self._direct_key_press(key, duration)
            if not success:
                logger.warning(f"[KEY-{operation_id}] Direct key press failed, falling back to normal key press")
                
                # Press the key down
                logger.debug(f"[KEY-{operation_id}] Pressing key down")
                if not self._key_down(key):
                    logger.error(f"[KEY-{operation_id}] Failed to press key down")
                    return False
                    
                # Wait for the specified duration
                logger.debug(f"[KEY-{operation_id}] Waiting for {duration} seconds")
                time.sleep(duration)
                
                # Release the key
                logger.debug(f"[KEY-{operation_id}] Releasing key")
                if not self._key_up(key):
                    logger.error(f"[KEY-{operation_id}] Failed to release key")
                    return False
            
            logger.info(f"[KEY-{operation_id}] Successfully pressed key '{key}'")
            
            # Add a small delay after key press to ensure it's processed
            time.sleep(0.1)
            
            return True
        except Exception as e:
            logger.error(f"[KEY-{operation_id}] Error in key_press for key '{key}': {e}")
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

    def ensure_key_released(self, key: str) -> bool:
        """Ensure a key is released (not being held down).
        
        Args:
            key: Key to ensure is released
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Map key to virtual key code
            vk_code = self._map_key_to_vk(key)
            if vk_code is None:
                logger.error(f"Could not map key '{key}' to virtual key code")
                return False
                
            # Check if key is down and send keyup if needed
            key_state = ctypes.windll.user32.GetAsyncKeyState(vk_code)
            if key_state & 0x8000:  # Key is down
                logger.warning(f"Key '{key}' appears to be held down, sending keyup event")
                
                # Send key up using the new method
                if not self._key_up(key):
                    logger.error(f"Failed to release held key '{key}'")
                    return False
                
                # Brief pause to let it register
                time.sleep(0.05)
                
            return True
        except Exception as e:
            logger.error(f"Error in ensure_key_released for key '{key}': {e}")
            return False

    def _map_key_to_vk(self, key: str) -> Optional[int]:
        """Map a string key representation to a virtual key code.
        
        Args:
            key: String representation of the key
            
        Returns:
            Virtual key code or None if the key cannot be mapped
        """
        # First check if it's a one-character key (a-z, 0-9, etc.)
        if len(key) == 1:
            return ord(key)
            
        # Check if it's in our key_map
        if key.lower() in self.key_map:
            mapped_key = self.key_map[key.lower()]
            
            # If it's a Key object from pynput, handle it specially
            if isinstance(mapped_key, Key):
                # Map common special keys to their virtual key codes
                if mapped_key == Key.shift:
                    return 0x10  # VK_SHIFT
                elif mapped_key == Key.ctrl:
                    return 0x11  # VK_CONTROL
                elif mapped_key == Key.alt:
                    return 0x12  # VK_MENU (Alt)
                elif mapped_key == Key.space:
                    return 0x20  # VK_SPACE
                elif mapped_key == Key.enter:
                    return 0x0D  # VK_RETURN
                elif mapped_key == Key.esc:
                    return 0x1B  # VK_ESCAPE
                elif mapped_key == Key.tab:
                    return 0x09  # VK_TAB
                elif mapped_key == Key.backspace:
                    return 0x08  # VK_BACK
                elif mapped_key == Key.up:
                    return 0x26  # VK_UP
                elif mapped_key == Key.down:
                    return 0x28  # VK_DOWN
                elif mapped_key == Key.left:
                    return 0x25  # VK_LEFT
                elif mapped_key == Key.right:
                    return 0x27  # VK_RIGHT
                elif mapped_key == Key.home:
                    return 0x24  # VK_HOME
                elif mapped_key == Key.end:
                    return 0x23  # VK_END
                elif mapped_key == Key.page_up:
                    return 0x21  # VK_PRIOR
                elif mapped_key == Key.page_down:
                    return 0x22  # VK_NEXT
                elif mapped_key == Key.delete:
                    return 0x2E  # VK_DELETE
                elif mapped_key == Key.f1:
                    return 0x70  # VK_F1
                elif mapped_key == Key.f2:
                    return 0x71  # VK_F2
                elif mapped_key == Key.f3:
                    return 0x72  # VK_F3
                elif mapped_key == Key.f4:
                    return 0x73  # VK_F4
                elif mapped_key == Key.f5:
                    return 0x74  # VK_F5
                elif mapped_key == Key.f6:
                    return 0x75  # VK_F6
                elif mapped_key == Key.f7:
                    return 0x76  # VK_F7
                elif mapped_key == Key.f8:
                    return 0x77  # VK_F8
                elif mapped_key == Key.f9:
                    return 0x78  # VK_F9
                elif mapped_key == Key.f10:
                    return 0x79  # VK_F10
                elif mapped_key == Key.f11:
                    return 0x7A  # VK_F11
                elif mapped_key == Key.f12:
                    return 0x7B  # VK_F12
                else:
                    logger.warning(f"Unmapped special key: {mapped_key}")
                    return None
            else:
                # For regular character keys
                return ord(mapped_key)
            
        # Try to find it in Key enum
        try:
            if hasattr(Key, key.lower()):
                return self._map_key_to_vk(getattr(Key, key.lower()))
        except (AttributeError, TypeError):
            pass
            
        # If we get here, we couldn't map the key
        return None

    def _key_down(self, key: str) -> bool:
        """Press a key down.
        
        Args:
            key: Key to press down
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map the key to a virtual key code
            vk_code = self._map_key_to_vk(key)
            if vk_code is None:
                logger.error(f"Could not map key '{key}' to virtual key code")
                return False
            
            # Press the key down
            keyboard_input = self._create_keyboard_input(vk_code, 0x0000)  # KEYDOWN
            result = ctypes.windll.user32.SendInput(1, ctypes.byref(keyboard_input), ctypes.sizeof(keyboard_input))
            if result != 1:
                logger.error(f"SendInput failed for key down '{key}'")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error pressing key down for key '{key}': {e}")
            return False

    def _key_up(self, key: str) -> bool:
        """Release a key.
        
        Args:
            key: Key to release
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map the key to a virtual key code
            vk_code = self._map_key_to_vk(key)
            if vk_code is None:
                logger.error(f"Could not map key '{key}' to virtual key code")
                return False
            
            # Release the key
            keyboard_input = self._create_keyboard_input(vk_code, 0x0002)  # KEYUP
            result = ctypes.windll.user32.SendInput(1, ctypes.byref(keyboard_input), ctypes.sizeof(keyboard_input))
            if result != 1:
                logger.error(f"SendInput failed for key up '{key}'")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error releasing key for key '{key}': {e}")
            return False

    def _create_keyboard_input(self, vk_code: int, flags: int) -> ctypes.Structure:
        """Create a keyboard input structure.
        
        Args:
            vk_code: Virtual key code
            flags: Flags for the input
            
        Returns:
            INPUT structure for SendInput
        """
        # Define the required ctypes structures
        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
            ]

        class HARDWAREINPUT(ctypes.Structure):
            _fields_ = [
                ("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)
            ]

        class INPUT_UNION(ctypes.Union):
            _fields_ = [
                ("mi", MOUSEINPUT),
                ("ki", KEYBDINPUT),
                ("hi", HARDWAREINPUT)
            ]

        class INPUT(ctypes.Structure):
            _fields_ = [
                ("type", ctypes.c_ulong),
                ("union", INPUT_UNION)
            ]

        # Create and configure a keyboard INPUT structure
        extra = ctypes.c_ulong(0)
        
        # Get the scan code for virtual key - this makes key recognition more reliable
        scan_code = ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)  # MAPVK_VK_TO_VSC
        
        # Create the keyboard input structure
        kb_input = KEYBDINPUT()
        kb_input.wVk = vk_code  # Virtual key code
        kb_input.wScan = scan_code  # Use proper scan code
        kb_input.dwFlags = flags
        kb_input.time = 0
        kb_input.dwExtraInfo = ctypes.pointer(extra)
        
        # Create the INPUT structure
        input_struct = INPUT()
        input_struct.type = 1  # INPUT_KEYBOARD
        input_struct.union.ki = kb_input
        
        return input_struct

    def _direct_key_press(self, key: str, duration: float = 0.1) -> bool:
        """Press a key directly using Win32 API which can be more reliable than pynput.
        
        Args:
            key: Key to press
            duration: Duration to hold key
            
        Returns:
            bool: True if successful
        """
        try:
            # Map key to VK code
            vk_code = self._map_key_to_vk(key)
            if vk_code is None:
                logger.error(f"Could not map key '{key}' to virtual key code")
                return False
                
            # Constants for key events
            KEYEVENTF_KEYDOWN = 0x0000
            KEYEVENTF_KEYUP = 0x0002
            
            # Press key down
            kb_down = self._create_keyboard_input(vk_code, KEYEVENTF_KEYDOWN)
            result_down = ctypes.windll.user32.SendInput(1, ctypes.byref(kb_down), ctypes.sizeof(kb_down))
            if result_down != 1:
                logger.error(f"SendInput failed for key down '{key}', error code: {ctypes.GetLastError()}")
                return False
                
            # Wait for specified duration
            time.sleep(duration)
            
            # Release key
            kb_up = self._create_keyboard_input(vk_code, KEYEVENTF_KEYUP)
            result_up = ctypes.windll.user32.SendInput(1, ctypes.byref(kb_up), ctypes.sizeof(kb_up))
            if result_up != 1:
                logger.error(f"SendInput failed for key up '{key}', error code: {ctypes.GetLastError()}")
                return False
                
            # Additional wait to ensure key event is processed
            time.sleep(0.05)
            
            return True
        except Exception as e:
            logger.error(f"Error in direct key press for '{key}': {e}")
            return False

    # Add a helper method to check if a key is stuck
    def check_stuck_keys(self) -> None:
        """Check if any keys are stuck down and release them."""
        # Common keys to check
        keys_to_check = [
            'a', 'w', 's', 'd', 'q', 'e', 'shift', 'ctrl', 'alt', 
            'f', 'escape', 'space', '1', '2', '3', '4', '5'
        ]
        
        for key in keys_to_check:
            try:
                vk_code = self._map_key_to_vk(key)
                if vk_code is None:
                    continue
                    
                # Check if the key is down
                key_state = ctypes.windll.user32.GetAsyncKeyState(vk_code)
                if key_state & 0x8000:  # Key is down
                    logger.warning(f"Key '{key}' appears to be stuck, releasing it")
                    self._key_up(key)
                    time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error checking stuck key '{key}': {e}") 