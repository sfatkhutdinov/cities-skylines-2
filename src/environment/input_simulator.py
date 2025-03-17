from pynput import mouse, keyboard
import time
import win32gui
import win32con
import win32api
from typing import Tuple, List, Optional
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import ctypes

# Import Win32 user32 for more direct mouse control
user32 = ctypes.WinDLL('user32', use_last_error=True)

class InputSimulator:
    def __init__(self):
        """Initialize input simulator for keyboard and mouse control."""
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        
        # Store current display resolution
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        print(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Initialize virtual key code mapping for all standard keys
        self.key_map = {
            # Function keys
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,
            
            # Special keys
            'escape': Key.esc, 'tab': Key.tab, 'capslock': Key.caps_lock,
            'shift': Key.shift, 'ctrl': Key.ctrl, 'alt': Key.alt,
            'space': Key.space, 'enter': Key.enter, 'backspace': Key.backspace,
            
            # Navigation keys
            'insert': Key.insert, 'delete': Key.delete, 'home': Key.home, 'end': Key.end,
            'pageup': Key.page_up, 'pagedown': Key.page_down,
            'left': Key.left, 'up': Key.up, 'right': Key.right, 'down': Key.down,
        }
        
        # Add letter keys
        for c in 'abcdefghijklmnopqrstuvwxyz0123456789':
            self.key_map[c] = c
            
    def find_game_window(self) -> bool:
        """Find the Cities: Skylines II window handle."""
        game_hwnd = None
        window_titles = [
            "Cities: Skylines II",
            "Cities Skylines II", 
            "Cities Skylines 2",
            "Cities: Skylines 2"
        ]
        
        def enum_windows_callback(hwnd, _):
            nonlocal game_hwnd
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                for title in window_titles:
                    if title.lower() in window_text.lower():
                        game_hwnd = hwnd
                        
                        # Get window details for debugging
                        rect = win32gui.GetWindowRect(hwnd)
                        client_rect = win32gui.GetClientRect(hwnd)
                        print(f"Found window '{window_text}' - Handle: {hwnd}")
                        print(f"Window rect: {rect}")
                        print(f"Client rect: {client_rect}")
                        
                        return False
            return True
        
        # Add error handling and retry logic for EnumWindows
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                win32gui.EnumWindows(enum_windows_callback, None)
                break  # Success, exit the loop
            except Exception as e:
                # If this is the last attempt, re-raise the exception
                if attempt == max_attempts - 1:
                    print(f"Failed to enumerate windows after {max_attempts} attempts: {str(e)}")
                    return False
                # Otherwise, wait briefly and try again
                time.sleep(0.5)
        
        if game_hwnd:
            self.game_hwnd = game_hwnd
            
            # Extra debugging info
            if hasattr(self, 'screen_capture'):
                self.screen_capture.game_hwnd = game_hwnd
                
            return True
        else:
            # Try matching partial window titles as fallback
            def fallback_enum_callback(hwnd, _):
                nonlocal game_hwnd
                if win32gui.IsWindowVisible(hwnd):
                    window_text = win32gui.GetWindowText(hwnd)
                    if "skylines" in window_text.lower() or "cities" in window_text.lower():
                        game_hwnd = hwnd
                        print(f"Found fallback window: '{window_text}'")
                        return False
                return True
                
            # Try fallback search
            win32gui.EnumWindows(fallback_enum_callback, None)
            
            if game_hwnd:
                self.game_hwnd = game_hwnd
                return True
                
        return False
        
    def ensure_game_window_focused(self) -> bool:
        """Ensure the Cities: Skylines II window is focused and ready for input.
        
        Returns:
            bool: True if window is successfully focused, False otherwise
        """
        import win32gui
        import win32con
        
        # First make sure we have the window handle
        if not hasattr(self, 'game_hwnd') or self.game_hwnd is None:
            if not self.find_game_window():
                return False
                
        # Try to focus and bring window to foreground
        try:
            # Check if window still exists
            if not win32gui.IsWindow(self.game_hwnd):
                # Window no longer exists
                self.game_hwnd = None
                return False
                
            # Show window if minimized
            if win32gui.IsIconic(self.game_hwnd):
                win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                time.sleep(0.5)  # Give time for window to restore
                
            # Bring window to foreground
            win32gui.SetForegroundWindow(self.game_hwnd)
            
            # Activate window
            win32gui.BringWindowToTop(self.game_hwnd)
            win32gui.SetActiveWindow(self.game_hwnd)
            
            # Give time for window to become active
            time.sleep(0.5)
            
            # Verify window is active
            foreground_hwnd = win32gui.GetForegroundWindow()
            if foreground_hwnd == self.game_hwnd:
                return True
            else:
                # As fallback, try Alt+Tab
                self.key_combination(['alt', 'tab'])
                time.sleep(0.5)
                return True
                
        except Exception as e:
            print(f"Error focusing window: {e}")
            
        return False
        
    def mouse_move(self, x: int, y: int, relative: bool = False, use_win32: bool = True):
        """Move mouse to specified coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            relative (bool): If True, move relative to current position
            use_win32 (bool): If True, use Win32 API directly
        """
        # Print current position for debugging
        curr_x, curr_y = win32api.GetCursorPos()
        
        # Calculate target position
        if hasattr(self, 'screen_capture') and hasattr(self.screen_capture, 'client_position'):
            # Get client area of game window
            client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
            
            if not relative:
                # For absolute positioning within game window (x,y are relative to client area)
                target_x = client_left + x
                target_y = client_top + y
            else:
                # For relative movement
                target_x = curr_x + x
                target_y = curr_y + y
                
            # Ensure coordinates stay within client area bounds
            target_x = max(client_left, min(client_right, target_x))
            target_y = max(client_top, min(client_bottom, target_y))
        else:
            # Without client area info, use screen coordinates directly
            if not relative:
                target_x = x
                target_y = y
            else:
                target_x = curr_x + x
                target_y = curr_y + y
                
            # Ensure coordinates stay within screen bounds
            target_x = max(0, min(self.screen_width - 1, target_x))
            target_y = max(0, min(self.screen_height - 1, target_y))
        
        # Use Win32 API for more direct control
        if use_win32:
            # Print movement details for debugging
            print(f"Moving mouse: {curr_x},{curr_y} -> {target_x},{target_y}")
            
            # Use Win32 API to set cursor position
            user32.SetCursorPos(int(target_x), int(target_y))
        else:
            # Use pynput as fallback
            self.mouse.position = (target_x, target_y)
            
        # Small delay to let the OS process the mouse movement
        time.sleep(0.01)
        
    def mouse_click(self, x: int, y: int, button: str = 'left', double: bool = False):
        """Perform mouse click at specified coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            button (str): 'left', 'right', or 'middle'
            double (bool): Whether to perform double click
        """
        # Move to target position using our improved mouse_move
        self.mouse_move(x, y)
        time.sleep(0.1)
        
        # Map button string to pynput Button
        button_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        btn = button_map.get(button, Button.left)
        
        # Perform click(s)
        self.mouse.click(btn, 1)
        time.sleep(0.1)
        
        if double:
            time.sleep(0.1)
            self.mouse.click(btn, 1)
            time.sleep(0.1)
            
    def mouse_drag(self, start: Tuple[int, int], end: Tuple[int, int],
                  button: str = 'left', duration: float = 0.2):
        """Perform mouse drag operation.
        
        Args:
            start (Tuple[int, int]): Starting coordinates (x, y)
            end (Tuple[int, int]): Ending coordinates (x, y)
            button (str): 'left', 'right', or 'middle'
            duration (float): Duration of drag operation in seconds
        """
        x1, y1 = start
        x2, y2 = end
        
        # Map button string to pynput Button
        button_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        btn = button_map.get(button, Button.left)
        
        # Move to start position using improved mouse_move
        self.mouse_move(x1, y1)
        time.sleep(0.1)
        
        # Print drag operation details for debugging
        print(f"Dragging: ({x1},{y1}) -> ({x2},{y2}) with {button} button")
        
        # Press button
        self.mouse.press(btn)
        time.sleep(0.1)
        
        # Smooth movement
        steps = max(5, int(duration * 60))  # At least 5 steps, up to 60 updates per second
        for i in range(1, steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            self.mouse_move(x, y)
            time.sleep(duration / steps)
            
        # Release button
        self.mouse.release(btn)
        time.sleep(0.1)
        
    def mouse_scroll(self, clicks: int):
        """Scroll the mouse wheel.
        
        Args:
            clicks (int): Number of clicks (positive for up, negative for down)
        """
        self.mouse.scroll(0, clicks)
        
    def rotate_camera(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """Rotate camera using middle mouse button drag.
        
        Args:
            start_x (int): Starting X coordinate
            start_y (int): Starting Y coordinate
            end_x (int): Ending X coordinate
            end_y (int): Ending Y coordinate
            duration (float): Duration of rotation operation in seconds
        """
        # Print rotation operation details
        print(f"Rotating camera: ({start_x},{start_y}) -> ({end_x},{end_y})")
        
        # Move to start position
        self.mouse_move(start_x, start_y)
        time.sleep(0.1)
        
        # Press middle button
        self.mouse.press(Button.middle)
        time.sleep(0.1)
        
        # Smooth movement for rotation
        steps = max(5, int(duration * 60))  # At least 5 steps
        for i in range(1, steps + 1):
            t = i / steps
            x = int(start_x + (end_x - start_x) * t)
            y = int(start_y + (end_y - start_y) * t)
            self.mouse_move(x, y)
            time.sleep(duration / steps)
            
        # Release middle button
        self.mouse.release(Button.middle)
        time.sleep(0.1)
        
    def pan_camera(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5):
        """Pan camera using right mouse button drag.
        
        Args:
            start_x (int): Starting X coordinate
            start_y (int): Starting Y coordinate
            end_x (int): Ending X coordinate
            end_y (int): Ending Y coordinate
            duration (float): Duration of panning operation in seconds
        """
        # Print panning operation details
        print(f"Panning camera: ({start_x},{start_y}) -> ({end_x},{end_y})")
        
        # Move to start position
        self.mouse_move(start_x, start_y)
        time.sleep(0.1)
        
        # Press right button
        self.mouse.press(Button.right)
        time.sleep(0.1)
        
        # Smooth movement for panning
        steps = max(5, int(duration * 60))  # At least 5 steps
        for i in range(1, steps + 1):
            t = i / steps
            x = int(start_x + (end_x - start_x) * t)
            y = int(start_y + (end_y - start_y) * t)
            self.mouse_move(x, y)
            time.sleep(duration / steps)
            
        # Release right button
        self.mouse.release(Button.right)
        time.sleep(0.1)
        
    def key_press(self, key: str, duration: float = 0.1):
        """Press and hold a keyboard key.
        
        Args:
            key (str): Key name from key_map
            duration (float): Duration to hold the key in seconds
        """
        key_code = self.key_map.get(key.lower())
        if key_code is not None:
            self.keyboard.press(key_code)
            time.sleep(duration)
            self.keyboard.release(key_code)
            
    def key_combination(self, keys: List[str], duration: float = 0.1):
        """Press a combination of keys simultaneously.
        
        Args:
            keys (List[str]): List of key names from key_map
            duration (float): Duration to hold the keys in seconds
        """
        # Safety check: prevent dangerous key combinations
        dangerous_combinations = [
            ['alt', 'tab'],
            ['alt', 'f4'],
            ['ctrl', 'w'],
            ['alt', 'escape'],
            ['ctrl', 'alt', 'delete']
        ]
        
        # Check if requested combination contains a dangerous pattern
        for dangerous_combo in dangerous_combinations:
            if all(k.lower() in [key.lower() for key in keys] for k in dangerous_combo):
                print(f"WARNING: Blocked dangerous key combination: {keys}")
                return False
        
        # Press all keys
        pressed_keys = []
        for key in keys:
            key_code = self.key_map.get(key.lower())
            if key_code is not None:
                self.keyboard.press(key_code)
                pressed_keys.append(key_code)
                
        time.sleep(duration)
        
        # Release all keys in reverse order
        for key_code in reversed(pressed_keys):
            self.keyboard.release(key_code)
            
    def press_key(self, key: str, duration: float = 0.1):
        """Press a key with a short duration.
        
        Args:
            key (str): Key to press
            duration (float): How long to hold the key
        """
        # Prevent ESC key from being pressed
        if key.lower() in ['escape', 'esc']:
            print("WARNING: Blocked ESC key press")
            return False
            
        try:
            # Get the key from the map
            if key in self.key_map:
                k = self.key_map[key]
            else:
                # If not in map, try direct key
                k = key
                
            # Press and release with specified duration
            self.keyboard.press(k)
            time.sleep(duration)
            self.keyboard.release(k)
            time.sleep(0.05)  # Small delay after release
            return True
        except Exception as e:
            print(f"Error pressing key {key}: {str(e)}")
            return False
            
    def close(self):
        """Clean up resources and release all keys/buttons."""
        # Release all mouse buttons just in case
        try:
            self.mouse.release(Button.left)
            self.mouse.release(Button.right)
            self.mouse.release(Button.middle)
        except Exception as e:
            print(f"Error releasing mouse buttons: {e}")
            
        # Release common keyboard keys
        for key in ["shift", "ctrl", "alt", "space"]:
            try:
                if key in self.key_map:
                    self.keyboard.release(self.key_map[key])
            except Exception as e:
                print(f"Error releasing key {key}: {e}")
                
        print("Input simulator resources released") 