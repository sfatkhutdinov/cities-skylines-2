from pynput import mouse, keyboard
import time
import win32gui
import win32con
import win32api
from typing import Tuple, List, Optional
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController
import ctypes
import pyautogui
import torch
import logging

# Import Win32 user32 for more direct mouse control
user32 = ctypes.WinDLL('user32', use_last_error=True)
logger = logging.getLogger(__name__)

class InputSimulator:
    def __init__(self):
        """Initialize input simulator for keyboard and mouse control."""
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.screen_capture = None  # Will be set by the environment
        
        # Store current display resolution
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        logger.info(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
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
            
        # Store game window information
        self.game_hwnd = None
        self.game_rect = None
        self.client_rect = None
        
        # Block escape key by default to prevent accidental menu toggling
        self.block_escape = True
        
        # Movement speed for camera control
        self.movement_speed = 1.0
        
    def find_game_window(self) -> bool:
        """Find the Cities: Skylines II window handle."""
        game_hwnd = None
        window_titles = [
            "Cities: Skylines II",
            "Cities Skylines II",
            "Cities: Skylines",
            "Cities Skylines"
        ]
        
        print("Searching for Cities: Skylines II window...")
        
        def enum_windows_callback(hwnd, _):
            nonlocal game_hwnd
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                
                # Skip any windows that look like text editors/IDEs
                if ("cities skylines" in window_text.lower() and 
                    any(editor in window_text.lower() for editor in 
                    ["cursor", "vscode", "editor", ".py", "code", "notepad"])):
                    print(f"Skipping editor window: '{window_text}'")
                    return True
                
                print(f"Found window: '{window_text}'")
                for title in window_titles:
                    if title.lower() in window_text.lower():
                        game_hwnd = hwnd
                        
                        # Get window details for debugging
                        rect = win32gui.GetWindowRect(hwnd)
                        client_rect = win32gui.GetClientRect(hwnd)
                        print(f"Found game window '{window_text}' - Handle: {hwnd}")
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
            print("Game window not found in first pass. Trying fallback approach...")
            # Try matching partial window titles as fallback
            def fallback_enum_callback(hwnd, _):
                nonlocal game_hwnd
                if win32gui.IsWindowVisible(hwnd):
                    window_text = win32gui.GetWindowText(hwnd)
                    if ("skylines" in window_text.lower() or 
                        "cities" in window_text.lower() or
                        "colossal order" in window_text.lower() or
                        "paradox" in window_text.lower()):
                        game_hwnd = hwnd
                        print(f"Found fallback game window: '{window_text}' - Handle: {hwnd}")
                        return False
                return True
                
            # Try fallback search
            try:
                win32gui.EnumWindows(fallback_enum_callback, None)
            except Exception as e:
                print(f"Error in fallback window search: {str(e)}")
            
            if game_hwnd:
                self.game_hwnd = game_hwnd
                return True
            
            print("Game window not found. Make sure Cities: Skylines II is running and visible.")
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
            print("No game window handle found. Attempting to find game window...")
            if not self.find_game_window():
                print("Could not find game window. Make sure Cities: Skylines II is running.")
                return False
                
        # Try multiple methods to focus window
        success = False
        methods_tried = []
        
        # Method 1: Standard SetForegroundWindow
        try:
            # Check if window still exists
            if not win32gui.IsWindow(self.game_hwnd):
                print("Window no longer exists. Attempting to find game window again...")
                self.game_hwnd = None
                return self.ensure_game_window_focused()  # Try again from the beginning
                
            # Show window if minimized
            if win32gui.IsIconic(self.game_hwnd):
                print("Game window is minimized. Restoring...")
                win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                time.sleep(0.5)  # Give time for window to restore
            
            # Method 1: Try SetForegroundWindow
            print("Attempting to focus window with SetForegroundWindow...")
            win32gui.SetForegroundWindow(self.game_hwnd)
            methods_tried.append("SetForegroundWindow")
            
            # Verify success
            time.sleep(0.3)
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                print("Successfully focused game window")
                return True
        except Exception as e:
            print(f"Error using SetForegroundWindow: {e}")
        
        # Method 2: Try BringWindowToTop
        try:
            print("Attempting to focus window with BringWindowToTop...")
            win32gui.BringWindowToTop(self.game_hwnd)
            methods_tried.append("BringWindowToTop")
            
            # Verify success
            time.sleep(0.3)
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                print("Successfully focused game window")
                return True
        except Exception as e:
            print(f"Error using BringWindowToTop: {e}")
        
        # Method 3: Try SetActiveWindow
        try:
            print("Attempting to focus window with SetActiveWindow...")
            win32gui.SetActiveWindow(self.game_hwnd)
            methods_tried.append("SetActiveWindow")
            
            # Verify success
            time.sleep(0.3)
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                print("Successfully focused game window")
                return True
        except Exception as e:
            print(f"Error using SetActiveWindow: {e}")
        
        # Method 4: Alt+Tab as a fallback approach
        print("All standard methods failed. Trying Alt+Tab as fallback...")
        try:
            self.key_combination(['alt', 'tab'], allow_focus_keys=True)
            time.sleep(0.5)
            methods_tried.append("Alt+Tab")
            
            # Verify current foreground window
            foreground_hwnd = win32gui.GetForegroundWindow()
            foreground_title = win32gui.GetWindowText(foreground_hwnd)
            print(f"Current foreground window: '{foreground_title}'")
            
            if foreground_hwnd == self.game_hwnd:
                print("Alt+Tab successfully focused game window")
                return True
        except Exception as e:
            print(f"Error using Alt+Tab fallback: {e}")
        
        # Fallback: return False since we couldn't properly focus the window
        print(f"WARNING: Failed to focus game window after trying methods: {', '.join(methods_tried)}")
        
        # Check if window still exists at least
        if win32gui.IsWindow(self.game_hwnd):
            # If the window exists but we can't focus it, try once more
            try:
                win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                time.sleep(0.5)
                win32gui.SetForegroundWindow(self.game_hwnd)
                time.sleep(0.2)
                if win32gui.GetForegroundWindow() == self.game_hwnd:
                    return True
            except:
                pass
        else:
            # Window no longer exists, reset the handle
            self.game_hwnd = None
            print("Game window no longer exists. Will attempt to find it again next time.")
            
        # Return False to indicate failure
        return False
        
    def mouse_move(self, x: int, y: int, relative: bool = False, use_win32: bool = True):
        """Move the mouse to a position.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            relative (bool): If True, move relative to current position
            use_win32 (bool): If True, use win32api for direct positioning
        """
        # First call ensure_game_window_focused() to make sure we interact with the game
        self.ensure_game_window_focused()
        
        # Get current mouse position
        current_x, current_y = win32api.GetCursorPos()
        
        # Print movement
        if not relative:
            print(f"Moving mouse: {current_x},{current_y} -> {x},{y}")
        
        # Verify the coordinates are within screen bounds
        try:
            screen_width = win32api.GetSystemMetrics(0)
            screen_height = win32api.GetSystemMetrics(1)
            
            # Adjust to be within screen bounds
            if not relative:
                x = max(0, min(x, screen_width - 1))
                y = max(0, min(y, screen_height - 1))
        except Exception as e:
            print(f"Warning: Error getting screen metrics: {e}")
        
        # For absolute positioning, use direct win32 API for reliability
        if use_win32 and not relative:
            try:
                win32api.SetCursorPos((x, y))
                # Verify position after setting
                for attempt in range(3):  # Try up to 3 times
                    time.sleep(0.05)
                    new_x, new_y = win32api.GetCursorPos()
                    if abs(new_x - x) <= 3 and abs(new_y - y) <= 3:
                        # Position is close enough
                        break
                    # Try again
                    win32api.SetCursorPos((x, y))
                return
            except Exception as e:
                print(f"Win32 mouse positioning error: {e}")
                # Fall back to alternative methods
        
        try:
            # Determine position
            if relative:
                # Get current position
                current_x, current_y = win32api.GetCursorPos()
                target_x = current_x + x
                target_y = current_y + y
            else:
                target_x, target_y = x, y
                
            # Move mouse with pyautogui for fallback
            pyautogui.moveTo(target_x, target_y, duration=0.1)
            
            # Verify final position
            final_x, final_y = win32api.GetCursorPos()
            if abs(final_x - target_x) > 5 or abs(final_y - target_y) > 5:
                # If position is significantly off, try direct win32 setting as last resort
                try:
                    win32api.SetCursorPos((target_x, target_y))
                except:
                    pass
                    
        except Exception as e:
            print(f"Error moving mouse: {e}")
            # Try direct win32 as last resort
            try:
                win32api.SetCursorPos((x, y))
            except:
                pass
        
    def mouse_click(self, x: int, y: int, button: str = 'left', double: bool = False) -> bool:
        """Perform mouse click at specified coordinates.
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            button (str): 'left', 'right', or 'middle'
            double (bool): Whether to perform double click
            
        Returns:
            bool: True if click was successful, False otherwise
        """
        # First ensure the game window is focused
        if not self.ensure_game_window_focused():
            print(f"Failed to focus game window before mouse click at ({x},{y})")
            return False
            
        # Move to target position using our improved mouse_move
        try:
            self.mouse_move(x, y)
            time.sleep(0.1)
            
            # Verify cursor position after move
            current_x, current_y = win32api.GetCursorPos()
            if abs(current_x - x) > 5 or abs(current_y - y) > 5:
                print(f"Mouse position verification failed: requested ({x},{y}), got ({current_x},{current_y})")
                # Try one more time
                self.mouse_move(x, y)
                time.sleep(0.1)
                current_x, current_y = win32api.GetCursorPos()
                if abs(current_x - x) > 5 or abs(current_y - y) > 5:
                    print(f"Mouse position verification failed again, proceeding anyway")
            
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
                
            return True
        except Exception as e:
            print(f"Error during mouse click at ({x},{y}): {e}")
            return False
        
    def mouse_drag(self, start: Tuple[int, int], end: Tuple[int, int],
                  button: str = 'left', duration: float = 0.2) -> bool:
        """Perform mouse drag operation.
        
        Args:
            start (Tuple[int, int]): Starting coordinates (x, y)
            end (Tuple[int, int]): Ending coordinates (x, y)
            button (str): 'left', 'right', or 'middle'
            duration (float): Duration of drag operation in seconds
            
        Returns:
            bool: True if drag was successful, False otherwise
        """
        # First ensure the game window is focused
        if not self.ensure_game_window_focused():
            print(f"Failed to focus game window before mouse drag")
            return False
            
        x1, y1 = start
        x2, y2 = end
        
        try:
            # Map button string to pynput Button
            button_map = {
                'left': Button.left,
                'right': Button.right,
                'middle': Button.middle
            }
            btn = button_map.get(button, Button.left)
            
            # Verify coordinates are within screen bounds
            screen_width, screen_height = self.get_screen_dimensions()
            if not (0 <= x1 < screen_width and 0 <= y1 < screen_height and
                    0 <= x2 < screen_width and 0 <= y2 < screen_height):
                print(f"Warning: Drag coordinates out of bounds. Screen: {screen_width}x{screen_height}, " +
                      f"Start: ({x1},{y1}), End: ({x2},{y2})")
                # Adjust coordinates to be within bounds
                x1 = max(0, min(x1, screen_width - 1))
                y1 = max(0, min(y1, screen_height - 1))
                x2 = max(0, min(x2, screen_width - 1))
                y2 = max(0, min(y2, screen_height - 1))
            
            # Move to start position using improved mouse_move
            self.mouse_move(x1, y1)
            time.sleep(0.1)
            
            # Verify start position
            current_x, current_y = win32api.GetCursorPos()
            if abs(current_x - x1) > 5 or abs(current_y - y1) > 5:
                print(f"Mouse drag start position verification failed: requested ({x1},{y1}), got ({current_x},{current_y})")
                # Try one more time
                self.mouse_move(x1, y1)
                time.sleep(0.1)
            
            # Print drag operation details for debugging
            print(f"Dragging: ({x1},{y1}) -> ({x2},{y2}) with {button} button")
            
            # Press button
            self.mouse.press(btn)
            time.sleep(0.1)
            
            # Smooth movement with more steps for precision
            steps = max(10, int(duration * 120))  # At least 10 steps, up to 120 updates per second
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            # If distance is very short, use fewer steps
            if distance < 50:
                steps = max(5, steps // 2)
                
            for i in range(1, steps + 1):
                t = i / steps
                # Use a smoother easing function for more natural movement
                # This uses an ease-in-out curve
                if t < 0.5:
                    ease_t = 2 * t * t
                else:
                    ease_t = -1 + (4 - 2 * t) * t
                
                x = int(x1 + (x2 - x1) * ease_t)
                y = int(y1 + (y2 - y1) * ease_t)
                self.mouse_move(x, y)
                # Adjust timing based on step count to maintain total duration
                time.sleep(duration / steps)
                
            # Release button
            self.mouse.release(btn)
            time.sleep(0.1)
            
            # Verify end position as a final check
            current_x, current_y = win32api.GetCursorPos()
            end_pos_success = (abs(current_x - x2) <= 10 and abs(current_y - y2) <= 10)
            if not end_pos_success:
                print(f"Mouse drag end position verification: requested ({x2},{y2}), got ({current_x},{current_y})")
                
            return True
        except Exception as e:
            print(f"Error during mouse drag {start} -> {end}: {e}")
            # Make sure to release the button on error
            try:
                button_map = {
                    'left': Button.left,
                    'right': Button.right,
                    'middle': Button.middle
                }
                btn = button_map.get(button, Button.left)
                self.mouse.release(btn)
            except:
                pass
            return False
        
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
        
    def key_press(self, key: str, duration: float = 0.1) -> bool:
        """Press a key with a short duration.
        
        Args:
            key (str): Key to press
            duration (float): How long to hold the key
            
        Returns:
            bool: True if successful, False otherwise
        """
        # First ensure the game window is focused
        if not self.ensure_game_window_focused():
            print(f"Failed to focus game window before key press: {key}")
            # We'll still attempt the key press, but log the issue
        
        # Check for escape key with improved safety logic
        if key.lower() in ['escape', 'esc']:
            if hasattr(self, 'block_escape') and self.block_escape:
                print("WARNING: Blocked ESC key press - using safe menu handling instead")
                return self.safe_menu_handling()
        
        # Check for potentially problematic key presses and block them
        dangerous_keys = ['f4', 'alt+f4', 'ctrl+w', 'alt+tab']
        if key.lower() in dangerous_keys:
            print(f"WARNING: Blocked dangerous key press: {key}")
            return False
        
        try:
            # Get the key from the map
            if key in self.key_map:
                k = self.key_map[key]
            else:
                # If not in map, try direct key
                k = key
                
            # Log the key press action
            print(f"Pressing key: {key} for {duration:.2f}s")
            
            # Press and release with specified duration
            self.keyboard.press(k)
            
            # Use an adaptive timeout based on duration
            # For very short presses (< 0.1s), use a minimum of 0.05s
            actual_duration = max(0.05, duration)
            time.sleep(actual_duration)
            
            self.keyboard.release(k)
            time.sleep(0.05)  # Small delay after release
            
            # Verify that the key was released properly
            try:
                # For most keys we can't directly verify, but we can make sure the keyboard state is reset
                self.keyboard.release(k)  # Try releasing again to ensure it's released
            except:
                pass
            
            return True
        except Exception as e:
            print(f"Error pressing key {key}: {str(e)}")
            
            # Try to ensure key is released on error
            try:
                if key in self.key_map:
                    self.keyboard.release(self.key_map[key])
            except:
                pass
                
            return False
            
    def safe_menu_handling(self) -> bool:
        """Handle being stuck in a menu using a safe sequence of escape and clicks.
        
        Returns:
            bool: True if the menu was likely handled successfully
        """
        logger.info("Attempting safe menu handling routine")
        
        # Safety mechanism: try to close any menu by pressing ESC and clicking center
        success = False
        
        # Sequence 1: Press ESC a few times with delays
        logger.info("Trying escape key sequence")
        for _ in range(3):
            self.key_press("escape", duration=0.1)
            time.sleep(0.5)
        
        # Get screen dimensions
        width, height = self._get_screen_dimensions()
        center_x, center_y = width // 2, height // 2
        
        # Sequence 2: Click center of screen
        logger.info(f"Clicking center of screen ({center_x}, {center_y})")
        self.mouse_click(center_x, center_y)
        time.sleep(0.5)
        
        # Sequence 3: Try clicking at known menu button locations
        self._click_common_menu_buttons()
        
        # Sequence 4: Try pressing common menu navigation keys
        logger.info("Trying menu navigation keys")
        for key in ["tab", "enter", "space"]:
            self.key_press(key, duration=0.1)
            time.sleep(0.3)
        
        # Sequence 5: More aggressive approach - click in grid pattern
        if not success:
            logger.info("Trying grid pattern clicks")
            self._click_grid_pattern()
            
        return True
        
    def _click_common_menu_buttons(self) -> None:
        """Click at positions where common menu buttons are typically located."""
        # Get screen dimensions
        width, height = self._get_screen_dimensions()
        
        # Define common button positions (based on typical menu layouts)
        # Each position is a tuple of (x_ratio, y_ratio, description)
        button_positions = [
            # Resume button (center top)
            (0.5, 0.25, "center top - resume game"),
            # Center options
            (0.5, 0.5, "center - main button"),
            # Lower buttons
            (0.5, 0.75, "center bottom - confirm/ok"),
            # Left menu buttons
            (0.2, 0.3, "left menu - top item"),
            (0.2, 0.4, "left menu - second item"),
            (0.2, 0.5, "left menu - third item"),
            # Close buttons (top right)
            (0.95, 0.05, "top right - close button"),
            # Back button (bottom left)
            (0.1, 0.9, "bottom left - back button")
        ]
        
        # Click each position with a delay
        for x_ratio, y_ratio, desc in button_positions:
            x = int(width * x_ratio)
            y = int(height * y_ratio)
            logger.info(f"Clicking {desc} at ({x}, {y})")
            self.mouse_click(x, y)
            time.sleep(0.5)
            
    def _click_grid_pattern(self) -> None:
        """Click in a grid pattern to try to hit any interactive elements."""
        # Get screen dimensions
        width, height = self._get_screen_dimensions()
        
        # Define grid size
        grid_cols, grid_rows = 4, 3
        
        # Calculate spacing
        x_spacing = width // (grid_cols + 1)
        y_spacing = height // (grid_rows + 1)
        
        # Click at each grid intersection
        for row in range(1, grid_rows + 1):
            for col in range(1, grid_cols + 1):
                x = col * x_spacing
                y = row * y_spacing
                logger.info(f"Grid click at ({x}, {y})")
                self.mouse_click(x, y)
                time.sleep(0.3)
                
    def _get_screen_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the game screen.
        
        Returns:
            Tuple[int, int]: Width and height of the screen
        """
        # Use client position if available
        if hasattr(self, 'screen_capture') and self.screen_capture:
            if hasattr(self.screen_capture, 'client_position') and self.screen_capture.client_position:
                client_left, client_top, client_right, client_bottom = self.screen_capture.client_position
                return (client_right - client_left, client_bottom - client_top)
        
        # Fallback to system metrics
        return (self.screen_width, self.screen_height)
        
    def handle_menu_recovery(self, retries: int = 3) -> bool:
        """Comprehensive menu recovery with multiple strategies.
        
        Args:
            retries (int): Number of recovery attempts to make
            
        Returns:
            bool: True if recovery was likely successful
        """
        logger.info(f"Attempting menu recovery with {retries} retries")
        
        # Make sure game window is focused first
        if not self.ensure_game_window_focused():
            logger.warning("Failed to focus game window for menu recovery")
            return False
            
        # Try different recovery strategies in sequence
        for attempt in range(retries):
            logger.info(f"Menu recovery attempt {attempt+1}/{retries}")
            
            # Strategy 1: Standard ESC sequence
            self.key_press("escape", duration=0.1)
            time.sleep(0.5)
            self.key_press("escape", duration=0.1)
            time.sleep(0.5)
            
            # Strategy 2: Click known resume button location
            width, height = self._get_screen_dimensions()
            # Resume button is typically at (720, 513) in 1920x1080
            x_ratio, y_ratio = 720/1920, 513/1080
            x = int(width * x_ratio)
            y = int(height * y_ratio)
            
            logger.info(f"Clicking resume button at ({x}, {y})")
            self.mouse_click(x, y)
            time.sleep(1.0)
            
            # Strategy 3: Try common keys that dismiss dialog boxes
            for key in ["enter", "space", "y", "escape"]:
                self.key_press(key, duration=0.1)
                time.sleep(0.5)
                
            # Strategy 4: Click in center of screen
            self.mouse_click(width // 2, height // 2)
            time.sleep(0.5)
            
            # Strategy 5: Full safe menu handling
            if attempt == retries - 1:  # On last attempt
                self.safe_menu_handling()
                
            # Add a longer delay between retry attempts
            time.sleep(1.0)
            
        return True  # Assume we've done our best

    def get_screen_dimensions(self):
        """Get the dimensions of the game window or screen.
        
        Returns:
            tuple: (width, height)
        """
        if self.client_rect:
            width = self.client_rect[2] - self.client_rect[0]
            height = self.client_rect[3] - self.client_rect[1]
            return width, height
        else:
            # Fallback to primary monitor resolution
            return win32api.GetSystemMetrics(0), win32api.GetSystemMetrics(1)
            
    def key_combination(self, keys: List[str], duration: float = 0.1, allow_focus_keys: bool = False) -> bool:
        """Press a combination of keys simultaneously.
        
        Args:
            keys (List[str]): List of key names from key_map
            duration (float): Duration to hold the keys in seconds
            allow_focus_keys (bool): If True, allow certain dangerous combinations like Alt+Tab when used for window focusing
            
        Returns:
            bool: True if successful, False otherwise
        """
        # First ensure the game window is focused (unless this is a focus operation)
        if not allow_focus_keys and not self.ensure_game_window_focused():
            print(f"Failed to focus game window before key combination: {keys}")
            # We'll still attempt the key combo, but log the issue
        
        # Safety check: prevent dangerous key combinations
        dangerous_combinations = [
            ['alt', 'tab'],
            ['alt', 'f4'],
            ['ctrl', 'w'],
            ['alt', 'escape'],
            ['ctrl', 'alt', 'delete'],
            ['windows', 'l'],
            ['windows', 'd'],
            ['windows', 'e'],
            ['ctrl', 'shift', 'esc']
        ]
        
        # Convert all keys to lowercase for consistent comparison
        lower_keys = [k.lower() for k in keys]
        
        # If allow_focus_keys is True, we won't block alt+tab specifically when used for window focusing
        if allow_focus_keys and len(keys) == 2 and 'alt' in lower_keys and 'tab' in lower_keys:
            print("Allowing Alt+Tab for window focusing")
        else:
            # Check if requested combination contains a dangerous pattern
            for dangerous_combo in dangerous_combinations:
                # Convert to set for easier subset checking
                combo_set = set(dangerous_combo)
                keys_set = set(lower_keys)
                
                if combo_set.issubset(keys_set):
                    print(f"WARNING: Blocked dangerous key combination: {keys}")
                    return False
        
        try:
            # Log the key combination
            print(f"Pressing key combination: {'+'.join(keys)} for {duration:.2f}s")
            
            # Press all keys
            pressed_keys = []
            for key in keys:
                key_code = self.key_map.get(key.lower())
                if key_code is not None:
                    self.keyboard.press(key_code)
                    pressed_keys.append(key_code)
                    # Small delay between presses to ensure they register
                    time.sleep(0.02)
                else:
                    print(f"Warning: Key '{key}' not found in key map, skipping")
                    
            # Use an adaptive timeout based on duration
            # For very short presses (< 0.1s), use a minimum of 0.05s
            actual_duration = max(0.05, duration)
            time.sleep(actual_duration)
            
            # Release all keys in reverse order
            for key_code in reversed(pressed_keys):
                self.keyboard.release(key_code)
                # Small delay between releases
                time.sleep(0.02)
                
            # Additional safety: ensure all keys are released by releasing them again
            time.sleep(0.05)
            for key_code in pressed_keys:
                try:
                    self.keyboard.release(key_code)
                except:
                    pass
                    
            return True
        except Exception as e:
            print(f"Error executing key combination {keys}: {e}")
            
            # Emergency release of all keys on error
            for key in keys:
                try:
                    key_code = self.key_map.get(key.lower())
                    if key_code is not None:
                        self.keyboard.release(key_code)
                except:
                    pass
                    
            return False
            
    def press_key(self, key: str):
        """Press a key without releasing it.
        
        Args:
            key (str): Key to press down and hold
        """
        # Map string key to pynput Key if needed
        mapped_key = self.key_map.get(key, key)
        
        try:
            # Press the key
            self.keyboard.press(mapped_key)
        except Exception as e:
            print(f"Error pressing key {key}: {e}")
    
    def release_key(self, key: str):
        """Release a previously pressed key.
        
        Args:
            key (str): Key to release
        """
        # Map string key to pynput Key if needed
        mapped_key = self.key_map.get(key, key)
        
        try:
            # Release the key
            self.keyboard.release(mapped_key)
        except Exception as e:
            print(f"Error releasing key {key}: {e}")
            
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

    def press_mouse_button(self, button: str = 'left'):
        """Press a mouse button without releasing it.
        
        Args:
            button (str): 'left', 'right', or 'middle'
        """
        # Map button string to pynput Button
        button_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        btn = button_map.get(button, Button.left)
        
        # Press button
        self.mouse.press(btn)
        
    def release_mouse_button(self, button: str = 'left'):
        """Release a previously pressed mouse button.
        
        Args:
            button (str): 'left', 'right', or 'middle'
        """
        # Map button string to pynput Button
        button_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        btn = button_map.get(button, Button.left)
        
        # Release button
        self.mouse.release(btn)

    # Cities: Skylines 2 specific helper methods
    
    def cs2_toggle_pause(self) -> bool:
        """Toggle game pause state.
        
        Returns:
            bool: Success status
        """
        return self.key_press('space')
        
    def cs2_change_game_speed(self, speed: int) -> bool:
        """Change game speed in Cities Skylines 2.
        
        Args:
            speed (int): Speed level (0-4)
                0 = Pause
                1 = Normal
                2 = Fast
                3 = Faster
                4 = Fastest
                
        Returns:
            bool: Success flag
        """
        if not self.ensure_game_window_focused():
            logger.warning("Failed to focus game window for speed change")
            return False
            
        # Map speed level to key
        speed_keys = {
            0: "0",  # Pause
            1: "1",  # Normal
            2: "2",  # Fast 
            3: "3",  # Faster
            4: "4"   # Fastest
        }
        
        # Ensure speed is within valid range
        speed = max(0, min(4, speed))
        
        # Get the corresponding key
        key = speed_keys.get(speed, "1")  # Default to normal speed
        
        # Press the key to change speed
        success = self.key_press(key, duration=0.1)
        
        if success:
            logger.info(f"Changed game speed to level {speed}")
        else:
            logger.warning(f"Failed to change game speed to level {speed}")
            
        return success
        
    def cs2_toggle_info_view(self, view_type: str) -> bool:
        """Toggle an information view.
        
        Args:
            view_type (str): Type of view ('economy', 'progression', 'transport', etc.)
            
        Returns:
            bool: Success status
        """
        view_keys = {
            'progression': 'p',
            'economy': 'z', 
            'transportation': 'x',
            'information': 'c',
            'statistics': 'v',
            'map': 'm'
        }
        
        if view_type.lower() in view_keys:
            return self.key_press(view_keys[view_type.lower()])
        else:
            print(f"Unknown info view: {view_type}")
            return False
            
    def cs2_bulldoze(self) -> bool:
        """Activate bulldoze tool.
        
        Returns:
            bool: Success status
        """
        return self.key_press('b')
        
    def cs2_select_build_tool(self, tool: str) -> bool:
        """Select a building tool.
        
        Args:
            tool (str): Tool name ('road', 'zone', 'water', etc.)
            
        Returns:
            bool: Success status
        """
        # First ensure we're in the main game view
        if not self.ensure_game_window_focused():
            return False
            
        # Map of common tools and their typical UI locations (as ratios of screen dimensions)
        tool_positions = {
            'road': (0.1, 0.1),           # Top left menu, first icon
            'zone_residential': (0.1, 0.2),  # Top left, second row
            'zone_commercial': (0.1, 0.25),
            'zone_industrial': (0.1, 0.3),
            'water': (0.1, 0.4),
            'electricity': (0.1, 0.5),
            'services': (0.1, 0.6)
        }
        
        if tool.lower() in tool_positions:
            # Get screen dimensions
            width, height = self.get_screen_dimensions()
            
            # Calculate pixel positions
            x, y = tool_positions[tool.lower()]
            x_px = int(x * width)
            y_px = int(y * height)
            
            # Click on the tool icon
            return self.mouse_click(x_px, y_px)
        else:
            print(f"Unknown building tool: {tool}")
            return False
            
    def cs2_save_game(self) -> bool:
        """Quick save the game.
        
        Returns:
            bool: Success status
        """
        return self.key_press('f5')
        
    def cs2_load_game(self) -> bool:
        """Quick load the game.
        
        Returns:
            bool: Success status
        """
        # Be very careful with load operations - don't want to lose progress
        print("WARNING: Load game operation requested")
        return False  # Disabled for safety - require explicit UI navigation

    def verify_action_success(self, action_name: str, timeout: float = 2.0) -> bool:
        """Verify that an action was successful by checking for visual changes.
        
        This method requires that the input simulator has access to the screen_capture object.
        
        Args:
            action_name (str): Name of the action to verify (for logging)
            timeout (float): Maximum time to wait for visual confirmation in seconds
            
        Returns:
            bool: True if action appears successful, False otherwise
        """
        # Check if we have access to screen capture
        if not hasattr(self, 'screen_capture'):
            print(f"Cannot verify action '{action_name}' - no screen capture available")
            return True  # Assume success if we can't verify
        
        try:
            # Capture before state if not already captured
            if not hasattr(self, 'before_action_frame'):
                # Store current frame for comparison
                self.before_action_frame = self.screen_capture.capture_frame()
                print(f"Stored pre-action state for '{action_name}'")
                return True  # This is just initialization
                
            # Wait briefly for the game to respond
            time.sleep(0.2)
            
            # Capture the current frame
            current_frame = self.screen_capture.capture_frame()
            
            # Compare with before state
            if torch.is_tensor(current_frame) and torch.is_tensor(self.before_action_frame):
                # Calculate frame difference
                diff = torch.abs(current_frame - self.before_action_frame).mean().item()
                
                # Clear the stored frame to avoid reusing it
                del self.before_action_frame
                
                # Check if there was a significant change
                change_threshold = 0.01  # Adjust based on testing
                if diff > change_threshold:
                    print(f"Action '{action_name}' verified - detected visual change: {diff:.4f}")
                    return True
                else:
                    print(f"Action '{action_name}' may have failed - minimal visual change: {diff:.4f}")
                    return False
            else:
                print(f"Could not compare frames for action '{action_name}' - incompatible types")
                return True  # Assume success if we can't verify
                
        except Exception as e:
            print(f"Error verifying action '{action_name}': {e}")
            # Clean up if needed
            if hasattr(self, 'before_action_frame'):
                del self.before_action_frame
            return True  # Assume success on error
            
    def retry_on_failure(self, action_func, action_name: str, max_attempts: int = 3, *args, **kwargs):
        """Execute an action and retry if it fails.
        
        Args:
            action_func: The function to call
            action_name (str): Name of the action for logging
            max_attempts (int): Maximum number of retry attempts
            *args, **kwargs: Arguments to pass to the action function
            
        Returns:
            The result of the action function, or False if all attempts fail
        """
        # Store a frame before the action
        if hasattr(self, 'screen_capture'):
            self.before_action_frame = self.screen_capture.capture_frame()
            
        for attempt in range(max_attempts):
            # Execute the action
            result = action_func(*args, **kwargs)
            
            # If action failed in its own error handling, retry
            if not result:
                print(f"Action '{action_name}' failed on attempt {attempt+1}/{max_attempts}")
                time.sleep(0.5)  # Wait before retry
                continue
                
            # Verify success with visual feedback
            if self.verify_action_success(action_name):
                if attempt > 0:
                    print(f"Action '{action_name}' succeeded on attempt {attempt+1}")
                return result
            elif attempt < max_attempts - 1:
                print(f"Action '{action_name}' verification failed, retrying ({attempt+1}/{max_attempts})")
                time.sleep(0.5)  # Wait before retry
                
        # If we get here, all attempts failed
        print(f"Action '{action_name}' failed after {max_attempts} attempts")
        return False

    def connect_to_reward_system(self, reward_system):
        """Connect this input simulator to a reward system for reinforcement learning.
        
        Args:
            reward_system: The reward system object to use for feedback
        """
        self.reward_system = reward_system
        print(f"Input simulator connected to reward system")
        
    def execute_with_feedback(self, action_func, action_name: str, action_type: str, *args, **kwargs) -> Tuple[bool, float]:
        """Execute an action and get reward feedback for reinforcement learning.
        
        This method requires the input simulator to be connected to a reward system.
        
        Args:
            action_func: The function to call
            action_name (str): Name of the action for logging
            action_type (str): Type of action for reward calculation 
                              ('ui', 'camera', 'build', etc.)
            *args, **kwargs: Arguments to pass to the action function
            
        Returns:
            Tuple[bool, float]: (success status, reward)
        """
        # Check if we have a reward system
        if not hasattr(self, 'reward_system'):
            print(f"Warning: No reward system connected, can't get feedback for '{action_name}'")
            # Execute without feedback
            success = self.retry_on_failure(action_func, action_name, *args, **kwargs)
            return success, 0.0
            
        # Prepare action info for reward calculation
        action_info = {
            "type": action_type,
            "action": action_name,
            "args": args,
            "kwargs": kwargs
        }
        
        # Check if we have a screen capture system for before/after states
        if hasattr(self, 'screen_capture'):
            # Capture state before action
            before_state = self.screen_capture.capture_frame()
            
            # Execute action
            success = self.retry_on_failure(action_func, action_name, *args, **kwargs)
            
            # Capture state after action
            after_state = self.screen_capture.capture_frame()
            
            # Get reward based on the action and state change
            try:
                reward = self.reward_system.calculate_action_reward(
                    before_state, 
                    after_state,
                    action_info,
                    success
                )
                
                print(f"Action '{action_name}' received reward: {reward:.4f}")
                
                # Store successful actions with positive rewards for learning
                if success and reward > 0 and hasattr(self, 'successful_actions'):
                    self.successful_actions.append((action_name, action_type, reward))
                    
                    # Keep only the last 100 successful actions
                    if len(self.successful_actions) > 100:
                        self.successful_actions.pop(0)
                        
                return success, reward
                
            except Exception as e:
                print(f"Error calculating reward for action '{action_name}': {e}")
                return success, 0.0
        else:
            # No screen capture, just execute the action
            success = self.retry_on_failure(action_func, action_name, *args, **kwargs)
            return success, 0.0
            
    def get_action_suggestions(self, current_state=None, top_n: int = 5) -> List[Tuple[str, str]]:
        """Get suggested actions based on previous successes.
        
        Args:
            current_state: Optional state tensor to consider for context
            top_n (int): Number of suggestions to return
            
        Returns:
            List[Tuple[str, str]]: List of (action_name, action_type) suggestions
        """
        # Initialize successful actions list if not already present
        if not hasattr(self, 'successful_actions'):
            self.successful_actions = []
            
        # If we have no history, return some basic actions
        if not self.successful_actions:
            default_suggestions = [
                ('move_up', 'camera'),
                ('move_down', 'camera'),
                ('click_random', 'ui'),
                ('rotate_right', 'camera')
            ]
            return default_suggestions[:top_n]
            
        # Sort actions by reward
        sorted_actions = sorted(self.successful_actions, key=lambda x: x[2], reverse=True)
        
        # Return the top N actions (name and type only)
        return [(name, action_type) for name, action_type, _ in sorted_actions[:top_n]] 
    
    def set_movement_speed(self, speed: float):
        """Set the movement speed for camera control.
        
        Args:
            speed (float): Movement speed multiplier (0.1 to 2.0)
        """
        self.movement_speed = max(0.1, min(2.0, speed))
        logger.info(f"Set movement speed to {self.movement_speed}")

    def get_mouse_position(self) -> Tuple[int, int]:
        """Get the current mouse position.
        
        Returns:
            Tuple[int, int]: Current mouse position (x, y)
        """
        try:
            # Use win32api to get mouse position
            cursor_pos = win32api.GetCursorPos()
            
            # Make sure we have a valid tuple with two elements
            if isinstance(cursor_pos, tuple) and len(cursor_pos) == 2:
                x, y = cursor_pos
            else:
                # If not a valid tuple, use screen center
                screen_width, screen_height = self.screen_width, self.screen_height
                x, y = screen_width // 2, screen_height // 2
                logger.warning(f"Invalid cursor position format: {cursor_pos}, using screen center")
            
            # If we have client position information, translate to client coordinates
            if hasattr(self, 'screen_capture') and self.screen_capture:
                if hasattr(self.screen_capture, 'client_position') and self.screen_capture.client_position:
                    client_left, client_top, _, _ = self.screen_capture.client_position
                    x = x - client_left
                    y = y - client_top
            
            return (x, y)
        except Exception as e:
            logger.error(f"Error getting mouse position: {e}")
            # Return center of screen as fallback
            screen_width, screen_height = self.screen_width, self.screen_height
            return (screen_width // 2, screen_height // 2)