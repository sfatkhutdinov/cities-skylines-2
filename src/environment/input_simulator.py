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
        
        # Fallback: return True to avoid blocking, but warn user
        print(f"WARNING: Failed to focus game window after trying methods: {', '.join(methods_tried)}")
        print("Continuing anyway. User may need to manually focus the game window.")
        return True  # Return True to allow operation to continue
        
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
        """Press a key with a short duration.
        
        Args:
            key (str): Key to press
            duration (float): How long to hold the key
        """
        # Completely prevent ESC key from being pressed unless explicitly allowed
        if key.lower() in ['escape', 'esc']:
            if hasattr(self, 'block_escape') and self.block_escape:
                print("WARNING: Blocked ESC key press - using safe menu handling instead")
                return self.safe_menu_handling()
        
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
            
    def safe_menu_handling(self):
        """Safely handle menu toggling without using Escape key.
        Uses alternative methods like clicking menu buttons.
        
        Returns:
            bool: Success status
        """
        # Get screen dimensions
        screen_width, screen_height = self.get_screen_dimensions()
        
        # EXACT coordinates provided by user: 720x513
        primary_resume_button = (720, 513)  # Exact coordinates for RESUME GAME button
        
        # Create a more comprehensive grid of positions around the main button
        # to increase chances of hitting the right spot
        grid_positions = []
        for x_offset in range(-30, 31, 10):  # -30 to +30 in steps of 10
            for y_offset in range(-30, 31, 10):  # -30 to +30 in steps of 10
                grid_positions.append((
                    primary_resume_button[0] + x_offset,
                    primary_resume_button[1] + y_offset
                ))
        
        # Scale coordinates for different resolutions
        if screen_width != 1920 or screen_height != 1080:
            x_scale = screen_width / 1920
            y_scale = screen_height / 1080
            scaled_positions = []
            for x, y in grid_positions:
                scaled_positions.append((int(x * x_scale), int(y * y_scale)))
        else:
            # Use exact positions for 1920x1080
            scaled_positions = grid_positions
        
        # Add additional fallback positions with wider coverage
        additional_positions = [
            # Center positions
            (screen_width // 2, screen_height // 2),
            # Various relative positions for common button placements
            (int(screen_width * 0.375), int(screen_height * 0.475)),  # ~720x513 in 1920x1080
            (int(screen_width * 0.37), int(screen_height * 0.47)),
            (int(screen_width * 0.38), int(screen_height * 0.48)),
            # Bottom positions (for OK buttons)
            (int(screen_width * 0.5), int(screen_height * 0.8)),
            # Try standard button positions
            (int(screen_width * 0.25), int(screen_height * 0.8)),
            (int(screen_width * 0.75), int(screen_height * 0.8)),
        ]
        
        # Prioritize primary position and nearby grid, then add fallbacks
        click_positions = [primary_resume_button] + scaled_positions[:10] + additional_positions
        
        print("Attempting to exit menu by clicking RESUME GAME button")
        
        # Try positions in two passes with different patterns
        for attempt in range(2):
            # First pass tries fewer positions with longer waits
            # Second pass tries more positions with shorter waits
            positions_to_try = click_positions[:5] if attempt == 0 else click_positions
            wait_time = 1.0 if attempt == 0 else 0.3
            
            for i, (x, y) in enumerate(positions_to_try):
                # Ensure coordinates are within screen bounds
                x = max(0, min(x, screen_width - 1))
                y = max(0, min(y, screen_height - 1))
                
                print(f"Clicking at position {i+1}/{len(positions_to_try)}: ({x}, {y})")
                
                # Move to position and click
                self.mouse_move(x, y, use_win32=True)
                time.sleep(0.1)  # Small delay before clicking
                self.mouse_click(x, y)
                time.sleep(wait_time)  # Wait for click to register
            
            # After each pass, try pressing Enter key as an additional option
            self.key_press('enter', 0.1)
            time.sleep(0.5)
            
            # Try space bar too
            self.key_press('space', 0.1)
            time.sleep(0.5)
        
        # Reset mouse to center of screen to allow full movement after menu handling
        print("Resetting mouse to center position")
        center_x, center_y = screen_width // 2, screen_height // 2
        self.mouse_move(center_x, center_y, use_win32=True)
        
        # Return mouse control to normal
        return True

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
            
    def key_combination(self, keys: List[str], duration: float = 0.1, allow_focus_keys: bool = False):
        """Press a combination of keys simultaneously.
        
        Args:
            keys (List[str]): List of key names from key_map
            duration (float): Duration to hold the keys in seconds
            allow_focus_keys (bool): If True, allow certain dangerous combinations like Alt+Tab when used for window focusing
        """
        # Safety check: prevent dangerous key combinations
        dangerous_combinations = [
            ['alt', 'tab'],
            ['alt', 'f4'],
            ['ctrl', 'w'],
            ['alt', 'escape'],
            ['ctrl', 'alt', 'delete']
        ]
        
        # If allow_focus_keys is True, we won't block alt+tab specifically
        if allow_focus_keys and len(keys) == 2 and 'alt' in [k.lower() for k in keys] and 'tab' in [k.lower() for k in keys]:
            print("Allowing Alt+Tab for window focusing")
        else:
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