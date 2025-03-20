"""
Mouse input simulation for Cities: Skylines 2.

This module provides mouse input capabilities for the game environment.
"""

import time
import win32gui
import win32con
import win32api
import logging
import ctypes
from typing import Tuple, Optional, Dict, Any
from pynput.mouse import Button, Controller as MouseController

# Import Win32 user32 for more direct mouse control
user32 = ctypes.WinDLL('user32', use_last_error=True)
logger = logging.getLogger(__name__)

class MouseInput:
    """Handles mouse input simulation for the game."""
    
    def __init__(self):
        """Initialize mouse controller and state variables."""
        self.mouse = MouseController()
        
        # Store current display resolution
        self.screen_width = win32api.GetSystemMetrics(0)
        self.screen_height = win32api.GetSystemMetrics(1)
        logger.info(f"Screen resolution: {self.screen_width}x{self.screen_height}")
        
        # Store game window information
        self.game_hwnd = None
        self.game_rect = None
        self.client_rect = None
        
        # Movement speed adjustment for more realistic mouse movement
        self.movement_speed = 1.0
        
        # Button mapping for convenience
        self.button_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }
        
        logger.info("Mouse input initialized")
    
    def find_game_window(self, window_title: str = "Cities: Skylines II") -> bool:
        """Find the game window by title and store its dimensions.
        
        Args:
            window_title: Title of the game window to find
            
        Returns:
            True if found, False otherwise
        """
        try:
            self.game_hwnd = win32gui.FindWindow(None, window_title)
            if self.game_hwnd == 0:
                logger.warning(f"Game window '{window_title}' not found")
                return False
                
            # Get window and client area dimensions
            self.game_rect = win32gui.GetWindowRect(self.game_hwnd)
            self.client_rect = win32gui.GetClientRect(self.game_hwnd)
            
            logger.info(f"Found game window: handle={self.game_hwnd}")
            logger.info(f"Window rect: {self.game_rect}")
            logger.info(f"Client rect: {self.client_rect}")
            return True
        except Exception as e:
            logger.error(f"Error finding game window: {e}")
            return False
    
    def mouse_move(self, x: int, y: int, retry_count: int = 2) -> bool:
        """Move mouse pointer to the specified coordinates.
        
        Args:
            x: X coordinate in screen space
            y: Y coordinate in screen space
            retry_count: Number of retry attempts if movement fails
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure coordinates are integers
        x, y = int(x), int(y)
        
        # Verify coordinates are within screen boundaries
        if x < 0 or x >= self.screen_width or y < 0 or y >= self.screen_height:
            logger.warning(f"Mouse coordinates ({x}, {y}) outside screen boundaries ({self.screen_width}x{self.screen_height})")
            
            # Clamp coordinates to screen boundaries
            x = max(0, min(x, self.screen_width - 1))
            y = max(0, min(y, self.screen_height - 1))
            logger.info(f"Clamped coordinates to ({x}, {y})")
        
        # Try different movement methods with retry
        success = False
        errors = []
        
        for attempt in range(retry_count + 1):
            if attempt > 0:
                logger.info(f"Retrying mouse move to ({x}, {y}), attempt {attempt}/{retry_count}")
                time.sleep(0.1)  # Short delay between retries
            
            # Method 1: Win32 API for direct positioning
            try:
                win32api.SetCursorPos((x, y))
                
                # Verify position
                curr_x, curr_y = win32api.GetCursorPos()
                if abs(curr_x - x) <= 5 and abs(curr_y - y) <= 5:
                    logger.debug(f"Successfully moved mouse to ({x}, {y}) using win32api")
                    success = True
                    break
                else:
                    error_msg = f"Mouse position verification failed: requested ({x}, {y}), got ({curr_x}, {curr_y})"
                    logger.warning(error_msg)
                    errors.append(f"win32api: {error_msg}")
            except Exception as e:
                logger.warning(f"Error moving mouse using win32api: {e}")
                errors.append(f"win32api: {str(e)}")
            
            # Method 2: Fallback to pynput
            if not success:
                try:
                    self.mouse.position = (x, y)
                    
                    # Verify position
                    curr_x, curr_y = win32api.GetCursorPos()
                    if abs(curr_x - x) <= 5 and abs(curr_y - y) <= 5:
                        logger.debug(f"Successfully moved mouse to ({x}, {y}) using pynput")
                        success = True
                        break
                    else:
                        error_msg = f"Mouse position verification failed: requested ({x}, {y}), got ({curr_x}, {curr_y})"
                        logger.warning(error_msg)
                        errors.append(f"pynput: {error_msg}")
                except Exception as e:
                    logger.warning(f"Error moving mouse using pynput: {e}")
                    errors.append(f"pynput: {str(e)}")
            
            # Method 3: Use Win32 SendInput
            if not success:
                try:
                    # Create input event structure
                    import ctypes
                    x_norm = int(x * 65535 / self.screen_width)
                    y_norm = int(y * 65535 / self.screen_height)
                    
                    extra = ctypes.c_ulong(0)
                    ii_ = ctypes.c_ulong(0)
                    # Move mouse
                    x0, y0 = ctypes.c_long(x_norm), ctypes.c_long(y_norm)
                    
                    # Define input structure
                    class MOUSEINPUT(ctypes.Structure):
                        _fields_ = [("dx", ctypes.c_long),
                                    ("dy", ctypes.c_long),
                                    ("mouseData", ctypes.c_ulong),
                                    ("dwFlags", ctypes.c_ulong),
                                    ("time", ctypes.c_ulong),
                                    ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]
                                    
                    class INPUT(ctypes.Structure):
                        _fields_ = [("type", ctypes.c_ulong),
                                    ("mi", MOUSEINPUT),
                                    ("padding", ctypes.c_longlong)]
                    
                    pt = MOUSEINPUT(x0, y0, 0, 0x0001 | 0x8000, 0, ctypes.pointer(extra)) # MOUSEEVENTF_MOVE|MOUSEEVENTF_ABSOLUTE
                    command = INPUT(0, pt, 0)
                    ctypes.windll.user32.SendInput(1, ctypes.pointer(command), ctypes.sizeof(command))
                    
                    # Verify position
                    time.sleep(0.05)  # Wait for OS to process
                    curr_x, curr_y = win32api.GetCursorPos()
                    if abs(curr_x - x) <= 5 and abs(curr_y - y) <= 5:
                        logger.debug(f"Successfully moved mouse to ({x}, {y}) using SendInput")
                        success = True
                        break
                    else:
                        error_msg = f"Mouse position verification failed: requested ({x}, {y}), got ({curr_x}, {curr_y})"
                        logger.warning(error_msg)
                        errors.append(f"SendInput: {error_msg}")
                except Exception as e:
                    logger.warning(f"Error moving mouse using SendInput: {e}")
                    errors.append(f"SendInput: {str(e)}")
        
        if not success:
            logger.error(f"All mouse movement methods failed after {retry_count+1} attempts. Errors: {errors}")
        
        return success
    
    def mouse_click(self, x: int, y: int, button: str = 'left', double: bool = False, retry_count: int = 2) -> bool:
        """Move mouse to specified coordinates and perform click.
        
        Args:
            x: X coordinate in screen space
            y: Y coordinate in screen space
            button: Button to click ('left', 'right', 'middle')
            double: Whether to perform a double-click
            retry_count: Number of retry attempts if clicking fails
            
        Returns:
            True if successful, False otherwise
        """
        # Ensure mouse is at the target position first
        if not self.mouse_move(x, y):
            logger.warning(f"Failed to move mouse to ({x}, {y}) for click")
            return False
            
        # Get mapped button
        try:
            btn = self.button_map.get(button.lower(), Button.left)
        except Exception as e:
            logger.error(f"Error mapping button {button}: {e}")
            return False
            
        # Try different click methods with retry
        success = False
        errors = []
        
        for attempt in range(retry_count + 1):
            if attempt > 0:
                logger.info(f"Retrying mouse click at ({x}, {y}), attempt {attempt}/{retry_count}")
                # Ensure mouse is still at the position
                if not self.mouse_move(x, y):
                    logger.warning(f"Failed to move mouse to position for click retry")
                    continue
                    
                time.sleep(0.1)  # Short delay between retries
            
            # Method 1: pynput click
            try:
                if double:
                    self.mouse.click(btn, 2)
                else:
                    self.mouse.click(btn)
                
                success = True
                break
            except Exception as e:
                logger.warning(f"Error clicking using pynput: {e}")
                errors.append(f"pynput: {str(e)}")
            
            # Method 2: pyautogui (fallback)
            if not success:
                try:
                    import pyautogui
                    btn_map = {'left': 'left', 'right': 'right', 'middle': 'middle'}
                    pyautogui_btn = btn_map.get(button.lower(), 'left')
                    
                    # Ensure position is correct in pyautogui as well
                    pyautogui.moveTo(x, y)
                    
                    if double:
                        pyautogui.doubleClick(button=pyautogui_btn)
                    else:
                        pyautogui.click(button=pyautogui_btn)
                        
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Error clicking using pyautogui: {e}")
                    errors.append(f"pyautogui: {str(e)}")
                    
            # Method 3: Direct Win32 API call
            if not success:
                try:
                    # Define button down/up event constants
                    button_down_flags = {
                        'left': 0x0002,     # MOUSEEVENTF_LEFTDOWN
                        'right': 0x0008,    # MOUSEEVENTF_RIGHTDOWN
                        'middle': 0x0020    # MOUSEEVENTF_MIDDLEDOWN
                    }
                    
                    button_up_flags = {
                        'left': 0x0004,     # MOUSEEVENTF_LEFTUP
                        'right': 0x0010,    # MOUSEEVENTF_RIGHTUP
                        'middle': 0x0040    # MOUSEEVENTF_MIDDLEUP
                    }
                    
                    down_flag = button_down_flags.get(button.lower(), 0x0002)  # Default to left button
                    up_flag = button_up_flags.get(button.lower(), 0x0004)  # Default to left button
                    
                    # Perform the click
                    user32.mouse_event(down_flag, 0, 0, 0, 0)
                    time.sleep(0.05)
                    user32.mouse_event(up_flag, 0, 0, 0, 0)
                    
                    if double:
                        time.sleep(0.05)
                        user32.mouse_event(down_flag, 0, 0, 0, 0)
                        time.sleep(0.05)
                        user32.mouse_event(up_flag, 0, 0, 0, 0)
                    
                    success = True
                    break
                except Exception as e:
                    logger.warning(f"Direct Win32 click failed: {e}")
                    errors.append(f"win32: {str(e)}")
            
        if not success:
            logger.error(f"All click methods failed for {button} button at ({x}, {y}): {errors}")
            
        return success
    
    def mouse_drag(self, start: Tuple[int, int], end: Tuple[int, int], 
                 button: str = 'left', duration: float = 0.5) -> bool:
        """Perform a mouse drag operation from start to end coordinates.
        
        Args:
            start: (x, y) start coordinates
            end: (x, y) end coordinates
            button: Which button to use for dragging
            duration: How long the drag operation should take
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract coordinates
            x1, y1 = start
            x2, y2 = end
            
            # Map button string to pynput Button
            btn = self.button_map.get(button.lower(), Button.left)
            
            # Move to start position
            if not self.mouse_move(x1, y1):
                logger.warning(f"Failed to move mouse to start position ({x1}, {y1})")
                return False
                
            # Wait for position to stabilize
            time.sleep(0.1)
            
            # Press button
            self.mouse.press(btn)
            time.sleep(0.1)
            
            # Calculate steps based on distance and duration
            # More steps for longer distances for smoother movement
            distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            steps = max(10, min(100, int(distance / 10)))
            
            # At least 10 steps, up to 120 updates per second
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
                logger.warning(f"Mouse drag end position verification: requested ({x2},{y2}), got ({current_x},{current_y})")
                
            return True
        except Exception as e:
            logger.error(f"Error during mouse drag {start} -> {end}: {e}")
            # Make sure to release the button on error
            try:
                btn = self.button_map.get(button.lower(), Button.left)
                self.mouse.release(btn)
            except Exception as e2:
                logger.error(f"Error releasing mouse button during error handling: {e2}")
            return False
    
    def mouse_scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Perform mouse scroll operation.
        
        Args:
            clicks: Number of scroll "clicks" (positive = up, negative = down)
            x: Optional X position to move before scrolling
            y: Optional Y position to move before scrolling
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Move to position first if specified
            if x is not None and y is not None:
                if not self.mouse_move(x, y):
                    logger.warning(f"Failed to move mouse to ({x}, {y}) before scrolling")
                    return False
                time.sleep(0.1)
                
            # Perform scrolling
            self.mouse.scroll(0, clicks)
            return True
        except Exception as e:
            logger.error(f"Error during mouse scroll: {e}")
            return False
    
    def press_mouse_button(self, button: str = 'left') -> bool:
        """Press a mouse button without releasing it.
        
        Args:
            button: 'left', 'right', or 'middle'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map button string to pynput Button
            btn = self.button_map.get(button.lower(), Button.left)
            
            # Press button
            self.mouse.press(btn)
            return True
        except Exception as e:
            logger.error(f"Error pressing mouse button {button}: {e}")
            return False
        
    def release_mouse_button(self, button: str = 'left') -> bool:
        """Release a previously pressed mouse button.
        
        Args:
            button: 'left', 'right', or 'middle'
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Map button string to pynput Button
            btn = self.button_map.get(button.lower(), Button.left)
            
            # Release button
            self.mouse.release(btn)
            return True
        except Exception as e:
            logger.error(f"Error releasing mouse button {button}: {e}")
            return False
    
    def pan_camera(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> bool:
        """Pan camera using right mouse button drag.
        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration: Duration of panning operation in seconds
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Panning camera: ({start_x},{start_y}) -> ({end_x},{end_y})")
        return self.mouse_drag((start_x, start_y), (end_x, end_y), button='right', duration=duration)
    
    def click_at_normalized_position(self, x: float, y: float, button: str = 'left', double: bool = False) -> bool:
        """Click at a position specified by normalized coordinates (0-1).
        
        Args:
            x: Normalized X coordinate (0.0 to 1.0)
            y: Normalized Y coordinate (0.0 to 1.0)
            button: Which button to click
            double: Whether to perform a double-click
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clip coordinates to valid range
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))
            
            # Convert to screen coordinates
            screen_x = int(x * self.screen_width)
            screen_y = int(y * self.screen_height)
            
            # Perform click
            return self.mouse_click(screen_x, screen_y, button, double)
        except Exception as e:
            logger.error(f"Error clicking at normalized position ({x}, {y}): {e}")
            return False
    
    def close(self) -> None:
        """Release all mouse buttons to clean up resources."""
        try:
            # Release all mouse buttons just in case
            for button in ["left", "right", "middle"]:
                try:
                    btn = self.button_map.get(button, Button.left)
                    self.mouse.release(btn)
                except Exception as e:
                    logger.warning(f"Error releasing mouse button {button}: {e}")
        except Exception as e:
            logger.error(f"Error in mouse cleanup: {e}")
            
        logger.info("Mouse input resources released") 