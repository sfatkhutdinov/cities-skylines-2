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
    
    def mouse_move(self, x: int, y: int) -> bool:
        """Move mouse pointer to the specified coordinates.
        
        Args:
            x: X coordinate in screen space
            y: Y coordinate in screen space
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Use win32api for direct cursor positioning
            win32api.SetCursorPos((x, y))
            return True
        except Exception as e:
            logger.error(f"Error moving mouse using win32api: {e}")
            
            # Fallback to pynput
            try:
                self.mouse.position = (x, y)
                return True
            except Exception as e2:
                logger.error(f"Error moving mouse using pynput fallback: {e2}")
                return False
    
    def mouse_click(self, x: int, y: int, button: str = 'left', double: bool = False) -> bool:
        """Move mouse to specified coordinates and perform click.
        
        Args:
            x: X coordinate in screen space
            y: Y coordinate in screen space
            button: Which button to click ('left', 'right', 'middle')
            double: Whether to perform a double-click
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Move to position first
            if not self.mouse_move(x, y):
                logger.warning(f"Failed to move mouse to ({x}, {y}) before clicking")
                return False
                
            # Small delay for stability
            time.sleep(0.05)
            
            # Try to click using direct Win32 calls
            try:
                if button.lower() == 'left':
                    user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    time.sleep(0.05)
                    user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    
                    if double:
                        time.sleep(0.05)
                        user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        time.sleep(0.05)
                        user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                elif button.lower() == 'right':
                    user32.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                    time.sleep(0.05)
                    user32.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                    
                    if double:
                        time.sleep(0.05)
                        user32.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
                        time.sleep(0.05)
                        user32.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
                elif button.lower() == 'middle':
                    user32.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                    time.sleep(0.05)
                    user32.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                    
                    if double:
                        time.sleep(0.05)
                        user32.mouse_event(win32con.MOUSEEVENTF_MIDDLEDOWN, 0, 0, 0, 0)
                        time.sleep(0.05)
                        user32.mouse_event(win32con.MOUSEEVENTF_MIDDLEUP, 0, 0, 0, 0)
                    
                return True
            except Exception as e:
                logger.warning(f"Direct Win32 click failed: {e}, falling back to pynput")
            
            # Fallback to pynput
            btn = self.button_map.get(button.lower(), Button.left)
            
            if double:
                self.mouse.click(btn, 2)
            else:
                self.mouse.click(btn)
                
            return True
        except Exception as e:
            logger.error(f"Error during mouse click at ({x}, {y}): {e}")
            return False
    
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