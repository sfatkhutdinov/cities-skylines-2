"""
Optimized screen capture module for Cities Skylines 2 agent.

This module provides high-performance screen capture functionality
with various optimization techniques for game state observation.
"""

import time
import logging
import numpy as np
from typing import Tuple, Optional, Dict, Any

# Import MSS for fast screen capture
try:
    import mss
    import mss.tools
    MSS_AVAILABLE = True
except ImportError:
    MSS_AVAILABLE = False
    logging.warning("MSS library not available. Screen capture will be limited.")

# Try to import Windows-specific libraries for additional optimizations
try:
    import win32gui
    import win32con
    WIN32_AVAILABLE = True
except ImportError:
    WIN32_AVAILABLE = False
    logging.warning("Win32 libraries not available. Some features will be limited.")

logger = logging.getLogger(__name__)

class OptimizedScreenCapture:
    """Optimized screen capture implementation for Cities Skylines 2.
    
    This class handles efficient screen capture using platform-specific
    optimizations and supports mock mode for testing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the screen capture system.
        
        Args:
            config: Configuration dictionary with capture settings
        """
        self.config = config
        self.capture_config = config.get('capture', {})
        self.method = self.capture_config.get('capture_method', 'windows')
        self.target_fps = self.capture_config.get('capture_fps', 30)
        self.frame_interval = 1.0 / self.target_fps
        
        # Initialize capture state
        self.last_capture_time = 0
        self.use_mock = False
        self.window_title = "Cities: Skylines II"
        self.client_position = None
        self.sct = None
        self.initialized = False
        
        # Initialize screen capture
        self._initialize_capture()
    
    def _initialize_capture(self):
        """Initialize the appropriate screen capture method."""
        if self.use_mock:
            logger.info("Using mock screen capture")
            self.initialized = True
            return
            
        if not MSS_AVAILABLE:
            logger.error("MSS library is required for screen capture")
            raise ImportError("MSS library is required for screen capture")
            
        try:
            # Initialize MSS screen capture
            self.sct = mss.mss()
            
            # Get game window position if using Windows method
            if self.method == 'windows' and WIN32_AVAILABLE:
                self._find_game_window()
            else:
                # Fallback to full screen capture
                monitor = self.sct.monitors[0]  # Full screen
                self.client_position = (
                    monitor["left"], monitor["top"],
                    monitor["left"] + monitor["width"],
                    monitor["top"] + monitor["height"]
                )
                
            self.initialized = True
            logger.info(f"Screen capture initialized using {self.method} method")
            logger.debug(f"Capture area: {self.client_position}")
            
        except Exception as e:
            logger.error(f"Failed to initialize screen capture: {e}")
            raise
    
    def _find_game_window(self):
        """Find the Cities Skylines 2 game window."""
        if not WIN32_AVAILABLE:
            logger.warning("Win32 libraries not available for window detection")
            return
            
        try:
            hwnd = win32gui.FindWindow(None, self.window_title)
            if hwnd == 0:
                # Try partial match if exact title not found
                def callback(hwnd, windows):
                    if self.window_title in win32gui.GetWindowText(hwnd):
                        windows.append(hwnd)
                    return True
                    
                windows = []
                win32gui.EnumWindows(callback, windows)
                
                if windows:
                    hwnd = windows[0]
                    
            if hwnd != 0:
                # Get window rect
                left, top, right, bottom = win32gui.GetClientRect(hwnd)
                
                # Convert to screen coordinates
                left_screen, top_screen = win32gui.ClientToScreen(hwnd, (left, top))
                right_screen, bottom_screen = win32gui.ClientToScreen(hwnd, (right, bottom))
                
                self.client_position = (left_screen, top_screen, right_screen, bottom_screen)
                logger.info(f"Found game window: {self.client_position}")
            else:
                logger.warning(f"Game window '{self.window_title}' not found")
                # Fallback to primary monitor
                monitor = self.sct.monitors[1]  # Primary monitor
                self.client_position = (
                    monitor["left"], monitor["top"],
                    monitor["left"] + monitor["width"],
                    monitor["top"] + monitor["height"]
                )
                
        except Exception as e:
            logger.error(f"Error finding game window: {e}")
            # Fallback to primary monitor
            monitor = self.sct.monitors[1]  # Primary monitor
            self.client_position = (
                monitor["left"], monitor["top"],
                monitor["left"] + monitor["width"],
                monitor["top"] + monitor["height"]
            )
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the screen.
        
        Returns:
            numpy.ndarray: Captured frame as numpy array in RGB format,
                          or None if capture failed
        """
        if self.use_mock:
            # Return a mock frame for testing
            return self._generate_mock_frame()
            
        if not self.initialized:
            logger.error("Screen capture not initialized")
            return None
            
        try:
            # Throttle capture rate to target FPS
            current_time = time.time()
            elapsed = current_time - self.last_capture_time
            if elapsed < self.frame_interval:
                time.sleep(self.frame_interval - elapsed)
                
            # Capture screen
            if self.client_position:
                left, top, right, bottom = self.client_position
                monitor = {
                    "left": left,
                    "top": top,
                    "width": right - left,
                    "height": bottom - top
                }
                
                # Capture the specified region
                sct_img = self.sct.grab(monitor)
                
                # Convert to numpy array (BGRA format)
                img_np = np.array(sct_img)
                
                # Convert BGRA to RGB
                img_rgb = img_np[:, :, :3][:, :, ::-1]
                
                self.last_capture_time = time.time()
                return img_rgb
            else:
                logger.error("No valid capture area defined")
                return None
                
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    def _generate_mock_frame(self) -> np.ndarray:
        """Generate a mock frame for testing.
        
        Returns:
            numpy.ndarray: A mock frame
        """
        if self.client_position:
            width = self.client_position[2] - self.client_position[0]
            height = self.client_position[3] - self.client_position[1]
        else:
            width, height = 1920, 1080  # Default size
            
        # Create a simple gradient image for mock mode
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add a gradient background
        for y in range(height):
            for x in range(width):
                frame[y, x, 0] = int(255 * y / height)  # R
                frame[y, x, 1] = int(255 * x / width)   # G
                frame[y, x, 2] = int(127 + 128 * np.sin(time.time()))  # B with animation
                
        # Add a simulated UI element
        ui_x, ui_y = int(width * 0.8), int(height * 0.1)
        ui_width, ui_height = int(width * 0.15), int(height * 0.05)
        frame[ui_y:ui_y+ui_height, ui_x:ui_x+ui_width] = [200, 200, 200]
        
        return frame
    
    def get_resolution(self) -> Tuple[int, int]:
        """Get the current capture resolution.
        
        Returns:
            tuple: Width and height of the capture area
        """
        if self.client_position:
            width = self.client_position[2] - self.client_position[0]
            height = self.client_position[3] - self.client_position[1]
            return width, height
        else:
            # Default resolution if no window is detected
            return 1920, 1080
    
    def close(self):
        """Release resources used by the screen capture."""
        if self.sct:
            self.sct.close() 