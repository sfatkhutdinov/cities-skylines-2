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
        self.window_title = self.capture_config.get('window_title', "Cities: Skylines II")
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
            # Use the provided window title as the primary search target
            primary_title = self.window_title
            logger.info(f"Searching for game window with title: '{primary_title}'")
            
            # Try several possible window titles, with the provided title first
            possible_titles = [
                primary_title,
                "Cities: Skylines II",
                "Cities Skylines II", 
                "Cities: Skylines 2",
                "Cities Skylines 2",
                "CitiesSkylines2"
            ]
            
            # Remove duplicates while preserving order
            possible_titles = list(dict.fromkeys(possible_titles))
            logger.info(f"Will try these window titles: {possible_titles}")
            
            # First try exact matches
            hwnd = 0
            for title in possible_titles:
                hwnd = win32gui.FindWindow(None, title)
                if hwnd != 0:
                    logger.info(f"Found game window with exact title: {title}")
                    break
            
            # If no exact match, try partial match
            if hwnd == 0:
                # List all windows for logging
                all_windows = []
                def enum_callback(hwnd, windows):
                    if win32gui.IsWindowVisible(hwnd):
                        window_title = win32gui.GetWindowText(hwnd)
                        if window_title:
                            windows.append((hwnd, window_title))
                    return True
                
                win32gui.EnumWindows(enum_callback, all_windows)
                logger.info(f"Found {len(all_windows)} visible windows")
                
                # Log the first 10 windows for debugging
                for i, (_, title) in enumerate(all_windows[:10]):
                    logger.info(f"Window {i+1}: {title}")
                
                # First try to find windows containing the primary title
                logger.info(f"Looking for windows containing '{primary_title}'")
                for hwnd, title in all_windows:
                    if primary_title.lower() in title.lower():
                        logger.info(f"Found game window with partial match to primary title: {title}")
                        hwnd_candidate = hwnd
                        hwnd = hwnd_candidate
                        break
                
                # If still not found, try other possible titles
                if hwnd == 0:
                    for hwnd, title in all_windows:
                        for game_title in possible_titles:
                            if game_title.lower() in title.lower() or "cities" in title.lower():
                                logger.info(f"Found potential game window with partial match: {title}")
                                hwnd_candidate = hwnd
                                hwnd = hwnd_candidate
                                break
                        if hwnd != 0:
                            break
                        
            if hwnd != 0:
                # Get window rect
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    logger.info(f"Window rectangle: {rect}")
                    
                    # Check if window size is reasonable
                    width = right - left
                    height = bottom - top
                    logger.info(f"Window size: {width}x{height}")
                    
                    if width < 100 or height < 100:
                        logger.warning(f"Window size too small, may not be game window")
                        
                    # Try to get client rect
                    try:
                        client_rect = win32gui.GetClientRect(hwnd)
                        logger.info(f"Client rectangle: {client_rect}")
                        left, top, right, bottom = client_rect
                        
                        # Convert to screen coordinates
                        try:
                            left_screen, top_screen = win32gui.ClientToScreen(hwnd, (left, top))
                            right_screen, bottom_screen = win32gui.ClientToScreen(hwnd, (right, bottom))
                            
                            self.client_position = (left_screen, top_screen, right_screen, bottom_screen)
                            logger.info(f"Found game window: {self.client_position}")
                        except Exception as e:
                            logger.error(f"Error converting to screen coordinates: {e}")
                            self.client_position = rect
                            logger.info(f"Using window rect as fallback: {self.client_position}")
                    except Exception as e:
                        logger.error(f"Error getting client rect: {e}")
                        self.client_position = rect
                        logger.info(f"Using window rect as fallback: {self.client_position}")
                except Exception as e:
                    logger.error(f"Error getting window rect: {e}")
                    # Fallback to primary monitor
                    monitor = self.sct.monitors[1]  # Primary monitor
                    self.client_position = (
                        monitor["left"], monitor["top"],
                        monitor["left"] + monitor["width"],
                        monitor["top"] + monitor["height"]
                    )
                
                # Store the window handle
                self.game_hwnd = hwnd
                # Store the actual window title we found for future reference
                try:
                    self.actual_window_title = win32gui.GetWindowText(hwnd)
                    logger.info(f"Using window with title: '{self.actual_window_title}'")
                except:
                    self.actual_window_title = self.window_title
                
                # Focus the window
                self.focus_game_window()
            else:
                logger.warning(f"Game window not found after trying multiple titles")
                self.game_hwnd = None
                # Fallback to primary monitor
                monitor = self.sct.monitors[1]  # Primary monitor
                self.client_position = (
                    monitor["left"], monitor["top"],
                    monitor["left"] + monitor["width"],
                    monitor["top"] + monitor["height"]
                )
                
        except Exception as e:
            logger.error(f"Error finding game window: {e}")
            self.game_hwnd = None
            # Fallback to primary monitor
            monitor = self.sct.monitors[1]  # Primary monitor
            self.client_position = (
                monitor["left"], monitor["top"],
                monitor["left"] + monitor["width"],
                monitor["top"] + monitor["height"]
            )
    
    def focus_game_window(self):
        """Bring the game window to the foreground."""
        if not WIN32_AVAILABLE or not hasattr(self, 'game_hwnd') or self.game_hwnd is None:
            logger.warning("Cannot focus game window - window handle not available")
            return False
            
        try:
            # Get window info for logging
            try:
                window_title = win32gui.GetWindowText(self.game_hwnd)
                logger.info(f"Focusing window: '{window_title}' (hwnd: {self.game_hwnd})")
            except Exception as e:
                logger.error(f"Error getting window title: {e}")
            
            # Check if window exists
            if not win32gui.IsWindow(self.game_hwnd):
                logger.error(f"Window handle {self.game_hwnd} is not a valid window")
                return False
            
            # Check if window is visible
            if not win32gui.IsWindowVisible(self.game_hwnd):
                logger.warning(f"Window {self.game_hwnd} is not visible")
                try:
                    win32gui.ShowWindow(self.game_hwnd, win32con.SW_SHOW)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error showing window: {e}")
            
            # Check if window is minimized
            if win32gui.IsIconic(self.game_hwnd):
                logger.info("Window is minimized, restoring...")
                # Show the window if it's minimized
                try:
                    win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"Error restoring window: {e}")
                
            # Check if window is already in foreground
            try:
                foreground_hwnd = win32gui.GetForegroundWindow()
                foreground_title = win32gui.GetWindowText(foreground_hwnd)
                logger.info(f"Current foreground window: '{foreground_title}' (hwnd: {foreground_hwnd})")
                
                if foreground_hwnd == self.game_hwnd:
                    logger.info("Game window is already in focus")
                    return True
            except Exception as e:
                logger.error(f"Error checking foreground window: {e}")
                
            # Create a counter for retry attempts
            focus_attempts = 0
            max_focus_attempts = 3
            
            # Try different approaches with retry mechanism
            while focus_attempts < max_focus_attempts:
                focus_attempts += 1
                
                # Approach 1: Standard SetForegroundWindow
                try:
                    logger.info(f"Attempt {focus_attempts}: Focusing window with SetForegroundWindow...")
                    import ctypes
                    user32 = ctypes.windll.user32
                    user32.AllowSetForegroundWindow(self.game_hwnd)
                    
                    # Set window as foreground window
                    result = win32gui.SetForegroundWindow(self.game_hwnd)
                    logger.info(f"SetForegroundWindow result: {result}")
                    
                    # Small delay to allow OS to process
                    time.sleep(0.5)
                    
                    # Check if successful
                    if win32gui.GetForegroundWindow() == self.game_hwnd:
                        logger.info("Successfully focused window with SetForegroundWindow")
                        return True
                except Exception as e:
                    # If we got "Access is denied", just log it and continue to other approaches
                    if "Access is denied" in str(e):
                        logger.warning(f"SetForegroundWindow denied access: {e}")
                    else:
                        logger.error(f"Error using SetForegroundWindow: {e}")
                
                # Approach 2: Alternative using BringWindowToTop
                try:
                    logger.info(f"Attempt {focus_attempts}: Focusing window with BringWindowToTop...")
                    win32gui.BringWindowToTop(self.game_hwnd)
                    time.sleep(0.5)
                    
                    # Check if successful
                    if win32gui.GetForegroundWindow() == self.game_hwnd:
                        logger.info("Successfully focused window with BringWindowToTop")
                        return True
                except Exception as e:
                    logger.error(f"Error using BringWindowToTop: {e}")
                
                # Approach 3: Use ShowWindow to activate
                try:
                    logger.info(f"Attempt {focus_attempts}: Focusing window with ShowWindow...")
                    win32gui.ShowWindow(self.game_hwnd, win32con.SW_SHOW)
                    win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                    time.sleep(0.5)
                    
                    # Check if successful
                    if win32gui.GetForegroundWindow() == self.game_hwnd:
                        logger.info("Successfully focused window with ShowWindow")
                        return True
                except Exception as e:
                    logger.error(f"Error using ShowWindow: {e}")
                
                # Approach 4: Try using AttachThreadInput technique
                try:
                    logger.info(f"Attempt {focus_attempts}: Focusing with AttachThreadInput technique...")
                    import ctypes
                    user32 = ctypes.windll.user32
                    
                    # Get the threads of the foreground window and our target window
                    foreground_thread = user32.GetWindowThreadProcessId(foreground_hwnd, None)
                    target_thread = user32.GetWindowThreadProcessId(self.game_hwnd, None)
                    
                    if foreground_thread != target_thread:
                        # Attach the threads
                        user32.AttachThreadInput(foreground_thread, target_thread, True)
                        
                        # Set focus and bring to top
                        user32.SetForegroundWindow(self.game_hwnd)
                        user32.BringWindowToTop(self.game_hwnd)
                        
                        # Detach the threads
                        user32.AttachThreadInput(foreground_thread, target_thread, False)
                        
                        time.sleep(0.5)
                        
                        # Check if successful
                        if win32gui.GetForegroundWindow() == self.game_hwnd:
                            logger.info("Successfully focused window with AttachThreadInput")
                            return True
                except Exception as e:
                    logger.error(f"Error using AttachThreadInput: {e}")
                
                # Wait before retrying
                time.sleep(1.0)
            
            logger.warning(f"Failed to focus window after {max_focus_attempts} attempts")
            
            # Fallback method - if all else fails, try to continue without focus
            logger.info("Proceeding with capturing without window focus as a fallback")
            return False
            
        except Exception as e:
            logger.error(f"Failed to focus game window: {e}")
            return False
    
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
            # Ensure game window is in focus before capturing
            if hasattr(self, 'game_hwnd') and self.game_hwnd and WIN32_AVAILABLE:
                foreground_hwnd = win32gui.GetForegroundWindow()
                if foreground_hwnd != self.game_hwnd:
                    logger.warning("Game window not in focus, attempting to refocus")
                    self.focus_game_window()
            
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