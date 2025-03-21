"""
Optimized screen capture module for Cities Skylines 2 agent.

This module provides high-performance screen capture functionality
with various optimization techniques for game state observation.
"""

import time
import logging
import numpy as np
import win32gui
import win32con
import win32ui
import win32api
import ctypes
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
    import platform
    WIN32_AVAILABLE = platform.system() == "Windows"
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
    
    def focus_game_window(self, retries=3, verification_time=0.5) -> bool:
        """Focus the game window to ensure inputs go to the game.
        
        Args:
            retries: Number of retry attempts
            verification_time: Time to wait before verifying focus
            
        Returns:
            True if window is successfully focused
        """
        if not self.game_hwnd:
            logger.warning("Cannot focus window, no game window handle")
            return False
        
        window_title = win32gui.GetWindowText(self.game_hwnd)
        logger.info(f"Focusing window: '{window_title}' (hwnd: {self.game_hwnd})")
        
        # Check if window is already focused
        current_hwnd = win32gui.GetForegroundWindow()
        current_title = win32gui.GetWindowText(current_hwnd)
        logger.info(f"Current foreground window: '{current_title}' (hwnd: {current_hwnd})")
        
        if current_hwnd == self.game_hwnd:
            logger.info("Game window is already focused")
            return True
        
        # Try different focus techniques
        focused = False
        
        # Technique 1: SetForegroundWindow with input state
        logger.info("Trying focus technique 1...")
        try:
            # Try to simulate Alt key press to allow focus change
            # Define ASFW_ANY ourselves if it's not in win32con
            ASFW_ANY = 0x01  # ASFW_ANY constant value for allowing focus change
            
            # Attach foreground window rights
            user32 = ctypes.windll.user32
            user32.AllowSetForegroundWindow(ASFW_ANY)
            
            # Set game window to foreground
            result = win32gui.SetForegroundWindow(self.game_hwnd)
            focused = result != 0
            
            # Wait and verify
            time.sleep(verification_time)
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                logger.info("Focus technique 1 successful")
                return True
        except Exception as e:
            logger.error(f"Error using SetForegroundWindow: {e}")
        
        # If technique 1 failed or verification failed, try technique 2
        if not focused:
            logger.info("Trying focus technique 2...")
            try:
                # Restore window if minimized
                if win32gui.IsIconic(self.game_hwnd):
                    win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
                
                # Make window visible and activate
                win32gui.ShowWindow(self.game_hwnd, win32con.SW_SHOW)
                win32gui.SetActiveWindow(self.game_hwnd)
                
                # Alt+Tab simulation to switch to the window
                user32.keybd_event(0x12, 0, 0, 0)  # Alt down
                user32.keybd_event(0x09, 0, 0, 0)  # Tab down
                user32.keybd_event(0x09, 0, 2, 0)  # Tab up
                user32.keybd_event(0x12, 0, 2, 0)  # Alt up
                
                # Wait and verify
                time.sleep(verification_time)
                if win32gui.GetForegroundWindow() == self.game_hwnd:
                    logger.info("Focus technique 2 successful")
                    return True
                else:
                    logger.warning("Focus technique 2 reported success but verification failed")
            except Exception as e:
                logger.error(f"Error using Alt+Tab technique: {e}")
            
        # Technique 3: Manipulate window Z-order
        logger.info("Trying focus technique 3...")
        try:
            # Get current foreground window
            fg_hwnd = win32gui.GetForegroundWindow()
            
            # Temporarily disable window if it's the foreground window (force context switch)
            if fg_hwnd != 0:
                win32gui.EnableWindow(fg_hwnd, False)
                
            # Force our window to top and activate
            win32gui.BringWindowToTop(self.game_hwnd)
            win32gui.SetForegroundWindow(self.game_hwnd)
            
            # Re-enable previous window
            if fg_hwnd != 0:
                win32gui.EnableWindow(fg_hwnd, True)
                
            # Wait and verify
            time.sleep(verification_time)
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                logger.info("Focus technique 3 successful")
                return True
            else:
                logger.warning("Focus technique 3 reported success but verification failed")
        except Exception as e:
            logger.error(f"Error using BringWindowToTop technique: {e}")
        
        # Technique 4: Use thread attachment
        logger.info("Trying focus technique 4...")
        try:
            # Get IDs of foreground window thread and our thread
            cur_thread = ctypes.windll.user32.GetCurrentThreadId()
            if not cur_thread:
                raise Exception("function 'GetCurrentThreadId' not found")
            
            # For workaround, use an alternative
            import threading
            cur_thread = threading.get_ident()
            
            fg_hwnd = win32gui.GetForegroundWindow()
            fg_thread = ctypes.windll.user32.GetWindowThreadProcessId(fg_hwnd, None)
            
            # Attach threads
            result = ctypes.windll.user32.AttachThreadInput(cur_thread, fg_thread, True)
            
            # Set our window to foreground
            win32gui.BringWindowToTop(self.game_hwnd)
            win32gui.SetForegroundWindow(self.game_hwnd)
            
            # Detach threads
            ctypes.windll.user32.AttachThreadInput(cur_thread, fg_thread, False)
            
            # Wait and verify
            time.sleep(verification_time)
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                logger.info("Focus technique 4 successful")
                return True
        except Exception as e:
            logger.error(f"Error using thread attachment method: {e}")
        
        logger.error("All focus techniques failed")
        return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a frame from the screen.
        
        Returns:
            numpy.ndarray: Captured frame as numpy array in RGB format,
                          or None if capture failed
        """
        logger.critical("Starting screen capture")
        
        if self.use_mock:
            logger.critical("Using mock mode, generating mock frame")
            # Return a mock frame for testing
            mock_frame = self._generate_mock_frame()
            logger.critical(f"Mock frame generated: shape={mock_frame.shape}")
            return mock_frame
            
        if not self.initialized:
            logger.critical("ERROR: Screen capture not initialized")
            return None
            
        try:
            # Ensure game window is in focus before capturing
            if hasattr(self, 'game_hwnd') and self.game_hwnd and WIN32_AVAILABLE:
                foreground_hwnd = win32gui.GetForegroundWindow()
                if foreground_hwnd != self.game_hwnd:
                    logger.critical(f"Game window not in focus (current={foreground_hwnd}, game={self.game_hwnd}), attempting to refocus")
                    focus_success = self.focus_game_window()
                    logger.critical(f"Window focus attempt result: {focus_success}")
                else:
                    logger.critical("Game window is already in focus")
            
            # Throttle capture rate to target FPS
            current_time = time.time()
            elapsed = current_time - self.last_capture_time
            if elapsed < self.frame_interval:
                logger.critical(f"Throttling capture: elapsed={elapsed:.4f}s, sleeping for {self.frame_interval - elapsed:.4f}s")
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
                
                logger.critical(f"Capturing screen area: left={left}, top={top}, width={right-left}, height={bottom-top}")
                
                # Capture the specified region
                capture_start = time.time()
                sct_img = self.sct.grab(monitor)
                capture_duration = time.time() - capture_start
                logger.critical(f"Screen capture completed in {capture_duration:.4f}s")
                
                # Convert to numpy array (BGRA format)
                convert_start = time.time()
                img_np = np.array(sct_img)
                
                # Convert BGRA to RGB
                img_rgb = img_np[:, :, :3][:, :, ::-1]
                convert_duration = time.time() - convert_start
                
                logger.critical(f"Image conversion completed in {convert_duration:.4f}s: shape={img_rgb.shape}, dtype={img_rgb.dtype}")
                
                self.last_capture_time = time.time()
                return img_rgb
            else:
                logger.critical("ERROR: No valid capture area defined")
                return None
                
        except Exception as e:
            # Log detailed error information
            import traceback
            error_trace = traceback.format_exc()
            logger.critical(f"ERROR capturing frame: {e}")
            logger.critical(f"Error traceback: {error_trace}")
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