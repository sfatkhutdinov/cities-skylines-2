"""
Optimized screen capture module for Cities: Skylines 2.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import ImageGrab
import time
import logging
from typing import Optional, Tuple, List, Dict, Any, Union
import threading
import queue

logger = logging.getLogger(__name__)

class OptimizedScreenCapture:
    """Optimized screen capture with hardware acceleration when available."""
    
    def __init__(self, config):
        """Initialize screen capture with hardware configuration."""
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Pre-allocate tensors for frame processing
        # Use a smaller resolution for processing (320x240) regardless of game window size
        self.capture_resolution = (1920, 1080)  # Full HD game window
        self.process_resolution = (320, 240)    # Downsampled for neural network
        
        # Initialize buffers on appropriate device
        self.frame_buffer = torch.zeros((1, 3, self.process_resolution[1], self.process_resolution[0]), 
                                     dtype=self.dtype, device=self.device)
        
        # Create CUDA stream if using CUDA for parallel processing
        self.cuda_stream = torch.cuda.Stream() if torch.cuda.is_available() and 'cuda' in self.device.type else None
        
        # Create a mock frame for testing without screen capture
        self.mock_frame = torch.rand((3, self.process_resolution[1], self.process_resolution[0]), 
                                  dtype=torch.float32, device=self.device)
        self.use_mock = False
        
        # Store a history of frames for temporal processing
        self.frame_history: List[torch.Tensor] = []
        self.max_history_length = 4
        
        # Game window information
        self.game_hwnd = None
        self.game_window_position = None
        self.client_position = None
        
        # For multi-threaded capture
        self.capture_thread = None
        self.frame_queue = queue.Queue(maxsize=3)  # Buffer a few frames
        self.stop_capture_thread = threading.Event()
        self.capture_thread_lock = threading.Lock()  # Add thread lock for safety
        self.use_threading = True  # Can be disabled for debugging
        
        # Frame quality control
        self.blank_frame_threshold = 0.02  # Threshold for detecting blank frames (very low std dev)
        self.previous_valid_frame = None   # Store last valid frame for fallback
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5
        
        # Adaptive resolution
        self.dynamic_resolution = True
        self.performance_history = []
        self.min_resolution = (160, 120)
        self.max_resolution = (640, 480)
        
        logger.info(f"Screen capture initialized with resolution {self.process_resolution}")
        logger.info(f"CUDA acceleration: {'Enabled' if self.cuda_stream else 'Disabled'}")
        
        # Start capture thread if threading is enabled
        if self.use_threading:
            self._start_capture_thread()
            
    def __del__(self):
        """Ensure thread cleanup on deletion."""
        self.stop_capture_thread.set()
        if hasattr(self, 'capture_thread') and self.capture_thread:
            if self.capture_thread.is_alive():
                logger.info("Cleaning up capture thread on deletion")
                try:
                    self.capture_thread.join(timeout=1.0)
                except Exception:
                    pass  # Best effort to join thread
    
    def _start_capture_thread(self):
        """Start a background thread for continuous frame capture."""
        with self.capture_thread_lock:
            if self.capture_thread is not None and self.capture_thread.is_alive():
                return
                
            logger.info("Starting background capture thread")
            self.stop_capture_thread.clear()
            self.capture_thread = threading.Thread(target=self._capture_thread_worker, daemon=True)
            self.capture_thread.start()
    
    def _capture_thread_worker(self):
        """Worker function for continuous frame capture in background thread."""
        logger.info("Capture thread started")
        
        while not self.stop_capture_thread.is_set():
            try:
                # Capture frame
                frame = self._capture_frame_direct()
                
                # Validate frame quality
                if frame is not None:
                    # Check if frame is blank or invalid
                    is_valid = self._validate_frame(frame)
                    
                    if is_valid and not self.frame_queue.full():
                        # Don't block if queue is full, just skip this frame
                        try:
                            self.frame_queue.put(frame, block=False)
                        except queue.Full:
                            pass
                    elif not is_valid:
                        logger.warning("Captured invalid frame, using fallback")
                        # Use previous valid frame if available
                        if self.previous_valid_frame is not None and not self.frame_queue.full():
                            try:
                                self.frame_queue.put(self.previous_valid_frame, block=False)
                            except queue.Full:
                                pass
                        
                # Small sleep to avoid excessive CPU usage
                time.sleep(0.01)
            except Exception as e:
                logger.warning(f"Error in capture thread: {e}")
                time.sleep(0.1)  # Longer sleep on error
                
        logger.info("Capture thread stopped")
    
    def _validate_frame(self, frame: torch.Tensor) -> bool:
        """Validate frame quality to detect blank or invalid frames.
        
        Args:
            frame (torch.Tensor): Frame to validate
            
        Returns:
            bool: True if frame is valid, False otherwise
        """
        if frame is None:
            return False
            
        try:
            # Check for blank or low-variation frames
            std_dev = torch.std(frame).item()
            if std_dev < self.blank_frame_threshold:
                return False
                
            # Check for abnormal brightness (all white or all black frames)
            mean_val = torch.mean(frame).item()
            if mean_val < 0.02 or mean_val > 0.98:
                return False
                
            # Check for valid shape (ensure it's within expected range)
            if len(frame.shape) != 3 or frame.shape[0] != 3:
                return False
            
            # Frame is valid
            return True
        except Exception as e:
            logger.warning(f"Error validating frame: {e}")
            return False
            
    def capture_frame(self) -> torch.Tensor:
        """Capture and process a single frame.
        
        Returns:
            torch.Tensor: Processed frame as tensor [C, H, W]
        """
        # If in mock mode, return the mock frame with some noise
        if self.use_mock:
            # Add some random noise to simulate changes
            noise = torch.randn_like(self.mock_frame) * 0.05
            self.mock_frame = torch.clamp(self.mock_frame + noise, 0, 1)
            
            # Add to history
            self._update_frame_history(self.mock_frame)
            
            return self.mock_frame
            
        # If using threaded capture, get frame from queue
        if self.use_threading and hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.is_alive():
            try:
                # Try to get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.2)  # Increased timeout for better reliability
                
                # Validate frame
                if self._validate_frame(frame):
                    # Add to history
                    self._update_frame_history(frame)
                    
                    # Store as previous valid frame
                    self.previous_valid_frame = frame.clone()  # Use clone to prevent modification
                    self.consecutive_failures = 0
                    
                    return frame
                else:
                    logger.warning("Retrieved invalid frame from queue")
                    self.consecutive_failures += 1
            except queue.Empty:
                logger.warning("Frame queue empty, capturing directly")
                self.consecutive_failures += 1
                # Queue empty, capture directly
            except Exception as e:
                logger.warning(f"Error getting frame from queue: {e}")
                self.consecutive_failures += 1
                
        # Direct capture as fallback
        try:
            frame = self._capture_frame_direct()
            
            # Validate frame 
            if self._validate_frame(frame):
                # Store as previous valid frame
                self.previous_valid_frame = frame.clone()
                self.consecutive_failures = 0
                
                # Add to history
                self._update_frame_history(frame)
                
                return frame
            else:
                logger.warning("Direct capture returned invalid frame")
                self.consecutive_failures += 1
        except Exception as e:
            logger.warning(f"Error in direct capture: {e}")
            self.consecutive_failures += 1
            
        # If we have too many consecutive failures, restart the capture thread
        if self.consecutive_failures > self.max_consecutive_failures:
            logger.warning(f"Too many consecutive failures ({self.consecutive_failures}), restarting capture thread")
            self._restart_capture_thread()
            self.consecutive_failures = 0
        
        # Return previous valid frame as fallback if available
        if self.previous_valid_frame is not None:
            return self.previous_valid_frame
            
        # Last resort fallback: return a blank frame
        logger.warning("No valid frame available, returning blank frame")
        return torch.zeros((3, self.process_resolution[1], self.process_resolution[0]), 
                         dtype=self.dtype, device=self.device)
                         
    def _restart_capture_thread(self):
        """Restart the capture thread after failures."""
        with self.capture_thread_lock:
            logger.info("Restarting capture thread")
            
            # Stop existing thread
            self.stop_capture_thread.set()
            if self.capture_thread and self.capture_thread.is_alive():
                try:
                    self.capture_thread.join(timeout=1.0)
                except Exception:
                    pass  # Best effort
                    
            # Clear queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except Exception:
                    pass
                    
            # Start new thread
            self.stop_capture_thread.clear()
            self.capture_thread = threading.Thread(target=self._capture_thread_worker, daemon=True)
            self.capture_thread.start()
            logger.info("Capture thread restarted")
    
    def _capture_frame_direct(self) -> Optional[torch.Tensor]:
        """Internal method to capture frame directly without threading.
        
        Returns:
            Optional[torch.Tensor]: Captured frame or None on failure
        """
        # Try to capture screen
        try:
            # First attempt to capture the game window if possible
            try:
                import win32gui
                import win32ui
                import win32con
                from ctypes import windll
                
                # Get handle to the game window 
                if self.game_hwnd is None:
                    game_hwnd = None
                    window_titles = [
                        "Cities: Skylines II",
                        "Cities: Skylines",
                        "Cities Skylines II",
                        "Cities Skylines"
                    ]
                    
                    def enum_windows_callback(hwnd, _):
                        nonlocal game_hwnd
                        if win32gui.IsWindowVisible(hwnd):
                            window_text = win32gui.GetWindowText(hwnd)
                            for title in window_titles:
                                if title.lower() in window_text.lower():
                                    game_hwnd = hwnd
                                    return False
                        return True
                    
                    win32gui.EnumWindows(enum_windows_callback, None)
                    self.game_hwnd = game_hwnd
                
                if self.game_hwnd:
                    # Get window dimensions
                    try:
                        left, top, right, bottom = win32gui.GetWindowRect(self.game_hwnd)
                        width = right - left
                        height = bottom - top
                        
                        # Store window position for mouse coordinate translation
                        self.game_window_position = (left, top, right, bottom)
                        
                        # Get client area (excludes window borders, title bar)
                        client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(self.game_hwnd)
                        client_width = client_right - client_left
                        client_height = client_bottom - client_top
                        
                        # Calculate client area position within the window
                        client_left, client_top = win32gui.ClientToScreen(self.game_hwnd, (client_left, client_top))
                        client_right = client_left + client_width
                        client_bottom = client_top + client_height
                        
                        # Store client position for mouse coordinate conversion
                        self.client_position = (client_left, client_top, client_right, client_bottom)
                    except Exception as e:
                        logger.warning(f"Error getting window rect: {e}")
                        return None
                    
                    # Create device context and bitmap
                    try:
                        hwnd_dc = win32gui.GetWindowDC(self.game_hwnd)
                        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                        save_dc = mfc_dc.CreateCompatibleDC()
                        
                        # Create bitmap
                        save_bitmap = win32ui.CreateBitmap()
                        save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                        save_dc.SelectObject(save_bitmap)
                        
                        # Copy window content
                        result = windll.user32.PrintWindow(self.game_hwnd, save_dc.GetSafeHdc(), 3)
                        
                        # Convert to numpy array
                        bmpinfo = save_bitmap.GetInfo()
                        bmpstr = save_bitmap.GetBitmapBits(True)
                        
                        frame = np.frombuffer(bmpstr, dtype=np.uint8)
                        frame = frame.reshape((height, width, 4))
                        frame = frame[:, :, :3]  # Remove alpha channel
                        
                        # Clean up
                        save_dc.DeleteDC()
                        mfc_dc.DeleteDC()
                        win32gui.ReleaseDC(self.game_hwnd, hwnd_dc)
                        win32gui.DeleteObject(save_bitmap.GetHandle())
                        
                        # If successful, proceed with this frame
                        if result == 1:
                            # Resize to our target processing resolution using cv2 for better quality
                            frame_resized = cv2.resize(frame, 
                                                    (self.process_resolution[0], self.process_resolution[1]),
                                                    interpolation=cv2.INTER_AREA)
                        else:
                            # Fallback to screen capture
                            logger.warning("PrintWindow failed - falling back to screen capture")
                            return self._fallback_screen_capture()
                    except Exception as e:
                        logger.warning(f"Error capturing window: {e}")
                        return self._fallback_screen_capture()
                else:
                    # Fallback to screen capture
                    logger.warning("Game window not found - falling back to screen capture")
                    return self._fallback_screen_capture()
            except Exception as e:
                # Fallback to screen capture
                logger.warning(f"Error in window capture: {e}")
                return self._fallback_screen_capture()
            
            # Convert from BGR to RGB
            if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Process the frame with CUDA if available
            if self.cuda_stream:
                with torch.cuda.stream(self.cuda_stream):
                    # Convert to tensor on GPU directly
                    frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
                    frame_tensor = frame_tensor.to(self.device, non_blocking=True)
            else:
                # Process on CPU
                frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
                
                # Move to appropriate device
                if frame_tensor.device != self.device:
                    frame_tensor = frame_tensor.to(self.device)
            
            # Validate frame
            if not self._is_valid_frame(frame_tensor):
                logger.warning("Invalid frame captured")
                return None
                
            return frame_tensor
        except Exception as e:
            logger.error(f"Screen capture failed: {str(e)}")
            return None
    
    def _fallback_screen_capture(self) -> Optional[torch.Tensor]:
        """Fallback screen capture method using PIL.ImageGrab.
        
        Returns:
            Optional[torch.Tensor]: Captured frame or None on failure
        """
        try:
            # Capture entire screen
            screen = ImageGrab.grab()
            
            # Convert to numpy array
            frame = np.array(screen)
            
            # Resize to target resolution
            frame_resized = cv2.resize(frame, 
                                     (self.process_resolution[0], self.process_resolution[1]),
                                     interpolation=cv2.INTER_AREA)
            
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
            
            # Move to appropriate device
            if frame_tensor.device != self.device:
                frame_tensor = frame_tensor.to(self.device)
                
            return frame_tensor
        except Exception as e:
            logger.error(f"Fallback screen capture failed: {str(e)}")
            return None
            
    def _is_valid_frame(self, frame: torch.Tensor) -> bool:
        """Check if a frame is valid (not blank, frozen, etc.).
        
        Args:
            frame (torch.Tensor): Frame to check
            
        Returns:
            bool: True if frame is valid, False otherwise
        """
        if frame is None:
            return False
            
        # Check if frame is blank (very low standard deviation)
        std_dev = torch.std(frame).item()
        if std_dev < self.blank_frame_threshold:
            logger.warning(f"Blank frame detected (std_dev: {std_dev:.6f})")
            return False
            
        # If we have a previous frame, check if it's too similar (frozen)
        if len(self.frame_history) > 0:
            last_frame = self.frame_history[-1]
            diff = torch.mean(torch.abs(frame - last_frame)).item()
            if diff < 0.001:  # Almost identical frames
                # This might be valid for idle situations, so don't consider it an error
                logger.debug(f"Nearly identical frame (diff: {diff:.6f})")
                # Still consider it valid, just logging for debugging
            
        return True
            
    def _update_frame_history(self, frame: torch.Tensor):
        """Update frame history for temporal processing.
        
        Args:
            frame (torch.Tensor): Frame to add to history
        """
        if frame is None:
            return
            
        self.frame_history.append(frame.clone())
        while len(self.frame_history) > self.max_history_length:
            self.frame_history.pop(0)
    
    def get_frame_stack(self) -> torch.Tensor:
        """Get stacked frames for temporal processing.
        
        Returns:
            torch.Tensor: Stacked frames [C*n, H, W]
        """
        if not self.frame_history:
            return self.capture_frame().unsqueeze(0)
            
        # Ensure we have enough frames
        while len(self.frame_history) < self.max_history_length:
            self.frame_history.insert(0, self.frame_history[0].clone())
            
        # Stack frames along channel dimension
        return torch.cat(self.frame_history, dim=0)
        
    def compute_frame_difference(self) -> torch.Tensor:
        """Compute difference between current and previous frame.
        
        Returns:
            torch.Tensor: Absolute difference between frames
        """
        if len(self.frame_history) < 2:
            return torch.zeros_like(self.frame_history[0] if self.frame_history else self.capture_frame())
            
        return torch.abs(self.frame_history[-1] - self.frame_history[-2])
    
    def adjust_resolution(self, fps: float):
        """Dynamically adjust processing resolution based on FPS.
        
        Args:
            fps (float): Current FPS
        """
        if not self.dynamic_resolution:
            return
            
        # Add FPS to history
        self.performance_history.append(fps)
        if len(self.performance_history) > 30:  # 30 second window
            self.performance_history.pop(0)
            
        # Calculate average FPS
        avg_fps = sum(self.performance_history) / len(self.performance_history)
        
        # Adjust resolution based on FPS
        # Target is 30 FPS
        if avg_fps < 25:  # Too slow
            # Decrease resolution
            new_width = max(self.process_resolution[0] - 32, self.min_resolution[0])
            new_height = max(self.process_resolution[1] - 24, self.min_resolution[1])
            if (new_width, new_height) != self.process_resolution:
                logger.info(f"Reducing resolution to {new_width}x{new_height} (FPS: {avg_fps:.1f})")
                self.process_resolution = (new_width, new_height)
        elif avg_fps > 35 and self.process_resolution != self.max_resolution:  # Fast enough to increase
            # Increase resolution
            new_width = min(self.process_resolution[0] + 32, self.max_resolution[0])
            new_height = min(self.process_resolution[1] + 24, self.max_resolution[1])
            if (new_width, new_height) != self.process_resolution:
                logger.info(f"Increasing resolution to {new_width}x{new_height} (FPS: {avg_fps:.1f})")
                self.process_resolution = (new_width, new_height)
                
    def close(self):
        """Clean up resources."""
        # Stop capture thread
        if self.capture_thread and self.capture_thread.is_alive():
            logger.info("Stopping capture thread")
            self.stop_capture_thread.set()
            self.capture_thread.join(timeout=1.0)
            
        # Clear tensor memory
        self.frame_buffer = None
        self.frame_history = []
        self.previous_valid_frame = None
        
        # Clear CUDA memory if using GPU
        if hasattr(self.config, 'use_cuda') and self.config.use_cuda:
            logger.info("Clearing CUDA memory")
            torch.cuda.empty_cache()

    def get_resolution(self) -> Tuple[int, int]:
        """Get the current screen resolution.
        
        Returns:
            Tuple[int, int]: Width and height of the screen
        """
        # If we have stored the game window position, use that
        if hasattr(self, 'game_window_position') and self.game_window_position:
            left, top, right, bottom = self.game_window_position
            width = right - left
            height = bottom - top
            return width, height
        
        # If we have stored the client position, use that
        if hasattr(self, 'client_position') and self.client_position:
            client_left, client_top, client_right, client_bottom = self.client_position
            width = client_right - client_left
            height = client_bottom - client_top
            return width, height
        
        # Fallback to capture resolution
        if hasattr(self, 'capture_resolution'):
            return self.capture_resolution
        
        # Ultimate fallback: try to get screen size from PIL
        try:
            img = ImageGrab.grab()
            return img.width, img.height
        except Exception as e:
            logger.error(f"Failed to get screen resolution: {e}")
            # Return a default resolution
            return 1920, 1080 