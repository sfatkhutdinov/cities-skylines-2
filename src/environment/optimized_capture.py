"""
Optimized screen capture module for Cities: Skylines 2.
"""

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import ImageGrab
import time
from typing import Optional, Tuple, List

class OptimizedScreenCapture:
    """Optimized screen capture with hardware acceleration when available."""
    
    def __init__(self, config):
        """Initialize screen capture with hardware configuration."""
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Get resolution from config if available
        height, width = getattr(config, 'resolution', (240, 320))
        
        # Pre-allocate tensors for frame processing
        # Use the configured resolution for processing
        self.capture_resolution = (1920, 1080)  # Full HD game window
        self.process_resolution = (width, height)  # Use configuration resolution
        
        # Initialize buffers on appropriate device
        self.frame_buffer = torch.zeros((1, 3, self.process_resolution[1], self.process_resolution[0]), 
                                       dtype=self.dtype, device=self.device)
        self.prev_frame = torch.zeros((1, 3, self.process_resolution[1], self.process_resolution[0]), 
                                     dtype=self.dtype, device=self.device)
        
        # Create empty CUDA stream if using CUDA
        self.stream = torch.cuda.Stream() if hasattr(config, 'use_cuda') and config.use_cuda else None
        
        # Create a mock frame for testing without screen capture
        self.mock_frame = torch.rand((3, self.process_resolution[1], self.process_resolution[0]), 
                                    dtype=torch.float32, device=self.device)
        self.use_mock = False
        
        # Store a history of frames for temporal processing
        self.frame_history: List[torch.Tensor] = []
        self.max_history_length = 4
        
    def capture_frame(self) -> torch.Tensor:
        """Capture and process a single frame."""
        # If in mock mode, return the mock frame
        if self.use_mock:
            # Add some random noise to simulate changes
            noise = torch.randn_like(self.mock_frame) * 0.05
            self.mock_frame = torch.clamp(self.mock_frame + noise, 0, 1)
            
            # Add to history
            self._update_frame_history(self.mock_frame)
            
            return self.mock_frame
            
        # Try to capture screen, don't fall back to mock if it fails
        try:
            # First attempt to capture the game window if possible
            try:
                import win32gui
                import win32ui
                import win32con
                from ctypes import windll
                
                # Get handle to the game window 
                game_hwnd = None
                window_titles = [
                    "Cities: Skylines II"
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
                
                if game_hwnd:
                    # Get window dimensions - Fixed to get actual window dimensions properly
                    left, top, right, bottom = win32gui.GetWindowRect(game_hwnd)
                    width = right - left
                    height = bottom - top
                    
                    # Store window position for mouse coordinate translation
                    self.game_window_position = (left, top, right, bottom)
                    
                    # Get client area (excludes window borders, title bar)
                    client_left, client_top, client_right, client_bottom = win32gui.GetClientRect(game_hwnd)
                    client_width = client_right - client_left
                    client_height = client_bottom - client_top
                    
                    # Calculate client area position within the window
                    client_left, client_top = win32gui.ClientToScreen(game_hwnd, (client_left, client_top))
                    client_right = client_left + client_width
                    client_bottom = client_top + client_height
                    
                    # Store client position for mouse coordinate conversion
                    self.client_position = (client_left, client_top, client_right, client_bottom)
                    
                    # Log window dimensions for debugging
                    print(f"Game window: {self.game_window_position}")
                    print(f"Client area: {self.client_position}")
                    print(f"Client dimensions: {client_width}x{client_height}")
                    
                    # Create device context and bitmap
                    hwnd_dc = win32gui.GetWindowDC(game_hwnd)
                    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
                    save_dc = mfc_dc.CreateCompatibleDC()
                    
                    # Create bitmap
                    save_bitmap = win32ui.CreateBitmap()
                    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
                    save_dc.SelectObject(save_bitmap)
                    
                    # Copy window content
                    result = windll.user32.PrintWindow(game_hwnd, save_dc.GetSafeHdc(), 3)
                    
                    # Convert to numpy array
                    bmpinfo = save_bitmap.GetInfo()
                    bmpstr = save_bitmap.GetBitmapBits(True)
                    
                    frame = np.frombuffer(bmpstr, dtype=np.uint8)
                    frame = frame.reshape((height, width, 4))
                    frame = frame[:, :, :3]  # Remove alpha channel
                    
                    # Clean up
                    save_dc.DeleteDC()
                    mfc_dc.DeleteDC()
                    win32gui.ReleaseDC(game_hwnd, hwnd_dc)
                    win32gui.DeleteObject(save_bitmap.GetHandle())
                    
                    # If successful, proceed with this frame
                    if result == 1:
                        # Resize to our target processing resolution using cv2 for better quality
                        frame_resized = cv2.resize(frame, 
                                                 (self.process_resolution[0], self.process_resolution[1]),
                                                 interpolation=cv2.INTER_AREA)
                    else:
                        # Raise exception instead of falling back
                        raise RuntimeError("PrintWindow failed - could not capture game window")
                else:
                    # Raise exception instead of falling back
                    raise RuntimeError("Could not find Cities: Skylines II window")
            except Exception as e:
                # Raise the exception instead of falling back to full screen capture
                raise RuntimeError(f"Failed to capture game window: {str(e)}")
            
            # Convert from BGR to RGB if needed
            if len(frame_resized.shape) == 3 and frame_resized.shape[2] == 3:
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to tensor on CPU first and normalize
            frame_tensor = torch.from_numpy(frame_resized).float().permute(2, 0, 1) / 255.0
            
            # Move to appropriate device
            if frame_tensor.device != self.device:
                frame_tensor = frame_tensor.to(self.device)
                
            # Add to history
            self._update_frame_history(frame_tensor)
                
            return frame_tensor
        except Exception as e:
            # Raise the exception instead of falling back to mock mode
            raise RuntimeError(f"Screen capture failed: {str(e)}")
            
    def _update_frame_history(self, frame: torch.Tensor):
        """Update frame history for temporal processing."""
        self.frame_history.append(frame.clone())
        if len(self.frame_history) > self.max_history_length:
            self.frame_history.pop(0)
    
    def get_frame_stack(self) -> torch.Tensor:
        """Get stacked frames for temporal processing."""
        if not self.frame_history:
            return self.capture_frame().unsqueeze(0)
            
        # Ensure we have enough frames
        while len(self.frame_history) < self.max_history_length:
            self.frame_history.insert(0, self.frame_history[0].clone())
            
        # Stack frames along channel dimension
        return torch.cat(self.frame_history, dim=0)
        
    def compute_frame_difference(self) -> torch.Tensor:
        """Compute difference between current and previous frame."""
        if len(self.frame_history) < 2:
            return torch.zeros_like(self.frame_history[0])
            
        return torch.abs(self.frame_history[-1] - self.frame_history[-2])
        
    def close(self):
        """Clean up resources."""
        # Clear tensor memory
        self.frame_buffer = None
        self.prev_frame = None
        self.frame_history = []
        
        # Clear CUDA memory if using GPU
        if hasattr(self.config, 'use_cuda') and self.config.use_cuda:
            torch.cuda.empty_cache() 