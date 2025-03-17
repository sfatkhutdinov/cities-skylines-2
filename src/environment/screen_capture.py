import numpy as np
import mss
import cv2
from typing import Tuple, Optional

class ScreenCapture:
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        """Initialize screen capture with specified resolution."""
        self.resolution = resolution
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]  # Primary monitor
        
    def capture_frame(self) -> np.ndarray:
        """Capture a single frame from the screen.
        
        Returns:
            np.ndarray: RGB frame with shape (height, width, 3)
        """
        # Capture the screen
        screenshot = self.sct.grab(self.monitor)
        
        # Convert to numpy array and ensure RGB format
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        # Resize to desired resolution if necessary
        if frame.shape[:2] != self.resolution:
            frame = cv2.resize(frame, self.resolution)
            
        return frame
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for neural network input.
        
        Args:
            frame (np.ndarray): Raw RGB frame
            
        Returns:
            np.ndarray: Preprocessed frame normalized to [0, 1]
        """
        # Normalize to [0, 1]
        frame = frame.astype(np.float32) / 255.0
        return frame 