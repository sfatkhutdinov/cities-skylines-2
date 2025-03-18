import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class VisualChangeAnalyzer:
    """Analyzes visual changes between frames and learns to associate them with outcomes."""
    
    def __init__(self, memory_size: int = 1000):
        """Initialize the visual change analyzer.
        
        Args:
            memory_size (int): Maximum number of pattern-outcome pairs to remember
        """
        self.memory_size = memory_size
        self.pattern_memory = []  # Stores (pattern, outcome) pairs
        self.outcomes_memory = []  # Corresponding outcomes
        self.feature_downscale = (16, 16)  # Downsample size for memory efficiency
        
    def update_association(self, visual_change_pattern: np.ndarray, outcome: float) -> None:
        """Update association between a visual change pattern and observed outcome.
        
        Args:
            visual_change_pattern: Visual pattern (difference between frames)
            outcome: Observed outcome or reward
        """
        if visual_change_pattern is None:
            return
            
        # Preprocess pattern for memory efficiency
        pattern = self._preprocess_pattern(visual_change_pattern)
        
        # Store pattern and outcome
        self.pattern_memory.append(pattern)
        self.outcomes_memory.append(outcome)
        
        # Keep memory size under limit
        if len(self.pattern_memory) > self.memory_size:
            self.pattern_memory.pop(0)
            self.outcomes_memory.pop(0)
            
        logger.debug(f"Updated visual association memory, size: {len(self.pattern_memory)}")
        
    def predict_outcome(self, pattern: np.ndarray, k: int = 5) -> float:
        """Predict outcome for a given pattern based on past associations.
        
        Args:
            pattern: Visual pattern to predict for
            k: Number of neighbors to consider
            
        Returns:
            float: Predicted outcome
        """
        if not self.pattern_memory:
            return 0.0
            
        # Preprocess pattern
        pattern = self._preprocess_pattern(pattern)
        if pattern is None:
            return 0.0
            
        # Flatten pattern for easier distance calculation
        flattened = pattern.flatten()
        
        # Calculate distances to all stored patterns
        distances = []
        for stored_pattern in self.pattern_memory:
            # Handle potential shape mismatches
            if stored_pattern.size != flattened.size:
                continue
                
            # Compute distance
            distance = np.linalg.norm(stored_pattern - flattened)
            distances.append(distance)
            
        if not distances:
            return 0.0
            
        # Find k smallest distances
        indices = np.argsort(distances)[:k]
        
        # Weight by inverse distance
        weights = [1.0 / (distances[i] + 1e-6) for i in indices]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
            
        # Weighted average of outcomes
        prediction = sum(weights[i] * self.outcomes_memory[indices[i]] for i in range(k)) / total_weight
        
        return prediction
        
    def get_visual_change_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate a visual change score between frames.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            float: Visual change score (higher = more change)
        """
        # Input validation
        if prev_frame is None or curr_frame is None:
            return 0.0
            
        try:
            # Ensure frames are in RGB format
            if prev_frame.ndim == 4:
                prev_frame = prev_frame[0]  # Remove batch dimension
            if curr_frame.ndim == 4:
                curr_frame = curr_frame[0]
                
            # Convert to grayscale for basic change detection
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            
            # Calculate normalized change score
            change_score = np.mean(diff) / 255.0
            
            # Scale to make score more pronounced
            change_score = min(1.0, change_score * 5.0)
            
            return change_score
        except Exception as e:
            logger.error(f"Error computing visual change score: {e}")
            return 0.0
        
    def _preprocess_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Preprocess pattern for more efficient storage and comparison.
        
        Args:
            pattern: Raw visual pattern
            
        Returns:
            np.ndarray: Preprocessed pattern
        """
        try:
            # Ensure pattern is valid
            if pattern is None or pattern.size == 0:
                return None
                
            # Convert to grayscale if it has colors
            if pattern.ndim > 2 and pattern.shape[2] >= 3:
                pattern = cv2.cvtColor(pattern, cv2.COLOR_RGB2GRAY)
                
            # Resize for memory efficiency
            pattern = cv2.resize(pattern, self.feature_downscale)
            
            # Normalize
            pattern = pattern.astype(np.float32) / 255.0
            
            return pattern
        except Exception as e:
            logger.error(f"Error preprocessing pattern: {e}")
            return None
            
    def save_state(self, path: str) -> bool:
        """Save the analyzer state to disk.
        
        Args:
            path: Path to save the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            state = {
                'memory_size': self.memory_size,
                'pattern_memory': self.pattern_memory,
                'outcomes_memory': self.outcomes_memory,
                'feature_downscale': self.feature_downscale
            }
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save using pickle
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved visual change analyzer state with {len(self.pattern_memory)} patterns")
            return True
        except Exception as e:
            logger.error(f"Failed to save visual change analyzer state: {str(e)}")
            return False
            
    def load_state(self, path: str) -> bool:
        """Load the analyzer state from disk.
        
        Args:
            path: Path to load the state from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Visual change analyzer state file not found: {path}")
                return False
                
            # Load using pickle
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Restore state
            self.memory_size = state['memory_size']
            self.pattern_memory = state['pattern_memory']
            self.outcomes_memory = state['outcomes_memory']
            self.feature_downscale = state['feature_downscale']
            
            logger.info(f"Loaded visual change analyzer state with {len(self.pattern_memory)} patterns")
            return True
        except Exception as e:
            logger.error(f"Failed to load visual change analyzer state: {str(e)}")
            # Initialize with empty state
            self.pattern_memory = []
            self.outcomes_memory = []
            return False 