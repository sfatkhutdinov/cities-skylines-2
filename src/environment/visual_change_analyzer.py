import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Union, Any
import logging
from collections import deque

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
            k: Number of nearest neighbors to consider
            
        Returns:
            float: Predicted outcome
        """
        if not self.pattern_memory or pattern is None:
            return 0.0
            
        # Preprocess pattern
        pattern = self._preprocess_pattern(pattern)
        
        # Find k nearest neighbors
        distances = []
        for mem_pattern in self.pattern_memory:
            try:
                # Calculate pattern similarity
                distance = np.mean(np.abs(pattern - mem_pattern))
                distances.append(distance)
            except Exception as e:
                logger.error(f"Error calculating pattern distance: {e}")
                distances.append(float('inf'))
                
        # Get indices of k nearest neighbors
        if not distances:
            return 0.0
            
        k = min(k, len(distances))
        nearest_indices = np.argsort(distances)[:k]
        
        # Get corresponding outcomes
        nearest_outcomes = [self.outcomes_memory[i] for i in nearest_indices]
        nearest_distances = [distances[i] for i in nearest_indices]
        
        # Weight outcomes by inverse distance
        weights = []
        for dist in nearest_distances:
            if dist < 1e-10:  # Avoid division by zero
                weights.append(1.0)
            else:
                weights.append(1.0 / dist)
                
        # Normalize weights
        sum_weights = sum(weights)
        if sum_weights > 0:
            weights = [w / sum_weights for w in weights]
            
        # Calculate weighted average
        prediction = sum(w * o for w, o in zip(weights, nearest_outcomes))
        
        return prediction
        
    def get_visual_change_score(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
        """Calculate a visual change score between frames.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            float: Visual change score (0.0 to 1.0)
        """
        if prev_frame is None or curr_frame is None:
            return 0.0
            
        # Make sure frames have the same shape
        if prev_frame.shape != curr_frame.shape:
            try:
                curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
            except Exception as e:
                logger.error(f"Frame resize failed: {e}")
                return 0.0
                
        # Convert to grayscale if needed
        if prev_frame.ndim == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if curr_frame.ndim == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
            
        # Calculate difference
        try:
            diff = cv2.absdiff(prev_gray, curr_gray)
            score = np.mean(diff) / 255.0
            return score
        except Exception as e:
            logger.error(f"Error calculating visual change: {e}")
            return 0.0
    
    def _preprocess_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Preprocess pattern for memory efficiency.
        
        Args:
            pattern: Input pattern
            
        Returns:
            np.ndarray: Preprocessed pattern
        """
        if pattern is None:
            return np.zeros(self.feature_downscale, dtype=np.float32)
            
        try:
            # Convert to grayscale if needed
            if pattern.ndim == 3:
                pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
                
            # Resize for memory efficiency
            pattern = cv2.resize(pattern, self.feature_downscale, interpolation=cv2.INTER_AREA)
            
            # Normalize
            pattern = pattern.astype(np.float32) / 255.0
            
            return pattern
        except Exception as e:
            logger.error(f"Pattern preprocessing failed: {e}")
            return np.zeros(self.feature_downscale, dtype=np.float32) 