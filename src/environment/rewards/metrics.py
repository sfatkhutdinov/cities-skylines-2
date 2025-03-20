"""
Reward metrics for Cities: Skylines 2 environment.

This module provides tools for measuring and tracking reward-related metrics.
"""

import torch
import numpy as np
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)

class DensityEstimator:
    """Tracks density of visited states in feature space for novelty detection."""
    
    def __init__(self, feature_dim=512, history_size=2000, device=None):
        """Initialize state density estimator.
        
        Args:
            feature_dim: Dimension of feature vectors
            history_size: Maximum number of states to keep in memory
            device: Device for computations
        """
        self.feature_dim = feature_dim
        self.history_size = history_size
        self.device = device
        self.state_memory = []
        # Alias state_memory as memory for compatibility
        self.memory = self.state_memory
        self.mean_vector = None
        self.std_vector = None
        
    def compute_novelty(self, state_embedding: torch.Tensor) -> float:
        """Compute novelty score based on distance to nearest neighbors.
        
        Args:
            state_embedding: Feature representation of state
            
        Returns:
            float: Novelty score (higher = more novel)
        """
        if not self.state_memory:
            # First state is always novel
            return 1.0
            
        # Detach and move to CPU for numpy compatibility
        if state_embedding.requires_grad:
            state_embedding = state_embedding.detach()
        state_np = state_embedding.cpu().numpy()
        
        # Ensure we have a flat vector
        if state_np.ndim > 1:
            state_np = state_np.flatten()
            
        # Compute distances to all stored states
        distances = []
        for state in self.state_memory:
            if state.shape != state_np.shape:
                continue  # Skip incompatible states
            distance = np.linalg.norm(state - state_np)
            distances.append(distance)
            
        if not distances:
            return 1.0
            
        # Find minimum distance to any previously seen state
        min_distance = min(distances)
        
        # Normalize by mean if we have enough data
        if len(distances) > 10:
            mean_distance = np.mean(distances)
            min_distance = min_distance / mean_distance if mean_distance > 0 else min_distance
            
        # Higher distance = more novel
        # Scale to a reasonable range (0-1)
        novelty = min(1.0, min_distance)
        
        return float(novelty)
    
    def update(self, state_embedding: torch.Tensor) -> None:
        """Update memory with new state.
        
        Args:
            state_embedding: Feature representation of state
        """
        # Detach and move to CPU for numpy compatibility
        if state_embedding.requires_grad:
            state_embedding = state_embedding.detach()
        state_np = state_embedding.cpu().numpy()
        
        # Ensure we have a flat vector
        if state_np.ndim > 1:
            state_np = state_np.flatten()
            
        # Add to memory
        self.state_memory.append(state_np)
        
        # Keep memory size under limit
        if len(self.state_memory) > self.history_size:
            self.state_memory.pop(0)
            
        # Periodically update statistics
        if len(self.state_memory) % 100 == 0:
            self._update_statistics()
    
    def _update_statistics(self) -> None:
        """Update mean and standard deviation of states in memory."""
        if not self.state_memory:
            return
            
        # Stack all states
        states = np.stack(self.state_memory)
        
        # Compute mean and std
        self.mean_vector = np.mean(states, axis=0)
        self.std_vector = np.std(states, axis=0)
    
    def save_state(self, path: str) -> None:
        """Save density estimator state to disk.
        
        Args:
            path: Path to save the state
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save state using pickle
            state = {
                'feature_dim': self.feature_dim,
                'history_size': self.history_size,
                'state_memory': self.state_memory,
                'mean_vector': self.mean_vector,
                'std_vector': self.std_vector,
            }
            
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved density estimator state with {len(self.state_memory)} states")
        except Exception as e:
            logger.error(f"Error saving density estimator state: {e}")
    
    def load_state(self, path: str) -> None:
        """Load density estimator state from disk.
        
        Args:
            path: Path to load the state from
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Density estimator state file not found: {path}")
                return
                
            # Load state using pickle
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Restore state
            self.feature_dim = state['feature_dim']
            self.history_size = state['history_size']
            self.state_memory = state['state_memory']
            self.memory = self.state_memory  # Update alias
            self.mean_vector = state['mean_vector']
            self.std_vector = state['std_vector']
            
            logger.info(f"Loaded density estimator state with {len(self.state_memory)} states")
        except Exception as e:
            logger.error(f"Error loading density estimator state: {e}")


class TemporalAssociationMemory:
    """Associates state features with outcomes for reward prediction."""
    
    def __init__(self, feature_dim=512, history_size=1000, device=None):
        """Initialize temporal association memory.
        
        Args:
            feature_dim: Dimension of feature vectors
            history_size: Maximum history size
            device: Compute device
        """
        self.feature_dim = feature_dim
        self.history_size = history_size
        self.device = device
        
        # Storage for features and outcomes
        self.features = []
        self.outcomes = []
        
        # For efficient lookup
        self.kdtree = None
        self.kdtree_rebuild_freq = 50  # Rebuild after this many updates
        self.updates_since_rebuild = 0
    
    def store(self, feature: torch.Tensor, outcome: float) -> None:
        """Store feature-outcome association.
        
        Args:
            feature: Feature vector of state
            outcome: Outcome/reward value
        """
        try:
            # Process feature
            if feature.requires_grad:
                feature = feature.detach()
            feature_np = feature.cpu().numpy()
            
            # Flatten if needed
            if feature_np.ndim > 1:
                feature_np = feature_np.flatten()
                
            # Add to memory
            self.features.append(feature_np)
            self.outcomes.append(outcome)
            
            # Keep memory under limit
            if len(self.features) > self.history_size:
                self.features.pop(0)
                self.outcomes.pop(0)
                
            # Update KD-tree periodically
            self.updates_since_rebuild += 1
            if self.updates_since_rebuild >= self.kdtree_rebuild_freq:
                self._rebuild_kdtree()
                
        except Exception as e:
            logger.error(f"Error storing feature-outcome pair: {e}")
    
    def query(self, feature: torch.Tensor, k: int = 5) -> float:
        """Query memory for expected outcome of a state.
        
        Args:
            feature: Feature vector of state
            k: Number of nearest neighbors to consider
            
        Returns:
            float: Expected outcome
        """
        if not self.features:
            return 0.0  # No data yet
            
        try:
            # Process feature
            if feature.requires_grad:
                feature = feature.detach()
            feature_np = feature.cpu().numpy()
            
            # Flatten if needed
            if feature_np.ndim > 1:
                feature_np = feature_np.flatten()
                
            # Use KD-tree if available
            if self.kdtree is not None:
                return self._kdtree_search(feature_np, k)
            else:
                return self._linear_search(feature_np, k)
                
        except Exception as e:
            logger.error(f"Error querying association memory: {e}")
            return 0.0
    
    def _kdtree_search(self, feature: np.ndarray, k: int) -> float:
        """Search using KD-tree for efficiency.
        
        Args:
            feature: Feature vector
            k: Number of neighbors
            
        Returns:
            float: Expected outcome
        """
        # Find k nearest neighbors
        k = min(k, len(self.features))
        dists, indices = self.kdtree.query(feature, k=k)
        
        # Weight by inverse distance
        weights = 1.0 / (dists + 1e-6)  # Avoid division by zero
        total_weight = weights.sum()
        
        if total_weight == 0:
            return 0.0
            
        # Compute weighted average
        weighted_sum = sum(weights[i] * self.outcomes[indices[i]] for i in range(k))
        expected_outcome = weighted_sum / total_weight
        
        return float(expected_outcome)
    
    def _linear_search(self, feature: np.ndarray, k: int) -> float:
        """Fallback search when KD-tree is not available.
        
        Args:
            feature: Feature vector
            k: Number of neighbors
            
        Returns:
            float: Expected outcome
        """
        # Compute distances to all stored features
        distances = []
        for i, stored_feature in enumerate(self.features):
            if stored_feature.shape != feature.shape:
                continue  # Skip incompatible features
            distance = np.linalg.norm(stored_feature - feature)
            distances.append((distance, i))
            
        if not distances:
            return 0.0
            
        # Sort by distance
        distances.sort()
        
        # Take k nearest
        k = min(k, len(distances))
        weights = []
        indices = []
        
        for i in range(k):
            dist, idx = distances[i]
            weights.append(1.0 / (dist + 1e-6))  # Avoid division by zero
            indices.append(idx)
            
        total_weight = sum(weights)
        
        if total_weight == 0:
            return 0.0
            
        # Compute weighted average
        weighted_sum = sum(weights[i] * self.outcomes[indices[i]] for i in range(k))
        expected_outcome = weighted_sum / total_weight
        
        return float(expected_outcome)
    
    def _rebuild_kdtree(self) -> None:
        """Rebuild KD-tree for efficient search."""
        if not self.features:
            return
            
        try:
            # Check if all feature vectors have same shape
            shape = self.features[0].shape
            if not all(f.shape == shape for f in self.features):
                logger.warning("Cannot build KD-tree: inconsistent feature shapes")
                return
                
            # Build KD-tree
            self.kdtree = KDTree(np.array(self.features))
            self.updates_since_rebuild = 0
            
        except Exception as e:
            logger.error(f"Error rebuilding KD-tree: {e}")
            self.kdtree = None
    
    def save_state(self, path: str) -> None:
        """Save association memory state to disk.
        
        Args:
            path: Path to save the state
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save state using pickle
            state = {
                'feature_dim': self.feature_dim,
                'history_size': self.history_size,
                'features': self.features,
                'outcomes': self.outcomes,
            }
            
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved association memory with {len(self.features)} entries")
        except Exception as e:
            logger.error(f"Error saving association memory: {e}")
    
    def load_state(self, path: str) -> None:
        """Load association memory state from disk.
        
        Args:
            path: Path to load the state from
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Association memory state file not found: {path}")
                return
                
            # Load state using pickle
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Restore state
            self.feature_dim = state['feature_dim']
            self.history_size = state['history_size']
            self.features = state['features']
            self.outcomes = state['outcomes']
            
            # Rebuild KD-tree
            if self.features:
                self._rebuild_kdtree()
                
            logger.info(f"Loaded association memory with {len(self.features)} entries")
        except Exception as e:
            logger.error(f"Error loading association memory: {e}")


class CalibrationTracker:
    """Tracks and calibrates reward distribution over time."""
    
    def __init__(self, window_size: int = 1000, update_freq: int = 100):
        """Initialize calibration tracker.
        
        Args:
            window_size: Size of tracking window
            update_freq: How often to update calibration values
        """
        self.window_size = window_size
        self.update_freq = update_freq
        self.reward_history = []
        self.mean = 0.0
        self.std = 1.0
        self.min = float('inf')
        self.max = float('-inf')
        self.count = 0
        
    def update(self, reward: float) -> None:
        """Update tracker with new reward.
        
        Args:
            reward: New reward value
        """
        # Add to history
        self.reward_history.append(reward)
        if len(self.reward_history) > self.window_size:
            self.reward_history.pop(0)
            
        # Update min/max
        self.min = min(self.min, reward)
        self.max = max(self.max, reward)
        
        # Update count
        self.count += 1
        
        # Update statistics periodically
        if self.count % self.update_freq == 0:
            self._update_statistics()
            
    def _update_statistics(self) -> None:
        """Update mean and standard deviation."""
        if not self.reward_history:
            return
            
        # Calculate statistics
        self.mean = sum(self.reward_history) / len(self.reward_history)
        
        # Calculate standard deviation
        if len(self.reward_history) > 1:
            squared_diffs = [(r - self.mean) ** 2 for r in self.reward_history]
            self.std = (sum(squared_diffs) / len(self.reward_history)) ** 0.5
        else:
            self.std = 1.0
            
        # Ensure std is never zero
        self.std = max(self.std, 0.1)
        
    def normalize(self, reward: float) -> float:
        """Normalize a reward value based on tracked statistics.
        
        Args:
            reward: Raw reward value
            
        Returns:
            float: Normalized reward
        """
        # Z-score normalization with clipping
        if self.std == 0:
            return 0.0
            
        normalized = (reward - self.mean) / self.std
        
        # Clip to reasonable range
        normalized = max(-5.0, min(5.0, normalized))
        
        return normalized
    
    def denormalize(self, normalized_reward: float) -> float:
        """Convert normalized reward back to original scale.
        
        Args:
            normalized_reward: Normalized reward value
            
        Returns:
            float: Denormalized reward
        """
        return normalized_reward * self.std + self.mean
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics.
        
        Returns:
            Dict: Statistics about tracked rewards
        """
        return {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'count': self.count
        } 