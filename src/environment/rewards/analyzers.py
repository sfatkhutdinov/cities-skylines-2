"""
Visual change analyzers for Cities: Skylines 2 environment.

This module provides tools for analyzing visual changes between frames
and determining their significance for reward calculation.
"""

import torch
import numpy as np
import cv2
import logging
import os
import pickle
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class VisualChangeAnalyzer:
    """Analyzes visual changes between frames and learns to associate them with outcomes."""
    
    def __init__(self, 
                 history_size: int = 1000, 
                 association_threshold: float = 0.6,
                 device: Optional[torch.device] = None):
        """Initialize visual change analyzer.
        
        Args:
            history_size: Maximum number of visual change samples to keep
            association_threshold: Minimum correlation to consider changes associated
            device: Compute device to use
        """
        self.history_size = history_size
        self.association_threshold = association_threshold
        self.device = device
        
        # Storage for visual change patterns and their outcomes
        self.visual_changes = []  # List of visual change metrics
        self.outcomes = []        # List of corresponding outcomes/rewards
        
        # Statistical trackers
        self.change_mean = None
        self.change_std = None
        self.outcome_mean = 0.0
        self.outcome_std = 1.0
        
        # State variables
        self.update_count = 0
        self.stats_update_freq = 50
        
    def get_visual_change_score(self, 
                                frame1: np.ndarray, 
                                frame2: np.ndarray) -> float:
        """Calculate a score indicating degree of visual change between frames.
        
        Args:
            frame1: First frame (RGB format)
            frame2: Second frame (RGB format)
            
        Returns:
            float: Visual change score (0-1 range)
        """
        try:
            # Convert to grayscale
            if frame1.ndim == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = frame1
                
            if frame2.ndim == 3:
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = frame2
                
            # Ensure same shape
            if gray1.shape != gray2.shape:
                logger.warning(f"Frame shapes differ: {gray1.shape} vs {gray2.shape}")
                gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
                
            # Compute absolute difference
            diff = cv2.absdiff(gray1, gray2)
            
            # Apply threshold to identify changed pixels
            _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Count changed pixels
            change_pixels = np.count_nonzero(thresholded)
            total_pixels = thresholded.size
            
            # Calculate change percentage
            change_percentage = change_pixels / total_pixels
            
            # Compute additional metrics
            change_metrics = {
                'pixel_change_ratio': change_percentage,
                'mean_intensity_diff': np.mean(diff) / 255,
                'max_intensity_diff': np.max(diff) / 255,
                'std_intensity_diff': np.std(diff) / 255
            }
            
            # Compute weighted score
            weights = {
                'pixel_change_ratio': 0.5,
                'mean_intensity_diff': 0.2,
                'max_intensity_diff': 0.1,
                'std_intensity_diff': 0.2
            }
            
            weighted_score = sum(metric * weights[name] 
                                 for name, metric in change_metrics.items())
            
            # Scale to 0-1 range
            scaled_score = min(1.0, weighted_score * 5.0)
            
            return scaled_score, change_metrics
            
        except Exception as e:
            logger.error(f"Error computing visual change: {e}")
            return 0.0, {}
            
    def update_with_outcome(self, 
                           change_metrics: Dict[str, float], 
                           outcome: float) -> None:
        """Update analyzer with new visual change and its outcome.
        
        Args:
            change_metrics: Dict of visual change metrics
            outcome: Outcome/reward value associated with change
        """
        # Add to history
        self.visual_changes.append(change_metrics)
        self.outcomes.append(outcome)
        
        # Keep history within size limit
        if len(self.visual_changes) > self.history_size:
            self.visual_changes.pop(0)
            self.outcomes.pop(0)
            
        # Increment update counter
        self.update_count += 1
        
        # Update statistics periodically
        if self.update_count % self.stats_update_freq == 0:
            self._update_statistics()
            
    def _update_statistics(self) -> None:
        """Update statistical measures for normalization."""
        if not self.visual_changes:
            return
            
        # Extract data
        metrics_keys = self.visual_changes[0].keys()
        
        # Compute statistics for each metric
        self.change_mean = {}
        self.change_std = {}
        
        for key in metrics_keys:
            values = [change[key] for change in self.visual_changes]
            self.change_mean[key] = np.mean(values)
            self.change_std[key] = np.std(values) or 1.0  # Avoid zero std
            
        # Update outcome statistics
        self.outcome_mean = np.mean(self.outcomes)
        self.outcome_std = np.std(self.outcomes) or 1.0
        
    def predict_outcome(self, 
                       change_metrics: Dict[str, float], 
                       k: int = 5) -> float:
        """Predict outcome based on past associations.
        
        Args:
            change_metrics: Visual change metrics
            k: Number of nearest neighbors to consider
            
        Returns:
            float: Predicted outcome
        """
        if not self.visual_changes or len(self.visual_changes) < k:
            return 0.0
            
        try:
            # Compute similarity to all stored changes
            similarities = []
            
            for i, stored_metrics in enumerate(self.visual_changes):
                # Compute Euclidean distance between metrics
                distance = self._compute_metric_distance(change_metrics, stored_metrics)
                similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
                similarities.append((similarity, i))
                
            # Sort by similarity (descending)
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # Take k most similar
            k = min(k, len(similarities))
            weights = []
            indices = []
            
            for i in range(k):
                similarity, idx = similarities[i]
                weights.append(similarity)
                indices.append(idx)
                
            total_weight = sum(weights)
            
            if total_weight == 0:
                return 0.0
                
            # Compute weighted average outcome
            weighted_sum = sum(weights[i] * self.outcomes[indices[i]] for i in range(k))
            predicted_outcome = weighted_sum / total_weight
            
            return predicted_outcome
            
        except Exception as e:
            logger.error(f"Error predicting outcome: {e}")
            return 0.0
            
    def _compute_metric_distance(self, 
                                metrics1: Dict[str, float], 
                                metrics2: Dict[str, float]) -> float:
        """Compute distance between two sets of metrics.
        
        Args:
            metrics1: First set of metrics
            metrics2: Second set of metrics
            
        Returns:
            float: Distance value
        """
        # Ensure both have same keys
        common_keys = set(metrics1.keys()) & set(metrics2.keys())
        
        if not common_keys:
            return float('inf')
            
        # Compute normalized Euclidean distance
        squared_diffs = []
        
        for key in common_keys:
            # Skip keys with zero std to avoid division by zero
            if key not in self.change_std or self.change_std[key] == 0:
                continue
                
            # Normalize values
            if self.change_mean and self.change_std:
                value1 = (metrics1[key] - self.change_mean[key]) / self.change_std[key]
                value2 = (metrics2[key] - self.change_mean[key]) / self.change_std[key]
            else:
                value1 = metrics1[key]
                value2 = metrics2[key]
                
            squared_diff = (value1 - value2) ** 2
            squared_diffs.append(squared_diff)
            
        if not squared_diffs:
            return float('inf')
            
        # Compute root mean squared difference
        distance = (sum(squared_diffs) / len(squared_diffs)) ** 0.5
        
        return distance
        
    def get_association_strength(self, 
                                change_metrics: Dict[str, float], 
                                outcome: float) -> float:
        """Calculate how strongly a visual change is associated with an outcome.
        
        Args:
            change_metrics: Visual change metrics
            outcome: Observed outcome
            
        Returns:
            float: Association strength (0-1)
        """
        predicted = self.predict_outcome(change_metrics)
        
        # Normalize outcome
        if self.outcome_std:
            normalized_outcome = (outcome - self.outcome_mean) / self.outcome_std
            normalized_predicted = (predicted - self.outcome_mean) / self.outcome_std
        else:
            normalized_outcome = outcome
            normalized_predicted = predicted
            
        # Calculate agreement
        error = abs(normalized_outcome - normalized_predicted)
        max_error = 5.0  # Assume 5 standard deviations as max error
        
        # Convert to similarity (1 = perfect match, 0 = maximal disagreement)
        agreement = max(0.0, 1.0 - (error / max_error))
        
        return agreement
        
    def save_state(self, path: str) -> None:
        """Save analyzer state to disk.
        
        Args:
            path: Path to save the state
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare state
            state = {
                'history_size': self.history_size,
                'association_threshold': self.association_threshold,
                'visual_changes': self.visual_changes,
                'outcomes': self.outcomes,
                'change_mean': self.change_mean,
                'change_std': self.change_std,
                'outcome_mean': self.outcome_mean,
                'outcome_std': self.outcome_std,
                'update_count': self.update_count
            }
            
            # Save using pickle
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved visual change analyzer state with {len(self.outcomes)} entries")
            
        except Exception as e:
            logger.error(f"Error saving visual change analyzer: {e}")
            
    def load_state(self, path: str) -> None:
        """Load analyzer state from disk.
        
        Args:
            path: Path to load the state from
        """
        try:
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"Visual change analyzer state file not found: {path}")
                return
                
            # Load state using pickle
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Restore state
            self.history_size = state['history_size']
            self.association_threshold = state['association_threshold']
            self.visual_changes = state['visual_changes']
            self.outcomes = state['outcomes']
            self.change_mean = state['change_mean']
            self.change_std = state['change_std']
            self.outcome_mean = state['outcome_mean']
            self.outcome_std = state['outcome_std']
            self.update_count = state['update_count']
            
            logger.info(f"Loaded visual change analyzer state with {len(self.outcomes)} entries")
            
        except Exception as e:
            logger.error(f"Error loading visual change analyzer: {e}")


class VisualFeatureExtractor:
    """Extracts meaningful features from frames for reward analysis."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize feature extractor.
        
        Args:
            device: Compute device to use
        """
        self.device = device
        
        # Feature extraction parameters
        self.edge_threshold1 = 50
        self.edge_threshold2 = 150
        self.hist_bins = 32
        self.hist_channels = [0, 1, 2]  # RGB channels
        self.hist_ranges = [0, 256, 0, 256, 0, 256]
        
    def extract_features(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract visual features from a frame.
        
        Args:
            frame: Input frame (RGB format)
            
        Returns:
            Dict: Visual features extracted from frame
        """
        features = {}
        
        try:
            # Convert to grayscale for edge detection
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
                
            # Edge detection
            edges = cv2.Canny(gray, self.edge_threshold1, self.edge_threshold2)
            features['edges'] = edges
            
            # Color histogram
            if frame.ndim == 3:
                hist = cv2.calcHist([frame], self.hist_channels, None, 
                                    [self.hist_bins, self.hist_bins, self.hist_bins], 
                                    self.hist_ranges)
                hist = cv2.normalize(hist, hist).flatten()
                features['color_hist'] = hist
                
            # Calculate image stats
            features['mean_intensity'] = np.mean(gray)
            features['std_intensity'] = np.std(gray)
            
            # Edge density
            features['edge_density'] = np.count_nonzero(edges) / edges.size
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {}
            
    def compute_feature_difference(self, 
                                  features1: Dict[str, np.ndarray], 
                                  features2: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Compute difference metrics between two sets of features.
        
        Args:
            features1: First set of features
            features2: Second set of features
            
        Returns:
            Dict: Difference metrics
        """
        diff_metrics = {}
        
        try:
            # Edge difference
            if 'edges' in features1 and 'edges' in features2:
                edge_diff = cv2.absdiff(features1['edges'], features2['edges'])
                diff_metrics['edge_change'] = np.count_nonzero(edge_diff) / edge_diff.size
                
            # Histogram difference
            if 'color_hist' in features1 and 'color_hist' in features2:
                hist_diff = cv2.compareHist(
                    features1['color_hist'], features2['color_hist'], cv2.HISTCMP_BHATTACHARYYA)
                diff_metrics['hist_diff'] = hist_diff
                
            # Mean intensity change
            if 'mean_intensity' in features1 and 'mean_intensity' in features2:
                intensity_diff = abs(features1['mean_intensity'] - features2['mean_intensity'])
                diff_metrics['intensity_change'] = intensity_diff / 255.0
                
            # Edge density change
            if 'edge_density' in features1 and 'edge_density' in features2:
                density_diff = abs(features1['edge_density'] - features2['edge_density'])
                diff_metrics['edge_density_change'] = density_diff
                
            return diff_metrics
            
        except Exception as e:
            logger.error(f"Error computing feature difference: {e}")
            return {}
            
    def get_feature_importance(self, diff_metrics: Dict[str, float]) -> Dict[str, float]:
        """Get importance weights for different feature differences.
        
        Args:
            diff_metrics: Feature difference metrics
            
        Returns:
            Dict: Importance weights for each metric
        """
        # Default importance weights
        importance = {
            'edge_change': 0.4,
            'hist_diff': 0.3,
            'intensity_change': 0.2,
            'edge_density_change': 0.1
        }
        
        # Filter to only include available metrics
        filtered_importance = {k: v for k, v in importance.items() if k in diff_metrics}
        
        # Normalize weights to sum to 1
        if filtered_importance:
            total = sum(filtered_importance.values())
            if total > 0:
                filtered_importance = {k: v / total for k, v in filtered_importance.items()}
                
        return filtered_importance 