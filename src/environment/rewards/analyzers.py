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
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger(__name__)

class VisualChangeAnalyzer:
    """Analyzes visual changes between frames and learns to associate them with outcomes."""
    
    def __init__(self, config=None):
        """Initialize visual change analyzer.
        
        Args:
            config: Configuration for visual change analysis
        """
        self.config = config or {}
        self.min_change_threshold = self.config.get('min_change_threshold', 0.05)
        self.max_change_threshold = self.config.get('max_change_threshold', 0.8)
        self.feature_extractor = VisualFeatureExtractor(config)
        
    def get_visual_change_score(self, frame1, frame2):
        """Compute visual change score between two frames.
        
        Args:
            frame1: First frame (numpy array or torch tensor)
            frame2: Second frame (numpy array or torch tensor)
            
        Returns:
            float: Visual change score
            dict: Detailed metrics
        """
        try:
            # Convert tensors to numpy arrays if needed
            if isinstance(frame1, torch.Tensor):
                frame1_np = frame1.detach().cpu().numpy()
                if frame1_np.ndim == 3 and frame1_np.shape[0] == 3:  # CHW format
                    frame1_np = np.transpose(frame1_np, (1, 2, 0))
            else:
                frame1_np = frame1
                
            if isinstance(frame2, torch.Tensor):
                frame2_np = frame2.detach().cpu().numpy()
                if frame2_np.ndim == 3 and frame2_np.shape[0] == 3:  # CHW format
                    frame2_np = np.transpose(frame2_np, (1, 2, 0))
            else:
                frame2_np = frame2
            
            # Extract features from frames
            features1 = self.feature_extractor.extract_features(frame1_np)
            features2 = self.feature_extractor.extract_features(frame2_np)
            
            # Compute feature differences
            feature_diff = self.feature_extractor.compute_feature_difference(features1, features2)
            
            # Compute feature importance weights
            importance_weights = self.feature_extractor.get_feature_importance(feature_diff)
            
            # Compute weighted change score
            total_importance = sum(importance_weights.values()) or 1.0
            weighted_diff = sum(feature_diff[k] * importance_weights.get(k, 1.0) 
                               for k in feature_diff)
            change_score = weighted_diff / total_importance
            
            # Normalize change score to 0-1 range
            normalized_score = min(1.0, max(0.0, 
                                           (change_score - self.min_change_threshold) / 
                                           (self.max_change_threshold - self.min_change_threshold)))
            
            # Return score and detailed metrics
            metrics = {
                'raw_score': float(change_score),
                'normalized_score': float(normalized_score),
                **feature_diff
            }
            
            return normalized_score, metrics
            
        except Exception as e:
            logger.error(f"Error computing visual change score: {e}")
            return 0.0, {'error': str(e)}
            
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
    """Extract meaningful features from frames for reward analysis."""
    
    def __init__(self, config=None):
        """Initialize feature extractor.
        
        Args:
            config: Configuration for feature extraction
        """
        self.config = config or {}
        self.feature_history = []
        self.edge_threshold = self.config.get('edge_threshold', 100)
        self.hist_bins = self.config.get('histogram_bins', 32)
        
    def extract_features(self, frame):
        """Extract visual features from a frame.
        
        Args:
            frame: Input frame as numpy array or PyTorch tensor
            
        Returns:
            dict: Extracted features
        """
        try:
            # Ensure frame is a valid numpy array
            if frame is None:
                logger.warning("Received None frame in extract_features")
                return {'edges': np.array([]), 'color_hist': np.array([])}
                
            # Convert from tensor if needed
            if isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().numpy()
                # Handle different tensor formats (CHW vs HWC)
                if frame_np.ndim == 3 and frame_np.shape[0] == 3:  # CHW format
                    frame_np = np.transpose(frame_np, (1, 2, 0))
            else:
                frame_np = frame
                
            # Ensure frame is in correct format for OpenCV
            if frame_np.ndim < 2:
                logger.error(f"Invalid frame shape: {frame_np.shape}")
                return {'edges': np.array([]), 'color_hist': np.array([])}
                
            # Convert to grayscale for edge detection
            if frame_np.ndim == 3 and frame_np.shape[2] == 3:
                gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            elif frame_np.ndim == 2:
                gray = frame_np
            else:
                logger.error(f"Unexpected frame format: {frame_np.shape}")
                return {'edges': np.array([]), 'color_hist': np.array([])}
                
            # Edge detection
            edges = cv2.Canny(gray, self.edge_threshold, self.edge_threshold * 2)
            edge_pixels = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Color histogram (if color image)
            if frame_np.ndim == 3 and frame_np.shape[2] == 3:
                hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [self.hist_bins, self.hist_bins], 
                                    [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                hist_flat = hist.flatten()
            else:
                # For grayscale, create a simple intensity histogram
                hist = cv2.calcHist([gray], [0], None, [self.hist_bins], [0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                hist_flat = hist.flatten()
            
            # Store features
            features = {
                'edges': edge_pixels,
                'color_hist': hist_flat
            }
            
            # Add to history if needed
            if len(self.feature_history) >= 10:
                self.feature_history.pop(0)
            self.feature_history.append(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting visual features: {e}")
            return {'edges': np.array([]), 'color_hist': np.array([])}
    
    def compute_feature_difference(self, features1, features2):
        """Compute difference between feature sets.
        
        Args:
            features1: First feature set
            features2: Second feature set
            
        Returns:
            dict: Feature differences
        """
        try:
            if not features1 or not features2:
                return {'edge_diff': 0.0, 'color_diff': 0.0}
                
            # Edge difference
            edge_diff = abs(features1['edges'] - features2['edges'])
            
            # Color histogram difference
            hist1 = features1['color_hist']
            hist2 = features2['color_hist']
            
            if hist1.size == 0 or hist2.size == 0:
                color_diff = 0.0
            elif hist1.shape != hist2.shape:
                logger.warning(f"Histogram shapes don't match: {hist1.shape} vs {hist2.shape}")
                # Resize smaller histogram to match larger one
                if hist1.size < hist2.size:
                    hist1 = np.resize(hist1, hist2.shape)
                else:
                    hist2 = np.resize(hist2, hist1.shape)
                color_diff = np.sum(cv2.absdiff(hist1, hist2)) / max(hist1.size, 1)
            else:
                color_diff = np.sum(cv2.absdiff(hist1, hist2)) / max(hist1.size, 1)
            
            return {
                'edge_diff': float(edge_diff),
                'color_diff': float(color_diff)
            }
        except Exception as e:
            logger.error(f"Error computing feature difference: {e}")
            return {'edge_diff': 0.0, 'color_diff': 0.0}
            
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