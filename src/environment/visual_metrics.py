"""
Visual metrics estimation from raw game screenshots.
Uses only visual information to estimate game metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from src.config.hardware_config import HardwareConfig

class VisualMetricsEstimator:
    def __init__(self, config: HardwareConfig):
        """Initialize visual metrics estimator.
        
        Args:
            config (HardwareConfig): Hardware and training configuration
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # Initialize CNN for feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device, dtype=self.dtype)
        
        # Moving averages for stability
        self.moving_averages = {
            'building_density': [],
            'residential_areas': [],
            'traffic_density': [],
            'window_lights': []
        }
        self.ma_window = 10
        
    def estimate_population(self, frame: torch.Tensor) -> Tuple[int, Dict[str, float]]:
        """
        Estimate population from visual cues in the frame.
        Uses multiple visual indicators:
        - Building density
        - Residential area coverage
        - Traffic density
        - Window lights (night time activity)
        """
        # Ensure frame is on the correct device and dtype
        if frame.device != self.device:
            frame = frame.to(self.device)
        if frame.dtype != self.dtype:
            frame = frame.to(dtype=self.dtype)
        
        # Extract visual features
        features = self._extract_features(frame)
        
        # Compute various density metrics
        metrics = {
            'building_density': self._compute_building_density(frame),
            'residential_areas': self._compute_residential_areas(frame),
            'traffic_density': self._compute_traffic_density(frame),
            'window_lights': self._compute_window_lights(frame)
        }
        
        # Update moving averages
        for key, value in metrics.items():
            self.moving_averages[key].append(value)
            if len(self.moving_averages[key]) > self.ma_window:
                self.moving_averages[key].pop(0)
        
        # Compute smoothed metrics
        smoothed_metrics = {
            k: np.mean(v) for k, v in self.moving_averages.items()
        }
        
        # Estimate population based on smoothed metrics
        # This is a simplified model that learns to correlate visual patterns with population
        estimated_population = self._combine_metrics_for_population(smoothed_metrics)
        
        return estimated_population, smoothed_metrics
    
    def _extract_features(self, frame: torch.Tensor) -> torch.Tensor:
        """Extract visual features from frame using CNN."""
        with torch.no_grad():
            return self.feature_extractor(frame.unsqueeze(0))
    
    def _compute_building_density(self, frame: torch.Tensor) -> float:
        """
        Estimate building density from frame.
        Uses edge detection and color clustering to identify buildings.
        """
        # Convert to grayscale for edge detection
        gray = frame.mean(dim=0)
        
        # Simple Sobel edge detection
        edges_x = F.conv2d(gray.unsqueeze(0).unsqueeze(0), 
                          torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().to(self.device, dtype=self.dtype).unsqueeze(0),
                          padding=1)
        edges_y = F.conv2d(gray.unsqueeze(0).unsqueeze(0),
                          torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().to(self.device, dtype=self.dtype).unsqueeze(0),
                          padding=1)
        
        edges = torch.sqrt(edges_x ** 2 + edges_y ** 2)
        return float(edges.mean().item())
    
    def _compute_residential_areas(self, frame: torch.Tensor) -> float:
        """
        Estimate residential area coverage.
        Uses color patterns typical of residential zones.
        """
        # Simple color-based detection (assuming residential areas have certain color patterns)
        # This would need to be refined based on actual game visuals
        residential_mask = ((frame[0] > 0.4) & (frame[1] > 0.4) & (frame[2] > 0.4))
        return float(residential_mask.float().mean().item())
    
    def _compute_traffic_density(self, frame: torch.Tensor) -> float:
        """
        Estimate traffic density from movement patterns.
        Uses temporal differences between frames for movement detection.
        """
        # Simplified version - would need frame history for actual implementation
        # Currently returns a proxy based on road-like patterns
        road_mask = ((frame[0] < 0.3) & (frame[1] < 0.3) & (frame[2] < 0.3))
        return float(road_mask.float().mean().item())
    
    def _compute_window_lights(self, frame: torch.Tensor) -> float:
        """
        Estimate active buildings through window lights.
        Particularly useful for night-time population estimation.
        """
        # Look for bright spots in darker areas
        brightness = frame.mean(dim=0)
        light_spots = (brightness > 0.7).float()
        return float(light_spots.mean().item())
    
    def _combine_metrics_for_population(self, metrics: Dict[str, float]) -> int:
        """
        Combine various metrics to estimate population.
        Uses a weighted combination of different visual indicators.
        """
        weights = {
            'building_density': 0.4,
            'residential_areas': 0.3,
            'traffic_density': 0.2,
            'window_lights': 0.1
        }
        
        # Compute weighted sum and scale to reasonable population range
        weighted_sum = sum(metrics[k] * weights[k] for k in weights)
        
        # Scale to population (this would need calibration)
        # Assuming maximum reasonable starting city population around 10000
        estimated_population = int(weighted_sum * 10000)
        
        return max(0, estimated_population)  # Ensure non-negative
        
    def update_model(self, frame: torch.Tensor, reward: float):
        """
        Update the estimation model based on rewards.
        This allows the model to learn from the success/failure of its estimates.
        """
        # This would be expanded to actually update the feature extractor
        # based on reward signals from the environment
        pass 