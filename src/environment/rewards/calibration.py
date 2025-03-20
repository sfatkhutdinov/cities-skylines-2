"""
Reward calibration for Cities: Skylines 2 environment.

This module provides tools for calibrating and normalizing rewards
to ensure consistent scale and distribution during training.
"""

import numpy as np
import torch
import logging
import os
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)

class RewardNormalizer:
    """Normalizes rewards to a consistent scale for stable training."""
    
    def __init__(self, 
                window_size: int = 10000,
                clip_range: float = 10.0,
                epsilon: float = 1e-8,
                update_freq: int = 100):
        """Initialize reward normalizer.
        
        Args:
            window_size: Size of reward history window
            clip_range: Maximum absolute value after normalization
            epsilon: Small value to prevent division by zero
            update_freq: How often to update normalization statistics
        """
        self.window_size = window_size
        self.clip_range = clip_range
        self.epsilon = epsilon
        self.update_freq = update_freq
        
        # Running statistics
        self.reward_history = deque(maxlen=window_size)
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
        self.update_count = 0
        
    def normalize(self, reward: float) -> float:
        """Normalize a reward value.
        
        Args:
            reward: Raw reward value
            
        Returns:
            float: Normalized reward
        """
        # Update history
        self.reward_history.append(reward)
        self.count += 1
        
        # Update statistics periodically
        self.update_count += 1
        if self.update_count >= self.update_freq:
            self._update_statistics()
            self.update_count = 0
            
        # Apply normalization
        normalized = (reward - self.mean) / (self.std + self.epsilon)
        
        # Clip to range
        normalized = np.clip(normalized, -self.clip_range, self.clip_range)
        
        return float(normalized)
        
    def _update_statistics(self) -> None:
        """Update running statistics for normalization."""
        if not self.reward_history:
            return
            
        # Compute statistics
        self.mean = np.mean(self.reward_history)
        self.std = np.std(self.reward_history)
        
        # Ensure std is positive
        self.std = max(self.std, self.epsilon)
        
        logger.debug(f"Updated reward normalizer: mean={self.mean:.4f}, std={self.std:.4f}")
        
    def save_state(self, path: str) -> None:
        """Save normalizer state to disk.
        
        Args:
            path: Path to save state
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare state
            state = {
                'window_size': self.window_size,
                'clip_range': self.clip_range,
                'epsilon': self.epsilon,
                'mean': self.mean,
                'std': self.std,
                'count': self.count,
                'reward_history': list(self.reward_history)
            }
            
            # Save to file
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved reward normalizer state: mean={self.mean:.4f}, std={self.std:.4f}")
        except Exception as e:
            logger.error(f"Error saving reward normalizer state: {e}")
            
    def load_state(self, path: str) -> None:
        """Load normalizer state from disk.
        
        Args:
            path: Path to load state from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Reward normalizer state file not found: {path}")
                return
                
            # Load from file
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Restore state
            self.window_size = state['window_size']
            self.clip_range = state['clip_range']
            self.epsilon = state['epsilon']
            self.mean = state['mean']
            self.std = state['std']
            self.count = state['count']
            
            # Restore history
            self.reward_history = deque(state['reward_history'], maxlen=self.window_size)
            
            logger.info(f"Loaded reward normalizer state: mean={self.mean:.4f}, std={self.std:.4f}")
        except Exception as e:
            logger.error(f"Error loading reward normalizer state: {e}")


class RewardScaler:
    """Scales rewards to a target range based on observed distribution."""
    
    def __init__(self, 
                target_min: float = -1.0, 
                target_max: float = 1.0,
                window_size: int = 1000,
                adaptation_rate: float = 0.01):
        """Initialize reward scaler.
        
        Args:
            target_min: Target minimum value after scaling
            target_max: Target maximum value after scaling
            window_size: Size of history for estimating min/max
            adaptation_rate: Rate at which min/max are updated
        """
        self.target_min = target_min
        self.target_max = target_max
        self.window_size = window_size
        self.adaptation_rate = adaptation_rate
        
        # Observed min/max
        self.observed_min = float('inf')
        self.observed_max = float('-inf')
        
        # Running history for percentile-based scaling
        self.reward_history = deque(maxlen=window_size)
        
    def scale(self, reward: float) -> float:
        """Scale a reward value to target range.
        
        Args:
            reward: Raw reward value
            
        Returns:
            float: Scaled reward
        """
        # Update history
        self.reward_history.append(reward)
        
        # Update observed min/max with smoothing
        if reward < self.observed_min:
            self.observed_min = (1 - self.adaptation_rate) * self.observed_min + self.adaptation_rate * reward
        if reward > self.observed_max:
            self.observed_max = (1 - self.adaptation_rate) * self.observed_max + self.adaptation_rate * reward
            
        # Handle degenerate case
        if self.observed_min == self.observed_max:
            return (self.target_min + self.target_max) / 2
            
        # Apply scaling
        scaled = (reward - self.observed_min) / (self.observed_max - self.observed_min)
        scaled = scaled * (self.target_max - self.target_min) + self.target_min
        
        return float(scaled)
        
    def percentile_scale(self, reward: float, low_pct: float = 5, high_pct: float = 95) -> float:
        """Scale reward based on percentiles of observed distribution.
        
        Args:
            reward: Raw reward value
            low_pct: Lower percentile for scaling
            high_pct: Upper percentile for scaling
            
        Returns:
            float: Scaled reward
        """
        # Update history
        self.reward_history.append(reward)
        
        # Need enough history for percentiles
        if len(self.reward_history) < 10:
            return self.scale(reward)  # Fall back to simple scaling
            
        # Compute percentiles
        p_low = np.percentile(self.reward_history, low_pct)
        p_high = np.percentile(self.reward_history, high_pct)
        
        # Handle degenerate case
        if p_low == p_high:
            return (self.target_min + self.target_max) / 2
            
        # Clip to percentile range then scale
        clipped = max(p_low, min(reward, p_high))
        scaled = (clipped - p_low) / (p_high - p_low)
        scaled = scaled * (self.target_max - self.target_min) + self.target_min
        
        return float(scaled)
        
    def save_state(self, path: str) -> None:
        """Save scaler state to disk.
        
        Args:
            path: Path to save state
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare state
            state = {
                'target_min': self.target_min,
                'target_max': self.target_max,
                'adaptation_rate': self.adaptation_rate,
                'observed_min': self.observed_min,
                'observed_max': self.observed_max,
                'reward_history': list(self.reward_history)
            }
            
            # Save to file
            with open(path, 'wb') as f:
                pickle.dump(state, f)
                
            logger.info(f"Saved reward scaler state: min={self.observed_min:.4f}, max={self.observed_max:.4f}")
        except Exception as e:
            logger.error(f"Error saving reward scaler state: {e}")
            
    def load_state(self, path: str) -> None:
        """Load scaler state from disk.
        
        Args:
            path: Path to load state from
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Reward scaler state file not found: {path}")
                return
                
            # Load from file
            with open(path, 'rb') as f:
                state = pickle.load(f)
                
            # Restore state
            self.target_min = state['target_min']
            self.target_max = state['target_max']
            self.adaptation_rate = state['adaptation_rate']
            self.observed_min = state['observed_min']
            self.observed_max = state['observed_max']
            
            # Restore history
            self.reward_history = deque(state['reward_history'], maxlen=self.window_size)
            
            logger.info(f"Loaded reward scaler state: min={self.observed_min:.4f}, max={self.observed_max:.4f}")
        except Exception as e:
            logger.error(f"Error loading reward scaler state: {e}")


class RewardShaper:
    """Shapes rewards to encourage desired behavior using various heuristics."""
    
    def __init__(self):
        """Initialize reward shaper."""
        # Shaping parameters
        self.time_penalty_factor = 0.001  # Small penalty for time
        self.consecutive_bonus_factor = 0.1  # Bonus for consecutive positive rewards
        self.exploration_bonus_factor = 0.05  # Bonus for exploration
        
        # State tracking
        self.last_reward = 0.0
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        self.visited_states = set()
        
    def shape_reward(self, 
                    reward: float, 
                    state_hash: Optional[str] = None, 
                    time_step: int = 0) -> float:
        """Shape reward to encourage desired behavior.
        
        Args:
            reward: Base reward value
            state_hash: Hash representation of current state
            time_step: Current time step
            
        Returns:
            float: Shaped reward
        """
        shaped_reward = reward
        
        # Apply time penalty (small penalty for taking too long)
        shaped_reward -= self.time_penalty_factor * time_step
        
        # Consecutive action bonus
        if reward > 0:
            self.consecutive_positive += 1
            self.consecutive_negative = 0
            # Bonus for consistently good actions
            if self.consecutive_positive > 1:
                shaped_reward += self.consecutive_bonus_factor * min(self.consecutive_positive, 5)
        elif reward < 0:
            self.consecutive_negative += 1
            self.consecutive_positive = 0
            # Stronger penalty for consistently bad actions
            if self.consecutive_negative > 1:
                shaped_reward -= self.consecutive_bonus_factor * min(self.consecutive_negative, 5)
        else:
            # Reset consecutive counters on zero reward
            self.consecutive_positive = 0
            self.consecutive_negative = 0
            
        # Exploration bonus
        if state_hash is not None:
            if state_hash not in self.visited_states:
                self.visited_states.add(state_hash)
                shaped_reward += self.exploration_bonus_factor
                
            # Limit memory usage
            if len(self.visited_states) > 10000:
                self.visited_states = set(list(self.visited_states)[-5000:])
                
        # Update state
        self.last_reward = reward
        
        return shaped_reward
        
    def potential_based_shaping(self, 
                              current_potential: float, 
                              next_potential: float, 
                              discount_factor: float = 0.99) -> float:
        """Apply potential-based reward shaping (guaranteed to preserve optimal policy).
        
        Args:
            current_potential: Potential value of current state
            next_potential: Potential value of next state
            discount_factor: Discount factor for RL
            
        Returns:
            float: Shaping reward
        """
        # Potential-based shaping formula: F(s,a,s') = γΦ(s') - Φ(s)
        shaping_reward = discount_factor * next_potential - current_potential
        return shaping_reward
        
    def reset(self) -> None:
        """Reset shaper state between episodes."""
        self.last_reward = 0.0
        self.consecutive_positive = 0
        self.consecutive_negative = 0
        # Keep visited states to encourage long-term exploration


class RewardCalibrationManager:
    """Manages the calibration of rewards across training sessions."""
    
    def __init__(self, 
                 calibration_dir: str = "checkpoints/reward_calibration",
                 reward_stats_file: str = "reward_statistics.json"):
        """Initialize reward calibration manager.
        
        Args:
            calibration_dir: Directory for calibration data
            reward_stats_file: File to store reward statistics
        """
        self.calibration_dir = calibration_dir
        self.reward_stats_file = os.path.join(calibration_dir, reward_stats_file)
        
        # Ensure directory exists
        os.makedirs(calibration_dir, exist_ok=True)
        
        # Statistics
        self.reward_statistics = {
            'lifetime_mean': 0.0,
            'lifetime_std': 1.0,
            'lifetime_min': 0.0,
            'lifetime_max': 0.0,
            'lifetime_count': 0,
            'session_stats': {},
            'component_stats': {}
        }
        
        # Load existing statistics
        self._load_statistics()
        
    def _load_statistics(self) -> None:
        """Load reward statistics from disk."""
        try:
            if os.path.exists(self.reward_stats_file):
                with open(self.reward_stats_file, 'r') as f:
                    self.reward_statistics = json.load(f)
                logger.info(f"Loaded reward statistics with {self.reward_statistics['lifetime_count']} samples")
        except Exception as e:
            logger.error(f"Error loading reward statistics: {e}")
            
    def _save_statistics(self) -> None:
        """Save reward statistics to disk."""
        try:
            with open(self.reward_stats_file, 'w') as f:
                json.dump(self.reward_statistics, f, indent=2)
            logger.info(f"Saved reward statistics with {self.reward_statistics['lifetime_count']} samples")
        except Exception as e:
            logger.error(f"Error saving reward statistics: {e}")
            
    def update_statistics(self, 
                         rewards: List[float], 
                         session_id: str = "current",
                         component: Optional[str] = None) -> None:
        """Update reward statistics with new rewards.
        
        Args:
            rewards: List of reward values
            session_id: Identifier for training session
            component: Optional reward component identifier
        """
        if not rewards:
            return
            
        # Compute statistics
        count = len(rewards)
        mean = sum(rewards) / count
        std = np.std(rewards) if count > 1 else 1.0
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        # Update lifetime statistics using weighted average
        lifetime_count = self.reward_statistics['lifetime_count']
        new_count = lifetime_count + count
        
        # Update mean
        old_mean = self.reward_statistics['lifetime_mean']
        self.reward_statistics['lifetime_mean'] = (old_mean * lifetime_count + mean * count) / new_count
        
        # Update min/max
        self.reward_statistics['lifetime_min'] = min(self.reward_statistics['lifetime_min'], min_reward)
        self.reward_statistics['lifetime_max'] = max(self.reward_statistics['lifetime_max'], max_reward)
        
        # Update count
        self.reward_statistics['lifetime_count'] = new_count
        
        # Update std (approximate)
        old_std = self.reward_statistics['lifetime_std']
        self.reward_statistics['lifetime_std'] = ((old_std**2 * lifetime_count + std**2 * count) / new_count)**0.5
        
        # Update session stats
        if session_id not in self.reward_statistics['session_stats']:
            self.reward_statistics['session_stats'][session_id] = {
                'mean': 0.0,
                'std': 1.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
            
        session_stats = self.reward_statistics['session_stats'][session_id]
        session_count = session_stats['count']
        new_session_count = session_count + count
        
        # Update session mean
        session_mean = session_stats['mean']
        session_stats['mean'] = (session_mean * session_count + mean * count) / new_session_count
        
        # Update session min/max
        session_stats['min'] = min(session_stats['min'], min_reward) if session_count > 0 else min_reward
        session_stats['max'] = max(session_stats['max'], max_reward) if session_count > 0 else max_reward
        
        # Update session count
        session_stats['count'] = new_session_count
        
        # Update session std (approximate)
        session_std = session_stats['std']
        session_stats['std'] = ((session_std**2 * session_count + std**2 * count) / new_session_count)**0.5
        
        # Update component stats if provided
        if component:
            if 'component_stats' not in self.reward_statistics:
                self.reward_statistics['component_stats'] = {}
                
            if component not in self.reward_statistics['component_stats']:
                self.reward_statistics['component_stats'][component] = {
                    'mean': 0.0,
                    'std': 1.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0
                }
                
            comp_stats = self.reward_statistics['component_stats'][component]
            comp_count = comp_stats['count']
            new_comp_count = comp_count + count
            
            # Update component mean
            comp_mean = comp_stats['mean']
            comp_stats['mean'] = (comp_mean * comp_count + mean * count) / new_comp_count
            
            # Update component min/max
            comp_stats['min'] = min(comp_stats['min'], min_reward) if comp_count > 0 else min_reward
            comp_stats['max'] = max(comp_stats['max'], max_reward) if comp_count > 0 else max_reward
            
            # Update component count
            comp_stats['count'] = new_comp_count
            
            # Update component std (approximate)
            comp_std = comp_stats['std']
            comp_stats['std'] = ((comp_std**2 * comp_count + std**2 * count) / new_comp_count)**0.5
            
        # Save updated statistics
        self._save_statistics()
        
    def get_calibration_params(self, 
                              component: Optional[str] = None) -> Dict[str, float]:
        """Get calibration parameters based on collected statistics.
        
        Args:
            component: Optional component name for specialized calibration
            
        Returns:
            Dict: Calibration parameters
        """
        if component and component in self.reward_statistics.get('component_stats', {}):
            stats = self.reward_statistics['component_stats'][component]
        else:
            stats = self.reward_statistics
            
        # Extract parameters (with reasonable defaults)
        params = {
            'mean': stats.get('lifetime_mean', 0.0),
            'std': max(stats.get('lifetime_std', 1.0), 0.1),  # Avoid zero std
            'min': stats.get('lifetime_min', 0.0),
            'max': stats.get('lifetime_max', 1.0)
        }
        
        # Ensure min != max
        if params['min'] == params['max']:
            params['max'] = params['min'] + 1.0
            
        return params
        
    def get_component_weights(self) -> Dict[str, float]:
        """Get suggested weights for reward components based on statistics.
        
        Returns:
            Dict: Component weights
        """
        # Initialize with default weights
        weights = {'base': 1.0}
        
        # Return default if no component stats
        if 'component_stats' not in self.reward_statistics:
            return weights
            
        # Get all components
        components = self.reward_statistics['component_stats'].keys()
        
        # Compute weights based on relative std
        total_std = 0.0
        std_values = {}
        
        for comp in components:
            stats = self.reward_statistics['component_stats'][comp]
            std = stats.get('std', 1.0)
            total_std += std
            std_values[comp] = std
            
        # Normalize weights inversely proportional to std
        # Components with high variance get lower weight
        if total_std > 0:
            for comp in components:
                weights[comp] = 1.0 / (std_values[comp] / total_std)
                
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
                
        return weights


class RewardCalibrator:
    """Calibrates and normalizes rewards for reinforcement learning.
    
    This class combines normalization, scaling, and shaping to produce
    well-calibrated rewards for stable training.
    """
    
    def __init__(self, 
                config: Dict[str, Any] = None,
                window_size: int = 1000,
                use_running_normalization: bool = True,
                use_percentile_scaling: bool = False,
                clip_range: float = 5.0):
        """Initialize reward calibrator.
        
        Args:
            config: Configuration dictionary
            window_size: Size of history window
            use_running_normalization: Whether to normalize with running stats
            use_percentile_scaling: Whether to use percentile-based scaling
            clip_range: Range to clip rewards to
        """
        self.config = config or {}
        self.calibration_config = self.config.get('reward_calibration', {})
        
        # Extract parameters from config or use defaults
        self.window_size = self.calibration_config.get('window_size', window_size)
        self.use_running_normalization = self.calibration_config.get(
            'use_running_normalization', use_running_normalization)
        self.use_percentile_scaling = self.calibration_config.get(
            'use_percentile_scaling', use_percentile_scaling)
        self.clip_range = self.calibration_config.get('clip_range', clip_range)
        
        # Initialize components
        self.normalizer = RewardNormalizer(window_size=self.window_size, 
                                          clip_range=self.clip_range)
        self.scaler = RewardScaler(window_size=self.window_size)
        self.shaper = RewardShaper()
        
        # Statistics
        self.raw_rewards = []
        self.processed_rewards = []
        self.max_history = 10000
        
        logger.info("Initialized reward calibrator")
        
    def process_reward(self, 
                      reward: float, 
                      potential: float = None,
                      prev_potential: float = None,
                      state_hash: str = None,
                      time_step: int = 0) -> float:
        """Process a raw reward to produce calibrated reward.
        
        Args:
            reward: Raw reward value
            potential: Current state potential for shaping (optional)
            prev_potential: Previous state potential (optional)
            state_hash: Hash of current state (optional)
            time_step: Current time step (optional)
            
        Returns:
            float: Calibrated reward
        """
        # Store raw reward
        self.raw_rewards.append(reward)
        if len(self.raw_rewards) > self.max_history:
            self.raw_rewards.pop(0)
            
        # Apply processing pipeline
        processed = reward
        
        # 1. Apply shaping if potentials provided
        if potential is not None and prev_potential is not None:
            shaping = self.shaper.potential_based_shaping(
                prev_potential, potential)
            processed += shaping
            
        # 2. Apply contextual shaping if state hash provided
        if state_hash is not None:
            processed = self.shaper.shape_reward(
                processed, state_hash, time_step)
            
        # 3. Apply normalization if enabled
        if self.use_running_normalization:
            processed = self.normalizer.normalize(processed)
            
        # 4. Apply scaling if enabled
        if self.use_percentile_scaling:
            processed = self.scaler.percentile_scale(processed)
        else:
            processed = self.scaler.scale(processed)
            
        # Store processed reward
        self.processed_rewards.append(processed)
        if len(self.processed_rewards) > self.max_history:
            self.processed_rewards.pop(0)
            
        return processed
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about raw and processed rewards.
        
        Returns:
            dict: Statistics about rewards
        """
        stats = {}
        
        # Raw reward stats
        if self.raw_rewards:
            stats['raw'] = {
                'mean': float(np.mean(self.raw_rewards)),
                'std': float(np.std(self.raw_rewards)),
                'min': float(np.min(self.raw_rewards)),
                'max': float(np.max(self.raw_rewards)),
                'median': float(np.median(self.raw_rewards)),
                'count': len(self.raw_rewards)
            }
            
        # Processed reward stats
        if self.processed_rewards:
            stats['processed'] = {
                'mean': float(np.mean(self.processed_rewards)),
                'std': float(np.std(self.processed_rewards)),
                'min': float(np.min(self.processed_rewards)),
                'max': float(np.max(self.processed_rewards)),
                'median': float(np.median(self.processed_rewards)),
                'count': len(self.processed_rewards)
            }
            
        return stats
        
    def reset(self) -> None:
        """Reset calibrator state."""
        self.raw_rewards = []
        self.processed_rewards = []
        self.shaper.reset()
        
    def save_state(self, path: str) -> None:
        """Save calibrator state to disk.
        
        Args:
            path: Base path for saving components
        """
        try:
            # Create directory
            os.makedirs(path, exist_ok=True)
            
            # Save individual components
            self.normalizer.save_state(os.path.join(path, "normalizer.pkl"))
            self.scaler.save_state(os.path.join(path, "scaler.pkl"))
            
            # Save overall statistics
            stats = self.get_stats()
            with open(os.path.join(path, "calibrator_stats.json"), 'w') as f:
                json.dump(stats, f, indent=2)
                
            logger.info(f"Saved reward calibrator state to {path}")
        except Exception as e:
            logger.error(f"Error saving reward calibrator state: {e}")
            
    def load_state(self, path: str) -> None:
        """Load calibrator state from disk.
        
        Args:
            path: Base path for loading components
        """
        try:
            if not os.path.exists(path):
                logger.warning(f"Reward calibrator state directory not found: {path}")
                return
                
            # Load individual components
            self.normalizer.load_state(os.path.join(path, "normalizer.pkl"))
            self.scaler.load_state(os.path.join(path, "scaler.pkl"))
            
            logger.info(f"Loaded reward calibrator state from {path}")
        except Exception as e:
            logger.error(f"Error loading reward calibrator state: {e}") 